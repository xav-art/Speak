r"""
Speak - Privacy-First Voice-to-Text with Local AI Formatting
Author: xav-art (MIT License)
Features:
- 100% Offline & Private: Audio and text never leave your computer.
- Local Speech-to-Text via faster-whisper (CTranslate2 int8 on CPU).
- Local AI Post-Processing via Qwen 2.5 (cleans filler words, handles punctuation, formats bullet points).
- Fully Self-Contained: All models and configuration live within the Speak directory.
- Zero Admin Rights Required: Runs entirely in Windows user-space without firewall alerts.
- Dedicated Hotkey Support: Tap '<' (or '>', '<>', F8, Pause, etc.) to start/stop dictating.
- Universal Text Insertion: Pastes or types seamlessly into any app (Outlook, Word, Teams, Chrome, web apps, terminals).
- Personal Dictionary & Auto-Learning: Your names, jargon and product names are fed into both Whisper and the
  local LLM. Fix a word in any app, select the sentence, tap Ctrl+Alt+L and Speak learns the correction.
- Meeting Mode: Ctrl+Alt+M records the mic and the system audio (Teams, Zoom, in-room), then writes a
  transcript labelled "Me", "Person 1", "Person 2"... plus a summary and action items (speak_meeting.py).
"""

import os
import sys
import logging
import warnings

# Suppress Hugging Face Hub Windows symlink and token warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*HF Hub.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import queue
import threading
import time
import winsound
import numpy as np
import sounddevice as sd
import win32clipboard

# Dynamic directory paths (portable across any user account or path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import speak_text
from speak_dictionary import Dictionary
from speak_meeting import MeetingManager

# The engines, the insertion code and the shared basics live in their own
# modules (speak_engines, speak_insert, speak_base); the names other code and
# the tests use stay importable from speak.
import speak_base
import speak_engines
import speak_insert
from speak_base import ALLOWED_LANGUAGES, LOCAL_LLM_DIR, MODELS_DIR, _kernel32, _user32, log
from speak_engines import (CLEANUP_EXAMPLES, CLEANUP_SYSTEM_PROMPT, LANGUAGE_NOTES, LOCAL_LLM_MODELS,
                           MEETING_TURN_NOTE, WHISPER_MODELS, EnginesMixin, LiveTranscriber, SilenceWatcher,
                           build_cleanup_prompt, download_file, download_models, ensure_local_llm,
                           get_dir_size_str)
from speak_tray import TrayIcon, show_console
from speak_insert import (EXTENDED_KEYS, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_EXTENDEDKEY,
                          KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, PM_REMOVE, VK_C, VK_CONTROL, VK_INSERT, VK_LWIN,
                          VK_MENU, VK_RETURN, VK_RWIN, VK_SHIFT, VK_V, InsertionMixin, build_key_inputs,
                          chord, send_keys)
CONFIG_PATH = os.path.join(BASE_DIR, "speak_config.json")

# Windows Virtual Key Codes & Modifiers
WM_HOTKEY = 0x0312
MOD_NONE = 0x0000
MOD_SHIFT = 0x0004
MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
MOD_NOREPEAT = 0x4000

FUNCTION_KEYS = {
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73,
    "F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77,
    "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "PAUSE": 0x13, "CAPSLOCK": 0x14, "INSERT": 0x2D,
    "SCROLLLOCK": 0x91,
}

def prepare_console():
    """Window title 'Speak', and QuickEdit off: in the classic console, a stray
    click selects text and freezes all output, which stalls dictation until
    the user presses a key. (Windows Terminal is not affected.)"""
    try:
        _kernel32.SetConsoleTitleW("Speak - Local Voice Dictation")
        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080
        handle = _kernel32.GetStdHandle(STD_INPUT_HANDLE)
        mode = wintypes.DWORD()
        if _kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            _kernel32.SetConsoleMode(handle, (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS)
    except Exception:
        pass

_CONFIG_LOCK = threading.Lock()
_CONSOLE_HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
_kernel32.SetConsoleCtrlHandler.argtypes = [_CONSOLE_HANDLER, wintypes.BOOL]
_kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL


def load_config(path=None):
    path = path or CONFIG_PATH
    defaults = {
        "hotkey": "<",
        "model": "base",
        "language": None,
        "ai_mode": "local",
        "local_model": "1.5b",
        "paste_mode": "auto",
        "learn_hotkey": "CTRL+ALT+L",
        "whisper_threads": 4,
        "llm_threads": 6,
        "speculative_decoding": False,
        "llm_fast_path": True,
        "live_transcription": True,
        "hotkey_mode": "toggle",
        "auto_stop_silence_seconds": 0,
        "tray_icon": True,
        "language_check_short_phrases": True,
        "warm_up": True,
        "high_priority": True,
        "meeting_hotkey": "CTRL+ALT+M",
        "meeting_model": "small",
        "meeting_folder": None,
        "meeting_cleanup": True,
        "meeting_summary": True,
        "meeting_keep_audio": False,
        "meeting_mic": None,
        "meeting_speaker": None,
        "meeting_me_threshold": 0.5,
        "meeting_cluster_threshold": 0.5,
        "meeting_merge_threshold": 0.6,
        "meeting_num_speakers": None,
        "meeting_max_minutes": 240
    }
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON value is not an object")
            defaults.update(data)
        except Exception as e:
            # Never silently discard the user's settings: the caller saves the
            # config right after loading, which would overwrite the broken file.
            backup = f"{path}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
            try:
                os.replace(path, backup)
                log(f"[Config Warning] {os.path.basename(path)} could not be read ({e}).")
                log(f"                 Your old file was kept as: {backup}")
                log("                 Speak is starting with default settings.")
            except Exception as be:
                log(f"[Config Warning] {os.path.basename(path)} could not be read ({e}) "
                    f"and could not be backed up ({be}).")
    return defaults

def save_config(cfg, path=None):
    """Atomic write (tmp file + os.replace) so a crash or a concurrent save from
    the transcription thread can never leave a half-written config behind."""
    path = path or CONFIG_PATH
    tmp = path + ".tmp"
    with _CONFIG_LOCK:
        try:
            snapshot = dict(cfg)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            log(f"[Config Error] Could not save config: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

class SpeakApp(EnginesMixin, InsertionMixin):

    def __init__(self, config, load_speech_model=True):
        """load_speech_model=False (--transcribe-meeting) skips the dictation
        Whisper model: meetings load their own meeting_model."""
        self.config = config
        self.model_size = config.get("model", "base")
        self._auto_stop_requested = False

        # Speak idles at 0% CPU but needs the whole machine for ~2s after each
        # dictation. ABOVE_NORMAL keeps Chrome/Teams/Outlook from stealing those
        # cycles. (Not REALTIME/HIGH - those can starve the audio driver.)
        if config.get("high_priority", True):
            try:
                ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
                _kernel32.SetPriorityClass(_kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)
            except Exception:
                pass
        self.hotkey_req = self._normalize_key(config.get("hotkey", "<"))
        self.language = config.get("language", None)
        self.paste_mode = config.get("paste_mode", "auto").lower()
        self.sample_rate = 16000

        self.is_recording = False
        self.audio_frames = []
        self.record_stream = None
        self.lock = threading.Lock()
        self.registered_ids = []

        self.running = True
        self.hotkey_rebind_event = threading.Event()
        self.hotkey_vks = {}       # dictation hotkey id -> virtual key, for hold to talk
        self.tray = None
        self.silence = None
        self._holding = False      # the hold-to-talk key is down
        self._shut_down = False
        self._shutdown_done = threading.Event()

        # Engines (Whisper + LLM) are used by exactly one thread at a time:
        # the transcription worker, or the console when it swaps a model.
        # RLock so a menu action holding the lock can call init_ai().
        self.engine_lock = threading.RLock()
        # Dictations are processed one after another, in the order spoken.
        # Previously each one got its own thread, so a second dictation started
        # during processing ran Whisper/llama concurrently (llama.cpp is not
        # thread-safe) and could insert text out of order.
        self.jobs = queue.Queue()
        self.worker_thread = None
        # Serializes clipboard borrowing (paste + restore vs. learn hotkey copy).
        self.clipboard_lock = threading.Lock()
        # The clipboard restore that runs shortly after a paste (see
        # _schedule_clipboard_restore). Guarded by clipboard_lock.
        self._pending_restore = None
        self._pending_restore_timer = None

        # Personal dictionary + auto-learning from corrections
        self.dictionary = Dictionary(BASE_DIR)
        self.learn_hotkey_req = str(config.get("learn_hotkey", "CTRL+ALT+L")).upper().replace(" ", "")
        self.last_inserted_text = None
        self.last_language = None

        # Meeting mode: Ctrl+Alt+M records a meeting, transcribed with speaker labels
        self.meetings = MeetingManager(self, BASE_DIR, MODELS_DIR, download_file, log, ALLOWED_LANGUAGES)

        # Ensure model directories exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LOCAL_LLM_DIR, exist_ok=True)

        self.model_dir_name = f"models--Systran--faster-whisper-{self.model_size}"
        self.model_storage_path = os.path.join(MODELS_DIR, self.model_dir_name)

        # Initialize AI Formatting (Local or Cloud)
        self.init_ai()

        # Initialize Local Faster-Whisper using self-contained models directory
        self.whisper_model = None
        if load_speech_model:
            log(f"[Whisper] Loading local speech model 'faster-whisper-{self.model_size}' on CPU (int8)...")
            log(f"[Whisper] Storage Path: {self.model_storage_path}")
            self.whisper_model = self._load_whisper(self.model_size)
            log(f"[Whisper] Model 'faster-whisper-{self.model_size}' is ready.")

        if self.config.get("warm_up", True) and self.whisper_model is not None:
            self.warm_up()

        self.worker_thread = threading.Thread(target=self._worker_loop, name="speak-worker", daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Single consumer: processes queued dictations strictly in order."""
        while True:
            audio_data = self.jobs.get()
            try:
                if audio_data is None:   # shutdown sentinel
                    return
                self._process_worker(audio_data)
            except Exception as e:
                log(f"[Error] Worker failure: {e}")
            finally:
                self.jobs.task_done()
                self.refresh_tray()

    def _normalize_key(self, key_str):
        k = str(key_str).strip().lstrip("^").strip()
        if k in ("<", "LESS"):
            return "<"
        elif k in (">", "GREATER"):
            return ">"
        elif k in ("<>", "><", "BOTH"):
            return "<>"
        return k.upper()

    def play_sound(self, tone_type):
        try:
            if tone_type == "start":
                winsound.Beep(1200, 90)
            elif tone_type == "stop":
                winsound.Beep(950, 70)
                time.sleep(0.03)
                winsound.Beep(1350, 70)
            elif tone_type == "success":
                winsound.Beep(1600, 110)
            elif tone_type == "error":
                winsound.Beep(500, 250)
        except Exception:
            pass

    MAX_RECORDING_SECONDS = 300  # 5 minutes safety threshold

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            if status.input_overflow:
                log("[Audio Warning] Input overflow (system busy; some audio samples may have dropped).")
        if self.is_recording:
            self.audio_frames.append(indata.copy())
            # Auto-stop if recording is left active indefinitely
            # Blocks keep arriving until the stop thread has closed the
            # stream: start that thread once, not once per block.
            if (len(self.audio_frames) * frames / self.sample_rate > self.MAX_RECORDING_SECONDS
                    and not self._auto_stop_requested):
                self._auto_stop_requested = True
                log(f"[Audio Info] Maximum recording duration ({self.MAX_RECORDING_SECONDS}s) reached. Stopping automatically.")
                threading.Thread(target=self.stop_recording, daemon=True).start()

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        with self.lock:
            if self.is_recording:
                return
            self.audio_frames = []
            self._auto_stop_requested = False
            self.is_recording = True
            self.live = None
            try:
                self.record_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self.audio_callback
                )
                self.record_stream.start()
                if self.config.get("live_transcription", True) and self.whisper_model is not None:
                    # Long dictations: pieces are transcribed while the user speaks.
                    self.live = LiveTranscriber(self, self.audio_frames, self.sample_rate)
                    self.live.start()
                self.silence = None
                silence = self._auto_stop_seconds()
                if silence:
                    self.silence = SilenceWatcher(self.audio_frames, silence, self._stop_on_silence, self.sample_rate)
                    self.silence.start()
            except Exception as e:
                self.is_recording = False
                log(f"[Audio Error] Could not open microphone: {e}")
                log("              Check that a microphone is connected, and that Windows Settings > Privacy & security >")
                log("              Microphone > 'Let desktop apps access your microphone' is turned on.")
                threading.Thread(target=self.play_sound, args=("error",), daemon=True).start()
                return

        threading.Thread(target=self.play_sound, args=("start",), daemon=True).start()
        self.refresh_tray()
        if self.config.get("hotkey_mode") == "hold":
            log("\n[LISTENING...] Speak now... (Release the hotkey to insert; after a short tap, tap again)")
        else:
            log("\n[LISTENING...] Speak now... (Tap hotkey again to stop and insert)")

    def stop_recording(self):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            try:
                self.record_stream.stop()
                self.record_stream.close()
            except Exception:
                pass
            captured = list(self.audio_frames)
            live, self.live = self.live, None
            if self.silence is not None:
                self.silence.stop_event.set()
                self.silence = None

        threading.Thread(target=self.play_sound, args=("stop",), daemon=True).start()

        if not captured:
            self.refresh_tray()
            if live:
                live.stop_event.set()
            log("[Warning] No audio recorded.")
            return

        audio_data = np.concatenate(captured, axis=0).flatten()
        duration = len(audio_data) / self.sample_rate

        if duration < 0.35:
            if live:
                live.stop_event.set()
            log(f"[Ignored] Audio too short ({duration:.2f}s).")
            self.refresh_tray()
            return

        pending = self.jobs.unfinished_tasks
        if pending:
            log(f"[QUEUED] {duration:.1f}s of speech - will be inserted after the {pending} dictation(s) still processing.")
        else:
            log(f"[TRANSCRIBING] Processing {duration:.1f}s of speech using faster-whisper-{self.model_size}...")
        self.jobs.put((audio_data, live))
        self.refresh_tray()

    def _auto_stop_seconds(self):
        try:
            seconds = float(self.config.get("auto_stop_silence_seconds") or 0)
        except (TypeError, ValueError):
            return 0
        return seconds if seconds >= 1 else 0   # under a second would cut off normal pauses

    def _stop_on_silence(self, watcher):
        """The SilenceWatcher's call. Not while the hold-to-talk key is held
        (the user is still talking, or pausing on purpose), and only for the
        dictation that watcher belongs to."""
        if self._holding:
            return False
        with self.lock:
            if self.silence is not watcher or not self.is_recording:
                return True   # that dictation has ended: nothing to stop
        log(f"[Auto-stop] {self._auto_stop_seconds():g} s without speech: stopping.")
        threading.Thread(target=self.stop_recording, daemon=True).start()
        return True

    HOLD_SECONDS = 0.5   # held longer than this: release stops (hold to talk); shorter: a tap

    def hold_to_talk(self, vk):
        """hotkey_mode "hold": recording runs while the key is held. A short
        tap still starts a dictation that the next tap stops, so tapping
        works as before. Runs on its own thread per key press."""
        if self.is_recording:
            self.stop_recording()   # the tap that ends a tapped dictation
            return
        pressed = time.monotonic()
        self.start_recording()
        # Released while the microphone was opening (a slow device can take
        # half a second): that was a tap, whatever the clock says.
        if vk is None or not self.is_recording:
            return
        if not _user32.GetAsyncKeyState(vk) & 0x8000:
            if time.monotonic() - pressed >= self.HOLD_SECONDS:
                log("[Hotkey] The microphone took a moment to open, so this counts as a tap: "
                    "tap again to insert.")
            return
        self._holding = True
        try:
            while self.is_recording and _user32.GetAsyncKeyState(vk) & 0x8000:
                time.sleep(0.02)
        finally:
            self._holding = False
        if self.is_recording and time.monotonic() - pressed >= self.HOLD_SECONDS:
            self.stop_recording()

    def refresh_tray(self):
        """Shows what Speak is doing: recording, a meeting, working, or ready."""
        tray = self.tray
        if tray is None:
            return
        if self.is_recording:
            state = "recording"
        elif self.meetings.is_recording:
            state = "meeting"
        elif self.jobs.unfinished_tasks or self.meetings.jobs.unfinished_tasks:
            state = "working"
        else:
            state = "idle"
        tray.set_state(state)

    def _process_worker(self, job):
        audio_data, live = job if isinstance(job, tuple) else (job, None)
        t0 = time.time()
        try:
            # 1. Whisper. A long dictation was partly transcribed while it was
            #    spoken: finish() does the rest (and takes engine_lock itself).
            pieces = cleaned = stt = None
            if live is not None:
                raw_text, target_lang, lang_prob, cleaned = live.finish(audio_data)
                pieces, stt = live.during, live.stt_seconds
            # Steps 1 and 2 use the shared engines. Insertion does not, so a
            # meeting being cleaned up does not wait for the paste.
            with self.engine_lock:
                if live is None:
                    raw_text, target_lang, lang_prob = self.transcribe(audio_data)
                t_whisper = time.time() - t0 if stt is None else stt

                if not raw_text:
                    log("[Info] No speech detected.")
                    return

                self.last_language = target_lang
                if self.config.get("last_language") != target_lang:
                    self.config["last_language"] = target_lang
                    save_config(self.config)

                lang_name = ALLOWED_LANGUAGES.get(target_lang, target_lang.upper())
                in_pieces = f", after {pieces} piece(s) done while you spoke" if pieces else ""
                log(f"[Whisper STT {t_whisper:.2f}s{in_pieces}] [{lang_name} ({target_lang}) - {lang_prob*100:.0f}% confidence]: \"{raw_text}\"")

                # 2. Dictionary + Local AI Formatting & Cleanup (done piece by
                #    piece while recording, for a long dictation)
                if cleaned is not None:
                    final_text = cleaned
                    if self.ai_mode != "off" and self.local_llm:
                        log(f"[AI Formatted, piece by piece]: \"{final_text}\"")
                else:
                    final_text = self.clean_text(raw_text, target_lang)
                if not final_text:
                    log("[Info] Only filler words were heard. Nothing to insert.")
                    return

            # 3. Universal text insertion into active window
            with self.clipboard_lock:
                restore_token = self.insert_text(final_text)
                self.last_inserted_text = final_text
                elapsed = time.time() - t0
                threading.Thread(target=self.play_sound, args=("success",), daemon=True).start()
                log(f"[DONE] Inserted in {elapsed:.2f}s! [Ready for next dictation] "
                    f"(Fix a word? Select the sentence and tap {self.learn_hotkey_req} to teach Speak.)\n")
                # Put the user's own clipboard back once the target app has
                # pasted, without holding up the next dictation meanwhile.
                self._schedule_clipboard_restore(restore_token)

            # Usage counts rank the user's words for the Whisper prompt. This
            # writes the dictionary file, so it runs once the text is in and
            # holds neither lock. (Meeting turns are not counted.)
            self.dictionary.note_usage(final_text)

        except Exception as e:
            log(f"[Error] Transcription failed: {e}")
            threading.Thread(target=self.play_sound, args=("error",), daemon=True).start()

    LEARN_HOTKEY_ID = 9
    MEETING_HOTKEY_ID = 10

    def _parse_combo(self, combo):
        """'CTRL+ALT+L' / 'CTRL+SHIFT+F9' -> (modifier flags, virtual key) or None."""
        parts = [p for p in combo.upper().split("+") if p]
        if not parts:
            return None
        mods = 0
        for p in parts[:-1]:
            if p in ("CTRL", "CONTROL"):
                mods |= MOD_CONTROL
            elif p == "ALT":
                mods |= MOD_ALT
            elif p == "SHIFT":
                mods |= MOD_SHIFT
        key = parts[-1]
        if key in FUNCTION_KEYS:
            return mods, FUNCTION_KEYS[key]
        if len(key) == 1 and key.isalnum():
            return mods, ord(key)
        return None

    def learn_from_selection(self):
        """
        Called on the learn hotkey. The user has fixed the inserted text in their
        app and selected the corrected sentence. Copy it, diff it against what
        Speak inserted, and remember any corrected words.
        """
        with self.clipboard_lock:
            self._learn_from_selection()

    def _learn_from_selection(self):
        if not self.last_inserted_text:
            log("[Learn] Nothing dictated yet - dictate first, fix the text, select it, then tap the learn hotkey.")
            threading.Thread(target=self.play_sound, args=("error",), daemon=True).start()
            return

        # The user may still be holding Ctrl+Alt from the hotkey: wait for the
        # release, or the app receives Ctrl+Alt+C instead of Ctrl+C.
        self._release_modifiers(timeout=0.6)
        inherited = self._take_pending_restore()
        snapshot = inherited[0] if inherited else self._clipboard_snapshot()
        seq_before = _user32.GetClipboardSequenceNumber()
        send_keys(chord(VK_CONTROL, VK_C))

        # Wait for the app to actually copy (sequence number changes). Without
        # this, an empty selection made Speak diff stale clipboard contents.
        deadline = time.time() + 0.6
        time.sleep(0.15)
        while _user32.GetClipboardSequenceNumber() == seq_before and time.time() < deadline:
            time.sleep(0.03)
        copied = _user32.GetClipboardSequenceNumber() != seq_before

        corrected = None
        if copied:
            time.sleep(0.03)   # let the source app finish writing all formats
            if self._open_clipboard():
                try:
                    if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                        corrected = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
                except Exception:
                    corrected = None
                finally:
                    try:
                        win32clipboard.CloseClipboard()
                    except Exception:
                        pass
        if copied or inherited is not None:
            # The selection was only borrowed, or a dictation was still on the
            # clipboard waiting for its restore: give the user their clipboard back.
            self._restore_clipboard((snapshot, _user32.GetClipboardSequenceNumber(), time.time()), delay=0)

        if not corrected or not corrected.strip():
            log("[Learn] Nothing was copied - select the corrected text first, then tap the learn hotkey.")
            threading.Thread(target=self.play_sound, args=("error",), daemon=True).start()
            return

        added = self.dictionary.learn_from_correction(self.last_inserted_text, corrected)
        if added:
            saved = ("NOT saved: the dictionary file could not be read at startup, restart Speak"
                     if self.dictionary.read_only else "saved to personal dictionary")
            for heard, new in added:
                log(f"[Learned] '{heard}' -> '{new}'  ({saved}, {len(self.dictionary)} words)")
            log("          Whisper will now be primed with this spelling; the mishearing is auto-replaced once confirmed.")
            threading.Thread(target=self.play_sound, args=("success",), daemon=True).start()
        else:
            log("[Learn] No new words found in the selection (no changes, only common words, or a rewrite rather than a fix).")
            log("        Tip: you can always add words manually with the 'd' command.")
            threading.Thread(target=self.play_sound, args=("error",), daemon=True).start()

    def register_hotkeys(self):
        self.registered_ids = []
        hotkeys_to_bind = []

        if self.hotkey_req == "<":
            # Plain tap on '< >' key (0xE2) without Shift
            hotkeys_to_bind.append((1, MOD_NONE | MOD_NOREPEAT, 0xE2, "Tap '<' key (no shift)"))
            # Shift + Comma on US layouts
            hotkeys_to_bind.append((2, MOD_SHIFT | MOD_NOREPEAT, 0xBC, "Shift + Comma (US '<')"))
        elif self.hotkey_req == ">":
            # Shift + '< >' key (0xE2)
            hotkeys_to_bind.append((1, MOD_SHIFT | MOD_NOREPEAT, 0xE2, "Shift + '<' key (produces '>')"))
            # Shift + Period on US layouts
            hotkeys_to_bind.append((2, MOD_SHIFT | MOD_NOREPEAT, 0xBE, "Shift + Period (US '>')"))
        elif self.hotkey_req == "<>":
            # Both plain tap '<' and Shift tap '>'
            hotkeys_to_bind.append((1, MOD_NONE | MOD_NOREPEAT, 0xE2, "Plain '<' key"))
            hotkeys_to_bind.append((2, MOD_SHIFT | MOD_NOREPEAT, 0xE2, "Shift + '<' key ('>')"))
            hotkeys_to_bind.append((3, MOD_SHIFT | MOD_NOREPEAT, 0xBE, "Shift + Period (US '>')"))
        elif self.hotkey_req in FUNCTION_KEYS:
            vk = FUNCTION_KEYS[self.hotkey_req]
            hotkeys_to_bind.append((1, MOD_NONE | MOD_NOREPEAT, vk, f"Key {self.hotkey_req}"))
        else:
            if len(self.hotkey_req) == 1:
                vk = ord(self.hotkey_req.upper())
                hotkeys_to_bind.append((1, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, vk, f"Ctrl+Alt+{self.hotkey_req.upper()}"))

        # Learn hotkey (default Ctrl+Alt+L): teach Speak a corrected word
        learn = self._parse_combo(self.learn_hotkey_req)
        if learn:
            mods, vk = learn
            hotkeys_to_bind.append((self.LEARN_HOTKEY_ID, mods | MOD_NOREPEAT, vk, f"Learn correction ({self.learn_hotkey_req})"))

        # Meeting hotkey (default Ctrl+Alt+M): start / stop recording a meeting
        meeting = self._parse_combo(self.meetings.hotkey)
        if meeting:
            mods, vk = meeting
            hotkeys_to_bind.append((self.MEETING_HOTKEY_ID, mods | MOD_NOREPEAT, vk, f"Record meeting ({self.meetings.hotkey})"))

        success_count = 0
        self.hotkey_vks = {}
        for hid, mods, vk_code, desc in hotkeys_to_bind:
            if _user32.RegisterHotKey(None, hid, mods, vk_code):
                self.hotkey_vks[hid] = vk_code
                if hid in (self.LEARN_HOTKEY_ID, self.MEETING_HOTKEY_ID):
                    self.registered_ids.append(hid)
                    log(f"[Hotkey Registered] {desc} -> ACTIVE")
                    continue
                self.registered_ids.append(hid)
                success_count += 1
                log(f"[Hotkey Registered] {desc} -> ACTIVE")
            else:
                err = ctypes.get_last_error()
                log(f"[Hotkey Warning] Could not bind {desc} (WinError {err} - key may be held by another app).")

        if success_count == 0:
            log(f"[Hotkey Error] Failed to bind '{self.hotkey_req}'. Falling back to F8.")
            self.hotkey_req = "F8"
            f8_vk = FUNCTION_KEYS["F8"]
            if _user32.RegisterHotKey(None, 1, MOD_NONE | MOD_NOREPEAT, f8_vk):
                self.registered_ids.append(1)
                self.hotkey_vks[1] = f8_vk
                log("[Hotkey Registered] Fallback to F8 -> ACTIVE")

    def unregister_hotkeys(self):
        for hid in self.registered_ids:
            try:
                _user32.UnregisterHotKey(None, hid)
            except Exception:
                pass
        self.registered_ids = []

    def set_hotkey(self, new_key):
        normalized = self._normalize_key(new_key)
        self.hotkey_req = normalized
        self.config["hotkey"] = normalized
        save_config(self.config)
        self.hotkey_rebind_event.set()
        log(f"\n[HOTKEY UPDATED] New hotkey is: [{self.hotkey_req}] (Saved to config)")

    def set_paste_mode(self, new_mode):
        mode = new_mode.lower().strip()
        if mode in ("auto", "typing", "clipboard"):
            self.paste_mode = mode
            self.config["paste_mode"] = mode
            save_config(self.config)
            log(f"\n[PASTE MODE UPDATED] New paste mode: [{self.paste_mode.upper()}] (Saved to config)")
        else:
            log("Invalid mode. Choose 'auto', 'typing', or 'clipboard'.")

    def _listener_loop(self):
        self.register_hotkeys()
        if self.config.get("tray_icon", True):
            # On this thread: its message loop delivers the icon's clicks.
            self.tray = TrayIcon({
                "dictation": lambda: threading.Thread(target=self.toggle_recording, daemon=True).start(),
                "meeting": lambda: threading.Thread(target=self.meetings.toggle, daemon=True).start(),
                "show": show_console,
                "quit": self._quit_from_tray,
                "status": lambda: (self.is_recording, self.meetings.is_recording),
            })
            if not self.tray.create():
                self.tray = None
        msg = wintypes.MSG()
        while self.running:
            if self.hotkey_rebind_event.is_set():
                self.hotkey_rebind_event.clear()
                self.unregister_hotkeys()
                self.register_hotkeys()

            # Handle every queued message, not one per 20 ms.
            while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == WM_HOTKEY:
                    if msg.wParam == self.LEARN_HOTKEY_ID:
                        threading.Thread(target=self.learn_from_selection, daemon=True).start()
                    elif msg.wParam == self.MEETING_HOTKEY_ID:
                        threading.Thread(target=self.meetings.toggle, daemon=True).start()
                    elif self.config.get("hotkey_mode") == "hold":
                        vk = self.hotkey_vks.get(msg.wParam)
                        threading.Thread(target=self.hold_to_talk, args=(vk,), daemon=True).start()
                    else:
                        self.toggle_recording()
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))
            time.sleep(0.02)
        self.unregister_hotkeys()
        if self.tray is not None:
            self.tray.destroy()

    def _quit_from_tray(self):
        """The tray menu's Quit: the same shutdown as typing q. The console
        is waiting in input(), which cannot be woken cleanly, so the process
        ends here once everything is saved."""
        log("\nQuitting (from the tray icon)...")
        self.shutdown()
        os._exit(0)

    def get_ai_status_str(self):
        if self.ai_mode == "local" and self.local_llm:
            m_info = LOCAL_LLM_MODELS.get(self.local_model_choice, LOCAL_LLM_MODELS["1.5b"])
            return f"LOCAL ({m_info['name']} - 100% Offline, CPU)"
        elif self.ai_mode == "off":
            return "OFF (Raw Whisper verbatim transcription)"
        else:
            return "OFFLINE (Local LLM not loaded - raw Whisper verbatim transcription)"

    def print_status_banner(self):
        size_str = get_dir_size_str(self.model_storage_path)
        ai_status = self.get_ai_status_str()

        log("\n" + "=" * 68)
        log("                   SPEAK - LOCAL VOICE-TO-TEXT                 ")
        log("=" * 68)
        if self.config.get("hotkey_mode") == "hold":
            log(f"Active Hotkey  : [{self.hotkey_req}] (Hold while you speak, or tap to start and tap to stop)")
        else:
            log(f"Active Hotkey  : [{self.hotkey_req}] (Tap to start, Tap to stop & paste)")
        log(f"Whisper Model  : faster-whisper-{self.model_size} (Local CPU int8)")
        log(f"Whisper Path   : {self.model_storage_path} ({size_str})")
        log(f"AI Formatting  : {ai_status}")
        if self.language and self.language in ALLOWED_LANGUAGES:
            lang_str = f"{ALLOWED_LANGUAGES[self.language]} ({self.language})"
        else:
            lang_str = "Auto-detect (Restricted to: English, Danish, German, French)"
        log(f"Language       : {lang_str}")
        log(f"Paste Mode     : {self.paste_mode.upper()} (Auto handles Terminals & Apps)")
        voice = "voice sample recorded" if os.path.exists(self.meetings.profile_path) else "no voice sample yet (type r, then 2)"
        log(f"Meetings       : [{self.meetings.hotkey}] to record, {voice}")
        log("Dictation      : [ACTIVE SYSTEM-WIDE] Tap your hotkey to dictate.")
        log("Keep this window open while you use Speak (minimize it). Close it to stop Speak.")
        log("-" * 68)
        log("Interactive Commands:")
        log("  h  - Change Hotkey (e.g. <, >, <>, F8, F9, Pause, CapsLock)")
        log("  a  - AI Formatting (switch local models, toggle off, or test)")
        log("  d  - Personal Dictionary (add, remove, view, or import words)")
        log("  p  - Change Paste Mode (auto, typing, clipboard)")
        log("  m  - Change Whisper Model (medium, base, small, tiny)")
        log("  l  - Change Language (en, da, de, fr, or auto)")
        log("  r  - Meetings (record, voice sample, transcribe a file)")
        log("  q  - Quit Application")
        log("-" * 68)

    def print_welcome(self):
        key = self.hotkey_req
        log("")
        log("*" * 68)
        log("  WELCOME TO SPEAK - here is how to dictate:")
        log("")
        log("  1. Click where you want to write (an email, a document, a chat).")
        log(f"  2. Tap the {key} key once and speak." + ("  (It is next to Left Shift.)" if key in ("<", "<>") else ""))
        log(f"  3. Tap {key} again. Your text appears where you clicked.")
        log("")
        log("  You hear a short beep when Speak starts and stops listening.")
        log("  Speak works offline: your voice never leaves this computer.")
        log("*" * 68)
        log("")

    def meetings_menu(self):
        m = self.meetings
        log("\n--- MEETINGS ---")
        log(f"Recording      : {'IN PROGRESS' if m.is_recording else 'off'} (hotkey {m.hotkey})")
        created = m.profile_info()
        log(f"Voice sample   : {'recorded ' + str(created)[:10] if created else 'not recorded (you appear as a numbered person)'}")
        log(f"Transcripts    : {m.output_folder()}")
        if m.current_job or m.jobs.unfinished_tasks:
            log(f"Transcribing   : {m.jobs.unfinished_tasks} meeting(s) in progress or queued")
        leftovers = m.leftover_recordings()
        log("\nOptions:")
        log(f"  1 - {'Stop' if m.is_recording else 'Start'} recording a meeting (same as {m.hotkey})")
        log("  2 - Record my voice sample (30 seconds, so transcripts label you as \"Me\")")
        log("  3 - Transcribe an audio or video file (wav, mp3, m4a, mp4...)")
        if leftovers:
            log(f"  4 - Transcribe {len(leftovers)} recorded meeting(s) that were not transcribed yet")
        log("  5 - Delete my voice sample")
        log("  6 - Open the transcripts folder")
        log("  Enter - Return to main menu")
        sub = input("Choose option [1-6]: ").strip()
        if sub == "1":
            m.toggle()
        elif sub == "2":
            m.enroll_interactive()
        elif sub == "3":
            p = input("Path to the audio or video file: ").strip().strip('"')
            if os.path.isfile(p) or os.path.isdir(p):
                m.submit(p)
                log("[Meeting] Queued. Transcribing in the background; dictation keeps working meanwhile.")
            else:
                log("File not found.")
        elif sub == "4" and leftovers:
            for folder in leftovers:
                m.submit(folder)
            log(f"[Meeting] Queued {len(leftovers)} recording(s) for transcription.")
        elif sub == "5":
            m.delete_profile()
        elif sub == "6":
            os.makedirs(m.output_folder(), exist_ok=True)
            try:
                os.startfile(m.output_folder())
            except Exception as e:
                log(f"Could not open the folder: {e}")

    def _on_console_close(self, event):
        """Windows' console control handler. Closing the window, logging off
        or shutting down ends the process right after this returns: save a
        running meeting and remove the tray icon first. Ctrl+C is left to
        Python (it stops the menu loop)."""
        if event in (2, 5, 6):   # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            try:
                self.shutdown(wait=3.0)   # Windows ends the process about 5 s after this starts
            except Exception:
                pass
            return True
        return False

    def run(self, first_run=False):
        self._console_handler = _CONSOLE_HANDLER(self._on_console_close)   # kept alive: Windows calls it
        _kernel32.SetConsoleCtrlHandler(self._console_handler, True)
        listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        listener_thread.start()

        self.print_status_banner()
        if first_run:
            self.print_welcome()
        leftovers = self.meetings.leftover_recordings()
        if leftovers:
            log(f"[Meeting] {len(leftovers)} recorded meeting(s) not transcribed yet. Type r, then 4, to transcribe them.")

        try:
            while self.running:
                try:
                    cmd = input("Command [h/a/d/p/m/l/r/q] > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break

                if not cmd:
                    continue
                if cmd in ("h", "hotkey"):
                    log("\n--- CHANGE HOTKEY ---")
                    log("Common options:")
                    log("  <   - Plain tap on the key next to Left Shift (no Shift needed)")
                    log("  >   - Shift + the key next to Left Shift")
                    log("  <>  - Either plain tap OR Shift tap triggers dictation")
                    log("  F8, F9, F10, Pause, CapsLock, Insert")
                    new_h = input("Enter new hotkey [< / > / <> / F8 / etc.] (Enter keeps it): ").strip()
                    if new_h:
                        self.set_hotkey(new_h)
                    hold = self.config.get("hotkey_mode") == "hold"
                    log("Hold to talk: hold the hotkey while you speak and release it to insert the text.")
                    log("A short tap still starts a dictation that the next tap ends.")
                    answer = input(f"Hold to talk? [y/n] (now {'on' if hold else 'off'}, Enter keeps it): ").strip().lower()
                    if answer in ("y", "yes", "n", "no"):
                        self.config["hotkey_mode"] = "hold" if answer.startswith("y") else "toggle"
                        save_config(self.config)
                        log(f"[HOTKEY] Hold to talk is {'on' if answer.startswith('y') else 'off'} (saved).")
                    self.print_status_banner()

                elif cmd in ("a", "ai", "format"):
                    log("\n--- AI FORMATTING CONFIGURATION ---")
                    log("Current Status: " + self.get_ai_status_str())
                    log("Local AI cleans verbal fillers ('um', 'uh', 'øh', 'altså'),")
                    log("handles stutters, and formats spoken commands ('bullet point', 'new line')")
                    log("running 100% locally and privately on your CPU.")
                    log("\nOptions:")
                    log("  1 - Local Qwen 2.5 1.5B (Recommended ~2s, best for Danish + English)")
                    log("  2 - Local Qwen 2.5 0.5B (Ultra-fast ~1s, 468 MB)")
                    log("  3 - Turn AI Formatting OFF (Fastest: raw Whisper verbatim)")
                    log("  4 - Test Current AI formatting on sample dictation")
                    log("  Enter - Return to main menu")
                    sub = input("Choose option [1-4]: ").strip()
                    if sub == "1":
                        self.config["ai_mode"] = "local"
                        self.config["local_model"] = "1.5b"
                        save_config(self.config)
                        self.init_ai()
                    elif sub == "2":
                        self.config["ai_mode"] = "local"
                        self.config["local_model"] = "0.5b"
                        save_config(self.config)
                        self.init_ai()
                    elif sub == "3":
                        self.config["ai_mode"] = "off"
                        save_config(self.config)
                        self.init_ai()
                    elif sub == "4":
                        log("\n[Testing] Testing with: 'uh I think we should, bullet point deploy the change, bullet point notify the team'...")
                        with self.engine_lock:   # shared with dictation and meeting cleanup
                            test_out = self.clean_text("uh I think we should, bullet point deploy the change, bullet point notify the team", "en")
                        log(f"[Result]:\n{test_out}")
                    self.print_status_banner()

                elif cmd in ("d", "dict", "dictionary"):
                    log("\n--- PERSONAL DICTIONARY ---")
                    words = self.dictionary.words()
                    if words:
                        entries = sorted(self.dictionary.entries.values(), key=lambda e: e["word"].lower())
                        log(f"{len(words)} words (Whisper and the local AI are told to use these exact spellings):")
                        for e in entries:
                            tag = "learned" if e["source"] == "learned" else "manual"
                            log(f"  {e['word']:<28} [{tag}, used {e['uses']}x]")
                        if self.dictionary.corrections:
                            log("Learned mishearings (auto-replaced once confirmed):")
                            for c in sorted(self.dictionary.corrections.values(), key=lambda c: c["heard"].lower()):
                                log(f"  '{c['heard']}' -> '{c['word']}'  (corrected {c['count']}x)")
                    else:
                        log("Your dictionary is empty. Add names, product names, acronyms and jargon Whisper gets wrong.")
                    log("\nOptions:")
                    log("  a - Add words (comma separated, e.g. 'Annika, Salesforce, Kubernetes, PowerBI')")
                    log("  r - Remove a word")
                    log("  i - Import words from a text file (one per line or comma separated)")
                    log("  c - Clear the whole dictionary")
                    log(f"  Tip: after dictating, fix any wrong word in your app, select the sentence and tap {self.learn_hotkey_req}.")
                    log("  Enter - Return to main menu")
                    if self.dictionary.read_only:
                        log("  Note: the dictionary file could not be read at startup. Changes last until you restart Speak and are not saved.")
                    sub = input("Choose option [a/r/i/c]: ").strip().lower()
                    if sub == "a":
                        raw = input("Words to add: ").strip()
                        added = [w.strip() for w in raw.split(",") if w.strip() and self.dictionary.add(w.strip())]
                        log(f"Added {len(added)} word(s). Dictionary now has {len(self.dictionary)} words.")
                    elif sub == "r":
                        w = input("Word to remove: ").strip()
                        log("Removed." if self.dictionary.remove(w) else "Word not found.")
                    elif sub == "i":
                        p = input("Path to text file: ").strip().strip('"')
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                content = f.read().replace("\n", ",")
                            n = sum(1 for w in content.split(",") if w.strip() and self.dictionary.add(w.strip()))
                            log(f"Imported {n} new word(s).")
                        except Exception as e:
                            log(f"Could not import: {e}")
                    elif sub == "c":
                        if input("Type YES to clear the dictionary: ").strip() == "YES":
                            self.dictionary.clear()
                            log("Dictionary cleared.")
                    self.print_status_banner()

                elif cmd in ("p", "paste", "mode"):
                    log("\n--- CHANGE PASTE MODE ---")
                    log("  auto      - Smart detect (Ctrl+V for apps, direct type/Shift+Insert for terminals)")
                    log("  typing    - Direct Unicode typing (100% universal, works in legacy apps and any terminal)")
                    log("  clipboard - Standard Ctrl+V / Shift+Insert")
                    new_p = input("Enter paste mode [auto/typing/clipboard]: ").strip().lower()
                    if new_p:
                        self.set_paste_mode(new_p)
                    self.print_status_banner()

                elif cmd in ("m", "model"):
                    log("\n--- CHANGE WHISPER MODEL ---")
                    log("Options: medium (high accuracy, recommended), base (faster), small, tiny, large-v3 (most accurate, slow on CPU)")
                    new_m = input("Enter model size [tiny/base/small/medium/large-v3]: ").strip().lower()
                    if new_m in WHISPER_MODELS:
                        target_dir = os.path.join(MODELS_DIR, f"models--Systran--faster-whisper-{new_m}")
                        if not os.path.exists(target_dir):
                            confirm = input(f"Model '{new_m}' is not bundled locally. Download now via Hugging Face? (y/N): ").strip().lower()
                            if confirm != "y":
                                log("Model change canceled.")
                                self.print_status_banner()
                                continue
                        with self.engine_lock:
                            self.model_size = new_m
                            self.config["model"] = new_m
                            self.model_dir_name = f"models--Systran--faster-whisper-{self.model_size}"
                            self.model_storage_path = os.path.join(MODELS_DIR, self.model_dir_name)
                            save_config(self.config)
                            log(f"Reloading Whisper with '{new_m}' model from self-contained directory...")
                            log(f"Path: {self.model_storage_path}")
                            self.whisper_model = self._load_whisper(new_m)
                            log(f"Model 'faster-whisper-{new_m}' is now active!")
                    else:
                        log("Invalid model size.")
                    self.print_status_banner()

                elif cmd in ("l", "lang", "language"):
                    log("\n--- CHANGE LANGUAGE ---")
                    log("Supported languages:")
                    log("  auto - Auto-detect (Restricted to English, Danish, German, French)")
                    log("  en   - English")
                    log("  da   - Danish")
                    log("  de   - German")
                    log("  fr   - French")
                    new_l = input("Enter language code [auto/en/da/de/fr]: ").strip().lower()
                    if new_l in ("auto", "none", "", "all"):
                        self.language = None
                        self.config["language"] = None
                    elif new_l in ALLOWED_LANGUAGES:
                        self.language = new_l
                        self.config["language"] = new_l
                    else:
                        log(f"Invalid language '{new_l}'. Supported choices: auto, en, da, de, fr.")
                        time.sleep(1)
                        continue
                    save_config(self.config)
                    lang_label = ALLOWED_LANGUAGES.get(self.language, "Auto-detect (English, Danish, German, French)")
                    log(f"Language set to: {lang_label}")
                    self.print_status_banner()

                elif cmd in ("r", "rec", "meeting", "meetings"):
                    self.meetings_menu()
                    self.print_status_banner()

                elif cmd in ("q", "quit", "exit"):
                    log("Exiting...")
                    break
                else:
                    log("Unknown command. Type 'h' (hotkey), 'a' (AI formatting), 'd' (dictionary), 'p' (paste mode), 'm' (model), 'l' (language), 'r' (meetings), 'q' (quit).")

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self, wait=30.0):
        """Stops Speak: the tray icon and hotkeys go first (milliseconds), a
        dictation being recorded is kept, a running meeting is saved for
        later, dictations already spoken are typed in (for up to `wait`
        seconds) and the user's clipboard is given back. One shutdown at a
        time: a second caller waits for the first to finish."""
        with self.lock:
            first = not self._shut_down
            self._shut_down = True
        if not first:
            self._shutdown_done.wait(wait + 10)
            return
        try:
            self.running = False
            if self.tray is not None:
                self.tray.remove()   # the window itself belongs to the listener thread
            self.unregister_hotkeys()
            if self.is_recording:
                self.stop_recording()   # what is being said is queued like any dictation
            self.meetings.shutdown()
            deadline = time.monotonic() + wait
            if self.jobs.unfinished_tasks and wait > 0:
                log(f"Finishing {self.jobs.unfinished_tasks} dictation(s) first... (Ctrl+C skips)")
            try:
                while self.jobs.unfinished_tasks and time.monotonic() < deadline:
                    time.sleep(0.1)
                if self._pending_restore is not None:
                    # The target app may not have pasted yet: wait as a normal restore does.
                    time.sleep(self.CLIPBOARD_RESTORE_DELAY)
            except KeyboardInterrupt:
                log("Skipped.")
        finally:
            try:
                self.flush_clipboard_restore()
            finally:
                try:
                    self.jobs.put(None)
                except Exception:
                    pass
                self._shutdown_done.set()
                log("Speak stopped. Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speak - Local Voice Dictation")
    parser.add_argument("--hotkey", default=None,
                        help="Set global hotkey (e.g. '<', '>', '<>', 'F8', 'Pause'). Default reads from config.")
    parser.add_argument("--paste-mode", default=None, choices=["auto", "typing", "clipboard"],
                        help="Paste mode: 'auto', 'typing' (direct typewriter), or 'clipboard'.")
    parser.add_argument("--model", default=None, choices=WHISPER_MODELS,
                        help="Whisper model size. Default reads from config.")
    parser.add_argument("--language", default=None, choices=["en", "da", "de", "fr", "auto"],
                        help="Language code ('en', 'da', 'de', 'fr', 'auto'). Default reads from config.")
    parser.add_argument("--ai-mode", default=None, choices=["local", "off"],
                        help="AI formatting mode: 'local' (100%% offline) or 'off'.")
    parser.add_argument("--download-models", action="store_true",
                        help="Download the models the current settings need, then exit (used by the installer).")
    parser.add_argument("--transcribe-meeting", metavar="PATH", default=None,
                        help="Transcribe a meeting with speaker labels, then exit. PATH is an audio or video "
                             "file, or a Speak recording folder.")
    args = parser.parse_args()

    if args.download_models:
        sys.exit(0 if download_models(load_config()) else 1)

    if args.transcribe_meeting:
        # Runs next to a Speak window too: no hotkeys, no mutex, config not saved.
        cfg = load_config()
        cfg["warm_up"] = False
        app = SpeakApp(cfg, load_speech_model=False)
        try:
            app.meetings.process(os.path.abspath(args.transcribe_meeting))
        except Exception as e:
            log(f"[Meeting Error] {e}")
            sys.exit(1)
        sys.exit(0)

    prepare_console()
    # Single-instance check via Win32 Mutex
    ERROR_ALREADY_EXISTS = 183
    mutex = _kernel32.CreateMutexW(None, False, "Speak_Voice_To_Text_Mutex")
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        log("\n[WARNING] Another instance of Speak is already running!")
        log("Please close the other Speak window to avoid hotkey conflicts.")
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass
        sys.exit(0)

    first_run = not os.path.exists(CONFIG_PATH)
    cfg = load_config()
    if args.hotkey:
        cfg["hotkey"] = args.hotkey
    if args.paste_mode:
        cfg["paste_mode"] = args.paste_mode
    if args.model:
        cfg["model"] = args.model
    if args.language:
        cfg["language"] = None if args.language == "auto" else args.language
    if args.ai_mode:
        cfg["ai_mode"] = args.ai_mode

    save_config(cfg)
    app = SpeakApp(cfg)
    try:
        app.run(first_run=first_run)
    finally:
        if mutex:
            _kernel32.CloseHandle(mutex)
