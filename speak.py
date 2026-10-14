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
CONFIG_PATH = os.path.join(BASE_DIR, "speak_config.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOCAL_LLM_DIR = os.path.join(MODELS_DIR, "llm")

LOCAL_LLM_MODELS = {
    "1.5b": {
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "name": "Qwen 2.5 1.5B (Recommended: Best Danish & English, ~2s)",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "sha256": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        "size": "1.1 GB"
    },
    "0.5b": {
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "name": "Qwen 2.5 0.5B (Ultra-Fast ~1s, 468 MB)",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        "size": "470 MB"
    }
}

ALLOWED_LANGUAGES = {
    "en": "English",
    "da": "Danish",
    "de": "German",
    "fr": "French"
}

def log(msg=""):
    print(msg, flush=True)

# Win32 handles that preserve the thread's last-error code. ctypes.windll +
# kernel32.GetLastError() is unreliable: the interpreter can make other API
# calls in between and overwrite the code, so "already running" and hotkey
# conflict detection could silently report the wrong thing.
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL
_user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
_user32.RegisterHotKey.restype = wintypes.BOOL
_user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.UnregisterHotKey.restype = wintypes.BOOL
_user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
_user32.CountClipboardFormats.restype = ctypes.c_int
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.GetWindowTextLengthW.restype = ctypes.c_int
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetClassNameW.restype = ctypes.c_int
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD

_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
_kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.SetPriorityClass.restype = wintypes.BOOL
_kernel32.SetConsoleTitleW.argtypes = [wintypes.LPCWSTR]
_kernel32.SetConsoleTitleW.restype = wintypes.BOOL
_user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
_user32.PeekMessageW.restype = wintypes.BOOL
_user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
_user32.TranslateMessage.restype = wintypes.BOOL
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
_user32.DispatchMessageW.restype = ctypes.c_ssize_t

# Windows Virtual Key Codes & Modifiers
WM_HOTKEY = 0x0312
PM_REMOVE = 0x0001
MOD_NONE = 0x0000
MOD_SHIFT = 0x0004
MOD_CONTROL = 0x0002
MOD_ALT = 0x0001
MOD_NOREPEAT = 0x4000

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_RETURN = 0x0D
VK_V = 0x56
VK_INSERT = 0x2D
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1

# Windows SendInput 64-bit Structures
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong)
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_ulonglong)]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]

_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT
_user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
_user32.MapVirtualKeyW.restype = wintypes.UINT
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short

KEYEVENTF_EXTENDEDKEY = 0x0001
VK_C = 0x43
VK_LWIN = 0x5B
VK_RWIN = 0x5C
# Keys on the extended part of the keyboard. Without the flag, Insert can
# arrive as the numeric keypad's 0/Ins key.
EXTENDED_KEYS = {VK_INSERT, 0x2E, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28}


def build_key_inputs(steps):
    """[(virtual key, is_key_up)] -> INPUT array for a single SendInput call.
    Scan codes are filled in because some apps (remote desktops, Citrix)
    read those instead of the virtual key."""
    events = (INPUT * len(steps))()
    for event, (vk, key_up) in zip(events, steps):
        flags = (KEYEVENTF_KEYUP if key_up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_KEYS else 0)
        event.type = INPUT_KEYBOARD
        event.u.ki = KEYBDINPUT(vk, _user32.MapVirtualKeyW(vk, 0) & 0xFF, flags, 0, 0)
    return events


def send_keys(steps):
    """Sends a key sequence in one SendInput call, so the user's own key
    presses cannot land in the middle of it. Returns False when Windows did
    not accept it (a locked screen, for example). Input blocked by UIPI (the
    target app runs as administrator) is NOT reported: Windows hides it."""
    events = build_key_inputs(steps)
    return _user32.SendInput(len(events), events, ctypes.sizeof(INPUT)) == len(events)


def chord(*vks):
    """Press keys in order, release in reverse: chord(VK_CONTROL, VK_V) is Ctrl+V."""
    return [(vk, False) for vk in vks] + [(vk, True) for vk in reversed(vks)]

FUNCTION_KEYS = {
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73,
    "F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77,
    "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "PAUSE": 0x13, "CAPSLOCK": 0x14, "INSERT": 0x2D,
    "SCROLLLOCK": 0x91,
}

def get_dir_size_str(path):
    if not os.path.exists(path):
        return "Not downloaded yet"
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    if total > 1024 * 1024 * 1024:
        return f"{total / (1024 * 1024 * 1024):.2f} GB"
    return f"{total / (1024 * 1024):.1f} MB"

def download_file(url, dest, sha256, label):
    """One-time model download with progress, written to <dest>.part and only
    moved into place when the SHA-256 matches. urllib follows the Windows proxy
    settings. Returns True when dest is ready."""
    import hashlib
    import urllib.request
    part = dest + ".part"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Speak"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            digest, done, shown = hashlib.sha256(), 0, -1
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                done += len(block)
                if total:
                    pct = done * 100 // total
                    if pct // 5 != shown:
                        shown = pct // 5
                        print(f"\r[Download] {label}: {pct:3d}% of {total / 1e6:,.0f} MB", end="", flush=True)
        print(flush=True)
        if digest.hexdigest() != sha256:
            raise ValueError("checksum mismatch (the download is incomplete or was altered)")
        os.replace(part, dest)
        return True
    except Exception as e:
        print(flush=True)
        log(f"[Download Error] {label} could not be downloaded: {e}")
        try:
            os.remove(part)
        except OSError:
            pass
        return False

def ensure_local_llm(choice):
    """Path of the Qwen GGUF for this choice, downloading it once if missing.
    Returns None when the model is not available."""
    info = LOCAL_LLM_MODELS.get(choice, LOCAL_LLM_MODELS["1.5b"])
    path = os.path.join(LOCAL_LLM_DIR, info["file"])
    if os.path.exists(path):
        return path
    log(f"[AI] First-time setup: downloading the AI cleanup model ({info['size']}, one time only)...")
    url = f"https://huggingface.co/{info['repo']}/resolve/main/{info['file']}"
    return path if download_file(url, path, info["sha256"], info["file"]) else None

def download_models(cfg):
    """--download-models: fetch everything the current config needs, then exit.
    Used by the installer so the first start is instant."""
    ok = True
    size = cfg.get("model", "base")
    log(f"[Download] Speech model 'faster-whisper-{size}' (Hugging Face)...")
    try:
        from faster_whisper.utils import download_model
        download_model(size, cache_dir=MODELS_DIR)
        log(f"[Download] Speech model 'faster-whisper-{size}' is ready.")
    except Exception as e:
        log(f"[Download Error] Speech model could not be downloaded: {e}")
        ok = False
    if cfg.get("ai_mode", "local") == "local":
        path = ensure_local_llm(cfg.get("local_model", "1.5b"))
        if path:
            log(f"[Download] AI cleanup model is ready: {os.path.basename(path)}")
        else:
            ok = False
    return ok

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

# Whisper sizes offered by the m menu and --model.
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")

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

class SpeakApp:
    def init_ai(self):
        # Serialized with transcription: switching models from the menu waits
        # for a running dictation instead of swapping the LLM mid-inference.
        with self.engine_lock:
            self._init_ai()

    def _init_ai(self, mode=None):
        # mode overrides the configured ai_mode for this session only.
        self.ai_mode = mode or self.config.get("ai_mode", "local")
        if self.ai_mode not in ("local", "off"):
            # Older configs may still say "gemini": cloud mode was removed.
            self.ai_mode = "local"
        self.local_model_choice = self.config.get("local_model", "1.5b")
        self.local_llm = None

        if self.ai_mode == "off":
            log("[AI] AI Formatting is turned OFF (Raw Whisper transcription).")
            return

        if self.ai_mode == "local":
            model_info = LOCAL_LLM_MODELS.get(self.local_model_choice, LOCAL_LLM_MODELS["1.5b"])
            model_path = ensure_local_llm(self.local_model_choice)
            if not model_path:
                log(f"[AI Warning] Local model '{model_info['file']}' is not available in {LOCAL_LLM_DIR}.")
                log("[AI] Falling back to raw Whisper transcription.")
                return
            try:
                import llama_cpp
                log(f"[AI] Loading Local Mini LLM: {model_info['name']} (100% offline on CPU)...")
                t0 = time.time()
                draft_model = None
                if self.config.get("speculative_decoding", False):
                    # Prompt-lookup speculative decoding. Measured 2.5x SLOWER on
                    # CPU with llama-cpp-python 0.3.x (drafts are verified one
                    # by one in Python), so it is off by default. Kept as an
                    # opt-in for future library versions / GPU builds.
                    try:
                        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                        draft_model = LlamaPromptLookupDecoding(num_pred_tokens=10)
                    except Exception:
                        draft_model = None
                self.llm_ctx = 2048
                self.local_llm = llama_cpp.Llama(
                    model_path=model_path,
                    n_ctx=self.llm_ctx,
                    n_threads=int(self.config.get("llm_threads", 4)),
                    n_batch=512,
                    draft_model=draft_model,
                    verbose=False
                )
                log(f"[AI] Local Mini LLM ready in {time.time() - t0:.2f}s! "
                    f"(threads={self.local_llm.n_threads}, speculative={'on' if draft_model else 'off'})")
            except Exception as e:
                log(f"[AI Warning] Failed to initialize local LLM ({e}). Running raw Whisper.")
                self.local_llm = None

    def load_llm(self, n_ctx):
        """A separate Qwen instance with a bigger context, for meeting summaries.
        The GGUF is memory-mapped, so its weights are shared with the dictation
        instance; only the extra KV cache costs RAM. Returns None when AI is off."""
        if self.ai_mode == "off":
            return None
        path = ensure_local_llm(self.local_model_choice)
        if not path:
            return None
        try:
            import llama_cpp
            return llama_cpp.Llama(model_path=path, n_ctx=n_ctx, n_threads=int(self.config.get("llm_threads", 4)),
                                   n_batch=512, verbose=False)
        except Exception as e:
            log(f"[AI Warning] Could not load the summary model ({e}).")
            return None

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

    def _load_whisper(self, size):
        from faster_whisper import WhisperModel
        model = WhisperModel(
            size,
            download_root=MODELS_DIR,
            device="cpu",
            compute_type="int8",
            num_workers=1,
            cpu_threads=int(self.config.get("whisper_threads", 4))
        )
        self._install_whisper_speedups(model)
        return model

    def _install_whisper_speedups(self, model):
        """
        Two small hooks on the faster-whisper instance (faster-whisper 1.2.x):

        1. Encoder cache. With auto language detection faster-whisper runs the
           30s encoder pass twice per dictation: once inside detect_language()
           and once again in generate_segments() for the very same first
           window. Caching the last encoder output by feature fingerprint
           removes the second pass (~0.3s on base, ~1s+ on medium).
        2. detect_language override. Builds the detection segment exactly the
           way generate_segments builds its first segment (content frames,
           zero-padded) so the fingerprints match, and restricts the result to
           the 4 supported languages so a Danish speaker is never mis-detected
           as Norwegian or Swedish.

        Everything falls back to the stock faster-whisper path on any error.
        """
        import hashlib
        try:
            import faster_whisper
            fw_version = getattr(faster_whisper, "__version__", "?")
            if not str(fw_version).startswith("1.2."):
                log(f"[Whisper Warning] Speed patches were tested on faster-whisper 1.2.x, found {fw_version}. "
                    "Pin the version from requirements.txt if transcription misbehaves.")
            from faster_whisper.transcribe import pad_or_trim
        except Exception as e:
            log(f"[Whisper Warning] Speed patches disabled, running stock faster-whisper ({e}).")
            return

        original_encode = model.encode
        original_detect = model.detect_language
        cache = {"key": None, "out": None}

        def cached_encode(features):
            try:
                arr = np.ascontiguousarray(features)
                key = (arr.shape, hashlib.blake2b(arr.tobytes(), digest_size=16).digest())
            except Exception:
                return original_encode(features)
            if cache["key"] == key and cache["out"] is not None:
                return cache["out"]
            out = original_encode(features)
            cache["key"], cache["out"] = key, out
            return out

        real_model = model.model
        allowed_tokens = {f"<|{code}|>": code for code in ALLOWED_LANGUAGES}
        nb_max_frames = model.feature_extractor.nb_max_frames

        def fast_detect_language(audio=None, features=None, **kwargs):
            if features is None:
                return original_detect(audio=audio, features=features, **kwargs)
            try:
                content_frames = features.shape[-1] - 1
                segment = pad_or_trim(features[..., :min(nb_max_frames, content_frames)])
                results = real_model.detect_language(cached_encode(segment))[0]
                probs = [(allowed_tokens[tok], p) for tok, p in results if tok in allowed_tokens]
                if not probs:
                    probs = [(tok[2:-2], p) for tok, p in results]
                language, probability = probs[0]
                return language, probability, probs
            except Exception:
                return original_detect(audio=audio, features=features, **kwargs)

        model.encode = cached_encode
        model.detect_language = fast_detect_language
        model._speak_encode_cache = cache

    def warm_up(self):
        """First inference after loading is 2-3x slower (page-in, thread spin-up).
        Do it now, on silence, so the user's first dictation is already fast."""
        t0 = time.time()
        try:
            silence = np.zeros(16000, dtype=np.float32)
            list(self.whisper_model.transcribe(silence, beam_size=1, language="en", vad_filter=False)[0])
            if self.local_llm:
                self._format_text_local("warm up", self.config.get("last_language") or "en", max_tokens=4, quiet=True)
            log(f"[Ready] Engines warmed up in {time.time() - t0:.2f}s.")
        except Exception as e:
            log(f"[Warm-up] Skipped ({e}).")

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
            try:
                self.record_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self.audio_callback
                )
                self.record_stream.start()
            except Exception as e:
                self.is_recording = False
                log(f"[Audio Error] Could not open microphone: {e}")
                log("              Check that a microphone is connected, and that Windows Settings > Privacy & security >")
                log("              Microphone > 'Let desktop apps access your microphone' is turned on.")
                threading.Thread(target=self.play_sound, args=("error",), daemon=True).start()
                return

        threading.Thread(target=self.play_sound, args=("start",), daemon=True).start()
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

        threading.Thread(target=self.play_sound, args=("stop",), daemon=True).start()

        if not captured:
            log("[Warning] No audio recorded.")
            return

        audio_data = np.concatenate(captured, axis=0).flatten()
        duration = len(audio_data) / self.sample_rate

        if duration < 0.35:
            log(f"[Ignored] Audio too short ({duration:.2f}s).")
            return

        pending = self.jobs.unfinished_tasks
        if pending:
            log(f"[QUEUED] {duration:.1f}s of speech - will be inserted after the {pending} dictation(s) still processing.")
        else:
            log(f"[TRANSCRIBING] Processing {duration:.1f}s of speech using faster-whisper-{self.model_size}...")
        self.jobs.put(audio_data)

    # Language detection is unreliable on a phrase of a few seconds: "Det er
    # fint med mig" scores German 0.92, Danish 0.03 on Whisper base, yet is
    # transcribed perfectly once Danish is set. Up to this much speech, a
    # detection that disagrees with the user's last language is checked by
    # decoding in both. The last language must win by a margin: decoded in the
    # wrong language Whisper translates ("Non, merci" -> "No, thank you."),
    # and in the benchmark a translation scored at most 0.08 better than the
    # detected language, against 0.29 and more for real corrections.
    LANGUAGE_CHECK_MAX_SPEECH = 4.0
    LANGUAGE_CHECK_MARGIN = 0.25
    # With medium on the benchmark the check fixed no detection and made a
    # short phrase in another language 3.4 s slower. `true` checks on these
    # models only, "always" on all of them.
    LANGUAGE_CHECK_MODELS = ("tiny", "base")

    def _language_check_enabled(self):
        # Hand-edited configs: "Always", 1, "true", "yes" mean what they say.
        setting = str(self.config.get("language_check_short_phrases", True)).strip().lower()
        if setting == "always":
            return True
        return setting in ("true", "1", "yes", "on") and self.model_size in self.LANGUAGE_CHECK_MODELS

    def transcribe(self, audio_data):
        """Whisper speech-to-text for one dictation. The caller holds engine_lock.
        Returns (text, language, language probability); text is "" when no
        speech was found."""
        # language=None lets faster-whisper detect the language on the first
        # window; the detect_language patch restricts it to the 4 supported
        # languages and the encoder cache makes detection nearly free.
        target_lang = self.language if self.language in ALLOWED_LANGUAGES else None
        last = self.config.get("last_language")
        last = last if last in ALLOWED_LANGUAGES else None
        text, info, _ = self._decode(audio_data, target_lang)
        lang = info.language if info.language in ALLOWED_LANGUAGES else "en"
        prob = info.language_probability if self.language is None else 1.0

        if (target_lang is None and last and lang != last and text
                and self._language_check_enabled()
                and getattr(info, "duration_after_vad", 0.0) <= self.LANGUAGE_CHECK_MAX_SPEECH):
            # Both candidates are decoded greedily so their scores compare like
            # for like. Whisper's temperature fallback would re-decode up to 5
            # times in the wrong language (seen: 0.5 s -> 3.5 s), and the
            # encoder output is cached, so these two passes are decoder only.
            _, _, det_logprob = self._decode(audio_data, lang, fallback=False)
            alt_text, _, alt_logprob = self._decode(audio_data, last, fallback=False)
            if alt_text and alt_logprob > det_logprob + self.LANGUAGE_CHECK_MARGIN:
                log(f"[Language] Short phrase: {ALLOWED_LANGUAGES[last]} fits better than "
                    f"{ALLOWED_LANGUAGES[lang]} (\"{text}\" -> \"{alt_text}\").")
                text, lang = alt_text, last
                prob = dict(getattr(info, "all_language_probs", None) or []).get(last, 0.0)
        return text, lang, prob

    def _decode(self, audio_data, language, fallback=True):
        """One Whisper pass. Returns (text, info, average token log-probability).
        fallback=False decodes greedily only (no temperature fallback)."""
        extra = {} if fallback else {"temperature": 0.0}
        segments, info = self.whisper_model.transcribe(
            audio_data,
            beam_size=1,
            language=language,
            **extra,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            without_timestamps=True,
            condition_on_previous_text=False,
            # initial_prompt biases decoding toward the dictionary spellings.
            # (hotwords= was tested too, but it makes Whisper stutter on
            # multi-word names: "Salesforce Force".)
            initial_prompt=self.dictionary.whisper_prompt()
        )
        kept, weighted, tokens = [], 0.0, 0
        for seg in segments:
            if speak_text.is_hallucination(seg.text, seg.no_speech_prob):
                if seg.text.strip():
                    log(f"[Whisper] Ignored \"{seg.text.strip()}\" (Whisper invents this on silence or noise).")
                continue
            kept.append(seg.text)
            n = max(1, len(getattr(seg, "tokens", None) or []))
            weighted += getattr(seg, "avg_logprob", 0.0) * n
            tokens += n
        return " ".join(kept).strip(), info, (weighted / tokens if tokens else float("-inf"))

    def _process_worker(self, audio_data):
        t0 = time.time()
        try:
            # Steps 1 and 2 use the shared engines. Insertion does not, so a
            # meeting being cleaned up does not wait for the paste.
            with self.engine_lock:
                # 1. Local Whisper Speech-to-Text
                raw_text, target_lang, lang_prob = self.transcribe(audio_data)
                t_whisper = time.time() - t0

                if not raw_text:
                    log("[Info] No speech detected.")
                    return

                self.last_language = target_lang
                if self.config.get("last_language") != target_lang:
                    self.config["last_language"] = target_lang
                    save_config(self.config)

                lang_name = ALLOWED_LANGUAGES.get(target_lang, target_lang.upper())
                log(f"[Whisper STT {t_whisper:.2f}s] [{lang_name} ({target_lang}) - {lang_prob*100:.0f}% confidence]: \"{raw_text}\"")

                # 2. Dictionary + Local AI Formatting & Cleanup
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

    def clean_text(self, raw_text, lang, use_llm=True, verbose=True, instructions=None, stats=None):
        """Dictionary fixes, Qwen cleanup (or the fast path for short clean
        phrases), then dictionary casing. Shared by dictation and meeting
        transcripts; the caller holds engine_lock. Meetings pass verbose=False
        to skip the per-turn logs. instructions replaces the dictation system
        prompt (meetings use a stricter "do not rephrase" one). stats, when
        given, receives "llm": True if Qwen was called (used by the benchmark)."""
        if stats is not None:
            stats["llm"] = False
        # Learned mishearings: confirmed ones are replaced right here, the
        # rest become a one-line hint for the LLM (only when relevant).
        text, dict_hint, applied, hint_needs_llm = self.dictionary.prepare(raw_text, lang)
        if verbose:
            for heard, word in applied:
                log(f"[Dictionary] '{heard}' -> '{word}' (learned correction)")

        if use_llm and self.ai_mode != "off" and self.local_llm:
            # "Um, yes." is a short clean phrase once the leading filler is gone.
            stripped = speak_text.strip_leading_fillers(text)
            # A dictionary word that is already spelled right (or that
            # apply_casing fixes below) is no reason to wait for Qwen.
            if (self.config.get("llm_fast_path", True) and not hint_needs_llm
                    and not speak_text.needs_llm(stripped, lang)):
                text = speak_text.light_cleanup(stripped) if stripped else ""
                if verbose:
                    log(f"[Fast path] Short clean phrase, LLM skipped: \"{text}\"")
            else:
                if stats is not None:
                    stats["llm"] = True
                enhanced = self.format_text(text, lang, dict_hint, quiet=not verbose, instructions=instructions)
                if enhanced:
                    text = enhanced
                    if verbose:
                        log(f"[AI Formatted]: \"{text}\"")

        # Deterministic safety net: dictionary spellings always win.
        return self.dictionary.apply_casing(text)

    def format_text(self, text, lang, dict_hint="", quiet=False, instructions=None):
        if not text or not text.strip():
            return text
        if self.ai_mode == "local" and self.local_llm:
            return self._format_text_local(text, lang, quiet=quiet, dict_hint=dict_hint, instructions=instructions)
        return text

    # Long dictations are cleaned in sentence-aligned chunks so we never blow the
    # LLM context window (the old fixed max_tokens=150 silently truncated anything
    # longer than ~100 words).
    LLM_CHUNK_WORDS = 160

    def _split_for_llm(self, text):
        words = text.split()
        if len(words) <= self.LLM_CHUNK_WORDS:
            return [text]
        import re
        limit = self.LLM_CHUNK_WORDS
        sentences = []
        for s in re.split(r"(?<=[.!?])\s+", text):
            if len(s.split()) <= limit:
                sentences.append(s)
                continue
            # Whisper sometimes writes a long, fast stretch with only commas:
            # split it at clauses, and as a last resort every `limit` words.
            for clause in re.split(r"(?<=[,;:])\s+", s):
                cw = clause.split()
                if len(cw) <= limit:
                    sentences.append(clause)
                else:
                    # Equal pieces: [160, 1] would send Qwen a one-word "text".
                    size = -(-len(cw) // -(-len(cw) // limit))
                    sentences.extend(" ".join(cw[i:i + size]) for i in range(0, len(cw), size))
        chunks, current = [], []
        for s in sentences:
            if current and len(" ".join(current + [s]).split()) > self.LLM_CHUNK_WORDS:
                chunks.append(" ".join(current))
                current = []
            current.append(s)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _format_text_local(self, text, lang, max_tokens=None, quiet=False, dict_hint="", instructions=None):
        chunks = self._split_for_llm(text)
        if len(chunks) > 1:
            outputs = []
            for c in chunks:
                outputs.append(self._format_chunk_local(c, lang, max_tokens, quiet, dict_hint, instructions))
            return " ".join(o.strip() for o in outputs if o)
        return self._format_chunk_local(text, lang, max_tokens, quiet, dict_hint, instructions)

    def _format_chunk_local(self, text, lang, max_tokens=None, quiet=False, dict_hint="", instructions=None):
        # The hint sits at the very end of the system prompt: the whole prompt
        # before it stays byte-identical between dictations, so llama.cpp
        # reuses its KV cache and only the transcript itself is evaluated.
        dict_suffix = dict_hint or ""
        if instructions:
            prompt = f"""<|im_start|>system
{instructions}{dict_suffix}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
        elif lang == "da":
            prompt = f"""<|im_start|>system
Du er en intelligent dikteringsassistent. Rens den indtalte tekst:
- Fjern fyldord ('øh', 'altså', 'sådan', 'um', 'uh', 'du ved').
- Ret talefejl og gentagelser.
- Formater som naturlig, sammenhængende tekst med korrekt tegnsætning og store bogstaver.
- Lav kun punktopstilling hvis brugeren specifikt sagde 'punkt' eller 'bullet point'.
- Hvis brugeren sagde 'ny linje', indsæt et linjeskift.
- Skriv på fejlfrit DANSK.
- Udskriv KUN den rensede tekst uden forklaringer eller citationstegn.{dict_suffix}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
        elif lang == "de":
            prompt = f"""<|im_start|>system
Du bist ein intelligenter Diktierassistent. Bereinige das gesprochene Transkript:
- Entferne Füllwörter ('äh', 'ähm', 'also', 'weißt du', 'quasi').
- Korrigiere Versprecher und Wortwiederholungen flüssig.
- Formatiere als natürlichen, zusammenhängenden Text mit korrekter Zeichensetzung und Groß-/Kleinschreibung.
- Erstelle nur Aufzählungspunkte, wenn ausdrücklich 'Aufzählungspunkt' oder 'Punkt' diktiert wurde.
- Wenn 'neue Zeile' oder 'Absatz' diktiert wurde, füge einen Zeilenumbruch ein.
- Schreibe in fehlerfreiem DEUTSCH.
- Gib NUR den bereinigten Text ohne Erklärungen oder Anführungszeichen aus.{dict_suffix}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
        elif lang == "fr":
            prompt = f"""<|im_start|>system
Tu es un assistant de dictée vocale intelligent. Nettoie la transcription suivante:
- Supprime les tics de langage et mots de remplissage ('euh', 'ben', 'tu sais', 'en fait', 'genre').
- Corrige les hésitations et répétitions avec fluidité.
- Formate en texte continu naturel avec ponctuation et majuscules soignées.
- Ne crée des puces que si l'utilisateur a explicitement dit 'point' ou 'puce'.
- Si l'utilisateur a dit 'à la ligne' ou 'nouvelle ligne', insère un saut de ligne.
- Écris en FRANÇAIS impeccable.
- Renvoie UNIQUEMENT le texte nettoyé sans explications ni guillemets.{dict_suffix}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
You are an intelligent voice dictation assistant. Clean up this spoken transcript:
- Remove verbal filler words ('um', 'uh', 'you know', 'like').
- Fix false starts and repetitions smoothly.
- Format as natural continuous text with proper capitalization and punctuation.
- Only create bullet points if the user explicitly dictated 'bullet point'.
- If the user dictated 'new line', insert a line break.
- Preserve the exact language and meaning.
- Output ONLY the final cleaned text without explanations or quotes.{dict_suffix}<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
        try:
            t0 = time.time()
            if max_tokens is None:
                # Output is roughly as long as the input; budget 1.5x + slack and
                # never exceed what fits in the context window.
                n_in = len(self.local_llm.tokenize(text.encode("utf-8"), add_bos=False))
                n_prompt = len(self.local_llm.tokenize(prompt.encode("utf-8"), add_bos=False))
                max_tokens = max(16, min(int(n_in * 1.5) + 32, self.llm_ctx - n_prompt - 8))
            res = self.local_llm(prompt, max_tokens=max_tokens, temperature=0.1, stop=["<|im_end|>"])
            cleaned = res["choices"][0]["text"].strip()
            dur = time.time() - t0
            if cleaned:
                # Strip conversational model prefixes if any slipped through
                for prefix in ("Here is the cleaned text:", "Cleaned text:", "Transcribed text:", "Here is the transcript:", "Sure, here is"):
                    if cleaned.lower().startswith(prefix.lower()):
                        cleaned = cleaned[len(prefix):].strip()
                        if cleaned.startswith('"') and cleaned.endswith('"'):
                            cleaned = cleaned[1:-1].strip()

                # Safety check: if the model answered a question or hallucinated instead of cleaning
                in_words = len(text.split())
                out_words = len(cleaned.split())
                if in_words >= 4 and (out_words > in_words * 2.5 + 10 or out_words < max(1, in_words * 0.25)):
                    log(f"[AI Warning] Local AI output diverged suspiciously ({out_words} vs {in_words} words). Falling back to raw transcript.")
                    return text

                if not quiet:
                    n_out = res.get("usage", {}).get("completion_tokens", 0)
                    log(f"[Local AI {dur:.2f}s, {n_out} tokens]: \"{cleaned}\"")
                return cleaned
        except Exception as e:
            log(f"[AI Warning] Local AI formatting failed ({e}). Using raw text.")
        return text

    def get_foreground_info(self):
        hwnd = _user32.GetForegroundWindow()
        if not hwnd:
            return "", "", ""
        length = _user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value.lower()

        class_buff = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, class_buff, 256)
        cls = class_buff.value.lower()

        exe = ""
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h_proc = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
            if h_proc:
                try:
                    exe_buf = ctypes.create_unicode_buffer(1024)
                    size = wintypes.DWORD(1024)
                    if _kernel32.QueryFullProcessImageNameW(h_proc, 0, exe_buf, ctypes.byref(size)):
                        exe = os.path.basename(exe_buf.value).lower()
                finally:
                    _kernel32.CloseHandle(h_proc)
        return cls, title, exe

    TYPE_CHUNK = 32  # characters per SendInput call (one syscall instead of one per char)

    def type_unicode_text(self, text):
        import struct
        events = []
        for ch in text:
            if ch == "\r":
                continue
            down = INPUT()
            down.type = INPUT_KEYBOARD
            up = INPUT()
            up.type = INPUT_KEYBOARD
            if ch == "\n":
                down.u.ki = KEYBDINPUT(VK_RETURN, 0, 0, 0, 0)
                up.u.ki = KEYBDINPUT(VK_RETURN, 0, KEYEVENTF_KEYUP, 0, 0)
                events.extend((down, up))
            else:
                # Encode to UTF-16 LE code units to safely handle surrogate pairs for emojis/non-BMP chars
                raw_bytes = ch.encode("utf-16-le")
                code_units = struct.unpack(f"<{len(raw_bytes) // 2}H", raw_bytes)
                for code in code_units:
                    d = INPUT()
                    d.type = INPUT_KEYBOARD
                    d.u.ki = KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0)
                    u = INPUT()
                    u.type = INPUT_KEYBOARD
                    u.u.ki = KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)
                    events.extend((d, u))

        for i in range(0, len(events), self.TYPE_CHUNK * 2):
            batch = events[i:i + self.TYPE_CHUNK * 2]
            arr = (INPUT * len(batch))(*batch)
            _user32.SendInput(len(batch), arr, ctypes.sizeof(INPUT))
            time.sleep(0.002)

    # ---- Clipboard save / restore -------------------------------------------
    # Pasting needs the clipboard, but it belongs to the user. Speak saves the
    # text that was on it, pastes, and puts it back once the target app has had
    # time to read it. Only plain text can be restored (images/files-only
    # clipboards are left alone - same as before). Opt out with the optional
    # config key "restore_clipboard": false.
    CLIPBOARD_RESTORE_DELAY = 0.8   # seconds after the paste keystroke

    def _open_clipboard(self, attempts=5):
        for _ in range(attempts):
            try:
                win32clipboard.OpenClipboard()
                return True
            except Exception:
                time.sleep(0.04)
        return False

    def _clipboard_snapshot(self):
        """Returns {'kind': 'text'|'empty'|'other', 'text': ...} or None if disabled/unreadable."""
        if not self.config.get("restore_clipboard", True):
            return None
        if not self._open_clipboard():
            return None
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                return {"kind": "text", "text": win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)}
            return {"kind": "empty"} if _user32.CountClipboardFormats() == 0 else {"kind": "other"}
        except Exception:
            return None
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass

    def _set_clipboard_text(self, text):
        if not self._open_clipboard():
            return False
        try:
            win32clipboard.EmptyClipboard()
            if text is not None:
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            return True
        except Exception:
            return False
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass

    def _restore_clipboard(self, token, delay=None):
        """Put the saved clipboard back, unless something else has changed it since."""
        if not token:
            return
        snapshot, our_seq, t_used = token
        if not snapshot or snapshot["kind"] == "other":
            return
        wait = (self.CLIPBOARD_RESTORE_DELAY if delay is None else delay) - (time.time() - t_used)
        if wait > 0:
            time.sleep(wait)
        if _user32.GetClipboardSequenceNumber() != our_seq:
            return  # the user (or another app) copied something new - keep it
        self._set_clipboard_text(snapshot.get("text") if snapshot["kind"] == "text" else None)

    def _schedule_clipboard_restore(self, token):
        """Restores the user's clipboard CLIPBOARD_RESTORE_DELAY after a paste,
        on a timer thread. Waiting here instead held the transcription worker
        and the clipboard for 0.8 s after every dictation, which delayed the
        next queued dictation and the learn hotkey. Call with clipboard_lock held."""
        if not token:
            return
        delay = max(0.0, self.CLIPBOARD_RESTORE_DELAY - (time.time() - token[2]))
        timer = threading.Timer(delay, self._run_scheduled_restore, args=(token,))
        timer.daemon = True
        self._pending_restore, self._pending_restore_timer = token, timer
        timer.start()

    def _run_scheduled_restore(self, token):
        with self.clipboard_lock:
            if self._pending_restore is not token:
                return   # taken over by a newer paste or the learn hotkey
            self._pending_restore = self._pending_restore_timer = None
            self._restore_clipboard(token, delay=0)

    def _take_pending_restore(self):
        """Cancels a scheduled restore before Speak uses the clipboard again.
        Returns the pending restore token (snapshot, our_seq, time) while
        Speak's pasted text is still on the clipboard, so the next restore puts
        back the user's content and not the previous dictation. Returns None
        when nothing is pending or the user has copied something since (that
        is then the clipboard to keep). Call with clipboard_lock held."""
        token, timer = self._pending_restore, self._pending_restore_timer
        if not token:
            return None
        # An app reads the clipboard only when it handles the Ctrl+V, which can
        # take a moment when it is busy. Replacing the clipboard before then
        # made it paste the next dictation twice (seen end to end). So the
        # previous paste still gets its full CLIPBOARD_RESTORE_DELAY; the time
        # the next dictation spent in Whisper and Qwen already counts.
        wait = self.CLIPBOARD_RESTORE_DELAY - (time.time() - token[2])
        if wait > 0:
            time.sleep(wait)
        self._pending_restore = self._pending_restore_timer = None
        if timer:
            timer.cancel()
        return token if _user32.GetClipboardSequenceNumber() == token[1] else None

    def flush_clipboard_restore(self):
        """Restores a pending clipboard now (on exit, so it is not lost)."""
        with self.clipboard_lock:
            token, timer = self._pending_restore, self._pending_restore_timer
            self._pending_restore = self._pending_restore_timer = None
            if timer:
                timer.cancel()
            if token:
                self._restore_clipboard(token, delay=0)

    MODIFIER_KEYS = (VK_SHIFT, VK_CONTROL, VK_MENU)

    def _release_modifiers(self, timeout):
        """Waits until the user has let go of Shift, Ctrl, Alt and Win, which
        are often still down from the hotkey: a Ctrl+C sent while Alt is
        physically held reaches the app as Ctrl+Alt+C. If a key is still down
        after timeout, a key-up is sent for Shift/Ctrl/Alt (never Win: a lone
        Win key-up opens the Start menu). Returns True if all were released."""
        deadline = time.time() + timeout
        while True:
            held = [vk for vk in self.MODIFIER_KEYS + (VK_LWIN, VK_RWIN) if _user32.GetAsyncKeyState(vk) & 0x8000]
            if not held:
                return True
            if time.time() >= deadline:
                break
            time.sleep(0.01)
        release = [(vk, True) for vk in held if vk in self.MODIFIER_KEYS]
        if release:
            send_keys(release)
            time.sleep(0.03)
        return False

    def insert_text(self, text):
        """Inserts text into the focused app. Returns a clipboard restore token
        (pass to _schedule_clipboard_restore) or None when the clipboard was not
        used. Call with clipboard_lock held."""
        self._release_modifiers(timeout=0.3)

        cls, title, exe = self.get_foreground_info()
        terminal_exes = {
            "windowsterminal.exe", "wt.exe", "powershell.exe", "pwsh.exe",
            "cmd.exe", "mintty.exe", "conhost.exe", "alacritty.exe",
            "wezterm-gui.exe", "kitty.exe", "warp.exe"
        }
        terminal_classes = {
            "consolewindowclass", "cascadia_code_window_class", "mintty"
        }
        is_terminal = (exe in terminal_exes) or any(tc in cls for tc in terminal_classes) or any(t in title for t in ["powershell", "cmd.exe", "wsl"])

        target_desc = f"{exe or cls or title}"
        if self.paste_mode == "typing" or (self.paste_mode == "auto" and is_terminal and len(text) < 400):
            log(f"[Insert] Using Direct Unicode Typing (Target: '{target_desc}')...")
            self.type_unicode_text(text)
            return None

        inherited = self._take_pending_restore()
        snapshot = inherited[0] if inherited else self._clipboard_snapshot()
        if not self._set_clipboard_text(text):
            # Pasting now would insert whatever is on the clipboard (the user's
            # content or the previous dictation). Type the text instead.
            log("[Insert Warning] Could not write to the clipboard (held by another app); typing the text instead.")
            self.type_unicode_text(text)
            if inherited is None:
                return None   # the clipboard still holds the user's own content
            # A previous dictation is still on the clipboard: keep its restore,
            # with its own sequence number. The current one may already belong
            # to the app holding the clipboard, and restoring over that would
            # destroy what the user just copied.
            return (snapshot, inherited[1], time.time())
        our_seq = _user32.GetClipboardSequenceNumber()

        time.sleep(0.05)

        if is_terminal and "consolewindowclass" in cls:
            log("[Insert] Using Shift+Insert for classic console...")
            sent = send_keys(chord(VK_SHIFT, VK_INSERT))
        else:
            log(f"[Insert] Using Clipboard Paste (Target: '{cls or title}')...")
            sent = send_keys(chord(VK_CONTROL, VK_V))
        if not sent:
            log("[Insert Warning] Windows did not accept the paste keystroke (for example, the screen is "
                "locked). The text is on the clipboard: press Ctrl+V to paste it.")
            # Deliberate: the dictation stays on the clipboard so the user can
            # paste it by hand. Their earlier clipboard content is not restored.
            return None
        return (snapshot, our_seq, time.time()) if snapshot else None

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
        for hid, mods, vk_code, desc in hotkeys_to_bind:
            if _user32.RegisterHotKey(None, hid, mods, vk_code):
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
                    else:
                        self.toggle_recording()
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))
            time.sleep(0.02)
        self.unregister_hotkeys()

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

    def run(self, first_run=False):
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
                    new_h = input("Enter new hotkey [< / > / <> / F8 / etc.]: ").strip()
                    if new_h:
                        self.set_hotkey(new_h)
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
                        log("\n[Testing] Testing with: 'uh I think we should bullet point deploy the change new line bullet point notify the team'...")
                        with self.engine_lock:   # shared with dictation and meeting cleanup
                            test_out = self.format_text("uh I think we should bullet point deploy the change new line bullet point notify the team", "en")
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
            self.running = False
            self.meetings.shutdown()
            self.unregister_hotkeys()
            self.flush_clipboard_restore()
            try:
                self.jobs.put(None)
            except Exception:
                pass
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
