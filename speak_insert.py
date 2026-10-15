"""
Speak - putting text into the focused window: clipboard paste (with the
user's clipboard restored afterwards), direct typing with SendInput, and
the keyboard input helpers the hotkeys use. Mixed into SpeakApp.
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import threading
import time

import win32clipboard

from speak_base import _kernel32, _user32, log
PM_REMOVE = 0x0001

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

_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT
_user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
_user32.MapVirtualKeyW.restype = wintypes.UINT
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short


class InsertionMixin:
    """Text insertion for SpeakApp. Uses self.config, self.paste_mode,
    self.clipboard_lock and the _pending_restore fields set in __init__."""


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
