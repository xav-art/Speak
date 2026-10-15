"""
Speak - the notification-area (tray) icon: shows whether Speak is ready,
recording a dictation, working on one, or recording a meeting, and has a small
menu (start or stop dictation or a meeting, show the window, quit).

Pure Win32 through ctypes, no extra package. The icon's window lives on the
hotkey listener thread, whose message loop (PeekMessageW + DispatchMessageW)
delivers its messages; set_state() may be called from any thread.
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import threading

import numpy as np

from speak_base import BASE_DIR, _kernel32, _user32, log

_shell32 = ctypes.WinDLL("shell32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WM_DESTROY = 0x0002
WM_NULL = 0x0000
WM_COMMAND = 0x0111
WM_LBUTTONUP = 0x0202
WM_RBUTTONUP = 0x0205
WM_CONTEXTMENU = 0x007B
WM_APP = 0x8000
TRAY_MESSAGE = WM_APP + 1
NIM_ADD, NIM_MODIFY, NIM_DELETE, NIM_SETVERSION = 0, 1, 2, 4
NIF_MESSAGE, NIF_ICON, NIF_TIP = 0x1, 0x2, 0x4
NOTIFYICON_VERSION_4 = 4
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x10
MF_STRING, MF_SEPARATOR, MF_GRAYED = 0x0, 0x800, 0x1
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x2, 0x100, 0x80
SW_RESTORE = 9

CMD_DICTATION, CMD_MEETING, CMD_SHOW, CMD_QUIT = 1, 2, 3, 4


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT), ("szInfoTitle", wintypes.WCHAR * 64),
                ("dwInfoFlags", wintypes.DWORD), ("guidItem", GUID), ("hBalloonIcon", wintypes.HICON)]


class ICONINFO(ctypes.Structure):
    _fields_ = [("fIcon", wintypes.BOOL), ("xHotspot", wintypes.DWORD), ("yHotspot", wintypes.DWORD),
                ("hbmMask", wintypes.HBITMAP), ("hbmColor", wintypes.HBITMAP)]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


_user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
_user32.RegisterClassExW.restype = wintypes.ATOM
_user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                    wintypes.HINSTANCE, wintypes.LPVOID]
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.DefWindowProcW.restype = LRESULT
_user32.DestroyWindow.argtypes = [wintypes.HWND]
_user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
_user32.GetSystemMetrics.argtypes = [ctypes.c_int]
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.FindWindowW.restype = wintypes.HWND
_kernel32.GetConsoleTitleW.argtypes = [wintypes.LPWSTR, wintypes.DWORD]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_ENUM_PROC, wintypes.LPARAM]


def small_icon_size():
    """The small-icon size of the display in real pixels (24 at 150 %).
    python.exe is not DPI-aware, so Windows would report 16 and stretch the
    icon: ask on this thread as a DPI-aware one, then switch back."""
    try:
        set_context = _user32.SetThreadDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = ctypes.c_void_p
        old = set_context(ctypes.c_void_p(-4))   # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        try:
            return _user32.GetSystemMetrics(49) or 16   # SM_CXSMICON
        finally:
            if old:
                set_context(ctypes.c_void_p(old))
    except Exception:
        return _user32.GetSystemMetrics(49) or 16
_user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
_user32.RegisterWindowMessageW.restype = wintypes.UINT
_user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int,
                               wintypes.UINT]
_user32.LoadImageW.restype = wintypes.HANDLE
_user32.CreateIconIndirect.argtypes = [ctypes.POINTER(ICONINFO)]
_user32.CreateIconIndirect.restype = wintypes.HICON
_user32.DestroyIcon.argtypes = [wintypes.HICON]
_user32.CreatePopupMenu.restype = wintypes.HMENU
_user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_size_t, wintypes.LPCWSTR]
_user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.LPVOID]
_user32.TrackPopupMenu.restype = wintypes.BOOL
_user32.DestroyMenu.argtypes = [wintypes.HMENU]
_user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
_shell32.Shell_NotifyIconW.restype = wintypes.BOOL
_gdi32.CreateBitmap.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.UINT, ctypes.c_void_p]
_gdi32.CreateBitmap.restype = wintypes.HBITMAP
_gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetConsoleWindow.restype = wintypes.HWND

# (tooltip, dot colour as BGRA or None for Speak's own icon)
STATES = {
    "idle": ("Speak - ready", None),
    "recording": ("Speak - listening", (40, 40, 220, 255)),
    "working": ("Speak - writing your text", (20, 160, 245, 255)),
    "meeting": ("Speak - recording a meeting", (180, 40, 200, 255)),
}


def dot_pixels(size, bgra):
    """A filled anti-aliased circle as BGRA rows, top row first, with straight
    (not premultiplied) alpha as 32-bit icons use (pure numpy, tested)."""
    y, x = np.mgrid[0:size, 0:size] + 0.5
    r = size / 2 - 1
    distance = np.hypot(x - size / 2, y - size / 2)
    alpha = np.clip(r + 0.5 - distance, 0.0, 1.0)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = np.where(alpha[..., None] > 0, np.array(bgra[:3], np.uint8), 0)
    out[..., 3] = (bgra[3] * alpha).astype(np.uint8)
    return out


def _make_dot_icon(bgra, size=16):
    pixels = np.ascontiguousarray(dot_pixels(size, bgra))
    color = _gdi32.CreateBitmap(size, size, 1, 32, pixels.ctypes.data)
    stride = (size + 15) // 16 * 2   # 1-bit rows are WORD-aligned
    mask = _gdi32.CreateBitmap(size, size, 1, 1, (ctypes.c_ubyte * (stride * size))())
    info = ICONINFO(True, 0, 0, mask, color)
    icon = _user32.CreateIconIndirect(ctypes.byref(info))
    _gdi32.DeleteObject(color)
    _gdi32.DeleteObject(mask)
    return icon


class TrayIcon:
    """One tray icon. create() on the thread that pumps messages; the rest
    from anywhere. actions: {"dictation", "meeting", "show", "quit"} ->
    callables, and "status" -> a callable returning (recording, meeting)."""

    CLASS_NAME = "SpeakTrayWindow"

    def __init__(self, actions):
        self.actions = actions
        self.hwnd = None
        self.icons = {}
        self.state = "idle"
        self._wndproc = WNDPROC(self._window_proc)   # kept alive: Windows calls it
        self._taskbar_created = 0
        self._lock = threading.Lock()
        self._instance = None

    def create(self):
        """Adds the icon. Returns False (and logs) when Windows refuses."""
        try:
            instance = self._instance = _kernel32.GetModuleHandleW(None)
            wc = WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
            wc.lpfnWndProc = self._wndproc
            wc.hInstance = instance
            wc.lpszClassName = self.CLASS_NAME
            _user32.RegisterClassExW(ctypes.byref(wc))   # fails harmlessly if already registered
            self.hwnd = _user32.CreateWindowExW(0, self.CLASS_NAME, "Speak", 0, 0, 0, 0, 0, None, None, instance, None)
            if not self.hwnd:
                raise OSError(f"CreateWindowExW failed ({ctypes.get_last_error()})")
            # Explorer broadcasts this when it restarts: the icon must be added again.
            self._taskbar_created = _user32.RegisterWindowMessageW("TaskbarCreated")
            # The small-icon size of this display (16 px at 100 %, larger when scaled).
            size = small_icon_size()
            path = os.path.join(BASE_DIR, "assets", "speak.ico")
            self.icons["idle"] = _user32.LoadImageW(None, path, IMAGE_ICON, size, size, LR_LOADFROMFILE)
            for name, (_, color) in STATES.items():
                if color is not None:
                    self.icons[name] = _make_dot_icon(color, size)
            if not self._notify(NIM_ADD):
                # No taskbar yet (Speak started before Explorer): the icon is
                # added when the taskbar announces itself (TaskbarCreated).
                log("[Tray] The taskbar is not ready yet; the icon will appear when it is.")
            return True
        except Exception as e:
            log(f"[Tray Warning] No notification-area icon ({e}). Speak works as usual.")
            self.destroy()
            return False

    def _data(self):
        data = NOTIFYICONDATAW()
        data.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        data.hWnd, data.uID = self.hwnd, 1
        data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        data.uCallbackMessage = TRAY_MESSAGE
        data.hIcon = self.icons.get(self.state) or self.icons.get("idle")
        data.szTip = STATES[self.state][0][:127]
        return data

    def _notify(self, action):
        if not self.hwnd:
            return False
        data = self._data()
        return bool(_shell32.Shell_NotifyIconW(action, ctypes.byref(data)))

    def set_state(self, state):
        """'idle', 'recording', 'working' or 'meeting'. Safe from any thread."""
        with self._lock:
            if state in STATES and state != self.state:
                self.state = state
                self._notify(NIM_MODIFY)

    def remove(self):
        """Takes the icon off the taskbar. Works from any thread (shutdown);
        destroy() must run on the thread that created the window."""
        with self._lock:
            self._notify(NIM_DELETE)

    def destroy(self):
        if self.hwnd:
            self.remove()
            _user32.DestroyWindow(self.hwnd)
            self.hwnd = None
            # The class keeps this instance's window procedure: unregister it,
            # so a later TrayIcon registers its own.
            _user32.UnregisterClassW(self.CLASS_NAME, self._instance)
        for icon in self.icons.values():
            if icon:
                _user32.DestroyIcon(icon)
        self.icons = {}

    def _window_proc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == TRAY_MESSAGE:
                event = lparam & 0xFFFF
                if event == WM_LBUTTONUP:
                    self._run("show")
                elif event in (WM_RBUTTONUP, WM_CONTEXTMENU):
                    self._menu()
                return 0
            if msg == self._taskbar_created and self._taskbar_created:
                self._notify(NIM_ADD)
                return 0
        except Exception as e:
            log(f"[Tray Warning] {e}")
        return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run(self, name):
        action = self.actions.get(name)
        if action:
            action()

    def menu_items(self):
        """[(command id, label)], with None for a separator (tested)."""
        recording, meeting = self.actions["status"]() if "status" in self.actions else (False, False)
        return [(CMD_DICTATION, "Stop dictation" if recording else "Start dictation"),
                (CMD_MEETING, "Stop meeting recording" if meeting else "Record a meeting"),
                None,
                (CMD_SHOW, "Show Speak window"),
                (CMD_QUIT, "Quit Speak")]

    def _menu(self):
        menu = _user32.CreatePopupMenu()
        try:
            for item in self.menu_items():
                if item is None:
                    _user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
                else:
                    _user32.AppendMenuW(menu, MF_STRING, item[0], item[1])
            point = POINT()
            _user32.GetCursorPos(ctypes.byref(point))
            # The documented dance: without the foreground window and the
            # WM_NULL after, the menu does not close when clicking elsewhere.
            _user32.SetForegroundWindow(self.hwnd)
            command = _user32.TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY,
                                             point.x, point.y, 0, self.hwnd, None)
            _user32.PostMessageW(self.hwnd, WM_NULL, 0, 0)
        finally:
            _user32.DestroyMenu(menu)
        self._run({CMD_DICTATION: "dictation", CMD_MEETING: "meeting", CMD_SHOW: "show",
                   CMD_QUIT: "quit"}.get(command))


def _visible_window_titled(title):
    """The first visible top-level window with exactly this title. (A hidden
    one, such as conhost's pseudo console under Windows Terminal or the
    tray's own window, cannot be brought to the front.)"""
    found = []

    def visit(hwnd, _):
        if _user32.IsWindowVisible(hwnd):
            buffer = ctypes.create_unicode_buffer(512)
            _user32.GetWindowTextW(hwnd, buffer, 512)
            if buffer.value == title:
                found.append(hwnd)
                return False
        return True
    _user32.EnumWindows(_ENUM_PROC(visit), 0)
    return found[0] if found else None


def show_console():
    """Brings the console window to the front (the tray's left click). Under
    Windows Terminal GetConsoleWindow is a hidden pseudo window, so the
    terminal window is looked up by its title (the console's title) first."""
    title = ctypes.create_unicode_buffer(512)
    hwnd = None
    if _kernel32.GetConsoleTitleW(title, 512) and title.value:
        hwnd = _visible_window_titled(title.value)
    hwnd = hwnd or _kernel32.GetConsoleWindow()
    if hwnd:
        _user32.ShowWindow(hwnd, SW_RESTORE)
        _user32.SetForegroundWindow(hwnd)
