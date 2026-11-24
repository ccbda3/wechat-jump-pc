import ctypes
import threading
from ctypes import wintypes

import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32
winmm = ctypes.windll.winmm

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
winmm.timeBeginPeriod(1)

LRESULT = ctypes.c_ssize_t
LPARAM = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, LPARAM)

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
LLMHF_INJECTED = 1
WM_KEYDOWN = 0x100
WM_SYSKEYDOWN = 0x104
WM_LBUTTONDOWN = 0x201
WM_LBUTTONUP = 0x202
VK_SPACE = 0x20


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("pt", wintypes.POINT), ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("extra", ctypes.c_size_t)]


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("extra", ctypes.c_size_t)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def list_windows():
    out = []

    def cb(h, l):
        if user32.IsWindowVisible(h):
            n = user32.GetWindowTextLengthW(h)
            if n > 0:
                b = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(h, b, n + 1)
                out.append((int(h), b.value))
        return True

    CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(CB(cb), 0)
    return out


def window_title(hwnd):
    n = user32.GetWindowTextLengthW(hwnd)
    b = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, b, n + 1)
    return b.value


def is_wechat_process(hwnd):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return False
    h = kernel32.OpenProcess(0x1000, False, pid.value)
    if not h:
        return False
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(1024)
        if not kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return False
        name = buf.value.replace("/", "\\").split("\\")[-1].lower()
        return "weixin" in name or "wechat" in name
    finally:
        kernel32.CloseHandle(h)


def grab_frame(hwnd):
    rc = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
        return None
    w, h = rc.right, rc.bottom
    if w < 8 or h < 8:
        return None
    hdc = user32.GetWindowDC(hwnd)
    if not hdc:
        return None
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    ok = user32.PrintWindow(hwnd, mem, 3)
    bi = BITMAPINFOHEADER(40, w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    got = gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
    gdi32.SelectObject(mem, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    if not ok or not got:
        return None
    arr = np.frombuffer(buf, np.uint8, count=w * h * 4).reshape(h, w, 4)
    return arr[:, :, :3].copy()


def client_origin(hwnd):
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


def bring_front(hwnd):
    if user32.GetForegroundWindow() == hwnd:
        return
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    kernel32.Sleep(60)


def get_foreground():
    return user32.GetForegroundWindow()


def set_cursor(x, y):
    user32.SetCursorPos(int(x), int(y))


def _mouse(flags):
    user32.mouse_event(flags, 0, 0, 0, None)


def press_hold(x, y, ms):
    ms = max(80.0, float(ms))
    user32.SetCursorPos(int(x), int(y))
    kernel32.Sleep(18)
    _mouse(2)
    freq = ctypes.c_longlong()
    kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
    t0 = ctypes.c_longlong()
    t = ctypes.c_longlong()
    kernel32.QueryPerformanceCounter(ctypes.byref(t0))
    target = ms / 1000.0
    while True:
        kernel32.QueryPerformanceCounter(ctypes.byref(t))
        rem = target - (t.value - t0.value) / freq.value
        if rem <= 0:
            break
        if rem > 0.003:
            kernel32.Sleep(int((rem - 0.002) * 1000))
        else:
            kernel32.Sleep(0)
    _mouse(4)
    kernel32.QueryPerformanceCounter(ctypes.byref(t))
    return (t.value - t0.value) * 1000.0 / freq.value


def click_at(x, y):
    user32.SetCursorPos(int(x), int(y))
    kernel32.Sleep(25)
    _mouse(2)
    kernel32.Sleep(45)
    _mouse(4)


class MouseHook:
    def __init__(self, state):
        self.state = state
        self._proc = None
        self.handle = None

    def start(self):
        self._proc = HOOKPROC(self._cb)
        self.handle = user32.SetWindowsHookExW(
            WH_MOUSE_LL, self._proc, kernel32.GetModuleHandleW(None), 0)
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        if self.handle:
            user32.UnhookWindowsHookEx(self.handle)
            self.handle = None

    def _cb(self, code, wp, lp):
        try:
            st = self.state
            if code == 0 and st.manual and st.running and not st.paused:
                m = ctypes.cast(lp, ctypes.POINTER(MSLLHOOKSTRUCT)).contents
                if not (m.flags & LLMHF_INJECTED):
                    if wp == WM_LBUTTONDOWN:
                        hwnd = st.target_hwnd
                        if hwnd and user32.IsWindow(hwnd):
                            pt = wintypes.POINT(m.pt.x, m.pt.y)
                            user32.ScreenToClient(hwnd, ctypes.byref(pt))
                            rc = wintypes.RECT()
                            user32.GetClientRect(hwnd, ctypes.byref(rc))
                            if 0 <= pt.x < rc.right and 0 <= pt.y < rc.bottom:
                                if not st.busy:
                                    st.tap_x = int(pt.x)
                                    st.tap_y = int(pt.y)
                                    st.got = True
                                st.swallow_up = True
                                return 1
                    elif wp == WM_LBUTTONUP and st.swallow_up:
                        st.swallow_up = False
                        return 1
        except Exception:
            pass
        return user32.CallNextHookEx(None, code, wp, lp)

    def _loop(self):
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass


class KeyboardHook:
    def __init__(self, state, on_space):
        self.state = state
        self.on_space = on_space
        self._proc = None
        self.handle = None

    def start(self):
        self._proc = HOOKPROC(self._cb)
        self.handle = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._proc, kernel32.GetModuleHandleW(None), 0)
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        if self.handle:
            user32.UnhookWindowsHookEx(self.handle)
            self.handle = None

    def _cb(self, code, wp, lp):
        try:
            if code == 0 and wp in (WM_KEYDOWN, WM_SYSKEYDOWN):
                k = ctypes.cast(lp, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if k.vkCode == VK_SPACE and self.state.running:
                    fg = get_foreground()
                    if fg in (self.state.main_hwnd, self.state.target_hwnd):
                        ctrl = user32.GetKeyState(0x11) & 0x8000
                        alt = user32.GetKeyState(0x12) & 0x8000
                        shift = user32.GetKeyState(0x10) & 0x8000
                        if not (ctrl or alt or shift):
                            self.on_space()
                            return 1
        except Exception:
            pass
        return user32.CallNextHookEx(None, code, wp, lp)

    def _loop(self):
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass
