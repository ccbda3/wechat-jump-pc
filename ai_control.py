# -*- coding: utf-8 -*-
"""AI 模式底层 I/O：窗口级抓图 + 高精度鼠标按压。

与原版 common/pc_control.py 完全独立，不修改原文件，可随时切回经典模式。

关键改进：
1. PrintWindow 抓窗口自身内容 —— 游戏窗口被其他窗口遮挡时，仍能拿到真实画面
   （原版 ImageGrab 抓的是屏幕合成结果，被挡住就全错）。
2. perf_counter 忙等 + 1ms 系统时钟精度 —— 按压时长误差从 ~15ms 降到 ~1ms
   （原版 time.sleep() 受 Windows 调度粒度影响）。
"""
import ctypes
import time
from ctypes import wintypes

import numpy as np
from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
winmm = ctypes.windll.winmm

try:
    user32.SetProcessDPIAware()
except Exception:
    pass

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
INPUT_MOUSE = 0
PW_RENDERFULLCONTENT = 2


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT_STRUCT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.c_void_p, ctypes.c_int]
user32.SendInput.restype = wintypes.UINT


def get_window_rect(hwnd):
    """返回窗口物理像素矩形 (left, top, right, bottom)。"""
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def grab_window(hwnd):
    """用 PrintWindow 抓窗口自身内容，返回 BGR uint8 数组（含标题栏）。

    返回整窗图像，因此图像坐标 (x, y) 对应屏幕坐标
    (window_left + x, window_top + y)。
    """
    l, t, r, b = get_window_rect(hwnd)
    w, h = r - l, b - t
    if w <= 0 or h <= 0:
        raise ValueError("invalid window size: %dx%d" % (w, h))

    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    try:
        ok = user32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT)
        if not ok:
            raise RuntimeError("PrintWindow failed")

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h          # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0     # BI_RGB

        buf = ctypes.create_string_buffer(w * h * 4)
        got = gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        if not got:
            raise RuntimeError("GetDIBits failed")
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(arr[:, :, :3])   # BGRA -> BGR
    finally:
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)


class ScreenGrabber:
    """mss 屏幕区域抓图（PrintWindow 不可用时的兜底）。"""

    def __init__(self):
        import mss
        self._sct = mss.mss()

    def grab(self, region):
        l, t, r, b = [int(v) for v in region]
        raw = self._sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})
        arr = np.asarray(raw)                # BGRA
        return np.ascontiguousarray(arr[:, :, :3])

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass


def _send_mouse(flags):
    inp = INPUT_STRUCT()
    inp.type = INPUT_MOUSE
    inp.u.mi.dwFlags = flags
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def precise_press(x, y, duration_ms, no_click=False):
    """按鼠标左键 duration_ms 毫秒，返回实测按压时长（毫秒）。

    no_click=True 时只走计时不发鼠标事件，供离线自测使用。
    """
    duration_ms = max(float(duration_ms), 50.0)
    target_s = duration_ms / 1000.0
    winmm.timeBeginPeriod(1)
    try:
        if not no_click:
            user32.SetCursorPos(int(x), int(y))
            time.sleep(0.02)                  # 光标到位，不计入按压
        t0 = time.perf_counter()
        if not no_click:
            _send_mouse(MOUSEEVENTF_LEFTDOWN)
        target = t0 + target_s
        while True:
            remain = target - time.perf_counter()
            if remain <= 0:
                break
            if remain > 0.003:
                time.sleep(min(remain - 0.002, 0.05))
            # 最后 3ms 自旋，避免 sleep 调度粒度带来的过冲
        if not no_click:
            _send_mouse(MOUSEEVENTF_LEFTUP)
        return (time.perf_counter() - t0) * 1000.0
    finally:
        winmm.timeEndPeriod(1)


def to_pil(bgr):
    """BGR numpy 数组 -> RGB PIL Image，供 GUI 预览。"""
    return Image.fromarray(bgr[:, :, ::-1].copy(), "RGB")


def bgr_to_pil_rgb(bgr):
    return to_pil(bgr)
