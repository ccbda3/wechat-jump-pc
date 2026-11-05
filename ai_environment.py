# -*- coding: utf-8 -*-
"""环境感知：监测窗口与画面的状态，决定「这一跳能不能按」。

原版对运行环境毫无感知：窗口被挡住、缩放、结算页、黑屏，都照按不误。
这里只做轻量判断，不改动原版的物理公式：
  * 窗口尺寸 / DPI 变化 -> 按原版规则重算 head_diameter（窗口宽 / 8）
  * 画面全黑、全白、纯色、被别的窗口盖住 -> 判定无效，本轮不点击
  * 亮度突变（夜间模式、全屏特效）-> 提示，避免使用脏样本学习
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnvironmentState:
    valid: bool
    resized: bool
    width: int
    height: int
    head_diameter: float
    brightness: float
    contrast: float
    brightness_jump: float
    note: str = ""

    def describe(self):
        if not self.valid:
            return self.note or "画面无效"
        if self.resized:
            return "窗口尺寸变化"
        if self.brightness_jump > 28:
            return "亮度突变"
        return "正常"


class EnvironmentMonitor:
    """逐帧调用 update()，内部保留上一帧的窗口尺寸与亮度。"""

    def __init__(self, head_diameter=None, resize_threshold=0.05):
        self._fixed_hd = head_diameter          # 用户在 GUI 里手填的头部直径
        self._hd = head_diameter
        self._last_size = None
        self._last_brightness = None
        self.resize_threshold = float(resize_threshold)
        self.frames = 0

    def set_head_diameter(self, hd):
        self._fixed_hd = float(hd) if hd else None
        self._hd = self._fixed_hd

    def update(self, bgr, window_rect=None):
        h, w = bgr.shape[:2]
        self.frames += 1
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (min(160, w), min(300, h)))
        brightness = float(small.mean())
        contrast = float(small.std())
        dark = float((small < 12).mean())
        white = float((small > 246).mean())

        # 画面有效性：纯色 / 黑屏 / 白屏 / 抓到别的程序界面（对比度异常）
        note = ""
        valid = True
        if dark > 0.80:
            valid, note = False, "画面几乎全黑（窗口未渲染或最小化）"
        elif white > 0.90:
            valid, note = False, "画面几乎全白（可能抓到别的窗口）"
        elif contrast < 6.0:
            valid, note = False, "画面接近纯色，无法识别"

        # 窗口尺寸变化（缩放 / 分辨率 / DPI）
        resized = False
        if self._last_size is not None:
            lw, lh = self._last_size
            if (abs(w - lw) / max(lw, 1) > self.resize_threshold or
                    abs(h - lh) / max(lh, 1) > self.resize_threshold):
                resized = True
        self._last_size = (w, h)

        if self._hd is None:
            # 整个会话只按首帧客户区宽算一次，之后固定不变：
            # 参数飘来飘去会让每次效果都不一样，自适应只交给系数 k / near
            self._hd = w / 8.0
        hd = max(float(self._hd), 1.0)

        jump = 0.0
        if self._last_brightness is not None:
            jump = abs(brightness - self._last_brightness)
        self._last_brightness = brightness

        return EnvironmentState(valid=valid, resized=resized, width=w, height=h,
                                head_diameter=hd, brightness=brightness,
                                contrast=contrast, brightness_jump=jump, note=note)
