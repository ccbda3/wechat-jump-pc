import asyncio

import cv2
import numpy as np

KEYWORDS = [
    "再玩一局", "再来一局", "再玩一次", "再一次", "重新开始", "重新挑战",
    "继续游戏", "开始游戏", "点击开始", "立即开始", "开始挑战",
]

OVER_WORDS = ["本次得分", "历史最高分", "游戏结束", "挑战失败", "再玩一局"]
LIST_WORDS = ["好友排行", "排行榜", "每周一凌晨刷新", "我的排名", "查看全部排行",
              "本周最高分", "精彩回放", "用键鼠玩游戏", "添加到桌面", "置顶小程序",
              "已开启声音", "福利", "更多游戏"]


class OcrReader:
    def __init__(self):
        self._engine = None
        self._loop = None
        self._ready = None

    def _init(self):
        if self._ready is not None:
            return self._ready
        try:
            from winrt.windows.media.ocr import OcrEngine
            from winrt.windows.globalization import Language
            self._loop = asyncio.new_event_loop()
            engine = OcrEngine.try_create_from_language(Language("zh-Hans"))
            if engine is None:
                engine = OcrEngine.try_create_from_user_profile_languages()
            self._engine = engine
            self._ready = engine is not None
        except Exception:
            self._ready = False
        return self._ready

    def read(self, bgr):
        if not self._init():
            return []
        try:
            from winrt.windows.graphics.imaging import (
                SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode)
            from winrt.windows.security.cryptography import CryptographicBuffer
            h, w = bgr.shape[:2]
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[:, :, 0] = bgr[:, :, 2]
            rgba[:, :, 1] = bgr[:, :, 1]
            rgba[:, :, 2] = bgr[:, :, 0]
            rgba[:, :, 3] = 255
            buf = CryptographicBuffer.create_from_byte_array(bytes(rgba.tobytes()))
            sb = SoftwareBitmap.create_copy_from_buffer(
                buf, BitmapPixelFormat.BGRA8, int(w), int(h),
                BitmapAlphaMode.PREMULTIPLIED)
            result = self._loop.run_until_complete(self._engine.recognize_async(sb))
        except Exception:
            return []
        lines = []
        for line in result.lines:
            words = list(line.words)
            if not words:
                continue
            x0 = min(wd.bounding_rect.x for wd in words)
            y0 = min(wd.bounding_rect.y for wd in words)
            x1 = max(wd.bounding_rect.x + wd.bounding_rect.width for wd in words)
            y1 = max(wd.bounding_rect.y + wd.bounding_rect.height for wd in words)
            text = line.text.replace(" ", "").replace("\u3000", "")
            lines.append((text, (x0 + x1) / 2.0, (y0 + y1) / 2.0,
                          x1 - x0, y1 - y0))
        return lines

    def locate(self, bgr, allow_shape=True):
        lines = self.read(bgr)
        text = "".join(t for t, *_ in lines)
        over = any(kw in text for kw in OVER_WORDS)
        listed = any(kw in text for kw in LIST_WORDS)
        for kw in KEYWORDS:
            for t, cx, cy, wd, ht in lines:
                if kw in t:
                    return {"found": 1, "kind": "start", "x": float(cx),
                            "y": float(cy), "text": t, "source": "ocr"}
        if listed:
            h, w = bgr.shape[:2]
            return {"found": 1, "kind": "list", "x": w * 0.06,
                    "y": h * 0.055, "text": "排行榜", "source": "ocr"}
        if over:
            if allow_shape:
                box = find_button(bgr)
                if box:
                    return {"found": 1, "kind": "over", "x": float(box[0]),
                            "y": float(box[1]), "text": "结算按钮", "source": "shape"}
            return {"found": 1, "kind": "over", "x": None, "y": None,
                    "text": "结算页", "source": "ocr"}
        if allow_shape:
            box = find_button(bgr)
            if box:
                return {"found": 1, "kind": "button", "x": float(box[0]),
                        "y": float(box[1]), "text": "按钮", "source": "shape"}
        return None

    def text_of(self, bgr):
        return " ".join(t for t, *_ in self.read(bgr))


def find_button(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 215).astype(np.uint8)
    top = int(h * 0.40)
    roi = mask[top:, :]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(roi, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if bw < 0.22 * w or bh < 0.028 * h or bh > 0.22 * h:
            continue
        if area < 0.006 * w * h:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.8 or aspect > 12.0:
            continue
        score = area * (1.0 if 2.2 < aspect < 7.0 else 0.6)
        if best is None or score > best[0]:
            best = (score, x, y, bw, bh)
    if best is None:
        return None
    _, x, y, bw, bh = best
    return (x + bw / 2.0, y + bh / 2.0 + top)
