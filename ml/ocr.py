import asyncio
import os

import cv2
import numpy as np

TPL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "score_templates.npz")
TPL_SHAPE = (34, 26)
TPL_TARGET_H = 32

KEYWORDS = [
    "再玩一局", "再来一局", "再玩一次", "再一次", "重新开始", "重新挑战",
    "继续游戏", "开始游戏", "点击开始", "立即开始", "开始挑战",
]

OVER_WORDS = ["再玩一局"]
LIST_WORDS = ["好友排行榜", "查看群排行", "我的排名", "查看全部排行",
              "每周一凌晨刷新", "本周最高分", "精彩回放"]


def merge_boxes(mask, min_h=30, max_h=90):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    raw = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 40 or bh < 12:
            continue
        raw.append([x, y, bw, bh])
    merged = True
    while merged:
        merged = False
        for i in range(len(raw)):
            for j in range(i + 1, len(raw)):
                a, b = raw[i], raw[j]
                ax2, ay2 = a[0] + a[2], a[1] + a[3]
                bx2, by2 = b[0] + b[2], b[1] + b[3]
                dx = max(0, max(a[0], b[0]) - min(ax2, bx2))
                dy = max(0, max(a[1], b[1]) - min(ay2, by2))
                if dx <= 5 and dy <= 5:
                    raw[i] = [min(a[0], b[0]), min(a[1], b[1]),
                              max(ax2, bx2) - min(a[0], b[0]),
                              max(ay2, by2) - min(a[1], b[1])]
                    raw.pop(j)
                    merged = True
                    break
            if merged:
                break
    raw.sort(key=lambda b: b[0])
    return [b for b in raw if min_h <= b[3] <= max_h]


def normalize_glyph(gray, box):
    x, y, bw, bh = box
    g = gray[y:y + bh, x:x + bw]
    g = cv2.resize(g, (max(1, round(bw * TPL_TARGET_H / max(1, bh))), TPL_TARGET_H),
                   interpolation=cv2.INTER_AREA)
    canvas = np.full(TPL_SHAPE, 255, np.uint8)
    hh, ww = g.shape
    if ww > TPL_SHAPE[1]:
        g = g[:, :TPL_SHAPE[1]]
        hh, ww = g.shape
    ox = (TPL_SHAPE[1] - ww) // 2
    oy = (TPL_SHAPE[0] - hh) // 2
    canvas[oy:oy + hh, ox:ox + ww] = g
    return (canvas < 150).astype(np.float32)


class OcrReader:
    _tpl = None

    def __init__(self):
        self._engine = None
        self._loop = None
        self._ready = None

    def templates(self):
        if OcrReader._tpl is not None:
            return OcrReader._tpl
        try:
            data = np.load(TPL_PATH)
            OcrReader._tpl = {int(k[1:]): data[k] for k in data.files}
        except Exception:
            OcrReader._tpl = {}
        return OcrReader._tpl

    def score(self, frame):
        sub, _ = game_crop(frame)
        v, _ = self.score_crop(sub)
        return v

    def score_crop(self, sub):
        tpl = self.templates()
        if not tpl:
            return None, []
        h, w = sub.shape[:2]
        roi = sub[int(h * 0.10):int(h * 0.27), int(w * 0.02):int(w * 0.52)]
        if roi.size == 0:
            return None, []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = (gray < 150).astype(np.uint8)
        boxes = merge_boxes(mask)
        if not boxes:
            return None, []
        digits = ""
        for box in boxes:
            g = normalize_glyph(gray, box)
            scores = {}
            for d, arrs in tpl.items():
                for t in arrs:
                    inter = float(np.minimum(g, t).sum())
                    union = float(np.maximum(g, t).sum())
                    iou = inter / union if union > 0 else 0.0
                    if iou > scores.get(d, 0.0):
                        scores[d] = iou
            if not scores:
                return None, []
            best = max(scores.items(), key=lambda kv: kv[1])
            second = max([v for k, v in scores.items() if k != best[0]] or [0.0])
            if best[1] < 0.62 or (best[1] - second) < 0.02:
                return None, []
            digits += str(best[0])
        if not digits:
            return None, []
        off_x = int(w * 0.02)
        off_y = int(h * 0.10)
        out = [(b[0] + off_x, b[1] + off_y, b[2], b[3]) for b in boxes]
        return int(digits), out

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
        if over:
            if allow_shape:
                box = find_button(bgr)
                if box:
                    return {"found": 1, "kind": "over", "x": float(box[0]),
                            "y": float(box[1]), "text": "结算按钮", "source": "shape"}
            return {"found": 1, "kind": "over", "x": None, "y": None,
                    "text": "结算页", "source": "ocr"}
        if listed:
            box = find_round_back(bgr)
            if box:
                return {"found": 1, "kind": "list", "x": float(box[0]),
                        "y": float(box[1]), "text": "排行榜返回", "source": "arrow"}
            return {"found": 1, "kind": "list", "x": None, "y": None,
                    "text": "排行榜", "source": "ocr"}
        if allow_shape:
            box = find_button(bgr)
            if box:
                return {"found": 1, "kind": "button", "x": float(box[0]),
                        "y": float(box[1]), "text": "按钮", "source": "shape"}
        return None

    def text_of(self, bgr):
        return " ".join(t for t, *_ in self.read(bgr))


def game_crop(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cm = gray.mean(axis=0)
    if cm[:max(1, int(w * 0.10))].mean() < 235:
        return frame, 0
    limit = int(w * 0.72)
    for x in range(int(w * 0.15), limit):
        if cm[x] < 228 and cm[x:x + 24].mean() < 222:
            x0 = min(x + 6, w - 8)
            return frame[:, x0:w - 6], x0
    return frame, 0


def bg_color(bgr):
    small = cv2.resize(bgr, (max(1, bgr.shape[1] // 4), max(1, bgr.shape[0] // 4)),
                       interpolation=cv2.INTER_AREA)
    q = (small // 16) * 16
    keys = (q[:, :, 0].astype(np.int64) * 65536 +
            q[:, :, 1].astype(np.int64) * 256 + q[:, :, 2].astype(np.int64))
    vals, counts = np.unique(keys, return_counts=True)
    sel = keys == vals[int(np.argmax(counts))]
    return np.median(small[sel], axis=0).astype(np.int16)


def find_round_back(bgr):
    sub, x0 = game_crop(bgr)
    h, w = sub.shape[:2]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    mask = (gray > 232).astype(np.uint8)
    top = int(h * 0.70)
    left = int(w * 0.02)
    right = int(w * 0.34)
    roi = mask[top:, left:right]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(roi, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if bw < 0.045 * w or bw > 0.17 * w:
            continue
        if bh < 0.020 * h or bh > 0.10 * h:
            continue
        ar = bw / max(bh, 1)
        if ar < 0.80 or ar > 1.25:
            continue
        if area < 0.58 * bw * bh:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, bw, bh)
    if best is None:
        return None
    _, x, y, bw, bh = best
    return (x0 + left + x + bw / 2.0, top + y + bh / 2.0)


def find_button(bgr):
    sub, x0 = game_crop(bgr)
    h, w = sub.shape[:2]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    mask = (gray > 225).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if bw < 0.12 * w or bw > 0.50 * w:
            continue
        if bh < 0.025 * h or bh > 0.10 * h:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 2.2 or aspect > 7.0:
            continue
        if area < 0.60 * bw * bh:
            continue
        cy = y + bh / 2.0
        if cy < 0.45 * h or cy > 0.90 * h:
            continue
        score = area
        if best is None or score > best[0]:
            best = (score, x, y, bw, bh)
    if best is None:
        return None
    _, x, y, bw, bh = best
    return (x0 + x + bw / 2.0, y + bh / 2.0)
