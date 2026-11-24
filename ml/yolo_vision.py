import math
import os

import cv2
import numpy as np

from ai_vision import FrameResult, face_center_y, find_top_spot
from ocr import bg_color

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "yolo_jump.pt")


class YoloVision:
    def __init__(self, conf=0.35):
        self.conf = float(conf)
        self.model = None
        self.names = {}
        self._tried = False

    def load(self):
        if self._tried:
            return self.model is not None
        self._tried = True
        try:
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            self.names = self.model.names
        except Exception:
            self.model = None
        return self.model is not None

    def raw(self, bgr):
        if not self.load():
            return None
        try:
            res = self.model.predict(bgr, conf=self.conf, verbose=False, device="cpu")[0]
        except Exception:
            return None
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            boxes.append((str(self.names[int(b.cls)]), float(b.conf), x1, y1, x2, y2))
        return boxes

    def _face_like(self, bgr, p, bg):
        h, w = bgr.shape[:2]
        x1, y1 = max(0, int(p[1])), max(0, int(p[2]))
        x2, y2 = min(w, int(p[3])), min(h, int(p[4]))
        bw, bh = x2 - x1, y2 - y1
        if bw < 24 or bh < 12:
            return False
        yA = y1 + int(bh * 0.10)
        yB = y1 + int(bh * 0.55)
        roi = bgr[yA:yB, x1:x2]
        if roi.size == 0:
            return False
        diff = np.abs(roi.astype(np.int16) - np.asarray(bg, np.int16)).sum(axis=2)
        if float(diff.mean()) < 40:
            return False
        on = diff > 45
        widths = np.zeros(on.shape[0], np.int32)
        for r in range(on.shape[0]):
            xs = np.nonzero(on[r])[0]
            if xs.size:
                widths[r] = xs[-1] - xs[0] + 1
        if float((widths >= 0.70 * bw).mean()) < 0.25:
            return False
        return True

    def _pick(self, plats, px, py, origin, extra=True):
        best = None
        for p in plats:
            bw = p[3] - p[1]
            if bw < 55:
                continue
            face_y = p[2] + (p[4] - p[2]) * 0.30
            if abs(face_y - py) < 45.0:
                continue
            if p[1] + 6 <= px <= p[3] - 6 and p[4] > py + 4:
                continue
            cx = (p[1] + p[3]) / 2.0
            if origin is not None:
                tc = (cx, p[2] + (p[4] - p[2]) * 0.30)
                if math.hypot(tc[0] - origin[0], tc[1] - origin[1]) < 0.62 * bw:
                    continue
            dist = math.hypot(cx - px, p[2] - py)
            if best is None or dist < best[0]:
                best = (dist, p)
        return best

    def analyze(self, bgr, ref=None):
        boxes = self.raw(bgr)
        if not boxes:
            return None
        piece = None
        plats = []
        for name, conf, x1, y1, x2, y2 in boxes:
            if name == "piece":
                if piece is None or conf > piece[0]:
                    piece = (conf, x1, y1, x2, y2)
            elif name == "platform":
                plats.append((conf, x1, y1, x2, y2))
        if piece is None or not plats:
            return None
        px = (piece[1] + piece[3]) / 2.0
        py = piece[4] - 2.0
        bg = bg_color(bgr)
        origin = None
        if ref is not None:
            prev_piece, prev_board = ref
            cur = None
            for p in plats:
                bw = p[3] - p[1]
                bh = p[4] - p[2]
                if bw < 55:
                    continue
                face_y = p[2] + bh * 0.30
                if abs(face_y - py) < 45.0 and p[1] - 6 <= px <= p[3] + 6:
                    if cur is None or bw > (cur[3] - cur[1]):
                        cur = p
            if cur is not None:
                dx = (cur[1] + cur[3]) / 2.0 - prev_board[0]
                dy = (cur[2] + cur[4]) / 2.0 - prev_board[1]
                if math.hypot(dx, dy) < 480:
                    origin = (prev_piece[0] + dx, prev_piece[1] + dy)
        best = self._pick(plats, px, py, origin)
        if best is not None and not self._face_like(bgr, best[1], bg):
            filtered = [p for p in plats if self._face_like(bgr, p, bg)]
            best = self._pick(filtered, px, py, origin)
            if best is None:
                best = self._pick(plats, px, py, origin)
        if best is None:
            best = self._pick(plats, px, py, None)
        if best is None:
            return None
        p = best[1]
        spot = find_top_spot(bgr, (p[1], p[2], p[3] - p[1], p[4] - p[2]))
        if spot:
            tx, ty = spot
        else:
            tx = (p[1] + p[3]) / 2.0
            ty = face_center_y(bgr, (p[1], p[2], p[3] - p[1], p[4] - p[2]), bg)
        d = math.hypot(tx - px, ty - py)
        return FrameResult(piece=(px, py), board=(tx, ty), distance=d,
                           confidence=min(piece[0], p[0]), method="yolo",
                           box=(p[1], p[2], p[3] - p[1], p[4] - p[2]))
