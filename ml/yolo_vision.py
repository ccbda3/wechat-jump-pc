import math
import os

import cv2
import numpy as np

from ai_vision import FrameResult, face_center_y
from correct import Corrector
from ocr import bg_color

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "yolo_jump.pt")
IMGSZ = 640
RETRY_IMGSZ = 1152


class YoloVision:
    def __init__(self, conf=0.35):
        self.conf = float(conf)
        self.model = None
        self.names = {}
        self._tried = False
        self.calib = Corrector()

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

    def _boxes(self, res):
        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            boxes.append((str(self.names[int(b.cls)]), float(b.conf), x1, y1, x2, y2))
        return boxes

    def raw(self, bgr):
        if not self.load():
            return None
        try:
            res = self.model.predict(bgr, conf=self.conf, imgsz=IMGSZ,
                                     verbose=False, device="cpu")[0]
            boxes = self._boxes(res)
            has_plat = any(b[0] == "platform" for b in boxes)
            has_piece = any(b[0] == "piece" for b in boxes)
            if not (has_plat and has_piece):
                res = self.model.predict(bgr, conf=self.conf, imgsz=RETRY_IMGSZ,
                                         verbose=False, device="cpu")[0]
                boxes = self._boxes(res)
            return boxes
        except Exception:
            return None

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

    def _grow_up(self, bgr, p, bg):
        h, w = bgr.shape[:2]
        x1 = max(0, int(p[1]))
        y1 = max(0, int(p[2]))
        x2 = min(w, int(p[3]))
        y2 = min(h, int(p[4]))
        if x2 - x1 < 24 or y2 - y1 < 12:
            return p
        xa = x1 + int((x2 - x1) * 0.22)
        xb = x2 - int((x2 - x1) * 0.22)
        limit = int((y2 - y1) * 1.5)
        while y1 > 0 and (int(p[2]) - y1) < limit:
            ya = max(0, y1 - 4)
            if ya >= y1:
                break
            row = bgr[ya:y1, xa:xb]
            if row.size == 0:
                break
            diff = np.abs(row.astype(np.int16) - np.asarray(bg, np.int16)).sum(axis=2)
            if float((diff > 45).mean()) < 0.5:
                break
            y1 = ya
        if y1 == int(p[2]):
            return p
        return (p[0], float(x1), float(y1), float(x2), float(y2))

    def _open_above(self, bgr, p, bg):
        h, w = bgr.shape[:2]
        x1 = max(0, int(p[1]))
        y1 = max(0, int(p[2]))
        x2 = min(w, int(p[3]))
        y2 = min(h, int(p[4]))
        bw, bh = x2 - x1, y2 - y1
        band = max(6, int(bh * 0.14))
        ya = max(0, y1 - band)
        yb = y1 - 1
        if yb - ya < 3:
            return True
        xa = x1 + int(bw * 0.18)
        xb = x2 - int(bw * 0.18)
        if xb - xa < 8:
            xa, xb = x1, x2
        roi = bgr[ya:yb, xa:xb]
        if roi.size == 0:
            return True
        diff = np.abs(roi.astype(np.int16) - np.asarray(bg, np.int16)).sum(axis=2)
        return float((diff <= 40).mean()) >= 0.6

    def _iou(self, p, box):
        ax1, ay1, ax2, ay2 = p[1], p[2], p[3], p[4]
        bx1, by1 = box[0], box[1]
        bx2, by2 = box[0] + box[2], box[1] + box[3]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / area if area > 0 else 0.0

    def _board_point(self, bgr, p, bg):
        tx = (p[1] + p[3]) / 2.0
        ty = face_center_y(bgr, (p[1], p[2], p[3] - p[1], p[4] - p[2]), bg)
        return tx, ty

    def _pick(self, plats, px, py, old_box=None, extra=True):
        best = None
        for p in plats:
            bw = p[3] - p[1]
            if bw < 55:
                continue
            face_y = p[2] + (p[4] - p[2]) * 0.30
            if abs(face_y - py) < 40.0 and p[1] - 6 <= px <= p[3] + 6:
                continue
            if p[1] + 6 <= px <= p[3] - 6 and p[4] > py + 4:
                continue
            if old_box is not None and self._iou(p, old_box) > 0.30:
                continue
            cx = (p[1] + p[3]) / 2.0
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
        if piece is None:
            return None
        px = (piece[1] + piece[3]) / 2.0
        py = piece[4] - 2.0
        if not plats:
            return FrameResult(piece=(px, py), confidence=piece[0], method="yolo")
        bg = bg_color(bgr)
        plats = [self._grow_up(bgr, p, bg) for p in plats]
        old_box = None
        if ref is not None:
            prev_box = ref[2] if len(ref) > 2 else None
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
            if cur is not None and prev_box is not None:
                dx = cur[1] - prev_box[0]
                dy = cur[2] - prev_box[1]
                if math.hypot(dx, dy) < 480:
                    old_box = (prev_box[0] + dx, prev_box[1] + dy,
                               prev_box[2], prev_box[3])
        best = self._pick(plats, px, py, old_box)
        penalty = 1.0
        if best is not None:
            ok = (self._face_like(bgr, best[1], bg)
                  and self._open_above(bgr, best[1], bg))
            if not ok:
                filtered = [p for p in plats
                            if self._face_like(bgr, p, bg)
                            and self._open_above(bgr, p, bg)]
                alt = self._pick(filtered, px, py, old_box)
                if alt is not None:
                    best = alt
                else:
                    penalty = 0.55
        if best is None:
            best = self._pick(plats, px, py, old_box)
            penalty = 0.5
        if best is None:
            return None
        p = best[1]
        tx, ty = self._board_point(bgr, p, bg)
        d = math.hypot(tx - px, ty - py)
        method = "yolo"
        box = (p[1], p[2], p[3] - p[1], p[4] - p[2])
        conf = min(piece[0], p[0]) * penalty
        if self.calib.count:
            s0 = self.calib.score(bgr, (tx, ty))
            hit = self.calib.locate(bgr)
            if hit is not None and hit[2] >= 0.62 and hit[2] > s0 + 0.12:
                hx, hy = hit[0], hit[1]
                if (math.hypot(hx - px, hy - py) > 60.0
                        and 20 < hx < bgr.shape[1] - 20
                        and 20 < hy < bgr.shape[0] - 20):
                    tx, ty = hx, hy
                    d = math.hypot(tx - px, ty - py)
                    method = "calib"
                    box = None
                    conf = max(conf, 0.75)
        return FrameResult(piece=(px, py), board=(tx, ty), distance=d,
                           confidence=conf, method=method, box=box)
