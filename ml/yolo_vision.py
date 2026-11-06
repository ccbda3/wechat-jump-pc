import math
import os

from ai_vision import FrameResult, find_top_spot

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

    def analyze(self, bgr):
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
        best = None
        for p in plats:
            if p[2] >= py - 4.0:
                continue
            cx = (p[1] + p[3]) / 2.0
            dist = math.hypot(cx - px, p[2] - py)
            if best is None or dist < best[0]:
                best = (dist, p)
        if best is None:
            return None
        p = best[1]
        spot = find_top_spot(bgr, (p[1], p[2], p[3] - p[1], p[4] - p[2]))
        if spot:
            tx, ty = spot
        else:
            tx = (p[1] + p[3]) / 2.0
            ty = p[2] + 0.30 * (p[4] - p[2])
        d = math.hypot(tx - px, ty - py)
        return FrameResult(piece=(px, py), board=(tx, ty), distance=d,
                           confidence=min(piece[0], p[0]), method="yolo",
                           box=(p[1], p[2], p[3] - p[1], p[4] - p[2]))
