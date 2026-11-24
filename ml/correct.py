import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.environ.get(
    "JUMP_CALIB_PATH") or os.path.join(ROOT, "ml", "vision_calib.npz")
PATCH = 96
HALF = PATCH // 2
LIMIT = 60


def _grad(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)


class Corrector:
    def __init__(self, path=CALIB_PATH):
        self.path = path
        self.patches = []
        self.shapes = []
        self._load()

    @property
    def count(self):
        return len(self.patches)

    def _load(self):
        try:
            data = np.load(self.path)
            patches = np.asarray(data["patches"])
            shapes = np.asarray(data["shapes"])
            for i in range(min(len(patches), len(shapes))):
                self.patches.append(patches[i].astype(np.uint8))
                self.shapes.append((int(shapes[i][0]), int(shapes[i][1])))
        except Exception:
            self.patches = []
            self.shapes = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            np.savez_compressed(
                self.path,
                patches=np.asarray(self.patches, dtype=np.uint8),
                shapes=np.asarray(self.shapes, dtype=np.int32))
        except Exception:
            pass

    def _patch(self, frame, point):
        h, w = frame.shape[:2]
        x = int(round(point[0]))
        y = int(round(point[1]))
        x0 = x - HALF
        y0 = y - HALF
        if x0 < 0 or y0 < 0 or x0 + PATCH > w or y0 + PATCH > h:
            return None
        roi = frame[y0:y0 + PATCH, x0:x0 + PATCH]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        patch = _grad(gray)
        if float(patch.std()) < 4.0:
            return None
        return patch

    def add(self, frame, point):
        patch = self._patch(frame, point)
        if patch is None:
            return self.count
        self.patches.append(patch)
        self.shapes.append((int(frame.shape[1]), int(frame.shape[0])))
        if len(self.patches) > LIMIT:
            self.patches = self.patches[-LIMIT:]
            self.shapes = self.shapes[-LIMIT:]
        self._save()
        return self.count

    def _batch(self, frame):
        shape = (int(frame.shape[1]), int(frame.shape[0]))
        return [p for p, s in zip(self.patches, self.shapes) if s == shape]

    def score(self, frame, point):
        patch = self._patch(frame, point)
        batch = self._batch(frame)
        if patch is None or not batch:
            return 0.0
        a = patch.astype(np.float32)
        a -= a.mean()
        da = float(np.sqrt((a * a).sum()))
        if da < 1e-6:
            return 0.0
        best = 0.0
        for tpl in batch:
            b = tpl.astype(np.float32)
            b -= b.mean()
            db = float(np.sqrt((b * b).sum()))
            if db < 1e-6:
                continue
            best = max(best, float((a * b).sum() / (da * db)))
        return best

    def locate(self, frame):
        batch = self._batch(frame)
        if not batch:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feat = _grad(gray)
        best = None
        for tpl in batch:
            res = cv2.matchTemplate(feat, tpl, cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-1.0)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[2]:
                best = (loc[0] + HALF, loc[1] + HALF, float(mx))
        return best
