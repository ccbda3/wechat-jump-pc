import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.environ.get("JUMP_BLOCK_DIR") or os.path.join(ROOT, "ml", "blocks")
LIB_CSV = os.environ.get("JUMP_BLOCK_LIB") or os.path.join(ROOT, "ml", "block_lib.csv")
CHECK_CSV = os.environ.get(
    "JUMP_BLOCK_CHECKS") or os.path.join(ROOT, "ml", "block_checks.csv")
LIMIT = 60
DEDUP = 0.06
MATCH = 0.10


def _load_lib():
    rows = []
    try:
        with open(LIB_CSV, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 11:
                    continue
                try:
                    idx = int(parts[0])
                    feat = [float(v) for v in parts[2:11]]
                except Exception:
                    continue
                rows.append((idx, parts[1], feat))
    except Exception:
        pass
    return rows


def _load_checks():
    out = {}
    try:
        with open(CHECK_CSV, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        out[int(parts[0])] = int(parts[1])
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def checked_count():
    return sum(1 for v in _load_checks().values() if v)


def _dist(a, b):
    return float(np.linalg.norm(
        np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def observe(sub, block, piece, feat):
    if block is None or feat is None or len(feat) < 9:
        return None
    rows = _load_lib()
    if len(rows) >= LIMIT:
        return None
    for _, _, old in rows:
        if _dist(feat, old) < DEDUP:
            return None
    h, w = sub.shape[:2]
    x0 = max(0, int(block["left"]) - 8)
    x1 = min(w, int(block["right"]) + 8)
    y0 = max(0, int(block["top"]) - 8)
    y1 = min(h, int(block["bottom"]) + 8)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    try:
        os.makedirs(LIB_DIR, exist_ok=True)
        idx = max([r[0] for r in rows], default=0) + 1
        name = "b%04d.bmp" % idx
        crop = sub[y0:y1, x0:x1]
        thumb = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(LIB_DIR, name), thumb)
        with open(LIB_CSV, "a", encoding="utf-8") as f:
            f.write("%d,%s,%s\n" % (
                idx, name, ",".join("%.4f" % float(v) for v in feat)))
        return idx
    except Exception:
        return None


def is_special(feat):
    if feat is None or len(feat) < 9:
        return False
    checks = _load_checks()
    if not any(checks.values()):
        return False
    for idx, _, old in _load_lib():
        if checks.get(idx) and _dist(feat, old) <= MATCH:
            return True
    return False
