import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_PATH = os.environ.get(
    "JUMP_SPECIAL_MEMORY") or os.path.join(ROOT, "ml", "special_memory.json")
MEMORY_LIMIT = 40
MATCH_DIST = 0.18


def _crop_keep(sub, block, piece):
    h, w = sub.shape[:2]
    x0 = max(0, int(block["left"]) + 4)
    x1 = min(w, int(block["right"]) - 4)
    y0 = max(0, int(block["top"]) + 4)
    y1 = min(h, int(block["bottom"]) - 3)
    if x1 - x0 < 24 or y1 - y0 < 18:
        return None, None
    crop = sub[y0:y1, x0:x1]
    px, py = int(piece[0]) - x0, int(piece[1]) - y0
    keep = np.ones(crop.shape[:2], bool)
    keep[max(0, py - 96):min(crop.shape[0], py + 12),
         max(0, px - 30):min(crop.shape[1], px + 30)] = False
    if keep.sum() < 200:
        return None, None
    return crop, keep


def block_features(sub, block, piece):
    crop, keep = _crop_keep(sub, block, piece)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2][keep].astype(np.int16)
    s = hsv[:, :, 1][keep]
    hu = hsv[:, :, 0][keep]
    sat_mask = (s > 100) & (v > 60)
    hues = hu[sat_mask]
    hstd = float(hues.std()) / 90.0 if hues.size else 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120) > 0
    return [
        float((v < 45).mean()),
        float((v < 95).mean()),
        float(sat_mask.mean()),
        min(1.0, hstd),
        float((v > 200).mean()),
        float(edges[keep].mean()),
    ]


def _load_memory():
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [[float(v) for v in row] for row in data if len(row) == 6]
    except Exception:
        pass
    return []


def memory_count():
    return len(_load_memory())


def remember(features):
    if features is None:
        return memory_count()
    rows = _load_memory()
    rows.append([float(v) for v in features])
    rows = rows[-MEMORY_LIMIT:]
    try:
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f)
    except Exception:
        pass
    return len(rows)


def matches_memory(features):
    if features is None:
        return False
    rows = _load_memory()
    if not rows:
        return False
    a = np.asarray(features, np.float32)
    for row in rows:
        b = np.asarray(row, np.float32)
        if float(np.linalg.norm(a - b)) <= MATCH_DIST:
            return True
    return False


def block_texture(sub, block, piece):
    h, w = sub.shape[:2]
    x0 = max(0, int(block["left"]) + 4)
    x1 = min(w, int(block["right"]) - 4)
    y0 = max(0, int(block["top"]) + 4)
    y1 = min(h, int(block["bottom"]) - 3)
    if x1 - x0 < 24 or y1 - y0 < 18:
        return None
    crop = sub[y0:y1, x0:x1]
    px, py = int(piece[0]) - x0, int(piece[1]) - y0
    keep = np.ones(crop.shape[:2], bool)
    keep[max(0, py - 96):min(crop.shape[0], py + 12),
         max(0, px - 30):min(crop.shape[1], px + 30)] = False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120) > 0
    if keep.sum() < 200:
        return None
    return float(edges[keep].mean())


def special_block(sub, block, piece):
    h, w = sub.shape[:2]
    x0 = max(0, int(block["left"]) + 4)
    x1 = min(w, int(block["right"]) - 4)
    y0 = max(0, int(block["top"]) + 4)
    y1 = min(h, int(block["bottom"]) - 3)
    if x1 - x0 < 24 or y1 - y0 < 18:
        return None
    crop = sub[y0:y1, x0:x1]
    px, py = int(piece[0]) - x0, int(piece[1]) - y0
    keep = np.ones(crop.shape[:2], bool)
    keep[max(0, py - 96):min(crop.shape[0], py + 12),
         max(0, px - 30):min(crop.shape[1], px + 30)] = False
    if keep.sum() < 200:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2][keep].astype(np.int16)
    s = hsv[:, :, 1][keep]
    hu = hsv[:, :, 0][keep]
    if float((v < 45).mean()) > 0.22:
        return "黑胶唱片"
    if float((v < 95).mean()) > 0.32:
        return "井盖"
    sat = (s > 100) & (v > 60)
    if float(sat.mean()) > 0.18:
        hues = hu[sat]
        if float(hues.std()) > 26.0:
            return "魔方"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120) > 0
    bright = float((v > 200).mean())
    if bright > 0.22 and float(sat.mean()) < 0.10 and float(edges[keep].mean()) > 0.07:
        return "便利店"
    return None
