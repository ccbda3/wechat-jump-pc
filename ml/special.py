import cv2
import numpy as np


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
