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
