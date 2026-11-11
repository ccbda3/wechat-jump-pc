import cv2
import numpy as np


def project_center(before, after, piece_xy, box):
    if before is None or after is None or box is None:
        return None
    h, w = before.shape[:2]
    px, py = int(piece_xy[0]), int(piece_xy[1])
    rx = 72
    x0 = max(0, px - rx)
    x1 = min(w, px + rx)
    y0 = max(0, py + 8)
    y1 = min(h, py + 8 + 76)
    if x1 - x0 < 60 or y1 - y0 < 30:
        return None
    tpl = cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    tgt = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    if tgt.shape[0] < tpl.shape[0] or tgt.shape[1] < tpl.shape[1]:
        return None
    res = cv2.matchTemplate(tgt, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < 0.45:
        return None
    dx = loc[0] - x0
    if abs(dx) > 420:
        return None
    exp = px - (box[0] + box[2] / 2.0)
    if abs(dx - exp) > 0.5 * abs(exp) + 70:
        return None
    return float(box[0] + box[2] / 2.0) + dx


def block_under(bgr, piece):
    h, w = bgr.shape[:2]
    px, py = int(piece[0]), int(piece[1])
    y0 = py + 2
    y1 = min(h - 1, py + 60)
    x0 = max(0, int(px - w * 0.34))
    x1 = min(w, int(px + w * 0.34))
    if y1 - y0 < 10 or x1 - x0 < 60:
        return None
    rows = []
    for y in range(y0, y1):
        row = bgr[y].astype(np.int16)
        med = np.median(row[::3, :], axis=0)
        on = np.abs(row - med).sum(axis=1) > 26
        xs = np.nonzero(on[x0:x1])[0]
        xs = xs + x0
        if xs.size < 8:
            continue
        segs = np.split(xs, np.where(np.diff(xs) > 4)[0] + 1)
        seg = None
        for s in segs:
            if s[0] <= px <= s[-1]:
                seg = s
                break
        if seg is None:
            near = [s for s in segs
                    if min(abs(int(s[0]) - px), abs(int(s[-1]) - px)) <= 14]
            if near:
                seg = min(near, key=lambda s: abs((int(s[0]) + int(s[-1])) / 2.0 - px))
        if seg is None:
            continue
        if int(seg[-1]) - int(seg[0]) < 18:
            continue
        rows.append((y, int(seg[0]), int(seg[-1])))
    if len(rows) < 6:
        return None
    mids = np.array([(a + b) / 2.0 for _, a, b in rows])
    spans = np.array([b - a for _, a, b in rows], dtype=np.float64)
    med_span = float(np.median(spans))
    if med_span < max(60.0, w * 0.10):
        return None
    keep = spans <= med_span * 1.3
    mids = mids[keep]
    spans = spans[keep]
    if mids.size < 4:
        return None
    center = float(np.median(mids))
    width = float(np.median(spans))
    if width > w * 0.55:
        return None
    left = center - width / 2.0
    right = center + width / 2.0
    top = float(rows[0][0])
    bottom = float(rows[-1][0])
    return {"x": center, "top": top,
            "bottom": bottom, "width": width,
            "left": left, "right": right}


def landing_reward(bgr, piece):
    block = block_under(bgr, piece)
    if block is None:
        return None, None, None
    err = abs(float(piece[0]) - block["x"])
    scale = max(16.0, block["width"] * 0.24)
    reward = float(np.exp(-(err / scale) ** 2))
    return reward, err, block
