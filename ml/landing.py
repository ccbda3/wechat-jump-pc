import numpy as np

from ocr import bg_color


def block_under(bgr, piece):
    h, w = bgr.shape[:2]
    px, py = int(piece[0]), int(piece[1])
    ref = bg_color(bgr)
    for dy in (3, 8, 14, 22, 34, 48):
        y = py + dy
        if y >= h - 1:
            break
        row = bgr[y].astype(np.int16)
        on = np.abs(row - ref).sum(axis=1) > 45
        if not on[px]:
            continue
        spans = np.nonzero(on)[0]
        segs = np.split(spans, np.where(np.diff(spans) > 3)[0] + 1)
        seg = None
        for s in segs:
            if s[0] <= px <= s[-1]:
                seg = s
                break
        if seg is None:
            continue
        x0, x1 = int(seg[0]), int(seg[-1])
        if x1 - x0 < max(55, w * 0.14):
            continue
        d = np.abs(row - ref).sum(axis=1)
        strong = np.where(d[x0:x1 + 1] > max(55.0, 0.55 * float(d[x0:x1 + 1].max())))[0]
        if len(strong) > 0:
            x0, x1 = x0 + int(strong[0]), x0 + int(strong[-1])
        top = y
        for yy in range(y - 1, max(0, y - 90), -1):
            row2 = bgr[yy].astype(np.int16)
            d2 = np.abs(row2 - ref).sum(axis=1)
            if d2[max(0, x0):min(w, x1 + 1)].mean() < 30:
                break
            top = yy
        cx = (x0 + x1) / 2.0
        return {"x": cx, "top": float(top), "bottom": float(y),
                "width": float(x1 - x0), "left": float(x0), "right": float(x1)}
    return None


def landing_reward(bgr, piece):
    block = block_under(bgr, piece)
    if block is None:
        return None, None, None
    err = abs(float(piece[0]) - block["x"])
    scale = max(16.0, block["width"] * 0.24)
    reward = float(np.exp(-(err / scale) ** 2))
    return reward, err, block
