# -*- coding: utf-8 -*-
"""AI 模式感知层：稳健定位棋子与目标台面，并判断落稳 / 结算页。

替代原版的三个脆弱假设：
  * 硬编码 RGB (50-60, 53-63, 95-110) 找棋子  -> HSV 范围 + 连通域形状过滤
  * 逐行颜色突变找目标 + 固定斜率 25.5/43.5 推算 Y -> 直接检测台面轮廓中心
  * 完全不判断游戏状态 -> 结算页 / 广告区识别，避免拿失败样本污染校准器

所有阈值集中在 VisionConfig，可用 vision_config.json 覆盖。
命令行：
    python ai_vision.py --debug debug_samples/xxx.png     # 标注并打印识别结果
    python ai_vision.py --collect 20                      # 开局后采集真实画面样本
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_config.json")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_samples")


@dataclass
class VisionConfig:
    # 棋子（紫色小人）：OpenCV HSV，H 取半角 0-179
    piece_h_low: int = 105
    piece_h_high: int = 140
    piece_s_low: int = 60
    piece_v_low: int = 45
    piece_v_high: int = 200       # 棋子是深紫色；浅蓝背景 V 接近 240，必须排除
    piece_area_ratio_min: float = 0.00015

    # 台面：与背景色差异（BGR 三通道 L1 之和）
    bg_bin: int = 16
    bg_diff_thresh: int = 45
    board_area_ratio_min: float = 0.0006
    board_area_ratio_max: float = 0.12
    board_w_min_ratio: float = 0.07
    board_w_max_ratio: float = 0.58
    board_h_min_ratio: float = 0.012
    board_h_max_ratio: float = 0.24
    roi_top_ratio: float = 0.10          # 跳过顶部标题栏 / 分数区
    roi_bottom_ratio: float = 0.86       # 跳过底部广告区
    settled_threshold: float = 2.5       # 帧差均值，低于此视为画面静止

    @classmethod
    def load(cls, path=None):
        path = path or CONFIG_PATH
        cfg = cls()
        try:
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        except Exception:
            pass
        return cfg

    def save(self, path=None):
        with open(path or CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4, ensure_ascii=False)


@dataclass
class FrameResult:
    piece: tuple = None          # (x, y) 棋子底部中心
    board: tuple = None          # (x, y) 目标台面中心
    distance: float = 0.0        # 像素距离
    confidence: float = 0.0
    game_over: bool = False
    method: str = "failed"       # cv / legacy / failed / game_over


# ── 背景估计 ────────────────────────────────────────────────

def estimate_background(bgr, cfg):
    """量化直方图取主导色，再取该色类的中位色，避免渐变背景被算成噪声。"""
    small = cv2.resize(bgr, (max(1, bgr.shape[1] // 4), max(1, bgr.shape[0] // 4)),
                       interpolation=cv2.INTER_AREA)
    q = (small // cfg.bg_bin) * cfg.bg_bin
    keys = (q[:, :, 0].astype(np.int64) * 65536 +
            q[:, :, 1].astype(np.int64) * 256 +
            q[:, :, 2].astype(np.int64))
    vals, counts = np.unique(keys, return_counts=True)
    sel = keys == vals[int(np.argmax(counts))]
    return np.median(small[sel], axis=0).astype(np.int16)


# ── 棋子 ────────────────────────────────────────────────────

def detect_piece(bgr, cfg=None):
    """返回 (底部中心 x, 底边 y, 置信度, bbox) 或 None。"""
    cfg = cfg or VisionConfig.load()
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg.piece_h_low, cfg.piece_s_low, cfg.piece_v_low], dtype=np.uint8)
    upper = np.array([cfg.piece_h_high, 255, cfg.piece_v_high], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask[:int(h * 0.10), :] = 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < max(24, w * h * cfg.piece_area_ratio_min):
            continue
        if not (0.20 * h < y + bh / 2 < 0.92 * h):
            continue
        # 真实画面里底部浅蓝渐变会被误当成大片"棋子"，这些形状约束专门挡它
        if bh > 0.30 * h or bw > 0.35 * w or area > 0.05 * w * h:
            continue
        if y + bh > 0.95 * h:
            continue
        aspect = bw / max(bh, 1)
        if not (0.25 < aspect < 3.5):
            continue
        score = area * (1.0 if 0.4 < aspect < 1.8 else 0.5)
        if best is None or score > best[0]:
            best = (score, x, y, bw, bh, area)
    if best is None:
        return None
    _, x, y, bw, bh, area = best
    conf = min(1.0, 0.55 + area / max(1.0, w * h * 0.004) * 0.45)
    return (x + bw / 2.0, y + bh - 2.0, conf, (x, y, bw, bh))


# ── 目标台面 ────────────────────────────────────────────────

def detect_board(bgr, piece, cfg=None):
    """直接检测台面轮廓并取其顶面中心，返回 (x, y, 置信度, bbox) 或 None。"""
    cfg = cfg or VisionConfig.load()
    h, w = bgr.shape[:2]
    bg = estimate_background(bgr, cfg)
    diff = np.abs(bgr.astype(np.int16) - bg).sum(axis=2)
    mask = (diff > cfg.bg_diff_thresh).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    roi_top = int(h * cfg.roi_top_ratio)
    roi_bottom = int(h * cfg.roi_bottom_ratio)
    mask[:roi_top, :] = 0
    mask[roi_bottom:, :] = 0

    if piece is not None:
        px, py = piece[0], piece[1]
        pad_x, pad_y = int(w * 0.10), int(h * 0.075)
        mask[max(0, int(py - pad_y * 2)):min(h, int(py + pad_y)),
             max(0, int(px - pad_x)):min(w, int(px + pad_x))] = 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    px = piece[0] if piece is not None else w / 2.0
    side = 1 if px < w / 2.0 else -1
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        ar = area / float(w * h)
        if not (cfg.board_area_ratio_min < ar < cfg.board_area_ratio_max):
            continue
        if not (w * cfg.board_w_min_ratio < bw < w * cfg.board_w_max_ratio):
            continue
        if not (h * cfg.board_h_min_ratio < bh < h * cfg.board_h_max_ratio):
            continue
        aspect = bw / max(bh, 1)
        if not (0.7 < aspect < 7.0):
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        if (cx - px) * side < w * 0.06:      # 必须在棋子的对侧半边
            continue
        if abs(cx - px) < w * 0.08:
            continue
        score = ar * 100.0
        score *= 1.0 + 0.25 * max(0.0, 1.0 - abs(aspect - 2.2) / 4.0)
        score /= 1.0 + 0.35 * abs(cx - px) / w
        if score > (best[0] if best else -1):
            best = (score, x, y, bw, bh, cx, cy)
    if best is None:
        return None
    _, x, y, bw, bh, cx, cy = best
    # 立体方块的落脚点在顶面中心（轮廓上部约 30% 处），而非整个连通域质心
    board_y = y + bh * 0.30
    conf = min(1.0, 0.5 + best[0] * 1.5)
    return (cx, board_y, conf, (x, y, bw, bh))


# ── 结算页 ──────────────────────────────────────────────────

def detect_game_over(bgr, piece_found):
    """结算页特征：中下部出现大片近白按钮 + 找不到棋子。"""
    h, w = bgr.shape[:2]
    roi = bgr[int(h * 0.55):int(h * 0.88), int(w * 0.20):int(w * 0.90)]
    if roi.size == 0:
        return False
    near_white = (roi.min(axis=2) > 215).mean()
    button = bgr[int(h * 0.68):int(h * 0.83), int(w * 0.32):int(w * 0.88)]
    button_white = (button.min(axis=2) > 220).mean() if button.size else 0.0
    # 两个条件同时成立才算结算页：真实结算页实测 button_white≈0.33、
    # near_white≈0.23；游戏画面分别是 0.00 / 0.02。只靠单一条件容易被
    # 白色方块或画面上的棋子误检触发，那会让程序平白停下来。
    return bool(button_white > 0.28 and near_white > 0.12)


# ── 原版算法兜底 ────────────────────────────────────────────

def legacy_locate(bgr):
    """numpy 复刻原版识别，仅在 OpenCV 路线失败时兜底（可随时对照原版行为）。"""
    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1].astype(np.int16)
    pm = ((rgb[:, :, 0] > 50) & (rgb[:, :, 0] < 60) &
          (rgb[:, :, 1] > 53) & (rgb[:, :, 1] < 63) &
          (rgb[:, :, 2] > 95) & (rgb[:, :, 2] < 110))
    ys, xs = np.nonzero(pm)
    if len(xs) == 0:
        return None
    ymax = int(ys.max())
    px = float(xs[ys == ymax].mean())
    py = ymax - 20.0
    bx = None
    xs_all = np.arange(w)
    for i in range(int(h / 3), int(h * 2 / 3)):
        last = rgb[i, 0]
        d1 = np.abs(rgb[i] - last).sum(axis=1)
        d2 = np.abs(rgb[min(i + 5, h - 1)] - last).sum(axis=1)
        cand = (d1 > 10) & (d2 > 10) & (np.abs(xs_all - px) >= 70)
        cand &= (xs_all >= px) if px < w / 2 else (xs_all <= px)
        if cand.any():
            bx = float(xs_all[cand].mean())
            break
    if bx is None:
        return None
    cx = w / 2 + (24 / 1080) * w
    cy = h / 2 + (17 / 1920) * h
    slope = 25.5 / 43.5
    by = round(slope * (bx - cx) + cy) if px > cx else round(-slope * (bx - cx) + cy)
    return (px, py), (bx, float(by))


# ── 统一入口 ────────────────────────────────────────────────

def analyze(bgr, cfg=None, prefer="cv"):
    """分析一帧，返回 FrameResult。

    prefer="legacy" 时优先使用原版识别算法（RGB 扫棋子 + 边缘扫台面 +
    固定斜率推 Y），保证第一跳与原脚本完全一致；OpenCV 只在原版失败时兜底。
    """
    cfg = cfg or VisionConfig.load()
    piece_raw = detect_piece(bgr, cfg)
    piece = (piece_raw[0], piece_raw[1]) if piece_raw else None
    game_over = detect_game_over(bgr, piece_raw is not None)
    if game_over:
        return FrameResult(piece=piece, game_over=True, method="game_over")

    if prefer == "legacy":
        legacy = legacy_locate(bgr)
        if legacy:
            lp, lb = legacy
            return FrameResult(piece=lp, board=lb,
                               distance=math.hypot(lb[0] - lp[0], lb[1] - lp[1]),
                               confidence=0.8, method="legacy")

    board_raw = detect_board(bgr, piece_raw, cfg)
    board = (board_raw[0], board_raw[1]) if board_raw else None
    conf = 0.0
    method = "failed"
    if piece and board:
        conf = min(piece_raw[2], board_raw[2])
        method = "cv"
    else:
        # cv 只成功一半时绝不与 legacy 混用：曾把底部浅蓝背景误检成棋子，
        # 再配上 legacy 的台面坐标，距离直接翻倍 → 持续按住跳飞。
        legacy = legacy_locate(bgr)
        if legacy:
            piece, board = legacy
            conf = 0.6
            method = "legacy"
        else:
            board = None
            method = "failed"
    if piece and board:
        d = math.hypot(board[0] - piece[0], board[1] - piece[1])
        return FrameResult(piece=piece, board=board, distance=d,
                           confidence=conf, method=method)
    return FrameResult(piece=piece, method="failed")


def estimate_shift(prev_bgr, cur_bgr):
    """估计两帧之间的全局背景平移（相机移动），返回 (dx, dy, response)。

    跳一跳的相机会跟随棋子移动，所以「棋子在画面里的位移」并不等于
    「棋子实际跳了多远」。真实位移 = 棋子位移 - 背景平移，这个公式
    在相机完全跟随或完全不跟随两种极端下都成立。

    response 越高越可信；低于 0.1 说明画面变化太大，不要拿它做补偿。
    """
    if prev_bgr is None or cur_bgr is None or prev_bgr.shape != cur_bgr.shape:
        return 0.0, 0.0, 0.0
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g0.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    try:
        (dx, dy), resp = cv2.phaseCorrelate(g0, g1, win)
    except Exception:
        return 0.0, 0.0, 0.0
    return float(dx), float(dy), float(resp)


def is_settled(prev_bgr, cur_bgr, cfg=None):
    """连续两帧差异足够小 = 棋子已落稳，可以读取落点。"""
    cfg = cfg or VisionConfig.load()
    if prev_bgr is None or prev_bgr.shape != cur_bgr.shape:
        return False
    a = cv2.resize(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY), (160, 300))
    b = cv2.resize(cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY), (160, 300))
    return float(cv2.absdiff(a, b).mean()) < cfg.settled_threshold


def annotate(bgr, result, extra=None):
    """画识别结果，返回 BGR 图（供 GUI 预览 / 调试）。"""
    out = bgr.copy()
    if result.piece:
        x, y = int(result.piece[0]), int(result.piece[1])
        cv2.circle(out, (x, y), 12, (60, 60, 255), 2)
        cv2.line(out, (x, 0), (x, out.shape[0]), (60, 60, 255), 1)
    if result.board:
        x, y = int(result.board[0]), int(result.board[1])
        cv2.circle(out, (x, y), 12, (255, 140, 60), 2)
        cv2.line(out, (0, y), (out.shape[1], y), (255, 140, 60), 1)
    if result.piece and result.board:
        cv2.line(out, tuple(map(int, result.piece)), tuple(map(int, result.board)),
                 (80, 220, 100), 2)
        mid = ((int(result.piece[0]) + int(result.board[0])) // 2,
               (int(result.piece[1]) + int(result.board[1])) // 2)
        cv2.putText(out, "d=%.0f  %s  c=%.2f" % (result.distance, result.method,
                                                  result.confidence),
                    (mid[0] + 6, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (80, 220, 100), 1, cv2.LINE_AA)
    if result.game_over:
        cv2.putText(out, "GAME OVER", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (60, 60, 255), 2, cv2.LINE_AA)
    if extra:
        for i, line in enumerate(extra):
            cv2.putText(out, str(line), (12, 24 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


# ── 命令行工具 ──────────────────────────────────────────────

def collect_samples(count, title="跳一跳", interval=0.45):
    """开局后运行：连续抓窗口帧，只在画面变化明显时保存，便于调阈值。"""
    from ai_control import grab_window
    from common.pc_control import enum_visible_windows
    wins = [w for w in enum_visible_windows() if w[1].strip() == title]
    if not wins:
        print("未找到标题为 %r 的窗口，请先在电脑微信里打开跳一跳" % title)
        return
    hwnd = wins[0][0]
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    prev, saved = None, 0
    print("采集中，请在游戏里跳几步… (Ctrl+C 结束)")
    try:
        while saved < count:
            frame = grab_window(hwnd)
            if prev is None or not is_settled(prev, frame, VisionConfig.load()):
                name = "sample_%02d_%d.png" % (saved, int(time.time()))
                cv2.imwrite(os.path.join(SAMPLE_DIR, name), frame)
                saved += 1
                print("  已保存", name)
            prev = frame
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    print("共保存 %d 张到 %s" % (saved, SAMPLE_DIR))


def debug_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        print("无法读取", path)
        return
    res = analyze(bgr)
    print("尺寸:", bgr.shape[1], "x", bgr.shape[0])
    print("棋子:", res.piece, "台面:", res.board)
    print("距离:", round(res.distance, 1), "置信度:", round(res.confidence, 2),
          "方法:", res.method, "结算页:", res.game_over)
    out = annotate(bgr, res)
    dst = os.path.splitext(path)[0] + "_annotated.png"
    cv2.imwrite(dst, out)
    print("标注图:", dst)


def main():
    ap = argparse.ArgumentParser(description="跳一跳 AI 感知调试工具")
    ap.add_argument("--debug", metavar="IMAGE", help="分析一张截图并生成标注图")
    ap.add_argument("--collect", type=int, metavar="N", help="采集 N 张真实游戏画面")
    ap.add_argument("--title", default="跳一跳", help="游戏窗口标题")
    ap.add_argument("--init-config", action="store_true", help="生成默认 vision_config.json")
    args = ap.parse_args()
    if args.debug:
        debug_image(args.debug)
    elif args.collect:
        collect_samples(args.collect, args.title)
    elif args.init_config:
        VisionConfig().save()
        print("已生成", CONFIG_PATH)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
