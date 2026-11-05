# -*- coding: utf-8 -*-
"""AI 模式离线自测：不碰真实鼠标、不需要游戏在跑。

    .venv/Scripts/python.exe selftest_ai.py

覆盖：感知精度 / 结算页识别 / 落稳判定 / 校准器收敛 / 按压计时精度 /
      引擎闭环流程 / GUI 集成。
"""
import math
import os
import random
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))


def synthetic_frame(piece=(200, 820), board=(430, 690)):
    """生成一张接近真实比例的模拟游戏画面（BGR）。"""
    w, h = 631, 1159
    top = np.array([120, 118, 112], np.int16)
    bot = np.array([70, 72, 78], np.int16)
    grad = np.linspace(0, 1, h)[:, None, None]
    bgr = np.repeat((top * (1 - grad) + bot * grad).astype(np.uint8), w, axis=1)
    bgr[:76, :, :] = 245
    cv2.putText(bgr, "128", (285, 170), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (235, 235, 235), 3)
    for cx, cy, col in [(piece[0], piece[1] + 15, (180, 190, 200)),
                        (board[0], board[1], (210, 220, 235))]:
        cv2.rectangle(bgr, (cx - 62, cy), (cx + 62, cy + 52), col, -1)
        cv2.ellipse(bgr, (cx, cy), (62, 22), 0, 0, 360,
                    tuple(min(255, c + 12) for c in col), -1)
        cv2.ellipse(bgr, (cx, cy), (62, 22), 0, 0, 360,
                    tuple(max(0, c - 25) for c in col), 2)
    px, py = piece
    purple = (102, 58, 55)
    cv2.ellipse(bgr, (px, py - 48), (22, 26), 0, 0, 360, purple, -1)
    cv2.fillPoly(bgr, [np.array([[px - 12, py - 26], [px + 12, py - 26],
                                 [px + 17, py], [px - 17, py]], np.int32)], purple)
    cv2.ellipse(bgr, (px, py), (18, 6), 0, 0, 360, purple, -1)
    return bgr


def test_vision():
    from ai_vision import (analyze, annotate, detect_piece, is_settled,
                           VisionConfig)
    frame = synthetic_frame()
    res = analyze(frame)
    ok = (res.piece and res.board and res.method == "cv" and not res.game_over)
    if ok:
        ep = math.hypot(res.piece[0] - 200, res.piece[1] - 820)
        eb = math.hypot(res.board[0] - 430, res.board[1] - 690)
        ok = ep < 14 and eb < 20 and res.distance > 100
        detail = "piece误差%.1fpx board误差%.1fpx d=%.0f conf=%.2f" % (ep, eb, res.distance, res.confidence)
    else:
        detail = str(res)
    check("感知：合成画面棋子/台面定位", ok, detail)

    sample = os.path.join(ROOT, "debug_samples", "printwindow_sample.png")
    if os.path.exists(sample):
        real = cv2.imread(sample)
        r2 = analyze(real)
        check("感知：真实结算页识别为 game_over", r2.game_over, "method=%s" % r2.method)
    check("感知：游戏画面不误判为结算页", not res.game_over)
    block = synthetic_frame()
    cv2.rectangle(block, (260, 810), (360, 870), (250, 250, 250), -1)
    check("感知：单个白色方块不误判为结算页", not analyze(block).game_over)

    cfg = VisionConfig()
    check("落稳：相同帧判定静止", is_settled(frame, frame.copy(), cfg))
    moved = synthetic_frame(piece=(300, 700), board=(500, 620))
    check("落稳：画面大幅变化判定未静止", not is_settled(frame, moved, cfg))

    live = os.path.join(ROOT, "debug_samples", "live_client.png")
    if os.path.exists(live):
        img = cv2.imread(live)
        lh = img.shape[0]
        pr = detect_piece(img, VisionConfig())
        check("真实画面：OpenCV 不再把底部浅蓝背景当成棋子",
              pr is None or pr[1] < 0.9 * lh,
              "piece=%s" % str(pr and tuple(round(v, 1) for v in pr[:2])))
        rl = analyze(img, prefer="legacy")
        rc = analyze(img, prefer="cv")
        check("真实画面：两种识别都给出合理的棋子位置",
              bool(rl.piece) and rl.piece[1] < 0.9 * lh
              and bool(rc.piece) and rc.piece[1] < 0.9 * lh,
              "legacy=%s cv=%s" % (rl.piece, rc.piece))
        check("真实画面：cv 结果不与 legacy 混用（距离不翻倍）",
              rl.distance > 0 and rc.distance < rl.distance * 1.6,
              "legacy d=%.0f  cv d=%.0f" % (rl.distance, rc.distance))
    else:
        print("SKIP  真实画面回归（debug_samples/live_client.png 不存在）")

    out = annotate(frame, res, ["selftest"])
    check("感知：标注图可生成", out.shape == frame.shape)


def test_calibration():
    """验证冷启动：首跳等于经典模式，之后只在基准附近微调，跳飞立刻回拉。"""
    from ai_calibration import AdaptiveCoefficient, legacy_press_ms, MAX_PRESS_MS
    random.seed(7)
    np.random.seed(7)
    hd = 631 / 8.0
    d = 300.0
    base_unit = legacy_press_ms(d, hd, 1.0)
    classic = legacy_press_ms(d, hd, 2.0)          # 经典模式，系数 2.0
    calib = AdaptiveCoefficient(path=None, base_factor=2.0)
    first = calib.predict(base_unit)
    check("冷启动：首跳与经典模式完全一致", abs(first - classic) < 1.0,
          "AI %.1fms / 经典 %.1fms" % (first, classic))
    check("冷启动：按压远低于硬上限", first < MAX_PRESS_MS * 0.75,
          "%.0fms / 上限 %.0fms" % (first, MAX_PRESS_MS))

    # 仿真：当前环境真正需要的系数比用户基准高 17.5%
    k_true = 2.35
    errs = []
    for i in range(30):
        d_req = random.uniform(180, 420)
        base = legacy_press_ms(d_req, hd, 1.0)
        t = calib.predict(base)
        d_act = d_req * (t / (base * k_true)) + np.random.normal(0, 2.0)
        calib.observe(t, d_act, d_req, base, confidence=0.9)
        errs.append(abs(d_act - d_req))
    s = calib.stats()
    early, late = float(np.mean(errs[:8])), float(np.mean(errs[-8:]))
    check("自适应：系数收敛到环境真值", abs(s["factor"] - k_true) < 0.12,
          "k 2.000 -> %.3f (真 %.2f) 样本%d" % (s["factor"], k_true, s["samples"]))
    check("自适应：越跳越准", late < early * 0.5,
          "前8跳偏差%.1fpx -> 后8跳%.1fpx" % (early, late))
    check("自适应：系数始终被夹在基准 ±40%",
          2.0 * 0.6 <= s["factor"] <= 2.0 * 1.4, "k=%.3f" % s["factor"])

    k_before = calib.stats()["factor"]
    ok, reason, _ = calib.observe(600, 2 * d, d, base_unit, confidence=0.9)
    k_after = calib.stats()["factor"]
    check("跳飞保护：超远落点拒绝学习并回拉系数",
          (not ok) and k_after < k_before, "%s  k %.3f -> %.3f" %
          (reason, k_before, k_after))
    check("硬上限：极端输入也不会长按", calib.predict(1e6) <= MAX_PRESS_MS,
          "%.0fms" % calib.predict(1e6))


def test_near_calibration():
    """近方块要单独学一个乘数，不能沿用中远距离调好的系数。"""
    from ai_calibration import AdaptiveCoefficient, legacy_press_ms
    base_near = legacy_press_ms(150, 77, 1.0)
    c = AdaptiveCoefficient(path=None, base_factor=2.0)
    k0 = c.stats()["factor"]
    for _ in range(10):
        t = c.predict(base_near, 150)
        c.observe(t, 150 * 1.25, 150, base_near, confidence=0.9)
    s = c.stats()
    check("近距离：单独学出更小的乘数，中远距离系数不受影响",
          s["near_factor"] < 0.98 and abs(s["factor"] - k0) < 1e-9,
          "near=%.3f k=%.3f" % (s["near_factor"], s["factor"]))
    near_before = c.stats()["near_factor"]
    base_far = legacy_press_ms(300, 77, 1.0)
    for _ in range(5):
        t = c.predict(base_far, 300)
        c.observe(t, 300 * 1.05, 300, base_far, confidence=0.9)
    check("近距离：中远距离样本不会污染近距乘数",
          abs(c.stats()["near_factor"] - near_before) < 1e-9,
          "near=%.3f" % c.stats()["near_factor"])
    check("近距离：近方块按压允许低于原版 200ms 下限",
          c.predict(legacy_press_ms(80, 77, 1.0), 80) < 200,
          "%.0fms" % c.predict(legacy_press_ms(80, 77, 1.0), 80))


def test_precise_press():
    from ai_control import precise_press
    worst = 0.0
    details = []
    for ms in (250.0, 600.0, 900.0):
        actual = precise_press(0, 0, ms, no_click=True)
        err = abs(actual - ms)
        worst = max(worst, err)
        details.append("%.0fms->%.1fms" % (ms, actual))
    check("按压：计时精度 < 5ms（且不真的点鼠标）", worst < 5.0,
          "最大误差%.2fms  %s" % (worst, " ".join(details)))


def test_engine():
    """用固定合成帧跑通完整闭环（dry_run 不发鼠标事件）。"""
    from ai_calibration import AdaptiveCoefficient
    from ai_engine import AIJumpEngine
    frame = synthetic_frame()
    logs, frames, counts, stats = [], [], [], []
    calib = AdaptiveCoefficient(path=None, base_factor=2.0)
    eng = AIJumpEngine(
        window_rect=(0, 0, frame.shape[1], frame.shape[0]),
        calibrator=calib, frame_source=lambda: frame.copy(),
        dry_run=True, max_jumps=2, settle_timeout=0.8,
        on_log=logs.append, on_frame=frames.append,
        on_count=counts.append, on_stats=stats.append)
    eng.start()
    eng.join(timeout=25)
    check("引擎：闭环主循环正常跑完", not eng.is_alive() and eng.jumps == 2,
          "jumps=%d logs=%d frames=%d" % (eng.jumps, len(logs), len(frames)))
    check("引擎：日志/预览/统计回调均被触发",
          any("AI 引擎结束" in s for s in logs) and len(frames) >= 2 and len(stats) >= 2)
    check("引擎：固定画面不会产生虚假学习",
          calib.samples == 0, "样本=%d 丢弃=%d" % (calib.samples, calib.rejected))


def test_environment():
    from ai_environment import EnvironmentMonitor
    frame = synthetic_frame()
    mon = EnvironmentMonitor()
    s1 = mon.update(frame, (0, 0, 631, 1159))
    check("环境：正常游戏画面判定有效", s1.valid,
          "亮度%.0f 对比度%.0f hd=%.0f" % (s1.brightness, s1.contrast, s1.head_diameter))
    s2 = mon.update(np.zeros_like(frame), (0, 0, 631, 1159))
    check("环境：黑屏判定无效（不会瞎点击）", not s2.valid, s2.note)
    big = cv2.resize(frame, (700, 1200))
    s3 = mon.update(big, (0, 0, 700, 1200))
    check("环境：窗口缩放被检测并重算头部直径",
          s3.resized and abs(s3.head_diameter - 700 / 8.0) < 1.0,
          "resized=%s hd=%.0f" % (s3.resized, s3.head_diameter))


def synthetic_game_over():
    """模拟结算页：中下部大片白 + 再玩一局按钮区域白色。"""
    f = synthetic_frame()
    h, w = f.shape[:2]
    cv2.rectangle(f, (int(w * 0.32), int(h * 0.68)),
                  (int(w * 0.88), int(h * 0.83)), (250, 250, 250), -1)
    cv2.rectangle(f, (int(w * 0.22), int(h * 0.57)),
                  (int(w * 0.86), int(h * 0.66)), (240, 240, 240), -1)
    return f


def test_game_over_behavior():
    """结算页：不改系数、不退出程序、等下一局继续；启停只由按钮控制。"""
    import json
    import tempfile
    from ai_calibration import AdaptiveCoefficient
    from ai_engine import AIJumpEngine
    from ai_vision import analyze
    over, normal = synthetic_game_over(), synthetic_frame()
    check("结算页：模拟画面能被识别", analyze(over).game_over)

    seq = [0]

    def source():
        seq[0] += 1
        return (over if seq[0] <= 2 else normal).copy()

    logs = []
    calib = AdaptiveCoefficient(path=None, base_factor=2.0)
    eng = AIJumpEngine(
        window_rect=(0, 0, normal.shape[1], normal.shape[0]),
        calibrator=calib, frame_source=source, dry_run=True,
        max_jumps=1, settle_timeout=0.6, on_log=logs.append)
    eng.start()
    eng.join(timeout=20)
    check("结算页：程序不自行退出，等到新一局后继续跳",
          (not eng.is_alive()) and eng.jumps == 1,
          "jumps=%d" % eng.jumps)
    check("结算页：日志提示等待与恢复，且没有「回拉」",
          any("等待重新开局" in s for s in logs)
          and any("新一局" in s for s in logs)
          and not any("回拉" in s for s in logs))
    check("结算页：系数完全不受影响", calib.stats()["factor"] == 2.0,
          "k=%.3f overshoots=%d" % (calib.stats()["factor"],
                                    calib.stats()["overshoots"]))

    # 旧版本误回拉留下的脏记录应自动修回基准
    path = os.path.join(tempfile.gettempdir(), "wj_dirty_calib_test.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"factor": 1.477, "base_factor": 2.0, "samples": 1,
                   "overshoots": 2, "rejected": 0}, f)
    repaired = AdaptiveCoefficient.load(path, base_factor=2.0)
    check("结算页：旧版脏记录自动修回基准",
          repaired.stats()["factor"] == 2.0 and repaired.stats()["overshoots"] == 0,
          "k=%.3f" % repaired.stats()["factor"])
    os.remove(path)


def test_gui():
    import gui_app
    app = gui_app.JumpGUI()
    app.root.withdraw()
    app.root.update_idletasks()
    ok = (gui_app.AI_AVAILABLE and hasattr(app, "ai_mode")
          and app.ai_mode.get() is False and hasattr(app, "_reset_ai"))
    check("GUI：AI 模式开关已接入且默认关闭", ok,
          "AI_AVAILABLE=%s" % gui_app.AI_AVAILABLE)
    app.root.destroy()


def test_persistence():
    """学习结果必须跨重启保留；正常上涨不能被误重置。"""
    import json
    import shutil
    import tempfile
    from ai_calibration import AdaptiveCoefficient, legacy_press_ms
    tmpdir = tempfile.mkdtemp(prefix="wj_persist_")
    path = os.path.join(tmpdir, "calib.json")
    base = legacy_press_ms(300, 77, 1.0)
    c = AdaptiveCoefficient(path=path, base_factor=2.0)
    for _ in range(6):
        t = c.predict(base)
        c.observe(t, 300 * 1.12, 300, base, confidence=0.9)
    c.save()
    c2 = AdaptiveCoefficient.load(path, base_factor=2.0)
    check("持久化：重启后系数、样本数、保存时间都保留",
          abs(c2.stats()["factor"] - c.stats()["factor"]) < 1e-6
          and c2.samples == c.samples and bool(c2.stats()["loaded_at"]),
          "k=%.3f samples=%d saved=%s" %
          (c2.stats()["factor"], c2.samples, c2.stats()["loaded_at"]))

    p2 = os.path.join(tmpdir, "up.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump({"factor": 2.50, "base_factor": 2.0, "samples": 2}, f)
    c3 = AdaptiveCoefficient.load(p2, base_factor=2.0)
    check("持久化：正常上涨的系数不会被误重置",
          abs(c3.stats()["factor"] - 2.50) < 1e-6, "k=%.3f" % c3.stats()["factor"])

    p3 = os.path.join(tmpdir, "bad.json")
    with open(p3, "w", encoding="utf-8") as f:
        json.dump({"factor": 1.477, "base_factor": 2.0, "samples": 1,
                   "overshoots": 2}, f)
    c4 = AdaptiveCoefficient.load(p3, base_factor=2.0)
    check("持久化：被旧版压低的脏记录仍会修回基准",
          abs(c4.stats()["factor"] - 2.0) < 1e-6, "k=%.3f" % c4.stats()["factor"])
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_camera_compensation():
    """相机跟随棋子移动时，真实位移 = 棋子画面位移 - 背景平移。"""
    from ai_vision import estimate_shift
    np.random.seed(3)
    img = np.random.randint(0, 255, (400, 600, 3), np.uint8)
    for true_dx, true_dy in [(140.0, -70.0), (-90.0, 40.0)]:
        m = np.float32([[1, 0, -true_dx], [0, 1, -true_dy]])
        shifted = cv2.warpAffine(img, m, (600, 400))
        dx, dy, resp = estimate_shift(img, shifted)
        est_dx, est_dy = -dx, -dy          # 相机完全跟随时，棋子画面位移为 0
        check("落点：相机位移补偿能还原真实跳跃距离",
              abs(est_dx - true_dx) < 3 and abs(est_dy - true_dy) < 3 and resp > 0.3,
              "真值(%.0f,%.0f) 估计(%.0f,%.0f) resp=%.2f" %
              (true_dx, true_dy, est_dx, est_dy, resp))


def test_window_position_independence():
    """窗口级截图必须与窗口在屏幕上的位置无关。"""
    import ctypes
    import tkinter as tk
    from ai_control import client_area, grab_window
    root = tk.Tk()
    root.geometry("240x180+1600+200")          # 屏幕右侧
    c = tk.Canvas(root, bg="#336699", highlightthickness=0)
    c.pack(fill=tk.BOTH, expand=True)
    c.create_rectangle(20, 20, 140, 110, fill="#ffcc00", outline="")
    c.create_text(120, 150, text="probe", fill="white")
    root.update()
    time.sleep(0.2)
    top = ctypes.windll.user32.GetAncestor(int(root.winfo_id()), 2)
    full = grab_window(top)
    ox, oy, cw, ch = client_area(top)
    crop = full[oy:oy + ch, ox:ox + cw]
    check("截图：窗口在屏幕右侧也能按客户区正确抓取",
          crop.shape[1] == cw and crop.shape[0] == ch and crop.std() > 3,
          "window=%dx%d client=%dx%d std=%.1f" %
          (full.shape[1], full.shape[0], cw, ch, crop.std()))

    root.geometry("+120+220")                   # 移到左侧
    root.update()
    time.sleep(0.2)
    full2 = grab_window(top)
    crop2 = full2[oy:oy + ch, ox:ox + cw]
    check("截图：窗口移动后画面仍正确（不依赖旧坐标）",
          crop2.shape == crop.shape and crop2.std() > 3,
          "shape=%s std=%.1f" % (crop2.shape, crop2.std()))
    root.destroy()


def test_window_grab():
    from ai_control import grab_window
    from common.pc_control import enum_visible_windows
    wins = [w for w in enum_visible_windows() if w[1].strip() == "跳一跳"]
    if not wins:
        print("SKIP  窗口抓图（当前未打开跳一跳窗口）")
        return
    arr = grab_window(wins[0][0])
    check("抓图：PrintWindow 返回有效画面", arr.ndim == 3 and arr.std() > 5,
          "size=%dx%d std=%.1f" % (arr.shape[1], arr.shape[0], arr.std()))


def main():
    t0 = time.time()
    for fn in (test_vision, test_calibration, test_near_calibration, test_precise_press,
               test_environment, test_game_over_behavior, test_engine,
               test_gui, test_persistence, test_camera_compensation,
               test_window_position_independence, test_window_grab):
        try:
            fn()
        except Exception as e:
            import traceback
            check(fn.__name__ + " 异常", False, repr(e))
            traceback.print_exc()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n%d/%d passed  (%.1fs)" % (len(RESULTS) - len(failed), len(RESULTS),
                                       time.time() - t0))
    if failed:
        print("FAILED:", ", ".join(failed))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
