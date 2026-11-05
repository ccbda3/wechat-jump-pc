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
    from ai_vision import analyze, annotate, is_settled, VisionConfig
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

    cfg = VisionConfig()
    check("落稳：相同帧判定静止", is_settled(frame, frame.copy(), cfg))
    moved = synthetic_frame(piece=(300, 700), board=(500, 620))
    check("落稳：画面大幅变化判定未静止", not is_settled(frame, moved, cfg))

    out = annotate(frame, res, ["selftest"])
    check("感知：标注图可生成", out.shape == frame.shape)


def test_calibration():
    """仿真真实物理 t = A*d + B，从错误初值出发，验证在线学习收敛且越跳越准。"""
    from ai_calibration import OnlineCalibrator
    random.seed(7)
    np.random.seed(7)
    TRUE_A, TRUE_B = 1.35, 240.0
    calib = OnlineCalibrator(path=None)
    start = calib.stats()
    errs = []
    for i in range(45):
        d_req = random.uniform(150, 450)
        t = calib.predict(d_req)
        d_act = max(1.0, (t - TRUE_B) / TRUE_A + np.random.normal(0, 3.0))
        calib.observe(t, d_act, d_req, confidence=0.9)
        errs.append(abs(d_act - d_req))
    s = calib.stats()
    early = float(np.mean(errs[:10]))
    late = float(np.mean(errs[-10:]))
    check("校准：从错误初值收敛到真实映射",
          abs(s["a"] - TRUE_A) < 0.18 and abs(s["b"] - TRUE_B) < 70,
          "a %.2f->%.3f (真1.35)  b %.0f->%.0f (真240)  样本%d" %
          (start["a"], s["a"], start["b"], s["b"], s["samples"]))
    check("校准：越跳越准（后期偏差显著小于前期）", late < early * 0.5,
          "前10跳平均偏差%.1fpx -> 后10跳%.1fpx" % (early, late))
    bad = OnlineCalibrator(path=None)
    for _ in range(10):
        bad.observe(600, 320, 300, confidence=0.9)
    ok, reason, _ = bad.observe(600, 20, 300, confidence=0.1)
    check("校准：低置信度/异常样本被拒绝", not ok, "reason=%s" % reason)


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
    from ai_calibration import OnlineCalibrator
    from ai_engine import AIJumpEngine
    frame = synthetic_frame()
    logs, frames, counts, stats = [], [], [], []
    calib = OnlineCalibrator(path=None)
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
    for fn in (test_vision, test_calibration, test_precise_press,
               test_engine, test_gui, test_window_grab):
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
