import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.policy import ACTIONS, FEATURES, HIDDEN, MS_PER_PX, Policy, SCALES
from ai_vision import VisionConfig, analyze

W, H = 600, 900

_TEXTURE = np.random.default_rng(9).integers(0, [W, H], size=(240, 2))


def make_frame(piece_dx=0.0, piece_dy=0.0):
    img = np.zeros((H, W, 3), np.uint8)
    top = np.array([238, 228, 214], np.float32)
    bottom = np.array([206, 196, 182], np.float32)
    for y in range(H):
        t = y / (H - 1)
        img[y, :] = (top * (1 - t) + bottom * t).astype(np.uint8)
    for x, y in _TEXTURE:
        cv2.circle(img, (int(x), int(y)), 2, (150, 148, 142), -1)
    px = int(120 + piece_dx)
    py = int(560 + piece_dy)
    cv2.rectangle(img, (px - 28, py - 90), (px + 28, py), (102, 58, 55), -1)
    cv2.rectangle(img, (px - 34, py - 14), (px + 34, py), (96, 54, 52), -1)
    bx0, by0, bx1, by1 = 420, 420, 560, 468
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (154, 150, 146), -1)
    cv2.rectangle(img, (bx0, by0), (bx1, by0 + 12), (176, 172, 168), -1)
    return img


def game_over_frame():
    from PIL import Image, ImageDraw, ImageFont
    img = make_frame()
    img2 = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
    cv2.rectangle(img2, (150, 560), (450, 620), (245, 245, 245), -1)
    pil = Image.fromarray(img2[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
        d.text((238, 574), "再玩一局", font=font, fill=(40, 40, 40))
    except Exception:
        pass
    return np.array(pil)[:, :, ::-1].copy()


def to_payload(bgr):
    a = np.dstack([bgr, np.full((bgr.shape[0], bgr.shape[1]), 255, np.uint8)])
    return np.ascontiguousarray(a).tobytes()


def send(proc, obj, frame=None):
    if frame is not None:
        phase = obj.get("phase", "before")
        path = os.path.join(ROOT, "logs", "frame_%s.bin" % phase)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(to_payload(frame))
    proc.stdin.write(json.dumps(obj).encode("utf-8") + b"\n")
    proc.stdin.flush()


def read(proc, quiet=False):
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker closed")
        obj = json.loads(line.decode("utf-8"))
        if obj.get("t") == "log":
            if not quiet:
                print("  [worker] " + obj["msg"])
            continue
        return obj


def test_policy():
    tmp = os.path.join(tempfile.gettempdir(), "jump_rl_selftest_model.json")
    if os.path.exists(tmp):
        os.remove(tmp)
    policy = Policy(path=tmp, lr=0.12, seed=2, frozen=False)
    x = np.zeros(FEATURES)
    x[0] = 1.0
    x[1] = 0.30
    x[2] = 0.7
    x[8] = 0.31
    ideal = 5
    base = 300.0
    rewards = []
    for step in range(500):
        a, scale, ent = policy.act(x)
        ratio = scale / SCALES[ideal]
        err = abs(ratio - 1.0) * base * 1.6
        sigma = 33.0
        r = math.exp(-(err / sigma) ** 2)
        policy.update(x, a, r)
        rewards.append(r)
    early = float(np.mean(rewards[:40]))
    late = float(np.mean(rewards[-40:]))
    _, p = policy.forward(x)
    best = int(np.argmax(p))
    ok = (best == ideal) and (late > early + 0.15) and (late > 0.75)
    print("策略学习：最优动作 %d（期望 %d），前40跳平均 %.3f，后40跳平均 %.3f"
          % (best, ideal, early, late))
    if not ok:
        raise SystemExit("策略学习未收敛")
    saved = policy.jumps
    policy2 = Policy(path=tmp)
    if policy2.jumps != saved:
        raise SystemExit("模型续训加载失败")
    for _ in range(30):
        a, scale, ent = policy2.act(x)
        policy2.update(x, a, 1.0)
    print("续训校验：加载后 %d 跳，续训后 %d 跳" % (saved, policy2.jumps))
    os.remove(tmp)


def test_vision():
    frame = make_frame()
    res = analyze(frame, VisionConfig.load(), prefer="legacy")
    print("识别：棋子 %s 台面 %s 距离 %.0f 方法 %s"
          % (res.piece, res.board, res.distance, res.method))
    if not res.piece or not res.board:
        raise SystemExit("原版识别算法失败")
    if res.distance < 100:
        raise SystemExit("识别距离异常")


def test_calib():
    from ml.correct import Corrector
    tmp = os.path.join(tempfile.gettempdir(), "jump_calib_selftest.npz")
    if os.path.exists(tmp):
        os.remove(tmp)
    frame = make_frame()
    cv2.rectangle(frame, (480, 680), (520, 720), (20, 30, 200), -1)
    cv2.circle(frame, (500, 700), 10, (0, 255, 255), -1)
    cal = Corrector(path=tmp)
    n = cal.add(frame, (500, 700))
    hit = cal.locate(frame)
    s_hit = cal.score(frame, (500, 700))
    s_miss = cal.score(frame, (300, 300))
    print("修正样本：%d 个，命中分 %.3f，偏移分 %.3f，定位 %s"
          % (n, s_hit, s_miss, None if hit is None else
             (round(hit[0]), round(hit[1]), round(hit[2], 3))))
    if n != 1 or s_hit < 0.99 or s_miss > 0.9:
        raise SystemExit("修正样本模板评分异常")
    if hit is None or abs(hit[0] - 500) > 2 or abs(hit[1] - 700) > 2:
        raise SystemExit("修正样本定位失败")
    os.remove(tmp)


def test_topface():
    from ai_vision import face_center_y
    from ml.yolo_vision import YoloVision
    from ocr import bg_color

    img = np.zeros((360, 400, 3), np.uint8)
    img[:, :] = (216, 231, 232)
    top = np.array([[200, 70], [320, 130], [200, 190], [80, 130]], np.int32)
    cv2.fillPoly(img, [top], (198, 236, 100))
    front = np.array([[80, 130], [200, 190], [200, 300], [80, 240]], np.int32)
    cv2.fillPoly(img, [front], (183, 218, 93))
    right = np.array([[320, 130], [200, 190], [200, 300], [320, 240]], np.int32)
    cv2.fillPoly(img, [right], (170, 205, 80))
    cv2.ellipse(img, (135, 220), (20, 34), 0, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(img, (265, 220), (20, 34), 0, 0, 360, (235, 235, 235), -1)
    v = YoloVision()
    bg = bg_color(img)
    side = (0.9, 60.0, 210.0, 340.0, 310.0)
    grown = v._grow_up(img, side, bg)
    ty = face_center_y(img, (grown[1], grown[2], grown[3] - grown[1],
                             grown[4] - grown[2]), bg)
    bx, by = v._board_point(img, grown, bg)
    print("顶面修正：侧面框 y=%.0f → 扩展后 y=%.0f，顶面中心 (%.0f, %.0f)"
          % (side[2], grown[2], bx, by))
    if grown[2] > 120 or not v._open_above(img, grown, bg):
        raise SystemExit("侧面框未能向上扩展到顶面")
    if not (95 < ty < 180):
        raise SystemExit("顶面中心估计异常 %.0f" % ty)
    if abs(bx - 200) > 10 or by > 170:
        raise SystemExit("侧面白点影响了目标点 (%.0f, %.0f)" % (bx, by))
    same = v._pick([(0.9, 260.0, 250.0, 400.0, 330.0)], 200, 300)
    if same is None:
        raise SystemExit("同高度近方块被误排除")
    under = v._pick([(0.9, 120.0, 180.0, 280.0, 330.0)], 200, 300)
    if under is not None:
        raise SystemExit("脚下方块未被排除")


def test_worker():
    worker = os.path.join(ROOT, "ml", "worker.py")
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    calib = os.path.join(tempfile.gettempdir(), "jump_calib_worker.npz")
    if os.path.exists(calib):
        os.remove(calib)
    env = os.environ.copy()
    env["JUMP_CALIB_PATH"] = calib
    proc = subprocess.Popen(
        [py, worker], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, env=env)
    try:
        send(proc, {"t": "init"})
        init = read(proc, quiet=True)
        print("初始化：kind=%s 模型=%s" % (init["kind"], init["stats"]["model"]))

        send(proc, {"t": "frame", "id": 1, "w": W, "h": H, "phase": "before",
                    "base": 1.0, "greedy": 0, "settled": 1}, make_frame())
        press = read(proc)
        print("决策：kind=%s ms=%.0f 动作=%d 缩放=%.2f 距离=%.0f"
              % (press["kind"], press.get("ms", 0), press.get("action", -1),
                 press.get("scale", 0), press["det"].get("d", 0)))
        if press["kind"] != "press":
            raise SystemExit("未输出按压动作")
        want = MS_PER_PX * press["det"]["d"] * press.get("scale", 1.0)
        if abs(press["ms"] - want) > max(60, 0.10 * want):
            raise SystemExit("按压时长与距离模型不符 %.0f != %.0f" % (press["ms"], want))

        send(proc, {"t": "frame", "id": 1, "w": W, "h": H, "phase": "after",
                    "base": 1.0, "greedy": 0, "settled": 1},
             make_frame(382.0, -116.0))
        after = read(proc)
        print("反馈：kind=%s 奖励=%s 偏差=%s"
              % (after["kind"], after.get("reward"), after.get("err")))
        if after["kind"] != "ok" or after.get("reward") is None:
            raise SystemExit("落点反馈未产生奖励")
        base_unit = press["ms"]

        send(proc, {"t": "frame", "id": 2, "w": W, "h": H, "phase": "before",
                    "base": 1.0, "greedy": 1, "settled": 1}, game_over_frame())
        over = read(proc)
        print("结算：kind=%s 文本=%s 点击=(%.0f, %.0f)"
              % (over["kind"], over.get("text", ""), over.get("x", -1),
                 over.get("y", -1)))
        if over["kind"] != "restart":
            raise SystemExit("结算页未触发自动重开")
        send(proc, {"t": "frame", "id": 3, "w": W, "h": H, "phase": "before",
                    "base": 1.0, "greedy": 0, "settled": 1,
                    "manual": 1, "has_tap": 0}, make_frame())
        wait = read(proc)
        print("手动等待：kind=%s" % wait["kind"])
        if wait["kind"] != "wait":
            raise SystemExit("手动模式无点击时不应按压")

        send(proc, {"t": "frame", "id": 4, "w": W, "h": H, "phase": "before",
                    "base": 1.0, "greedy": 0, "settled": 1,
                    "manual": 1, "has_tap": 1, "tx": 490, "ty": 444},
             make_frame())
        tap = read(proc)
        print("手动跳跃：kind=%s ms=%.0f 距离=%.0f"
              % (tap["kind"], tap.get("ms", 0), tap["det"].get("d", 0)))
        if tap["kind"] != "press":
            raise SystemExit("手动模式点击后未按压")
        want_tap = MS_PER_PX * math.hypot(490 - 120, 444 - 560)
        if abs(tap["ms"] - want_tap) > max(60, 0.10 * want_tap):
            raise SystemExit("手动模式按压时长不符 %.0f != %.0f" % (tap["ms"], want_tap))
        if abs(tap.get("x", -1) - 490) > 2 or abs(tap.get("y", -1) - 444) > 2:
            raise SystemExit("手动模式按压点未落在点击位置 (%s, %s)"
                             % (tap.get("x"), tap.get("y")))

        send(proc, {"t": "frame", "id": 5, "w": W, "h": H, "phase": "before",
                    "base": 1.0, "greedy": 0, "settled": 1,
                    "assist": 1, "has_tap": 1, "tx": 490, "ty": 444},
             make_frame())
        fix = read(proc)
        print("人工修正：kind=%s ms=%.0f 距离=%.0f 点=(%.0f, %.0f)"
              % (fix["kind"], fix.get("ms", 0), fix["det"].get("d", 0),
                 fix.get("x", -1), fix.get("y", -1)))
        if fix["kind"] != "press":
            raise SystemExit("人工修正点击后未按压")
        want_fix = MS_PER_PX * math.hypot(490 - 120, 444 - 560)
        if abs(fix["ms"] - want_fix) > max(60, 0.10 * want_fix):
            raise SystemExit("人工修正按压时长不符 %.0f != %.0f"
                             % (fix["ms"], want_fix))
        if abs(fix.get("x", -1) - 490) > 2 or abs(fix.get("y", -1) - 444) > 2:
            raise SystemExit("人工修正按压点未落在点击位置")
        if not os.path.exists(calib):
            raise SystemExit("人工修正未保存样本")

        print("协议联调通过，首跳按压=%.0fms" % base_unit)
    finally:
        try:
            send(proc, {"t": "quit"})
        except Exception:
            pass
        proc.wait(timeout=10)
        err = proc.stderr.read()
        if err:
            print("stderr:", err.decode("utf-8", "ignore")[-400:])
        if os.path.exists(calib):
            os.remove(calib)


def main():
    print("== 策略学习 ==")
    test_policy()
    print("== 视觉识别 ==")
    test_vision()
    print("== 顶面识别 ==")
    test_topface()
    print("== 修正样本 ==")
    test_calib()
    print("== 管道联调 ==")
    test_worker()
    print("全部自测通过")


if __name__ == "__main__":
    main()
