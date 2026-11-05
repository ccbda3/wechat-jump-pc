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

from ml.policy import ACTIONS, FEATURES, HIDDEN, Policy, SCALES, legacy_press_ms
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
    policy = Policy(path=tmp, lr=0.06, seed=2)
    x = np.zeros(FEATURES)
    x[0] = 1.0
    x[1] = 0.30
    x[2] = 0.7
    x[8] = 0.31
    ideal = 14
    base = 300.0
    rewards = []
    for step in range(700):
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


def test_worker():
    worker = os.path.join(ROOT, "ml", "worker.py")
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    proc = subprocess.Popen(
        [py, worker], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0)
    try:
        send(proc, {"t": "init"})
        init = read(proc, quiet=True)
        print("初始化：kind=%s 模型=%s" % (init["kind"], init["stats"]["model"]))

        send(proc, {"t": "frame", "id": 1, "w": W, "h": H, "phase": "before",
                    "base": 2.0, "greedy": 0, "settled": 1}, make_frame())
        press = read(proc)
        print("决策：kind=%s ms=%.0f 动作=%d 缩放=%.2f"
              % (press["kind"], press.get("ms", 0), press.get("action", -1),
                 press.get("scale", 0)))
        if press["kind"] != "press":
            raise SystemExit("未输出按压动作")

        base_unit = legacy_press_ms(press["det"]["d"], W / 8.0, 1.0)
        send(proc, {"t": "frame", "id": 1, "w": W, "h": H, "phase": "after",
                    "base": 2.0, "greedy": 0, "settled": 1},
             make_frame(382.0, -116.0))
        after = read(proc)
        print("反馈：kind=%s 奖励=%s 偏差=%s" % (after["kind"], after.get("reward"),
                                              after.get("err")))
        if after["kind"] != "ok" or after.get("reward") is None:
            raise SystemExit("落点反馈未产生奖励")

        send(proc, {"t": "frame", "id": 2, "w": W, "h": H, "phase": "before",
                    "base": 2.0, "greedy": 1, "settled": 1}, game_over_frame())
        over = read(proc)
        print("结算：kind=%s 文本=%s 点击=(%.0f, %.0f)"
              % (over["kind"], over.get("text", ""), over.get("x", -1),
                 over.get("y", -1)))
        if over["kind"] != "restart":
            raise SystemExit("结算页未触发自动重开")
        print("协议联调通过，base_unit=%.0f" % base_unit)
    finally:
        try:
            send(proc, {"t": "quit"})
        except Exception:
            pass
        proc.wait(timeout=10)
        err = proc.stderr.read()
        if err:
            print("stderr:", err.decode("utf-8", "ignore")[-400:])


def main():
    print("== 策略学习 ==")
    test_policy()
    print("== 视觉识别 ==")
    test_vision()
    print("== 管道联调 ==")
    test_worker()
    print("全部自测通过")


if __name__ == "__main__":
    main()
