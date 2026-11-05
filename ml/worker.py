import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PROTO = sys.stdout.buffer
sys.stdout = sys.stderr

import cv2
import numpy as np

from ai_vision import VisionConfig, analyze, detect_game_over
from policy import (MAX_PRESS_MS, MIN_PRESS_MS, Policy, legacy_press_ms)
from ocr import OcrReader, game_crop
from landing import landing_reward

MODEL_PATH = os.path.join(ROOT, "ml_model.json")
LOG_DIR = os.path.join(ROOT, "logs")


def emit(obj):
    try:
        PROTO.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        PROTO.flush()
    except Exception:
        pass


def log(msg):
    try:
        emit({"t": "log", "msg": str(msg).replace("\n", " ")})
    except Exception:
        pass


def features(res, w, h, base_unit, prev_reward, x0=0):
    d = res.distance
    diag = math.hypot(w, h)
    px, py = res.piece
    bx, by = res.board
    px -= x0
    bx -= x0
    return np.array([
        1.0,
        d / diag,
        min(d / 400.0, 2.0),
        1.0 if d < 200 else 0.0,
        (bx - px) / diag,
        (by - py) / diag,
        py / h,
        px / w,
        min(base_unit / 1000.0, 1.5),
        float(prev_reward),
    ], dtype=np.float64)


def analyze_view(cfg, frame):
    sub, x0 = game_crop(frame)
    res = analyze(sub, cfg, prefer="legacy")
    if x0:
        if res.piece:
            res.piece = (res.piece[0] + x0, res.piece[1])
        if res.board:
            res.board = (res.board[0] + x0, res.board[1])
        if res.distance:
            res.distance = math.hypot(res.board[0] - res.piece[0],
                                      res.board[1] - res.piece[1])
    return res, sub, x0


def det_of(res, over, reward=None, err=None, text=None):
    d = {
        "over": 1 if over else 0,
        "method": res.method if res else "failed",
        "has_piece": 1 if (res and res.piece) else 0,
        "has_board": 1 if (res and res.board) else 0,
        "d": round(float(res.distance), 1) if res else 0.0,
        "conf": round(float(res.confidence), 2) if res else 0.0,
    }
    if res and res.piece:
        d["px"] = round(float(res.piece[0]), 1)
        d["py"] = round(float(res.piece[1]), 1)
    if res and res.board:
        d["bx"] = round(float(res.board[0]), 1)
        d["by"] = round(float(res.board[1]), 1)
    if reward is not None:
        d["reward"] = round(float(reward), 3)
    if err is not None:
        d["err"] = round(float(err), 1)
    if text:
        d["ocr"] = text
    return d


class Runner:
    def __init__(self):
        self.cfg = VisionConfig.load()
        self.policy = Policy.load(MODEL_PATH)
        self.ocr = OcrReader()
        self.pending = {}
        self.episode_jumps = 0
        self.need_start = True
        self.episode_reward = 0.0
        self.jumps_in_episode = 0
        self.last_wait = None
        self.wait_count = 0
        self.last_ocr = 0.0
        self.last_click = 0.0
        self.episode_open = False
        self._init_log()

    def note_wait(self, reason, res=None, frame=None):
        if reason == self.last_wait:
            self.wait_count += 1
            if self.wait_count % 20 != 0:
                return
        else:
            self.last_wait = reason
            self.wait_count = 1
        extra = ""
        if res is not None:
            extra = " method=%s piece=%s board=%s" % (
                res.method, res.piece, res.board)
        self.logline("等待有效画面：%s%s" % (reason, extra))

    def _init_log(self):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            self._logf = open(os.path.join(
                LOG_DIR, "rl_%s.log" % __import__("time").strftime("%Y%m%d")),
                "a", encoding="utf-8")
        except Exception:
            self._logf = None

    def logline(self, msg):
        import time
        text = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        if self._logf:
            try:
                self._logf.write(text + "\n")
                self._logf.flush()
            except Exception:
                pass
        log(text)

    def base_stats(self):
        s = self.policy.stats()
        s["episode_jumps"] = self.jumps_in_episode
        return s

    def handle_init(self, req):
        s = self.policy.stats()
        if s["jumps"] > 0:
            self.logline("已加载训练模型：%d 局 / %d 跳 / 平均奖励 %.3f / 最佳 %.3f / 保存于 %s"
                         % (s["episodes"], s["jumps"], s["mean"], s["best"], s["model"]))
        else:
            self.logline("新模型：策略网络 %d 隐层，%d 个动作，奖励以落点离中心距离计算"
                         % (24, 21))
        if not self.ocr._init():
            self.logline("警告：系统缺少中文 OCR 语言包，结算页将退回按钮形状识别")
        emit({"t": "res", "kind": "ready", "stats": s})

    def handle_reset(self, req):
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
        except Exception:
            pass
        self.policy = Policy(path=MODEL_PATH)
        self.pending.clear()
        self.logline("学习记录已重置，策略网络回到初始状态")
        emit({"t": "res", "kind": "ok", "stats": self.policy.stats()})

    def close_episode(self):
        if self.episode_open:
            self.episode_open = False
            self.policy.end_episode()
            self.logline("本局结束：共 %d 跳" % self.jumps_in_episode)
        self.jumps_in_episode = 0
        self.episode_reward = 0.0
        self.need_start = True

    def tag_of(self, loc):
        if loc["source"] == "ocr":
            return "汉字「%s」" % loc["text"]
        return "结算按钮"

    def handle_before(self, req, frame, base_coef, greedy):
        res, sub, x0 = analyze_view(self.cfg, frame)
        over = bool(res.game_over) or detect_game_over(sub)
        w = frame.shape[1] - x0
        h = frame.shape[0]
        now = time.time()
        valid_game = (res.piece is not None and res.board is not None
                      and res.method != "game_over" and res.confidence >= 0.75)
        want_ocr = (not valid_game) and (over or self.need_start
                    or res.piece is None or res.board is None
                    or (now - self.last_ocr) >= 1.0)
        loc = None
        if want_ocr:
            self.last_ocr = now
            loc = self.ocr.locate(
                frame, allow_shape=(over or res.piece is None or self.need_start))
            kind = loc.get("kind") if loc else None
            if kind == "list":
                if loc["x"] is None:
                    self.note_wait("排行榜页未找到返回按钮", res)
                    emit({"t": "res", "kind": "wait", "det": det_of(res, False),
                          "stats": self.base_stats()})
                    return
                if now - self.last_click < 1.4:
                    emit({"t": "res", "kind": "wait", "det": det_of(res, False),
                          "stats": self.base_stats()})
                    return
                self.last_click = now
                self.logline("检测到汉字「%s」，点击返回按钮回到游戏" % loc["text"])
                emit({"t": "res", "kind": "restart", "x": loc["x"], "y": loc["y"],
                      "text": loc["text"], "det": det_of(res, False),
                      "stats": self.base_stats()})
                self.need_start = True
                return
            if kind == "start":
                if now - self.last_click < 1.4:
                    emit({"t": "res", "kind": "wait", "det": det_of(res, False),
                          "stats": self.base_stats()})
                    return
                self.last_click = now
                self.need_start = False
                self.logline("检测到汉字「%s」，自动点击 (%.0f, %.0f)"
                             % (loc["text"], loc["x"], loc["y"]))
                emit({"t": "res", "kind": "restart", "x": loc["x"], "y": loc["y"],
                      "text": loc["text"], "det": det_of(res, False),
                      "stats": self.base_stats()})
                return
            if kind == "over":
                over = True
        if over:
            self.close_episode()
            if loc and loc.get("x") is not None:
                if now - self.last_click < 1.4:
                    emit({"t": "res", "kind": "wait", "det": det_of(res, True),
                          "stats": self.base_stats()})
                    return
                self.last_click = now
                self.need_start = False
                self.logline("检测到%s，自动点击 (%.0f, %.0f) 重新开始"
                             % (self.tag_of(loc), loc["x"], loc["y"]))
                emit({"t": "res", "kind": "restart", "x": loc["x"], "y": loc["y"],
                      "text": loc["text"], "det": det_of(res, True, text=loc["text"]),
                      "stats": self.base_stats()})
                return
            self.note_wait("结算页等待按钮", res)
            emit({"t": "res", "kind": "wait", "det": det_of(res, True),
                  "stats": self.base_stats()})
            return
        if res.piece is None or res.board is None:
            self.note_wait("识别失败", res)
            emit({"t": "res", "kind": "wait", "det": det_of(res, False),
                  "stats": self.base_stats()})
            return
        if res.method != "legacy" and res.confidence < 0.8:
            self.note_wait("识别置信度过低 %.2f" % res.confidence, res)
            emit({"t": "res", "kind": "wait", "det": det_of(res, False),
                  "stats": self.base_stats()})
            return
        diag = math.hypot(w, h)
        d = float(res.distance)
        if d < 25 or d > 0.75 * diag:
            self.note_wait("距离异常 %.0fpx" % d, res)
            emit({"t": "res", "kind": "wait", "det": det_of(res, False),
                  "stats": self.base_stats()})
            return
        head = max(22.0, w / 8.0)
        base_unit = legacy_press_ms(d, head, 1.0)
        feat = features(res, w, h, base_unit, self.policy.last_reward, x0)
        action, scale, ent = self.policy.act(feat, greedy)
        ms = min(max(base_unit * base_coef * scale, MIN_PRESS_MS), MAX_PRESS_MS)
        self.pending[self.pending_key(req)] = {
            "frame": frame, "res": res, "feat": feat, "action": action,
            "scale": scale, "ms": ms, "d": d, "base_unit": base_unit,
        }
        self.episode_open = True
        self.need_start = False
        self.logline("第%02d跳 距离%.0fpx 动作%d 系数%.2f 按压%.0fms 熵%.2f"
                     % (self.jumps_in_episode + 1, d, action, scale, ms, ent))
        emit({"t": "res", "kind": "press", "ms": ms, "action": action,
              "scale": scale, "entropy": ent,
              "x": float(x0) + w * 0.5, "y": float(h) * 0.68,
              "det": det_of(res, False),
              "stats": self.base_stats()})

    def pending_key(self, req):
        return int(req.get("id", 0))

    def handle_after(self, req, frame):
        key = self.pending_key(req)
        info = self.pending.pop(key, None)
        if info is None:
            emit({"t": "res", "kind": "wait", "stats": self.base_stats()})
            return
        after, sub_after, ax0 = analyze_view(self.cfg, frame)
        over = bool(after.game_over) or detect_game_over(sub_after)
        if over or after.piece is None:
            self.policy.update(info["feat"], info["action"], 0.0)
            self.logline("跳跃失败：奖励 0.00")
            self.close_episode()
            loc = self.ocr.locate(frame, allow_shape=True)
            can_click = (loc and loc.get("x") is not None
                         and (loc["source"] == "ocr" or loc.get("over")))
            if can_click:
                self.need_start = False
                self.logline("检测到%s，自动点击 (%.0f, %.0f) 重新开始"
                             % (self.tag_of(loc), loc["x"], loc["y"]))
                emit({"t": "res", "kind": "restart", "x": loc["x"], "y": loc["y"],
                      "text": loc["text"],
                      "det": det_of(after, True, reward=0.0, text=loc["text"]),
                      "stats": self.base_stats()})
            else:
                emit({"t": "res", "kind": "ok", "reward": 0.0,
                      "det": det_of(after, True, reward=0.0),
                      "stats": self.base_stats()})
            return
        settled = int(req.get("settled", 1))
        if not settled:
            emit({"t": "res", "kind": "ok", "reward": None,
                  "det": det_of(after, False), "stats": self.base_stats()})
            return
        reward, err, block = landing_reward(
            sub_after, (after.piece[0] - ax0, after.piece[1]))
        if reward is None:
            self.logline("没有找到落点方砖，本跳不学习")
            emit({"t": "res", "kind": "ok", "reward": None,
                  "det": det_of(after, False), "stats": self.base_stats()})
            return
        self.policy.update(info["feat"], info["action"], reward)
        self.jumps_in_episode += 1
        self.episode_reward += reward
        self.logline("第%02d跳 目标%.0fpx 横向偏差%.0fpx 奖励%.3f 平均%.3f"
                     % (self.jumps_in_episode, info["d"], err, reward,
                        float(np.mean(self.policy.rewards[-20:]))))
        emit({"t": "res", "kind": "ok", "reward": reward, "err": err,
              "det": det_of(after, False, reward=reward, err=err),
              "stats": self.base_stats()})

    def handle_frame(self, req, frame):
        if req.get("phase") == "after":
            self.handle_after(req, frame)
            return
        self.handle_before(req, frame, float(req.get("base", 2.0)),
                           bool(int(req.get("greedy", 0))))

    def read_frame(self, req):
        path = os.path.join(LOG_DIR, "frame_%s.bin" % req.get("phase"))
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except Exception as e:
            self.logline("读取画面失败：%s" % e)
            return None
        w = int(req.get("w", 0))
        h = int(req.get("h", 0))
        need = w * h * 4
        if w <= 0 or h <= 0 or len(raw) < need:
            self.logline("画面尺寸不符：%d 字节，期望 %d" % (len(raw), need))
            return None
        arr = np.frombuffer(raw[:need], dtype=np.uint8)
        img = arr.reshape(h, w, 4)
        return np.ascontiguousarray(img[:, :, :3])

    def run(self):
        stdin = sys.stdin.buffer
        while True:
            line = stdin.readline()
            if not line:
                break
            try:
                req = json.loads(line.decode("utf-8", "ignore"))
            except Exception:
                continue
            t = req.get("t")
            if t == "init":
                self.handle_init(req)
            elif t == "reset":
                self.handle_reset(req)
            elif t == "frame":
                frame = self.read_frame(req)
                if frame is None:
                    emit({"t": "res", "kind": "wait", "stats": self.base_stats()})
                    continue
                try:
                    self.handle_frame(req, frame)
                except Exception as e:
                    self.logline("帧处理异常：%s" % e)
                    emit({"t": "res", "kind": "wait", "stats": self.base_stats()})
            elif t == "quit":
                break
        self.policy.save()
        if self._logf:
            try:
                self._logf.close()
            except Exception:
                pass


def main():
    runner = Runner()
    try:
        runner.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log("引擎异常退出：%s" % e)


if __name__ == "__main__":
    main()
