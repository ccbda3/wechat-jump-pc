# -*- coding: utf-8 -*-
"""AI 跳跃主循环：截图 -> 感知 -> 在线预测 -> 精确按压 -> 落点反馈 -> 学习。

与原版 _game_loop 完全并列，由 GUI 的「AI 自校准」开关选择，互不影响。
每一轮都会把 (实际跳了多远, 实际按了多久) 喂给 OnlineCalibrator，
所以系数会随窗口尺寸、DPI、机器性能自动收敛，不再需要人手调。
"""
import math
import os
import random
import threading
import time

from ai_calibration import OnlineCalibrator
from ai_control import (ScreenGrabber, get_window_rect, grab_window,
                        precise_press, to_pil)
from ai_vision import VisionConfig, analyze, annotate, is_settled

DEFAULT_STATE_PATH = "ai_calibration.json"


class AIJumpEngine(threading.Thread):
    """后台线程。所有回调都在本线程调用，GUI 侧请自行投递到 UI 队列。"""

    def __init__(self, window_rect, hwnd=None, screen_region=None,
                 calibrator=None, cfg=None, state_path=None,
                 on_log=None, on_frame=None, on_stats=None, on_count=None,
                 frame_source=None, dry_run=False, max_jumps=None,
                 settle_timeout=3.0, **kwargs):
        super().__init__(daemon=True, **kwargs)
        self.window_rect = tuple(window_rect)
        self.hwnd = hwnd
        self.screen_region = screen_region
        state_path = state_path or DEFAULT_STATE_PATH
        if not os.path.isabs(state_path):
            state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), state_path)
        self.calibrator = calibrator or OnlineCalibrator.load(state_path)
        self.cfg = cfg or VisionConfig.load()
        self.on_log = on_log or (lambda s: print("[AI]", s))
        self.on_frame = on_frame or (lambda img: None)
        self.on_stats = on_stats or (lambda d: None)
        self.on_count = on_count or (lambda n: None)
        self.frame_source = frame_source       # 离线自测用：callable -> BGR ndarray
        self.dry_run = dry_run                 # True 时只走计时、不发鼠标事件
        self.max_jumps = max_jumps
        self.settle_timeout = settle_timeout

        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self.jumps = 0
        self._screen = None

    # ── 控制 ────────────────────────────────────────────────
    def stop(self):
        self._running = False

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    @property
    def paused(self):
        with self._lock:
            return self._paused

    # ── 抓图 ────────────────────────────────────────────────
    def _grab(self):
        if self.frame_source is not None:
            return self.frame_source()
        if self.hwnd:
            try:
                return grab_window(self.hwnd)
            except Exception as e:
                self.on_log("窗口抓图失败(%s)，回退屏幕抓图" % e)
        if self._screen is None:
            self._screen = ScreenGrabber()
        return self._screen.grab(self.screen_region or self.window_rect)

    def _click_point(self):
        l, t, r, b = self.window_rect
        w, h = r - l, b - t
        return (int(l + w * random.uniform(0.35, 0.65)),
                int(t + h * random.uniform(0.55, 0.78)))

    # ── 主循环 ──────────────────────────────────────────────
    def run(self):
        self._running = True
        self.on_log("AI 自校准模式启动 | 已有学习样本 %d 条 | 当前 a=%.3f ms/px" %
                    (self.calibrator.samples, self.calibrator.stats()["a"]))
        misses = 0
        try:
            while self._running and (self.max_jumps is None or self.jumps < self.max_jumps):
                if self.paused:
                    time.sleep(0.25)
                    continue

                try:
                    frame = self._grab()
                except Exception as e:
                    self.on_log("抓图异常: %s" % e)
                    time.sleep(0.4)
                    continue

                res = analyze(frame, self.cfg)
                if res.game_over:
                    self.on_log("检测到结算页，停止跳跃（本局结束）")
                    self.on_frame(to_pil(annotate(frame, res, ["GAME OVER"])))
                    break
                if not res.piece or not res.board:
                    misses += 1
                    if misses in (1, 5, 15):
                        self.on_log("画面识别失败(%d)，等待有效游戏画面…" % misses)
                    time.sleep(0.2)
                    continue
                misses = 0

                d_req = res.distance
                t_ms = self.calibrator.predict(d_req)
                x, y = self._click_point()
                actual_t = precise_press(x, y, t_ms, no_click=self.dry_run)

                self.on_frame(to_pil(annotate(frame, res, [
                    "AI d=%.0fpx  t=%.0fms" % (d_req, actual_t),
                    "samples=%d  a=%.3f" % (self.calibrator.samples,
                                            self.calibrator.stats()["a"])])))

                self._wait_settled(frame)
                after = analyze(self._grab(), self.cfg)

                if after.piece:
                    vx = after.piece[0] - res.piece[0]
                    vy = after.piece[1] - res.piece[1]
                    ux = (res.board[0] - res.piece[0]) / max(d_req, 1e-6)
                    uy = (res.board[1] - res.piece[1]) / max(d_req, 1e-6)
                    d_act = vx * ux + vy * uy          # 沿目标方向投影
                    conf = min(res.confidence, 0.8 if after.confidence == 0 else after.confidence)
                    ok, reason, err = self.calibrator.observe(
                        actual_t, d_act, d_req, confidence=conf)
                    if ok:
                        self.on_log("第%02d跳 目标%.0fpx 实跳%.0fpx 偏差%+.0fpx 按压%.0fms | 已学习" %
                                    (self.jumps + 1, d_req, d_act, err, actual_t))
                    else:
                        self.on_log("第%02d跳 偏差%+.0fpx | 样本丢弃(%s)" %
                                    (self.jumps + 1, err if err is not None else 0, reason))
                else:
                    self.on_log("第%02d跳 落点识别失败，本轮不学习" % (self.jumps + 1))

                self.jumps += 1
                self.on_count(self.jumps)
                self.on_stats(self.calibrator.stats())
                time.sleep(random.uniform(0.35, 0.6))
        finally:
            self.calibrator.save()
            if self._screen:
                self._screen.close()
            self._running = False
            s = self.calibrator.stats()
            self.on_log("AI 引擎结束 | 共 %d 跳 | 学习样本 %d | a=%.3f b=%.1f" %
                        (self.jumps, s["samples"], s["a"], s["b"]))

    def _wait_settled(self, first_frame):
        """轮询到画面连续静止，避免把抛物线中的棋子当成落点。"""
        prev = first_frame
        stable = 0
        deadline = time.monotonic() + self.settle_timeout
        time.sleep(0.22)          # 等棋子离地，别把「还没动」误判成「已落稳」
        while self._running and time.monotonic() < deadline:
            time.sleep(0.12)
            try:
                cur = self._grab()
            except Exception:
                continue
            if is_settled(prev, cur, self.cfg):
                stable += 1
                if stable >= 2:
                    return True
            else:
                stable = 0
            prev = cur
        return False
