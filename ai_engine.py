# -*- coding: utf-8 -*-
"""AI 跳跃主循环：环境感知 -> 截图 -> 感知 -> 原版公式基线 -> 精确按压 -> 落点反馈。

与原版 _game_loop 并列，由 GUI 的「AI 自校准」开关选择，互不影响。
和原版唯一的区别是：按压时长 = 原版公式 × 一个在线微调的系数 k。
首跳 k 就等于用户当前滑块值，所以第一跳与经典模式完全一致，不会冷启动跳飞。
"""
import math
import os
import random
import threading
import time

import numpy as np

from ai_calibration import AdaptiveCoefficient, legacy_press_ms
from ai_control import (ScreenGrabber, client_area, client_rect_screen,
                        grab_window, precise_press, to_pil)
from ai_environment import EnvironmentMonitor
from ai_vision import VisionConfig, analyze, annotate, is_settled

DEFAULT_STATE_PATH = "ai_calibration.json"


class AIJumpEngine(threading.Thread):
    """后台线程。所有回调都在本线程调用，GUI 侧请自行投递到 UI 队列。"""

    def __init__(self, window_rect, hwnd=None, screen_region=None,
                 calibrator=None, cfg=None, state_path=None,
                 base_coefficient=2.0, head_diameter=None,
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
        self.calibrator = calibrator or AdaptiveCoefficient.load(
            state_path, base_factor=base_coefficient)
        self.env = EnvironmentMonitor(head_diameter=head_diameter)
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
                full = grab_window(self.hwnd)
                # 裁掉标题栏，只保留客户区：原版公式与 head_diameter 都基于客户区
                ox, oy, cw, ch = client_area(self.hwnd)
                if cw > 20 and ch > 20:
                    return np.ascontiguousarray(full[oy:oy + ch, ox:ox + cw])
                return full
            except Exception as e:
                self.on_log("窗口抓图失败(%s)，回退屏幕抓图" % e)
        if self._screen is None:
            self._screen = ScreenGrabber()
        return self._screen.grab(self.screen_region or self.window_rect)

    def _click_point(self):
        if self.hwnd:
            l, t, r, b = client_rect_screen(self.hwnd)
        else:
            l, t, r, b = self.screen_region or self.window_rect
        w, h = r - l, b - t
        return (int(l + w * random.uniform(0.35, 0.65)),
                int(t + h * random.uniform(0.55, 0.78)))

    # ── 主循环 ──────────────────────────────────────────────
    def run(self):
        self._running = True
        s0 = self.calibrator.stats()
        self.on_log("AI 模式启动 | 沿用原版公式 | 基准系数 %.3f 当前系数 %.3f | 已有样本 %d" %
                    (s0["base_factor"], s0["factor"], s0["samples"]))
        misses = 0
        self._waiting_game = False
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

                # 环境感知：画面无效时绝不点击
                state = self.env.update(frame, self.window_rect)
                if not state.valid:
                    if misses in (0, 1, 5, 15):
                        self.on_log("环境异常(%s)，本轮跳过不点击" % state.describe())
                    misses += 1
                    time.sleep(0.4)
                    continue
                if state.resized:
                    self.on_log("窗口尺寸变为 %dx%d，head_diameter 重算为 %.0fpx" %
                                (state.width, state.height, state.head_diameter))

                # 原版识别算法优先：第一跳与原脚本完全一致，AI 只在其结果上微调
                res = analyze(frame, self.cfg, prefer="legacy")
                if res.game_over:
                    # 一局结束不等于跳飞：不动系数、不退出程序，蹲着等下一局。
                    # 启停完全交给 GUI 的开始/停止按钮。
                    if not self._waiting_game:
                        self._waiting_game = True
                        self.on_log("结算页：程序保持运行，等待重新开局（点「停止」才会结束）")
                        self.on_frame(to_pil(annotate(frame, res, ["WAITING FOR NEXT ROUND"])))
                    time.sleep(0.6)
                    continue
                if self._waiting_game:
                    self._waiting_game = False
                    self.on_log("已检测到新一局，继续跳跃")
                if not res.piece or not res.board:
                    misses += 1
                    if misses in (1, 5, 15):
                        self.on_log("画面识别失败(%d)，等待有效游戏画面…" % misses)
                    time.sleep(0.2)
                    continue
                misses = 0

                d_req = res.distance
                # 距离合理性：识别到离谱距离时宁可不跳，也不按住一两秒
                diag = math.hypot(frame.shape[1], frame.shape[0])
                if d_req < 30 or d_req > 0.72 * diag:
                    self.on_log("第%02d跳 目标距离异常(%.0fpx, 识别=%s)，跳过不点击" %
                                (self.jumps + 1, d_req, res.method))
                    time.sleep(0.3)
                    continue

                # 关键：以原版公式为基线，只乘一个被限幅的自适应系数
                base_unit = legacy_press_ms(d_req, state.head_diameter, 1.0)
                k = self.calibrator.stats()["factor"]
                t_ms = self.calibrator.predict(base_unit)
                x, y = self._click_point()
                actual_t = precise_press(x, y, t_ms, no_click=self.dry_run)

                self.on_frame(to_pil(annotate(frame, res, [
                    "AI d=%.0fpx  t=%.0fms (base %.0f x %.3f)" % (d_req, actual_t, base_unit, k),
                    "k=%.3f base=%.3f samples=%d" %
                    (k, self.calibrator.stats()["base_factor"], self.calibrator.samples)])))

                self._wait_settled(frame)
                after = analyze(self._grab(), self.cfg, prefer="legacy")

                if after.piece and not after.game_over:
                    vx = after.piece[0] - res.piece[0]
                    vy = after.piece[1] - res.piece[1]
                    ux = (res.board[0] - res.piece[0]) / max(d_req, 1e-6)
                    uy = (res.board[1] - res.piece[1]) / max(d_req, 1e-6)
                    d_act = vx * ux + vy * uy          # 沿目标方向投影
                    conf = min(res.confidence, 0.8 if after.confidence == 0 else after.confidence)
                    ok, reason, err = self.calibrator.observe(
                        actual_t, d_act, d_req, base_unit, confidence=conf)
                    if ok:
                        self.on_log("第%02d跳 [%s conf=%.2f] 目标%.0fpx 实跳%.0fpx 偏差%+.0fpx"
                                    " | hd=%.0f base=%.0f k=%.3f t=%.0fms" %
                                    (self.jumps + 1, res.method, res.confidence, d_req,
                                     d_act, err, state.head_diameter, base_unit,
                                     self.calibrator.stats()["factor"], actual_t))
                    else:
                        self.on_log("第%02d跳 偏差%+.0fpx | %s，k=%.3f" %
                                    (self.jumps + 1, err if err is not None else 0,
                                     reason, self.calibrator.stats()["factor"]))
                else:
                    self.on_log("第%02d跳 落点识别失败或已进入结算页，本轮不学习" %
                                (self.jumps + 1))

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
            self.on_log("AI 引擎结束 | 共 %d 跳 | 系数 %.3f (基准 %.3f) | 样本 %d | 跳飞保护 %d 次" %
                        (self.jumps, s["factor"], s["base_factor"],
                         s["samples"], s["overshoots"]))

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
