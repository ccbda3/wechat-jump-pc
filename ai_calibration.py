# -*- coding: utf-8 -*-
"""在线校准器：用「跳完之后的真实落点」自主学习 按压时长 = f(目标距离)。

这是整个 AI 模式的核心。原版是开环：跳偏了不修正，误差永远不收敛。
这里把每次跳跃当作一个训练样本：
    观测 (实际跳跃距离 d_actual, 本次发出的按压 t_used)
    拟合  t = a * d + b
用带遗忘因子的递推最小二乘（RLS）在线更新，几十跳内自动收敛，
并且能自适应窗口尺寸、DPI、不同电脑、不同游戏版本。

注意：拟合的是「实际发生了什么」，不是「我们希望发生什么」，
所以感知层哪怕存在稳定的系统性偏差，也会被 a 自动吸收掉。
"""
import json
import math
import os
import threading

import numpy as np

D_SCALE = 100.0          # 距离归一化，保证 RLS 数值稳定
DEFAULT_A = 2.0          # ms / px 的初值（约等于原版默认系数 2.0 的等效值）
DEFAULT_B = 100.0        # ms
MIN_PRESS_MS = 200.0
MAX_PRESS_MS = 2200.0


class OnlineCalibrator:
    """线程安全的在线线性回归校准器（RLS + 遗忘因子 + 异常样本剔除）。"""

    def __init__(self, path=None, forgetting=0.97, auto_save=True):
        self.path = path
        self.forgetting = float(forgetting)
        self.auto_save = auto_save
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._theta = np.array([DEFAULT_A * D_SCALE, DEFAULT_B], dtype=float)
            self._P = np.eye(2) * 500.0
            self.samples = 0
            self.rejected = 0
            self._ewma_err = None
            self.history = []          # 最近若干个 (d_req, d_act, t_ms, err_px)
            self.last_reject_reason = ""

    # ── 预测 ────────────────────────────────────────────────
    def _raw(self, d_px):
        x = np.array([d_px / D_SCALE, 1.0])
        return float(x @ self._theta), x

    def predict(self, d_px):
        """给定目标距离（像素），返回建议按压时长（毫秒）。"""
        with self._lock:
            t, _ = self._raw(float(d_px))
        return float(np.clip(t, MIN_PRESS_MS, MAX_PRESS_MS))

    # ── 学习 ────────────────────────────────────────────────
    def observe(self, t_ms, d_actual_px, d_requested_px=None, confidence=1.0):
        """喂入一次真实跳跃的观测结果，返回 (accepted, reason, err_px)。

        t_ms            实际发出的按压时长（用 precise_press 的实测值，不要用名义值）
        d_actual_px     跳后棋子相对跳前沿目标方向的位移（像素）
        d_requested_px  本轮期望距离，用于异常过滤；None 时跳过比例检查
        confidence      落点检测置信度，过低不学习
        """
        t_ms = float(t_ms)
        d_act = float(d_actual_px)
        err = None if d_requested_px is None else d_act - float(d_requested_px)

        reason = self._reject_reason(t_ms, d_act, d_requested_px, confidence)
        if reason:
            with self._lock:
                self.rejected += 1
                self.last_reject_reason = reason
            return False, reason, err

        with self._lock:
            pred, x = self._raw(d_act)
            lam = self.forgetting
            denom = lam + float(x @ self._P @ x)
            K = (self._P @ x) / denom
            self._theta += K * (t_ms - pred)
            self._P = (self._P - np.outer(K, x @ self._P)) / lam
            self.samples += 1
            if err is not None:
                self._ewma_err = err if self._ewma_err is None else 0.7 * self._ewma_err + 0.3 * err
                self.history.append((round(float(d_requested_px), 1), round(d_act, 1),
                                     round(t_ms, 1), round(err, 1)))
                self.history = self.history[-60:]
            if self.auto_save and self.path:
                self._save_unlocked()
        return True, "ok", err

    def _reject_reason(self, t_ms, d_act, d_req, confidence):
        if not math.isfinite(d_act) or not math.isfinite(t_ms) or d_act <= 0:
            return "invalid"
        if confidence < 0.4:
            return "low_confidence"
        if d_req is not None:
            d_req = float(d_req)
            if d_act < 0.35 * d_req:
                return "fell_short"       # 多半是没跳上去 / 掉落，样本不可用
            if d_act > 2.0 * d_req + 120:
                return "implausible"      # 多半是落点识别错
        if self.samples > 8:
            with self._lock:
                pred, _ = self._raw(d_act)
            if abs(t_ms - pred) > max(280.0, 0.65 * t_ms):
                return "outlier"          # 与已学模型严重矛盾，别污染拟合
        return ""

    # ── 状态 ────────────────────────────────────────────────
    def stats(self):
        with self._lock:
            a = self._theta[0] / D_SCALE
            b = self._theta[1]
            return {
                "samples": self.samples,
                "rejected": self.rejected,
                "a": round(a, 3),             # ms per px
                "b": round(b, 1),
                "ewma_err": None if self._ewma_err is None else round(self._ewma_err, 1),
                "last_reject": self.last_reject_reason,
                "recent": self.history[-5:],
            }

    # ── 持久化 ──────────────────────────────────────────────
    def _save_unlocked(self):
        try:
            data = {"theta": self._theta.tolist(), "P": self._P.tolist(),
                    "samples": self.samples, "rejected": self.rejected,
                    "ewma_err": self._ewma_err, "history": self.history[-20:]}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def save(self):
        with self._lock:
            if self.path:
                self._save_unlocked()

    @classmethod
    def load(cls, path, **kwargs):
        obj = cls(path=path, **kwargs)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            obj._theta = np.array(data["theta"], dtype=float)
            obj._P = np.array(data["P"], dtype=float)
            obj.samples = int(data.get("samples", 0))
            obj.rejected = int(data.get("rejected", 0))
            obj._ewma_err = data.get("ewma_err")
            obj.history = data.get("history", [])
        except Exception:
            pass
        return obj
