# -*- coding: utf-8 -*-
"""环境自适应系数：以原版公式为基线，只学一个小修正量。

设计原则（避免冷启动「从零瞎按」）：
  1. 首跳与经典模式**完全一致**：按压 = 原版 calc_press_time(距离, 头部直径) × 用户已调好的系数。
  2. AI 只在线微调这个系数 k，并被夹在基准值的 ±40% 内，单步变化不超过 8%。
  3. 按压时长有绝对上限，任何异常都不可能出现「一直按住」。
  4. 跳飞（落点远超目标）立即把系数往回拉，绝不拿坏样本去学习。

所以它不是重新训练一个物理模型，而是在原项目已有的可靠参数之上，
根据当前窗口尺寸 / DPI / 机器计时 / 画面识别偏差做环境自适应。
"""
import json
import math
import os
import threading

MIN_PRESS_MS = 200.0
MAX_PRESS_MS = 1400.0          # 硬上限：原版典型值约 650ms，正常绝不会触顶


def legacy_press_ms(distance_px, head_diameter, coefficient=1.0):
    """原版 gui_app.calc_press_time 的等价实现，coefficient=1.0 时为裸基线。"""
    hd = max(float(head_diameter), 1.0)
    scale = 0.945 * 2 / hd
    actual = float(distance_px) * scale * (math.sqrt(6) / 2)
    t = (-945 + math.sqrt(945 ** 2 + 4 * 105 * 36 * actual)) / (2 * 105) * 1000
    return t * float(coefficient)


class AdaptiveCoefficient:
    """线程安全的在线系数自适应（EWMA + 限幅 + 跳飞保护）。"""

    def __init__(self, path=None, base_factor=2.0, factor=None, beta=0.28,
                 clamp_low=0.60, clamp_high=1.40, max_step=0.08,
                 auto_save=True):
        self.path = path
        self.base_factor = float(base_factor)      # 用户在 GUI 里调好的原版系数
        self.factor = float(factor) if factor is not None else self.base_factor
        self.beta = float(beta)
        self.clamp_low = float(clamp_low)
        self.clamp_high = float(clamp_high)
        self.max_step = float(max_step)
        self.auto_save = auto_save
        self._lock = threading.Lock()
        self.samples = 0
        self.rejected = 0
        self.overshoots = 0
        self._ewma_err = None
        self.history = []
        self.last_action = "init"
        if path:
            self._load()

    # ── 边界 ────────────────────────────────────────────────
    def _bounds(self):
        return (self.base_factor * self.clamp_low,
                self.base_factor * self.clamp_high)

    def _clip(self, value):
        return float(min(max(value, self._bounds()[0]), self._bounds()[1]))

    # ── 预测 ────────────────────────────────────────────────
    def predict(self, base_ms_unit):
        """base_ms_unit = legacy_press_ms(d, hd, 1.0)；返回本次按压毫秒。"""
        with self._lock:
            k = self.factor
        return float(min(max(base_ms_unit * k, MIN_PRESS_MS), MAX_PRESS_MS))

    def predict_for(self, distance_px, head_diameter):
        return self.predict(legacy_press_ms(distance_px, head_diameter, 1.0))

    # ── 学习 ────────────────────────────────────────────────
    def observe(self, t_used_ms, d_actual_px, d_requested_px,
                base_ms_unit, confidence=1.0):
        """用落点偏差反推「本来该按多久」，只微调系数 k。

        返回 (accepted, reason, err_px)。
        """
        t_used = float(t_used_ms)
        d_act = float(d_actual_px)
        d_req = float(d_requested_px)
        err = d_act - d_req
        reason = self._reject_reason(t_used, d_act, d_req, confidence)
        if reason:
            with self._lock:
                self.rejected += 1
                self.last_action = reason
                if reason in ("overshoot", "implausible"):
                    self.overshoots += 1
                    self.factor = self._clip(self.factor * 0.88)  # 立刻回拉
                    self.last_action = "overshoot_pullback"
                if self.auto_save and self.path:
                    self._save_unlocked()
            return False, reason, err

        with self._lock:
            # 局部线性近似：想跳 d_req，实际跳了 d_act，则应按 t_used * d_req/d_act
            t_target = t_used * d_req / max(d_act, 1.0)
            k_target = t_target / max(base_ms_unit, 1.0)
            lo, hi = self.factor * (1 - self.max_step), self.factor * (1 + self.max_step)
            k_target = min(max(k_target, lo), hi)
            self.factor = self._clip((1 - self.beta) * self.factor + self.beta * k_target)
            self.samples += 1
            self._ewma_err = err if self._ewma_err is None else 0.7 * self._ewma_err + 0.3 * err
            self.history.append((round(d_req, 1), round(d_act, 1),
                                 round(t_used, 1), round(err, 1)))
            self.history = self.history[-60:]
            self.last_action = "learn"
            if self.auto_save and self.path:
                self._save_unlocked()
        return True, "ok", err

    def _reject_reason(self, t_used, d_act, d_req, confidence):
        if not all(math.isfinite(v) for v in (t_used, d_act, d_req)) or d_act <= 0 or d_req <= 0:
            return "invalid"
        if confidence < 0.4:
            return "low_confidence"
        if d_act < 0.35 * d_req:
            return "fell_short"
        if d_act > 2.5 * d_req + 80:
            return "implausible"
        if d_act > 1.6 * d_req + 60:
            return "overshoot"           # 跳飞：不学习，立即回拉
        return ""

    def on_overshoot(self):
        """仅用于「落点远超目标」的真跳飞，把系数拉回保守值。

        注意：结算页（一局结束）不能调用这里 —— 跳近掉落反而需要加大系数。
        """
        with self._lock:
            self.factor = self._clip(self.factor * 0.85)
            self.overshoots += 1
            self.last_action = "overshoot_pullback"
            if self.auto_save and self.path:
                self._save_unlocked()

    # ── 状态 ────────────────────────────────────────────────
    def stats(self):
        with self._lock:
            return {
                "factor": round(self.factor, 3),
                "base_factor": round(self.base_factor, 3),
                "samples": self.samples,
                "rejected": self.rejected,
                "overshoots": self.overshoots,
                "ewma_err": None if self._ewma_err is None else round(self._ewma_err, 1),
                "last_action": self.last_action,
                "recent": self.history[-5:],
            }

    # ── 持久化 ──────────────────────────────────────────────
    def _save_unlocked(self):
        try:
            data = {"factor": self.factor, "base_factor": self.base_factor,
                    "samples": self.samples, "rejected": self.rejected,
                    "overshoots": self.overshoots, "ewma_err": self._ewma_err,
                    "history": self.history[-20:]}
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

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            saved_base = float(data.get("base_factor", self.base_factor)) or self.base_factor
            saved_k = float(data.get("factor", saved_base))
            # 用户后来调过滑块时，把已学结果按比例迁移，而不是从头学
            self.factor = self._clip(self.base_factor * (saved_k / saved_base))
            self.samples = int(data.get("samples", 0))
            self.rejected = int(data.get("rejected", 0))
            self.overshoots = int(data.get("overshoots", 0))
            self._ewma_err = data.get("ewma_err")
            self.history = data.get("history", [])
            # 旧版本把「结算页」误当跳飞，会连乘 0.85 把系数越拉越低。
            # 样本很少却大幅偏离基准的记录一定是脏数据，直接回到基准。
            if self.samples < 3 and abs(self.factor / self.base_factor - 1.0) > 0.20:
                self.factor = self.base_factor
                self.overshoots = 0
                self.rejected = 0
                self.history = []
                self._ewma_err = None
                self.last_action = "reset_unreliable"
        except Exception:
            pass

    @classmethod
    def load(cls, path, base_factor=2.0, **kwargs):
        return cls(path=path, base_factor=base_factor, **kwargs)

    def reset(self):
        with self._lock:
            self.factor = self.base_factor
            self.samples = 0
            self.rejected = 0
            self.overshoots = 0
            self._ewma_err = None
            self.history = []
            self.last_action = "reset"
