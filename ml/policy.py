import json
import math
import os
import time

import numpy as np

FEATURES = 10
HIDDEN = 24
ACTIONS = 11
SCALES = np.array([0.80 + 0.04 * i for i in range(ACTIONS)])
MIN_PRESS_MS = 140.0
MAX_PRESS_MS = 1150.0
MS_PER_PX = 2.16
MS_REF_W = 655.0
FREEZE_POLICY = True


def legacy_press_ms(distance_px, head_diameter, coefficient=1.0):
    hd = max(float(head_diameter), 1.0)
    scale = 0.945 * 2 / hd
    actual = float(distance_px) * scale * (math.sqrt(6) / 2)
    t = (-945 + math.sqrt(945 ** 2 + 4 * 105 * 36 * actual)) / (2 * 105) * 1000
    return t * float(coefficient)


class Policy:
    def __init__(self, path=None, lr=0.10, seed=1, baseline=0.55, frozen=None):
        self.frozen = FREEZE_POLICY if frozen is None else bool(frozen)
        self.path = path
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0.0, 0.12, (HIDDEN, FEATURES))
        self.b1 = np.zeros(HIDDEN)
        self.w2 = self.rng.normal(0.0, 0.04, (ACTIONS, HIDDEN))
        self.b2 = np.exp(-((np.arange(ACTIONS) - 5.0) / 2.2) ** 2) * 1.6
        self.baseline = float(baseline)
        self.jumps = 0
        self.episodes = 0
        self.best = 0.0
        self.total_reward = 0.0
        self.rewards = []
        self.action_hist = [0] * ACTIONS
        self.last_reward = 0.0
        self.last_entropy = math.log(ACTIONS)
        self.last_scale = 1.0
        self.updated_at = None
        if path:
            self._load()

    def forward(self, x):
        h = np.tanh(self.w1 @ x + self.b1)
        z = self.w2 @ h + self.b2
        z = z - z.max()
        e = np.exp(z)
        p = e / e.sum()
        return h, p

    def act(self, x, greedy=False):
        h, p = self.forward(x)
        ent = float(-(p * np.log(p + 1e-9)).sum())
        if greedy or self.frozen:
            a = int(np.argmax(p))
        else:
            a = int(self.rng.choice(ACTIONS, p=p))
        self.last_entropy = ent
        self.last_scale = float(SCALES[a])
        return a, float(SCALES[a]), ent

    def update(self, x, action, reward, entropy_bonus=0.004):
        adv = float(reward) - self.baseline
        adv = max(-1.5, min(1.5, adv))
        if not self.frozen:
            h, p = self.forward(x)
            g = -p.copy()
            g[action] += 1.0
            g *= adv
            if entropy_bonus:
                ent = float(-(p * np.log(p + 1e-9)).sum())
                g += entropy_bonus * (-p * (np.log(p + 1e-9) + ent))
            dh = (self.w2.T @ g) * (1.0 - h * h)
            self.w2 += self.lr * np.outer(g, h)
            self.b2 += self.lr * g
            self.w1 += self.lr * np.outer(dh, x)
            self.b1 += self.lr * dh
            self.w1 *= 0.9997
            self.w2 *= 0.9997
            self.baseline += 0.10 * (float(reward) - self.baseline)
        self.jumps += 1
        self.total_reward += float(reward)
        self.last_reward = float(reward)
        self.best = max(self.best, float(reward))
        self.rewards.append(float(reward))
        self.rewards = self.rewards[-80:]
        self.action_hist[action] += 1
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        return adv

    def end_episode(self):
        self.episodes += 1
        self.save()

    def stats(self):
        mean = float(np.mean(self.rewards[-40:])) if self.rewards else 0.0
        return {
            "episodes": self.episodes,
            "jumps": self.jumps,
            "mean": round(mean, 3),
            "best": round(self.best, 3),
            "entropy": round(self.last_entropy, 3),
            "scale": round(self.last_scale, 3),
            "reward": round(self.last_reward, 3),
            "baseline": round(self.baseline, 3),
            "model": self.updated_at or "--",
            "actions": list(self.action_hist),
        }

    def save(self):
        if not self.path:
            return
        try:
            data = {
                "w1": self.w1.tolist(),
                "b1": self.b1.tolist(),
                "w2": self.w2.tolist(),
                "b2": self.b2.tolist(),
                "baseline": self.baseline,
                "jumps": self.jumps,
                "episodes": self.episodes,
                "best": self.best,
                "total_reward": self.total_reward,
                "rewards": self.rewards[-80:],
                "action_hist": self.action_hist,
                "lr": self.lr,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
            self.updated_at = data["updated_at"]
        except Exception:
            pass

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            w1 = np.array(data["w1"], dtype=np.float64)
            b1 = np.array(data["b1"], dtype=np.float64)
            w2 = np.array(data["w2"], dtype=np.float64)
            b2 = np.array(data["b2"], dtype=np.float64)
            if w1.shape != (HIDDEN, FEATURES) or w2.shape != (ACTIONS, HIDDEN):
                return
            self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
            self.baseline = float(data.get("baseline", self.baseline))
            self.jumps = int(data.get("jumps", 0))
            self.episodes = int(data.get("episodes", 0))
            self.best = float(data.get("best", 0.0))
            self.total_reward = float(data.get("total_reward", 0.0))
            self.rewards = [float(v) for v in data.get("rewards", [])][-80:]
            hist = data.get("action_hist")
            if isinstance(hist, list) and len(hist) == ACTIONS:
                self.action_hist = [int(v) for v in hist]
            self.lr = float(data.get("lr", self.lr))
            self.updated_at = data.get("updated_at")
        except Exception:
            pass

    @classmethod
    def load(cls, path, **kwargs):
        return cls(path=path, **kwargs)
