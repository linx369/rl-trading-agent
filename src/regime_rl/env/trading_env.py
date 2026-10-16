"""Gymnasium environment: one step = one month, one action = one of eight portfolios."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from regime_rl import ACTIONS, N_ACTIONS
from regime_rl.config import Config
from regime_rl.env.gates import allowed_actions, resolve_action
from regime_rl.env.reward import Signals, step_reward, zscore_to_unit


@dataclass
class EpisodeData:
    """Everything the environment needs for one contiguous block of months."""

    dates: list[pd.Timestamp]
    states: np.ndarray  # (n, n_state) normalised observations
    returns: np.ndarray  # (n, 8) next-month net return per action
    bench: np.ndarray  # (n,) benchmark next-month return
    gate_probs: np.ndarray  # (n, 3) p_bull, p_bear, p_volatile used by the gates
    signal_probs: np.ndarray  # (n, 3) p_bull, p_bear, p_volatile used by the bonuses
    idx_dispersion: int
    idx_accel: int
    idx_hi52: int

    def __post_init__(self) -> None:
        n = len(self.dates)
        for name in ("states", "returns", "bench", "gate_probs", "signal_probs"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")

    def slice(self, mask: np.ndarray) -> EpisodeData:
        return EpisodeData(
            dates=[d for d, m in zip(self.dates, mask, strict=True) if m],
            states=self.states[mask],
            returns=self.returns[mask],
            bench=self.bench[mask],
            gate_probs=self.gate_probs[mask],
            signal_probs=self.signal_probs[mask],
            idx_dispersion=self.idx_dispersion,
            idx_accel=self.idx_accel,
            idx_hi52=self.idx_hi52,
        )


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}
    ALPHA_WINDOW = 3

    def __init__(self, data: EpisodeData, cfg: Config) -> None:
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.n = len(data.dates)
        # The original ended episodes one month early (the last month was never traded).
        self.n_steps = self.n - 1 if cfg.tfg else self.n
        if self.n_steps < 1:
            raise ValueError("Episode needs at least 2 months")
        n_obs = data.states.shape[1] + N_ACTIONS
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        base = self.data.states[min(self.t, self.n - 1)]
        alpha = np.array(
            [np.clip(np.mean(h) * 20, -3, 3) if h else 0.0 for h in self.strat_excess],
            dtype=np.float32,
        )
        return np.concatenate([base, alpha]).astype(np.float32)

    def _signals(self, t: int) -> Signals:
        st = self.data.states[t]
        bull, bear, vol = self.data.signal_probs[t]
        return Signals(
            bull=float(bull),
            bear=float(bear),
            volatile=float(vol),
            dispersion=zscore_to_unit(st[self.data.idx_dispersion]),
            accel=zscore_to_unit(st[self.data.idx_accel]),
            hi52=zscore_to_unit(st[self.data.idx_hi52]),
        )

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.port_val = self.peak = 1.0
        w = self.cfg.reward.window
        self.excess_hist: deque[float] = deque(maxlen=w)
        self.ret_hist: deque[float] = deque(maxlen=w)
        self.prev_actions: deque[int] = deque(maxlen=4)
        self.strat_excess = [deque(maxlen=self.ALPHA_WINDOW) for _ in range(N_ACTIONS)]
        return self._obs(), {}

    def step(self, action):
        t = self.t
        requested = int(action)
        p_bull, p_bear, p_vol = self.data.gate_probs[t]
        allowed = allowed_actions(p_bull, p_bear, p_vol, self.cfg.gates)
        action = resolve_action(requested, allowed, self.cfg.tfg)

        ret = float(self.data.returns[t, action])
        bench = float(self.data.bench[t])
        for s in range(N_ACTIONS):
            self.strat_excess[s].append(float(self.data.returns[t, s]) - bench)
        excess = ret - bench
        self.excess_hist.append(excess)
        self.ret_hist.append(ret)

        self.port_val *= 1 + ret
        self.peak = max(self.peak, self.port_val)
        drawdown = (self.port_val - self.peak) / self.peak

        # Penalise the 4th consecutive identical choice. The original never recorded
        # previous actions, so in tfg mode the penalty never fires.
        if not self.cfg.tfg:
            self.prev_actions.append(action)
        repeated = len(self.prev_actions) == 4 and len(set(self.prev_actions)) == 1

        signals = self._signals(t)
        reward, ir = step_reward(
            action,
            ret,
            list(self.excess_hist),
            list(self.ret_hist),
            drawdown,
            signals,
            repeated,
            self.cfg.reward,
        )

        self.t += 1
        terminated = self.t >= self.n_steps
        info = {
            "date": self.data.dates[t],
            "requested": requested,
            "action": action,
            "strategy": ACTIONS[action],
            "ret": ret,
            "bench_ret": bench,
            "excess": excess,
            "ir": ir,
            "reward": reward,
            "p_bull": signals.bull,
            "p_bear": signals.bear,
            "accel": signals.accel,
            "hi52": signals.hi52,
            "port_val": self.port_val,
            "drawdown": drawdown,
            "allowed": sorted(allowed),
        }
        return self._obs(), reward, terminated, False, info
