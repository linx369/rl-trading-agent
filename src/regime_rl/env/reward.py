"""Reward shaping, kept as pure functions so every term can be unit-tested.

    reward = w_ir * IR + w_calmar * Calmar - w_crash * crash + bonus - repeat

* IR      – annualised information ratio of the last <= 6 monthly excess returns.
* Calmar  – compound return of the last <= 6 months / (|drawdown from peak| + 1%).
* crash   – ``crash_scale * (|r| - 5%)`` for a monthly loss worse than -5%.
* bonus   – regime-conditional nudges towards concentration (bull), bonds (bear) and
            mean-reversion (mixed). Signals are on a 0–1 scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from regime_rl.config import RewardConfig


@dataclass(frozen=True)
class Signals:
    """0–1 scaled market signals used by the bonuses."""

    bull: float
    bear: float
    volatile: float
    dispersion: float
    accel: float
    hi52: float


def information_ratio(excess: Sequence[float], clip: float = 3.0) -> float:
    e = np.asarray(excess, dtype=float)
    if len(e) >= 3:
        return float(np.clip(e.mean() / (e.std() + 1e-6) * np.sqrt(12), -clip, clip))
    return float(np.clip(e[-1] * 12, -clip, clip)) if len(e) else 0.0


def calmar_signal(returns: Sequence[float], drawdown: float, clip: float = 3.0) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) >= 3:
        cum = float(np.prod(1 + r)) - 1
        return float(np.clip(cum / (abs(drawdown) + 0.01), -clip, clip))
    return float(np.clip(r[-1] / 0.05, -clip, clip)) if len(r) else 0.0


def crash_penalty(ret: float, cfg: RewardConfig) -> float:
    if ret < cfg.crash_threshold:
        return float(cfg.crash_scale * abs(ret - cfg.crash_threshold))
    return 0.0


def regime_bonus(action: int, s: Signals) -> float:
    bonus = 0.0
    if action in (0, 1):
        if s.bull > 0.5 and s.dispersion > 0.5:
            bonus = 0.4 * s.bull * s.dispersion
        if s.accel > 0.6 and s.hi52 > 0.6:
            bonus += 0.2 * s.accel * s.hi52
    elif action == 5:
        if s.bull > 0.75 and s.dispersion > 0.75:
            bonus = 0.6 * s.bull * s.dispersion
        if s.accel > 0.7 and s.hi52 > 0.7:
            bonus += 0.3 * s.accel * s.hi52
    elif action == 6:
        bonus = 0.4 * s.bear + 0.2 * (1 - s.bull)
    elif action == 7:
        mixed = max(0.0, 1.0 - s.bull - s.bear)
        bonus = 0.3 * mixed + 0.1 * s.volatile
    return bonus


def step_reward(
    action: int,
    ret: float,
    excess_hist: Sequence[float],
    ret_hist: Sequence[float],
    drawdown: float,
    signals: Signals,
    repeated: bool,
    cfg: RewardConfig,
) -> tuple[float, float]:
    """Return ``(reward, ir)`` for one month."""
    ir = information_ratio(excess_hist, cfg.clip)
    calmar = calmar_signal(ret_hist, drawdown, cfg.clip)
    reward = (
        cfg.w_ir * ir
        + cfg.w_calmar * calmar
        - cfg.w_crash * crash_penalty(ret, cfg)
        + regime_bonus(action, signals)
        - (cfg.repeat_penalty if repeated else 0.0)
    )
    return float(reward), ir


def zscore_to_unit(z: float) -> float:
    """Map a ±3 z-score to [0, 1] (0.5 = average of the trailing 36 months)."""
    return float(np.clip((z + 3) / 6, 0, 1))
