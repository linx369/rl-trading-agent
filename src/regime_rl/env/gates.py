"""Regime gates: which actions may be taken given the HMM probabilities.

The gates are the system's main defence against momentum crashes (Daniel & Moskowitz,
2016): concentrated momentum is switched off when a bear or volatile regime is likely.
"""

from __future__ import annotations

from regime_rl import N_ACTIONS
from regime_rl.config import GateConfig

ALL_ACTIONS = frozenset(range(N_ACTIONS))
SAFE_ACTION = 2  # Top 30 momentum is never gated


def allowed_actions(
    p_bull: float, p_bear: float, p_volatile: float, gates: GateConfig
) -> frozenset[int]:
    allowed = set(ALL_ACTIONS)
    if p_bear > gates.bear_blocks_concentrated:
        allowed -= {0, 1, 5}
    if p_volatile > gates.volatile_blocks_top:
        allowed -= {0, 5}
    if p_bull < gates.bull_min_for_top3:
        allowed -= {5}
    if p_bull > gates.bull_blocks_tlt and p_bear < gates.bear_max_for_tlt_block:
        allowed -= {6}
    return frozenset(allowed) if allowed else frozenset({SAFE_ACTION})


def resolve_action(action: int, allowed: frozenset[int], tfg: bool) -> int:
    """Map a blocked action to an allowed one.

    tfg: the allowed action with the nearest *number* (ties -> lower number), as originally.
    corrected: the never-gated broad portfolio (Top 30), a deliberate safe default.
    """
    if action in allowed:
        return action
    if tfg:
        return min(sorted(allowed), key=lambda a: abs(a - action))
    return SAFE_ACTION if SAFE_ACTION in allowed else min(allowed)
