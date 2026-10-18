"""Simple strategies the RL agent has to beat to justify its complexity.

The ladder adds one ingredient at a time:

1. ``spy``           buy and hold the benchmark (no costs).
2. ``momentum_30``   always hold action 2 (top 30 momentum) – momentum, nothing else.
3. ``best_fixed``    the single action with the best information ratio over the train and
                     validation months, then held unchanged through the test period.
4. ``equal_mix``     an equal split across all eight actions every month (no selection).
5. ``hmm_switch``    top 30 momentum, but 100% bonds whenever P(bear) exceeds the gate
                     threshold – the regime model as a plain rule, without RL.

Every baseline uses the same pre-computed, cost-adjusted monthly returns as the agent, on
exactly the months the agent traded, so the comparison is like for like. Only data that
was available before the test period is used to choose ``best_fixed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_rl import ACTIONS
from regime_rl.config import Config
from regime_rl.env.trading_env import EpisodeData
from regime_rl.evaluation.metrics import compute_metrics
from regime_rl.evaluation.robustness import block_bootstrap

MOMENTUM_ACTION = 2
BOND_ACTION = 6

LABELS = {
    "spy": "Buy & hold benchmark",
    "momentum_30": "Top 30 momentum, always",
    "best_fixed": "Best single strategy (chosen before test)",
    "equal_mix": "Equal mix of all 8 strategies",
    "hmm_switch": "Momentum, bonds when P(bear) high",
    "agent": "RL agent",
}


def _traded(data: EpisodeData, cfg: Config) -> EpisodeData:
    """Months the environment actually trades (the original skipped the last one)."""
    n = len(data.dates) - 1 if cfg.tfg else len(data.dates)
    return data.slice(np.arange(len(data.dates)) < n)


def information_ratio(rets: np.ndarray, bench: np.ndarray) -> float:
    exc = np.asarray(rets) - np.asarray(bench)
    return float(exc.mean() / (exc.std(ddof=1) + 1e-6) * np.sqrt(12)) if len(exc) > 2 else 0.0


def choose_best_fixed(history: EpisodeData) -> int:
    """Action with the highest information ratio on pre-test data."""
    irs = [information_ratio(history.returns[:, a], history.bench) for a in range(len(ACTIONS))]
    return int(np.argmax(irs))


def baseline_returns(
    test: EpisodeData, history: EpisodeData, cfg: Config
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Monthly returns of every baseline over the traded test months.

    Returns the returns frame (one column per baseline plus ``bench``) and a short note per
    baseline describing what it held.
    """
    t = _traded(test, cfg)
    best = choose_best_fixed(history)
    p_bear = t.gate_probs[:, 1]
    threshold = cfg.gates.bear_blocks_concentrated
    switch = np.where(p_bear > threshold, BOND_ACTION, MOMENTUM_ACTION)
    frame = pd.DataFrame(
        {
            "spy": t.bench,
            "momentum_30": t.returns[:, MOMENTUM_ACTION],
            "best_fixed": t.returns[:, best],
            "equal_mix": t.returns.mean(axis=1),
            "hmm_switch": t.returns[np.arange(len(switch)), switch],
            "bench": t.bench,
        },
        index=pd.DatetimeIndex(t.dates, name="date"),
    )
    notes = {
        "spy": "",
        "momentum_30": ACTIONS[MOMENTUM_ACTION],
        "best_fixed": f"{ACTIONS[best]} (action {best})",
        "equal_mix": "1/8 in each action",
        "hmm_switch": f"bonds in {int((switch == BOND_ACTION).sum())} of {len(switch)} months",
    }
    return frame, notes


def baseline_table(
    test: EpisodeData,
    history: EpisodeData,
    agent_record: pd.DataFrame,
    cfg: Config,
    agent_detail: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics (with a 90% block-bootstrap interval for alpha) for every baseline and the agent.

    Returns ``(table, monthly_returns)``.
    """
    frame, notes = baseline_returns(test, history, cfg)
    agent = agent_record["ret"].reindex(frame.index)
    if agent.isna().any():
        raise ValueError("Agent record does not cover the same months as the baselines")
    frame["agent"] = agent.to_numpy()
    notes["agent"] = agent_detail

    rb = cfg.robustness
    rows = []
    for key in ("spy", "momentum_30", "best_fixed", "equal_mix", "hmm_switch", "agent"):
        m = compute_metrics(frame[key], frame["bench"]) or {}
        ci = block_bootstrap(
            frame[key],
            frame["bench"],
            reps=rb.bootstrap_reps,
            block=min(rb.bootstrap_block, len(frame)),
            seed=rb.random_state,
        ).loc["alpha"]
        rows.append(
            {
                "strategy": key,
                "label": LABELS[key],
                "detail": notes[key],
                **m,
                "alpha_ci_lower": float(ci["ci_lower"]) if key != "spy" else 0.0,
                "alpha_ci_upper": float(ci["ci_upper"]) if key != "spy" else 0.0,
            }
        )
    table = pd.DataFrame(rows).set_index("strategy")
    return table, frame


def format_table(table: pd.DataFrame) -> str:
    """Plain-text table for the terminal."""
    lines = [
        f"{'Strategy':<44}{'Alpha':>9}{'IR':>7}{'Sharpe':>8}{'Max DD':>9}   90% CI for alpha",
    ]
    for key, r in table.iterrows():
        ci = "" if key == "spy" else f"[{r.alpha_ci_lower:+.1%}, {r.alpha_ci_upper:+.1%}]"
        ir = "" if key == "spy" else f"{r.ir:.2f}"
        alpha = "" if key == "spy" else f"{r.alpha:+.2%}"
        lines.append(f"{r.label:<44}{alpha:>9}{ir:>7}{r.sharpe:>8.2f}{r.max_dd:>9.1%}   {ci}")
    return "\n".join(lines)
