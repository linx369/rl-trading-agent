"""Portfolio weighting schemes. All return ``{ticker: weight}`` summing to 1."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def equal_weights(tickers: list[str]) -> dict[str, float]:
    uniq = list(dict.fromkeys(tickers))
    return {t: 1.0 / len(uniq) for t in uniq} if uniq else {}


def momentum_weights(hist: pd.DataFrame) -> dict[str, float]:
    """Weight proportional to the price ratio over the window (floored at 0.01)."""
    if hist.empty or hist.shape[1] == 0:
        return {}
    if hist.shape[0] < 2:
        return equal_weights(list(hist.columns))
    mom = (hist.iloc[-1] / hist.iloc[0]).clip(lower=0.01)
    return {t: float(mom[t] / mom.sum()) for t in hist.columns}


def max_sharpe_weights(hist: pd.DataFrame, pos_limit: float = 0.35) -> dict[str, float]:
    """Max-Sharpe weights from monthly prices (PyPortfolioOpt), with per-name bounds.

    Lower bound per name is ``max(2%, 1 / (2n))``. Falls back to equal weights when the
    problem is too small or infeasible (e.g. no asset beats the 2% risk-free rate).
    """
    cols = list(hist.columns)
    if hist.empty or hist.shape[0] < 3 or hist.shape[1] < 2:
        return equal_weights(cols)
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = expected_returns.mean_historical_return(hist, frequency=12)
            cov = risk_models.sample_cov(hist, frequency=12)
            lb = max(0.02, round(1 / (2 * len(cols)), 3))
            ef = EfficientFrontier(mu, cov, weight_bounds=(lb, pos_limit), solver="SCS")
            ef.max_sharpe()
            w = ef.clean_weights()
        out = {k: float(v) for k, v in w.items() if v > 0}
        total = sum(out.values())
        return {k: v / total for k, v in out.items()} if total > 0 else equal_weights(cols)
    except Exception as exc:  # infeasible / solver failure -> equal weights
        log.debug("max-Sharpe fallback to equal weights: %s", exc)
        return equal_weights(cols)


def tilt(weights: dict[str, float], boosted: set[str], factor: float = 2.0) -> dict[str, float]:
    """Multiply the weights of ``boosted`` names by ``factor`` and renormalise."""
    raw = {t: w * (factor if t in boosted else 1.0) for t, w in weights.items()}
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()} if total > 0 else {}


def portfolio_return(weights: dict[str, float], next_returns: pd.Series) -> float | None:
    """Weighted next-period return over names with a valid return (renormalised)."""
    valid = {
        t: w
        for t, w in weights.items()
        if t in next_returns.index and not np.isnan(next_returns[t])
    }
    if not valid:
        return None
    ws = pd.Series(valid)
    ws = ws / ws.sum()
    return float((next_returns[ws.index] * ws).sum())
