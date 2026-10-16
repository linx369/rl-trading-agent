"""Performance metrics from monthly return series.

Conventions (kept from the thesis): Sharpe has no risk-free rate; "alpha" is the
difference in annualised compound returns versus the benchmark (not a CAPM intercept).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("alpha", "ann_ret", "ann_vol", "sharpe", "max_dd", "calmar", "ir")


def annualised_return(r: pd.Series) -> float:
    n = len(r)
    return float((1 + r).prod() ** (12 / n) - 1) if n else float("nan")


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float(((wealth - wealth.cummax()) / wealth.cummax()).min()) if len(r) else 0.0


def compute_metrics(
    rets: pd.Series | np.ndarray,
    bench: pd.Series | np.ndarray,
) -> dict[str, float] | None:
    r, b = pd.Series(np.asarray(rets, float)), pd.Series(np.asarray(bench, float))
    if len(r) < 3:
        return None
    ann = annualised_return(r)
    vol = float(r.std() * np.sqrt(12))
    dd = max_drawdown(r)
    exc = r - b
    return {
        "alpha": ann - annualised_return(b),
        "ann_ret": ann,
        "ann_vol": vol,
        "sharpe": ann / vol if vol > 0 else float("nan"),
        "max_dd": dd,
        "calmar": ann / abs(dd) if dd < 0 else float("nan"),
        "ir": float(exc.mean() / (exc.std() + 1e-6) * np.sqrt(12)),
    }


def benchmark_metrics(bench: pd.Series) -> dict[str, float]:
    m = compute_metrics(bench, bench) or {}
    return {k: v for k, v in m.items() if k not in ("alpha", "ir")}


def rolling_ir(excess: pd.Series, window: int = 12) -> pd.Series:
    mu = excess.rolling(window).mean()
    sd = excess.rolling(window).std()
    return mu / (sd + 1e-6) * np.sqrt(12)
