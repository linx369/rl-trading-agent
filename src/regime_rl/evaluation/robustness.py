"""Statistical robustness: moving-block bootstrap and Monte Carlo projection.

Both use ``numpy.random.RandomState`` seeded like the original script, so ``tfg`` runs
reproduce its random draws exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_rl.evaluation.metrics import METRICS, compute_metrics


def block_indices(n_obs: int, block: int, rng: np.random.RandomState) -> np.ndarray:
    """Concatenate random contiguous blocks until ``n_obs`` indices are drawn."""
    if block > n_obs:
        raise ValueError(f"block ({block}) longer than the series ({n_obs})")
    idx: list[int] = []
    while len(idx) < n_obs:
        start = rng.randint(0, n_obs - block + 1)
        idx.extend(range(start, min(start + block, n_obs)))
    return np.asarray(idx[:n_obs])


def block_bootstrap(
    rets: pd.Series,
    bench: pd.Series,
    reps: int = 2000,
    block: int = 6,
    seed: int = 42,
    level: float = 0.90,
) -> pd.DataFrame:
    """Point estimate and ``level`` confidence interval for every metric.

    Months are resampled in blocks so volatility clustering and autocorrelation survive.
    """
    rng = np.random.RandomState(seed)
    r, b = np.asarray(rets, float), np.asarray(bench, float)
    draws: dict[str, list[float]] = {k: [] for k in METRICS}
    for _ in range(reps):
        i = block_indices(len(r), block, rng)
        m = compute_metrics(r[i], b[i])
        if m:
            for k in METRICS:
                if not np.isnan(m[k]):
                    draws[k].append(m[k])
    point = compute_metrics(r, b) or {}
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    rows = {
        k: {
            "point": point.get(k, np.nan),
            "ci_lower": np.percentile(v, lo) if v else np.nan,
            "median": np.percentile(v, 50) if v else np.nan,
            "ci_upper": np.percentile(v, hi) if v else np.nan,
            "p_positive": float(np.mean(np.asarray(v) > 0)) if v else np.nan,
        }
        for k, v in draws.items()
    }
    return pd.DataFrame(rows).T


def monte_carlo(
    rets: pd.Series,
    bench: pd.Series,
    paths: int = 1000,
    horizon: int = 24,
    seed: int = 42,
    compound_benchmark: bool = True,
) -> dict[str, object]:
    """IID-resampled wealth paths over ``horizon`` months (starting wealth = 1).

    ``compound_benchmark=False`` reproduces the original benchmark target
    (mean monthly return x horizon, not compounded).
    """
    rng = np.random.RandomState(seed)
    r = np.asarray(rets, float)
    wealth = np.ones((paths, horizon + 1))
    for p in range(paths):
        wealth[p, 1:] = np.cumprod(1 + rng.choice(r, horizon, replace=True))
    final = wealth[:, -1] - 1
    mb = float(np.mean(bench))
    bench_total = (1 + mb) ** horizon - 1 if compound_benchmark else mb * horizon
    peaks = np.maximum.accumulate(wealth, axis=1)
    dd = ((wealth - peaks) / peaks).min(axis=1)
    return {
        "p5": float(np.percentile(final, 5)),
        "median": float(np.percentile(final, 50)),
        "p95": float(np.percentile(final, 95)),
        "bench_total": float(bench_total),
        "p_profit": float((final > 0).mean()),
        "p_beat_bench": float((final > bench_total).mean()),
        "median_max_dd": float(np.median(dd)),
        "paths": wealth,
    }
