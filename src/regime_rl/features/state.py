"""Assemble the agent's monthly observation (before the 8 strategy-alpha features).

Blocks, in order: 12 cross-sectional gauges, 4 regime probabilities (lagged one month),
4 macro spreads, one return per sector ETF, sector spread, best sector, lagged benchmark
return. Every column is then turned into a trailing 36-month z-score clipped to ±3.
With 7 US sector ETFs this gives 30 columns; the environment appends 8 more (= 38).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regime_rl.features.indicators import rolling_normalise
from regime_rl.features.panel import log_monthly_returns
from regime_rl.markets import Market


@dataclass
class StateMatrix:
    raw: pd.DataFrame  # before normalisation (useful for inspection)
    normalised: pd.DataFrame  # what the agent sees

    @property
    def columns(self) -> list[str]:
        return list(self.normalised.columns)

    def index_of(self, column: str) -> int:
        return self.columns.index(column)


def macro_features(etf_adj: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Last month's bond, credit, small-cap and growth signals (shifted one month)."""
    tickers = [t for t in market.macro_etfs if t in etf_adj.columns]
    lr = log_monthly_returns(etf_adj[tickers])
    bench = lr.get(market.benchmark)

    def col(t: str) -> pd.Series:
        return lr[t] if t in lr else pd.Series(0.0, index=lr.index)

    out = pd.DataFrame(
        {
            "tlt_ret": col(market.bond_etf),
            "hyg_ret": col(market.credit_etf),
            "iwm_vs_spy": col(market.smallcap) - bench,
            "qqq_vs_spy": col(market.growth_etf) - bench,
        }
    )
    return out.shift(1).fillna(0.0)


def sector_features(sector_monthly_ret: pd.DataFrame) -> pd.DataFrame:
    lagged = sector_monthly_ret.shift(1)
    out = lagged.fillna(0.0).copy()
    out.columns = [f"sec_{c.lower().split('.')[0]}" for c in out.columns]
    out["sector_spread"] = lagged.max(axis=1) - lagged.min(axis=1)
    out["best_sector_ret"] = lagged.max(axis=1)
    return out.fillna(0.0)


def build_state(
    cross_section: pd.DataFrame,
    regimes_lagged: pd.DataFrame,
    etf_adj_daily: pd.DataFrame,
    sector_monthly_ret: pd.DataFrame,
    bench_monthly_ret: pd.Series,
    market: Market,
) -> StateMatrix:
    state = cross_section.join(regimes_lagged, how="inner")
    state = state.join(macro_features(etf_adj_daily, market), how="left").fillna(0.0)
    state = state.join(sector_features(sector_monthly_ret), how="left").fillna(0.0)
    state = state.join(bench_monthly_ret.shift(1).rename("spy_ret_lag"), how="left").fillna(0.0)
    normalised = rolling_normalise(state).dropna()
    return StateMatrix(raw=state.loc[normalised.index], normalised=normalised)
