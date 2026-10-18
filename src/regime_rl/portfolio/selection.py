"""Stock selection rules behind the eight actions.

Each function returns an ordered list of tickers for a given month-end ``date`` using only
data up to that date. Where the original script had defects, the ``tfg`` branch reproduces
them and the ``corrected`` branch implements the documented intent:

* ``ranked_momentum`` (tfg) applied *daily* window lengths (252/22/63) to *monthly* prices,
  so "momentum" was the return from the start of the data to 21 months ago, the crash filter
  looked at 21 months and the relative-strength filter at 62 months. It also dropped every
  ticker with any missing price since the data start (i.e. anything listed after ~2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_rl.config import Config
from regime_rl.features.panel import MonthlyData


@dataclass
class SelectionContext:
    cfg: Config
    monthly: MonthlyData
    panel: pd.DataFrame  # (date, ticker) features of the eligible universe
    bench_ret_3m: pd.Series  # benchmark 3-month return at each month end
    sector_ret_1m: pd.DataFrame  # sector ETF monthly returns
    sector_ret_3m: pd.DataFrame  # sector ETF 3-month returns
    sector_map: dict[str, tuple[str, ...]]

    def universe(self, date: pd.Timestamp) -> list[str]:
        try:
            return self.panel.xs(date, level="date").index.tolist()
        except KeyError:
            return []

    def history(self, date: pd.Timestamp, tickers: list[str], rows: int) -> pd.DataFrame:
        """Last ``rows`` month-end prices up to ``date``.

        tfg: drops tickers with *any* NaN since the data start (original behaviour).
        corrected: drops tickers with a NaN inside the requested window only.
        """
        valid = [t for t in dict.fromkeys(tickers) if t in self.monthly.adj.columns]
        hist = self.monthly.adj.loc[:date, valid]
        if self.cfg.tfg:
            return hist.dropna(axis=1).iloc[-rows:]
        return hist.iloc[-rows:].dropna(axis=1)


def _apply_filters(
    ctx: SelectionContext,
    date: pd.Timestamp,
    mom: pd.Series,
    last_month: pd.Series,
    ret_3m: pd.Series | None,
) -> pd.Series:
    """Crash filter then relative-strength filter, each only if >= N names survive."""
    u = ctx.cfg.universe
    f = mom[last_month.reindex(mom.index).fillna(0) > u.crash_threshold]
    if len(f) >= u.min_names_after_filter:
        mom = f
    if ret_3m is not None:
        sv = ctx.bench_ret_3m.loc[:date]
        if not sv.empty:
            s3 = float(sv.iloc[-1]) if not np.isnan(sv.iloc[-1]) else 0.0
            rs = mom[ret_3m.reindex(mom.index).fillna(0) > s3]
            if len(rs) >= u.min_names_after_filter:
                mom = rs
    return mom


def ranked_momentum(ctx: SelectionContext, tickers: list[str], date: pd.Timestamp) -> list[str]:
    """Universe sorted by skip-1 momentum (strongest first) after crash and RS filters."""
    if ctx.cfg.tfg:
        hist = ctx.history(date, tickers, 252)
        valid = [t for t in tickers if t in ctx.monthly.adj.columns]
        if not valid:
            return []
        if hist.shape[0] < 22:
            return valid
        mom = (hist.iloc[-22] / hist.iloc[0] - 1).dropna()
        if mom.empty:
            return valid
        last_month = hist.iloc[-1] / hist.iloc[-22] - 1
        ret_3m = hist.iloc[-1] / hist.iloc[-63] - 1 if hist.shape[0] >= 63 else None
    else:
        hist = ctx.history(date, tickers, 13)  # months t-12 .. t
        if hist.shape[0] < 13 or hist.empty:
            return []
        mom = (hist.iloc[-2] / hist.iloc[0] - 1).dropna()  # P(t-1)/P(t-12) - 1
        last_month = hist.iloc[-1] / hist.iloc[-2] - 1
        ret_3m = hist.iloc[-1] / hist.iloc[-4] - 1
    mom = _apply_filters(ctx, date, mom, last_month, ret_3m)
    return mom.sort_values(ascending=False, kind="stable").index.tolist()


def mega_cap_momentum(
    ctx: SelectionContext, tickers: list[str], date: pd.Timestamp, n: int = 10
) -> list[str]:
    """Top ``n`` by 0.6 x momentum + 0.4 x dollar volume (both min-max scaled)."""
    valid = [
        t for t in tickers if t in ctx.monthly.adj.columns and t in ctx.monthly.dollar_vol.columns
    ]
    if not valid:
        return []
    if ctx.cfg.tfg:
        hist = ctx.history(date, valid, 252)
        if hist.shape[0] < 22:
            return valid[:n]
        mom = (hist.iloc[-22] / hist.iloc[0] - 1).dropna()
    else:
        hist = ctx.history(date, valid, 13)
        if hist.shape[0] < 13:
            return valid[:n]
        mom = (hist.iloc[-2] / hist.iloc[0] - 1).dropna()
    if mom.empty:
        return valid[:n]
    dv = ctx.monthly.dollar_vol.loc[:date, valid].iloc[-3:].mean().reindex(mom.index).fillna(0)
    m_norm = (mom - mom.min()) / (mom.max() - mom.min() + 1e-9)
    d_norm = (dv - dv.min()) / (dv.max() - dv.min() + 1e-9)
    return (0.6 * m_norm + 0.4 * d_norm).nlargest(n).index.tolist()


def leading_sector(ctx: SelectionContext, date: pd.Timestamp) -> str | None:
    """Best sector ETF: last month's return (tfg) or 3-month return (corrected)."""
    src = ctx.sector_ret_1m if ctx.cfg.tfg else ctx.sector_ret_3m
    hist = src.loc[:date]
    if hist.empty:
        return None
    last = hist.iloc[-1].dropna()
    return None if last.empty else str(last.idxmax())


def sector_members(ctx: SelectionContext, date: pd.Timestamp) -> set[str]:
    best = leading_sector(ctx, date)
    return set(ctx.sector_map.get(best, ())) if best else set()


def mean_reversion(
    ctx: SelectionContext, date: pd.Timestamp, ranked: list[str], n: int = 15
) -> list[str]:
    """Oversold (RSI < 45) stocks still in an uptrend (6-month return > 0)."""
    try:
        s = ctx.panel.xs(date, level="date")
    except KeyError:
        s = pd.DataFrame()
    if not s.empty:
        cand = s[(s["rsi"] < 45) & (s["mom_6m"] > 0)]
        if len(cand) >= 5:
            return cand["mom_6m"].nlargest(n).index.tolist()
    return ranked[15:30] if len(ranked) >= 30 else ranked
