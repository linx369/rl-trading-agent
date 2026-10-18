"""Daily prices -> month-end stock panel (one row per month x eligible stock).

Output columns of :func:`build_panel` (index = (date, ticker), date = month end)::

    rsi, atr, macd, mom_skip1, mom_3m, mom_6m, mom_1m, hi52_prox, beta
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_rl.config import Config
from regime_rl.features import indicators as ind

MONTH_END = "ME"
PANEL_FEATURES = [
    "rsi",
    "atr",
    "macd",
    "mom_skip1",
    "mom_3m",
    "mom_6m",
    "mom_1m",
    "hi52_prox",
    "beta",
]


@dataclass
class PriceFrames:
    """Wide (date x ticker) daily frames for one asset group."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj_close: pd.DataFrame
    volume: pd.DataFrame

    @classmethod
    def from_long(cls, long: pd.DataFrame) -> PriceFrames:
        def piv(col: str) -> pd.DataFrame:
            w = long.pivot(index="date", columns="ticker", values=col).sort_index()
            w.columns = w.columns.astype(str)
            w.columns.name = None
            return w.astype(float)

        return cls(*(piv(c) for c in ("open", "high", "low", "close", "adj_close", "volume")))

    @property
    def tickers(self) -> list[str]:
        return list(self.adj_close.columns)


@dataclass
class MonthlyData:
    """Month-end series shared by the panel, the strategies and the environment."""

    adj: pd.DataFrame  # month-end adjusted close (all downloaded tickers)
    ret: pd.DataFrame  # monthly simple returns
    dollar_vol: pd.DataFrame  # mean daily dollar volume within the month ($m)


def monthly_data(px: PriceFrames) -> MonthlyData:
    adj = px.adj_close.resample(MONTH_END).last()
    dv = (px.adj_close * px.volume / 1e6).resample(MONTH_END).mean()
    return MonthlyData(adj=adj, ret=adj.pct_change(fill_method=None), dollar_vol=dv)


def daily_indicators(px: PriceFrames, cfg: Config) -> dict[str, pd.DataFrame]:
    """Compute every per-stock indicator on the daily grid."""
    a = px.adj_close
    # Tickers need a minimum number of complete OHLCV days (original: 50).
    complete = (
        px.open.notna()
        & px.high.notna()
        & px.low.notna()
        & px.close.notna()
        & a.notna()
        & px.volume.notna()
    ).sum()
    keep = complete[complete >= cfg.universe.min_history_days].index
    a = a[keep]
    h, lo, c = px.high[keep], px.low[keep], px.close[keep]

    zscore = ind.zscore_full if cfg.tfg else ind.zscore_rolling
    return {
        "rsi": ind.rsi(a, 20),
        "atr": zscore(ind.atr(h, lo, c, 14)),
        "macd": zscore(ind.macd(a, 12, 26)),
        "mom_skip1": ind.skip1_momentum(a),
        "mom_3m": a.pct_change(ind.TRADING_DAYS_QUARTER, fill_method=None),
        "mom_6m": a.pct_change(ind.TRADING_DAYS_HALF, fill_method=None),
        "mom_1m": a.pct_change(ind.TRADING_DAYS_MONTH, fill_method=None),
        "hi52_prox": ind.high52_proximity(a),
    }


def market_beta(
    stock_ret: pd.DataFrame,
    bench_ret: pd.Series,
    cfg: Config,
    window: int = 12,
    min_periods: int = 6,
) -> pd.DataFrame:
    """Rolling 12-month exposure to the benchmark, lagged one month.

    ``tfg``: rolling *correlation* (what the original code computed, labelled "beta").
    ``corrected``: rolling beta = cov(r_i, r_m) / var(r_m).
    """
    bench = bench_ret.reindex(stock_ret.index)
    if cfg.tfg:
        out = stock_ret.rolling(window, min_periods=min_periods).corr(bench)
    else:
        cov = stock_ret.rolling(window, min_periods=min_periods).cov(bench)
        var = bench.rolling(window, min_periods=min_periods).var()
        out = cov.div(var, axis=0)
    return out.shift(1)


def liquidity_rank(
    dollar_vol_m: pd.DataFrame, cfg: Config, members: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Rank of the trailing 60-month average dollar volume (1 = most liquid).

    ``tfg``: ranked *within each ticker across time* (the original groupby bug), so the
    filter drops each stock's least-liquid months instead of the least-liquid stocks.
    ``corrected``: ranked *across tickers on each date*.
    ``members`` (point-in-time mode): only stocks in the index on that date are ranked; the
    average itself still uses the stock's full past trading history.
    """
    w = cfg.universe.liquidity_window_months
    rv = dollar_vol_m.rolling(w, min_periods=12).mean()
    if members is not None:
        rv = rv.where(members.reindex(index=rv.index, columns=rv.columns, fill_value=False))
    axis = 0 if cfg.tfg else 1  # axis=0 ranks down each column (ticker over time)
    return rv.rank(axis=axis, ascending=False)


def build_panel(
    px: PriceFrames,
    monthly: MonthlyData,
    bench_monthly_ret: pd.Series,
    cfg: Config,
    members: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Month-end feature panel restricted to the liquid, seasoned universe.

    ``members`` is an optional (month-end x ticker) boolean frame of index membership; when
    given, a stock only enters the panel in months when it was actually in the index.
    """
    daily = daily_indicators(px, cfg)
    feats = {k: v.resample(MONTH_END).last() for k, v in daily.items()}
    feats["beta"] = market_beta(monthly.ret, bench_monthly_ret, cfg)
    tickers = sorted(daily["rsi"].columns)

    stacked = {k: v.reindex(columns=tickers).stack(future_stack=True) for k, v in feats.items()}
    panel = pd.DataFrame(stacked)
    panel.index.names = ["date", "ticker"]

    # Liquidity filter. The original keeps rank < N (strictly), which we preserve in tfg mode.
    rank = liquidity_rank(monthly.dollar_vol.reindex(columns=tickers), cfg, members)
    rank = rank.stack(future_stack=True).reindex(panel.index)
    n = cfg.universe.top_n_liquidity
    keep = rank < n if cfg.tfg else rank <= n
    if members is not None:
        is_member = members.reindex(columns=tickers, fill_value=False).stack(future_stack=True)
        keep = keep & is_member.reindex(panel.index, fill_value=False)

    # Seasoning: the original required 12 prior *rows* per ticker in the filtered panel
    # (a side-effect of computing 12-month returns then dropna). Corrected mode requires
    # 12 months of actual price history instead.
    beta_filled = panel["beta"].fillna(0.0)
    panel = panel.drop(columns="beta").assign(beta=beta_filled)
    base_cols = [c for c in PANEL_FEATURES if c != "beta"]
    panel = panel[keep.fillna(False)].dropna(subset=base_cols)
    if cfg.tfg:
        panel = panel[panel.groupby(level="ticker").cumcount() >= 12]
    else:
        hist = monthly.adj.notna().cumsum().stack(future_stack=True).reindex(panel.index)
        panel = panel[hist.fillna(0) > 12]
    return panel.sort_index()[PANEL_FEATURES]


def cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """Twelve universe-wide summary features per month (the agent's market gauges)."""
    g = panel.groupby(level="date")
    cs = pd.DataFrame(
        {
            "rsi_mean": g["rsi"].mean(),
            "rsi_top_pct": g["rsi"].apply(lambda x: (x > 60).mean()),
            "macd_pos_pct": g["macd"].apply(lambda x: (x > 0).mean()),
            "mom_12m_mean": g["mom_skip1"].mean(),
            "mom_dispersion": g["mom_skip1"].quantile(0.9) - g["mom_skip1"].quantile(0.1),
            "mom_3m_mean": g["mom_3m"].mean(),
            "mom_6m_mean": g["mom_6m"].mean(),
            "atr_mean": g["atr"].mean(),
            "beta_mean": g["beta"].mean(),
            "hi52_pct": g["hi52_prox"].apply(lambda x: (x > 0.90).mean()),
            "hi52_mean": g["hi52_prox"].mean(),
        }
    )
    cs.insert(7, "mom_accel", cs["mom_3m_mean"] - cs["mom_6m_mean"])
    return cs.fillna(0.0)


def log_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Sum of daily log returns within each month (rows with any NaN dropped, as original)."""
    return np.log(prices).diff().dropna().resample(MONTH_END).sum()
