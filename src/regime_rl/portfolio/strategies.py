"""The eight actions and the pre-computed next-month return of each.

Pre-computing ``returns[date, action]`` once means the RL environment is a pure table
lookup: training is fast and the agent cannot see a price after its decision date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from regime_rl import N_ACTIONS
from regime_rl.portfolio import selection as sel
from regime_rl.portfolio import weights as wts

log = logging.getLogger(__name__)

MAX_SHARPE_ACTIONS = {0, 1, 5}
DEFENSIVE_ACTION = 6


@dataclass
class StrategyBook:
    returns: pd.DataFrame  # index = decision date, columns 0..7, next-month net returns
    holdings: pd.DataFrame  # date, action, ticker, weight
    bench_fwd: pd.Series  # benchmark next-month return per decision date


def build_portfolios(ctx: sel.SelectionContext, date: pd.Timestamp) -> dict[int, dict[str, float]]:
    """Weights for each of the eight actions at ``date`` (empty dict = no position)."""
    universe = ctx.universe(date)
    if not universe:
        return {}
    ranked = sel.ranked_momentum(ctx, universe, date)
    if not ranked:
        return {}

    def top(n: int) -> list[str]:
        return ranked[:n] if len(ranked) >= n else ranked

    picks: dict[int, list[str]] = {
        0: top(5),
        1: top(15),
        2: top(30),
        3: sel.mega_cap_momentum(ctx, universe, date, n=10),
        5: ranked[:3] if len(ranked) >= 3 else ranked[:5],
        7: sel.mean_reversion(ctx, date, ranked),
    }
    if ctx.cfg.tfg:
        # Original: re-rank the ranked list, duplicate the leading sector's names, then the
        # momentum weighting de-duplicates them -> the tilt disappears.
        picks[4] = sel.ranked_momentum(ctx, ranked, date)[:30]
    else:
        picks[4] = top(30)

    out: dict[int, dict[str, float]] = {}
    for action, names in picks.items():
        if action in MAX_SHARPE_ACTIONS:
            out[action] = wts.max_sharpe_weights(ctx.history(date, names, 13), pos_limit=0.35)
        else:
            out[action] = wts.momentum_weights(ctx.history(date, names, 13))
    if not ctx.cfg.tfg:
        out[4] = wts.tilt(out[4], sel.sector_members(ctx, date), factor=2.0)
    out[DEFENSIVE_ACTION] = {ctx.cfg.market.bond_etf: 1.0}
    return out


def precompute_returns(
    ctx: sel.SelectionContext,
    dates: list[pd.Timestamp],
    etf_monthly_ret: pd.DataFrame,
    bench_monthly_ret: pd.Series,
) -> StrategyBook:
    """Next-month net return of every action at every decision date."""
    cost = ctx.cfg.transaction_cost
    bond = ctx.cfg.market.bond_etf
    stock_ret = ctx.monthly.ret
    rows, holdings, bench = {}, [], {}
    for i, d in enumerate(dates):
        nxt = stock_ret.index[stock_ret.index > d]
        if len(nxt) == 0:
            continue
        nd = nxt[0]
        ports = build_portfolios(ctx, d)
        if not ports:
            continue
        rets = {}
        for a in range(N_ACTIONS):
            w = ports.get(a, {})
            if a == DEFENSIVE_ACTION:
                bnext = etf_monthly_ret[bond] if bond in etf_monthly_ret else pd.Series(dtype=float)
                bnext = bnext[bnext.index > d]
                r = float(bnext.iloc[0]) if len(bnext) else None
            else:
                r = wts.portfolio_return(w, stock_ret.loc[nd])
            rets[a] = (r - cost) if r is not None else 0.0
            holdings.extend(
                {"date": d, "action": a, "ticker": t, "weight": wt} for t, wt in w.items()
            )
        rows[d] = rets
        bn = bench_monthly_ret[bench_monthly_ret.index > d]
        bench[d] = float(bn.iloc[0]) if len(bn) else 0.0
        if (i + 1) % 24 == 0:
            log.info("  pre-computed %d/%d months", i + 1, len(dates))
    returns = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    returns.index.name = "date"
    return StrategyBook(
        returns=returns, holdings=pd.DataFrame(holdings), bench_fwd=pd.Series(bench).sort_index()
    )
