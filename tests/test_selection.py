import numpy as np
import pandas as pd
import pytest

from regime_rl.config import Config, UniverseConfig
from regime_rl.features.panel import MonthlyData
from regime_rl.portfolio import selection as sel
from regime_rl.portfolio.strategies import build_portfolios


def make_ctx(
    prices: pd.DataFrame, cfg: Config, bench_3m: float = 0.0, min_names: int = 1
) -> sel.SelectionContext:
    cfg = cfg.replace(universe=UniverseConfig(min_names_after_filter=min_names))
    date = prices.index[-1]
    tickers = [t for t in prices.columns if prices[t].notna().iloc[-1]]
    panel = pd.DataFrame(
        {"rsi": 50.0, "mom_6m": 0.1},
        index=pd.MultiIndex.from_product([[date], tickers], names=["date", "ticker"]),
    )
    sectors = pd.DataFrame(
        {"XLK": [0.05] * len(prices), "XLE": [0.01] * len(prices)}, index=prices.index
    )
    sectors.iloc[-1] = [0.0, 0.06]  # XLE best last month, XLK best over 3 months
    return sel.SelectionContext(
        cfg=cfg,
        monthly=MonthlyData(
            adj=prices, ret=prices.pct_change(fill_method=None), dollar_vol=prices * 0 + 1.0
        ),
        panel=panel,
        bench_ret_3m=pd.Series(bench_3m, index=prices.index),
        sector_ret_1m=sectors,
        sector_ret_3m=sectors.rolling(3).sum(),
        sector_map={"XLK": ("WIN",), "XLE": ("FLAT",)},
    )


def test_corrected_momentum_is_12_minus_1(monthly_prices):
    ctx = make_ctx(monthly_prices, Config(), bench_3m=-1.0)  # relative-strength filter off
    d = monthly_prices.index[-1]
    ranked = sel.ranked_momentum(ctx, list(monthly_prices.columns), d)
    # JUMP's +60% happened in the skipped month, so it must not lead.
    assert ranked[0] in {"WIN", "LATE"}
    assert ranked.index("JUMP") > ranked.index("MID2")
    assert ranked[-1] == "LOSE"


def test_corrected_momentum_includes_recent_listings(monthly_prices):
    ctx = make_ctx(monthly_prices, Config())
    ranked = sel.ranked_momentum(ctx, list(monthly_prices.columns), monthly_prices.index[-1])
    assert "LATE" in ranked  # 14 months of history is enough for 12-1 momentum


def test_tfg_momentum_drops_recent_listings(monthly_prices):
    """Original defect: any ticker with a NaN since the data start is excluded."""
    ctx = make_ctx(monthly_prices, Config(fidelity="tfg"))
    ranked = sel.ranked_momentum(ctx, list(monthly_prices.columns), monthly_prices.index[-1])
    assert "LATE" not in ranked and "WIN" in ranked


def test_crash_filter_removes_last_month_losers(monthly_prices):
    prices = monthly_prices.copy()
    prices.loc[prices.index[-1], "WIN"] *= 0.8  # -20% last month
    ctx = make_ctx(prices, Config())
    ranked = sel.ranked_momentum(ctx, list(prices.columns), prices.index[-1])
    assert "WIN" not in ranked


def test_relative_strength_filter(monthly_prices):
    ctx = make_ctx(monthly_prices, Config(), bench_3m=0.10)
    ranked = sel.ranked_momentum(ctx, list(monthly_prices.columns), monthly_prices.index[-1])
    p = monthly_prices.iloc[-1] / monthly_prices.iloc[-4] - 1
    assert set(ranked) == set(p[p > 0.10].index)


def test_filters_skipped_when_too_few_survive(monthly_prices):
    ctx = make_ctx(monthly_prices, Config(), bench_3m=10.0, min_names=5)  # nobody beats +1000%
    ranked = sel.ranked_momentum(ctx, list(monthly_prices.columns), monthly_prices.index[-1])
    assert len(ranked) == monthly_prices.shape[1]


def test_no_future_prices_used(monthly_prices):
    ctx = make_ctx(monthly_prices, Config())
    d = monthly_prices.index[-3]
    before = sel.ranked_momentum(ctx, list(monthly_prices.columns), d)
    shocked = monthly_prices.copy()
    shocked.iloc[-2:] = shocked.iloc[-2:][::-1].to_numpy() * 5
    after = sel.ranked_momentum(make_ctx(shocked, Config()), list(shocked.columns), d)
    assert before == after


def test_leading_sector_window(monthly_prices):
    d = monthly_prices.index[-1]
    assert sel.leading_sector(make_ctx(monthly_prices, Config(fidelity="tfg")), d) == "XLE"
    assert sel.leading_sector(make_ctx(monthly_prices, Config()), d) == "XLK"


def test_mean_reversion_picks_oversold_uptrend():
    d = pd.Timestamp("2021-12-31")
    names = [f"S{i}" for i in range(8)]
    panel = pd.DataFrame(
        {
            "rsi": [30, 40, 44, 20, 35, 60, 70, 30],
            "mom_6m": [0.1, 0.2, 0.3, 0.05, 0.4, 0.5, 0.6, -0.1],
        },
        index=pd.MultiIndex.from_product([[d], names], names=["date", "ticker"]),
    )
    ctx = sel.SelectionContext(Config(), None, panel, None, None, None, {})
    assert sel.mean_reversion(ctx, d, ranked=[], n=3) == ["S4", "S2", "S1"]


@pytest.mark.parametrize("fidelity", ["tfg", "corrected"])
def test_build_portfolios_shapes(monthly_prices, fidelity):
    ctx = make_ctx(monthly_prices, Config(fidelity=fidelity))
    ports = build_portfolios(ctx, monthly_prices.index[-1])
    assert set(ports) == set(range(8))
    assert ports[6] == {"TLT": 1.0}
    for a, w in ports.items():
        assert np.isclose(sum(w.values()), 1.0), a
        assert all(v >= 0 for v in w.values())
    assert len(ports[5]) <= 3


def test_corrected_sector_tilt_doubles_leading_sector(monthly_prices):
    ctx = make_ctx(monthly_prices, Config())
    ports = build_portfolios(ctx, monthly_prices.index[-1])
    top30, tilted = ports[2], ports[4]
    ratio_win = tilted["WIN"] / top30["WIN"]
    ratio_mid = tilted["MID"] / top30["MID"]
    assert ratio_win == pytest.approx(2 * ratio_mid)


def test_tfg_sector_tilt_is_identical_to_top30(monthly_prices):
    """Original defect: duplicates were removed before weighting, so the tilt vanished."""
    ports = build_portfolios(
        make_ctx(monthly_prices, Config(fidelity="tfg")), monthly_prices.index[-1]
    )
    assert ports[4] == pytest.approx(ports[2])
