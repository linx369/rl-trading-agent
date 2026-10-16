import numpy as np
import pandas as pd
import pytest

from regime_rl.config import Config, UniverseConfig
from regime_rl.features.panel import cross_section, liquidity_rank, market_beta


def _dollar_volume() -> pd.DataFrame:
    idx = pd.date_range("2015-01-31", periods=30, freq="ME")
    # BIG is always 10x more liquid than SMALL; both grow over time.
    growth = np.linspace(1, 2, 30)
    return pd.DataFrame({"BIG": 1000 * growth, "SMALL": 100 * growth}, index=idx)


def test_corrected_liquidity_ranks_across_stocks():
    cfg = Config(universe=UniverseConfig(liquidity_window_months=12))
    r = liquidity_rank(_dollar_volume(), cfg).dropna()
    assert (r["BIG"] == 1).all() and (r["SMALL"] == 2).all()


def test_tfg_liquidity_ranks_each_stock_over_time():
    """The original bug: ranks are within each ticker, so both stocks get rank 1 somewhere."""
    cfg = Config(fidelity="tfg", universe=UniverseConfig(liquidity_window_months=12))
    r = liquidity_rank(_dollar_volume(), cfg)
    assert r["SMALL"].min() == 1 and r["BIG"].min() == 1
    assert r["SMALL"].iloc[-1] == 1  # most recent (largest) month of SMALL ranks first


def test_corrected_beta_is_regression_slope():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    mkt = pd.Series(rng.normal(0.01, 0.04, 60), index=idx)
    stocks = pd.DataFrame({"LEV": 2 * mkt, "NEG": -0.5 * mkt}, index=idx)
    beta = market_beta(stocks, mkt, Config()).dropna()
    assert np.allclose(beta["LEV"], 2.0) and np.allclose(beta["NEG"], -0.5)


def test_tfg_beta_is_correlation_and_lagged():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    mkt = pd.Series(rng.normal(0.01, 0.04, 60), index=idx)
    beta = market_beta(pd.DataFrame({"LEV": 2 * mkt}), mkt, Config(fidelity="tfg"))
    assert np.isnan(beta["LEV"].iloc[5])  # needs 6 obs, then shifted one month
    assert np.allclose(beta["LEV"].dropna(), 1.0)  # correlation, not slope


def test_cross_section_features():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-31"]), ["A", "B", "C", "D"]], names=["date", "ticker"]
    )
    panel = pd.DataFrame(
        {
            "rsi": [70, 50, 65, 30],
            "atr": [0, 1, 2, 3],
            "macd": [1, -1, 1, 1],
            "mom_skip1": [0.1, 0.2, 0.3, 0.4],
            "mom_3m": [0.1] * 4,
            "mom_6m": [0.04] * 4,
            "mom_1m": [0] * 4,
            "hi52_prox": [0.95, 0.5, 0.91, 0.2],
            "beta": [1] * 4,
        },
        index=idx,
    )
    cs = cross_section(panel).iloc[0]
    assert list(cross_section(panel).columns)[7] == "mom_accel"
    assert cs["rsi_top_pct"] == 0.5
    assert cs["macd_pos_pct"] == 0.75
    assert cs["hi52_pct"] == 0.5
    assert cs["mom_accel"] == pytest.approx(0.06)
    assert cs["mom_dispersion"] == pytest.approx(
        np.quantile([0.1, 0.2, 0.3, 0.4], 0.9) - np.quantile([0.1, 0.2, 0.3, 0.4], 0.1)
    )
