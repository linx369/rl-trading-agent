import numpy as np
import pandas as pd
import pytest

from regime_rl.portfolio import weights as w


@pytest.fixture
def hist() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-31", periods=13, freq="ME")
    drift = np.array([0.03, 0.02, 0.015, 0.01, 0.025])
    rets = drift + rng.normal(0, 0.03, (13, 5))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list("ABCDE"))


def test_equal_weights_deduplicates():
    assert w.equal_weights(["A", "B", "A"]) == {"A": 0.5, "B": 0.5}
    assert w.equal_weights([]) == {}


def test_momentum_weights_proportional_to_price_ratio(hist):
    out = w.momentum_weights(hist)
    ratio = hist.iloc[-1] / hist.iloc[0]
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["A"] / out["D"] == pytest.approx(ratio["A"] / ratio["D"])


def test_momentum_weights_floor():
    h = pd.DataFrame({"UP": [1.0, 2.0], "DEAD": [1.0, 0.0001]})
    out = w.momentum_weights(h)
    assert out["DEAD"] == pytest.approx(0.01 / 2.01)


@pytest.mark.parametrize("limit", [0.35, 0.5])
def test_max_sharpe_respects_bounds(hist, limit):
    out = w.max_sharpe_weights(hist, pos_limit=limit)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)
    assert max(out.values()) <= limit + 1e-3
    assert min(out.values()) >= 0.1 - 1e-3  # lower bound max(2%, 1/(2n)) = 10%


def test_max_sharpe_falls_back_to_equal_when_infeasible(hist):
    two = hist[["A", "B"]]
    assert w.max_sharpe_weights(two, pos_limit=0.35) == {"A": 0.5, "B": 0.5}


def test_max_sharpe_small_inputs():
    assert w.max_sharpe_weights(pd.DataFrame({"A": [1.0, 2.0]})) == {"A": 1.0}


def test_tilt():
    out = w.tilt({"A": 0.5, "B": 0.5}, {"A"}, 2.0)
    assert out == pytest.approx({"A": 2 / 3, "B": 1 / 3})


def test_portfolio_return_renormalises_over_valid_names():
    nxt = pd.Series({"A": 0.10, "B": np.nan, "C": -0.05})
    assert w.portfolio_return({"A": 0.5, "B": 0.25, "C": 0.25}, nxt) == pytest.approx(
        (0.5 * 0.10 + 0.25 * -0.05) / 0.75
    )
    assert w.portfolio_return({"B": 1.0}, nxt) is None
