import numpy as np
import pandas as pd
import pytest

from regime_rl.features import indicators as ind


@pytest.fixture
def prices() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)


def test_rsi_is_bounded(prices):
    r = ind.rsi(prices, 20).dropna()
    assert len(r) > 0
    assert r.between(0, 100).all()


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 60.0))
    down = up[::-1].reset_index(drop=True)
    assert ind.rsi(up, 14).dropna().eq(100).all()
    assert ind.rsi(down, 14).dropna().eq(0).all()


def test_rsi_first_value_after_warmup():
    r = ind.rsi(pd.Series(np.arange(1.0, 40.0)), 20)
    assert r.iloc[:20].isna().all() and r.iloc[20:].notna().all()


def test_true_range_uses_previous_close_for_gaps():
    high = pd.Series([10.0, 21.0])
    low = pd.Series([9.0, 20.0])
    close = pd.Series([9.5, 20.5])
    tr = ind.true_range(high, low, close)
    assert np.isnan(tr.iloc[0])
    assert tr.iloc[1] == pytest.approx(21.0 - 9.5)  # gap up dominates high-low


def test_atr_of_constant_range_equals_range():
    n = 60
    close = pd.Series(np.full(n, 100.0))
    atr = ind.atr(close + 1, close - 1, close, 14).dropna()
    assert np.allclose(atr, 2.0)


def test_ema_seeded_with_sma():
    s = pd.Series(np.arange(1.0, 31.0))
    e = ind.ema(s, 10)
    assert e.iloc[:9].isna().all()
    assert e.iloc[9] == pytest.approx(s.iloc[:10].mean())


def test_ema_dataframe_matches_series(prices):
    df = pd.DataFrame({"a": prices, "b": prices * 2})
    out = ind.ema(df, 12)
    pd.testing.assert_series_equal(out["a"], ind.ema(prices, 12), check_names=False)


def test_macd_zero_for_constant_price():
    m = ind.macd(pd.Series(np.full(80, 50.0))).dropna()
    assert np.allclose(m, 0.0)


def test_skip1_momentum_exact():
    s = pd.Series(np.arange(300, dtype=float) + 1)  # price on day i = i + 1
    mom = ind.skip1_momentum(s, lookback=252, skip=21)
    t = 299
    assert mom.iloc[t] == pytest.approx((t - 21 + 1) / (t - 252 + 1) - 1)
    assert mom.iloc[:252].isna().all()


def test_high52_proximity_is_one_at_new_high():
    s = pd.Series(np.arange(1.0, 300.0))
    p = ind.high52_proximity(s).dropna()
    assert p.eq(1.0).all()
    assert ind.high52_proximity(s[::-1].reset_index(drop=True)).dropna().lt(1).all()


def test_rolling_zscore_has_no_lookahead(prices):
    z1 = ind.zscore_rolling(prices)
    shocked = prices.copy()
    shocked.iloc[-50:] *= 3
    z2 = ind.zscore_rolling(shocked)
    pd.testing.assert_series_equal(z1.iloc[:-50], z2.iloc[:-50])


def test_full_zscore_does_look_ahead(prices):
    """Documents the defect that `tfg` mode reproduces."""
    shocked = prices.copy()
    shocked.iloc[-50:] *= 3
    assert not np.allclose(ind.zscore_full(prices).iloc[:-50], ind.zscore_full(shocked).iloc[:-50])


def test_rolling_normalise_clips_and_handles_constant():
    s = pd.Series(np.r_[np.zeros(40), 1000.0])
    z = ind.rolling_normalise(s)
    assert z.iloc[-1] == 3.0
    assert z.iloc[10] == 0.0  # zero std treated as 1
