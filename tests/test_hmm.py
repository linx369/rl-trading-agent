import numpy as np
import pandas as pd
import pytest

from regime_rl.config import HMMConfig
from regime_rl.regimes.hmm import RegimeDetector, monthly_regimes, regime_features


@pytest.fixture(scope="module")
def spy() -> pd.Series:
    """Prices with alternating calm-up and volatile-down years."""
    rng = np.random.default_rng(3)
    rets = []
    for year in range(10):
        mu, sd = (0.0008, 0.007) if year % 2 == 0 else (-0.0008, 0.022)
        rets.append(rng.normal(mu, sd, 252))
    r = np.concatenate(rets)
    idx = pd.bdate_range("2010-01-01", periods=len(r))
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


@pytest.fixture(scope="module")
def fitted(spy):
    feats = regime_features(spy)
    return RegimeDetector(HMMConfig(n_iter=100)).fit(feats), feats


def test_features_have_no_missing_values(spy):
    f = regime_features(spy)
    assert list(f.columns) == ["ret", "vol_20", "ret_63", "ret_252"]
    assert not f.isna().any().any()
    assert f.index[0] >= spy.index[252]


def test_labels_are_ordered_by_mean_return(fitted):
    det, _ = fitted
    means = det.profile()["means"]["ret"]
    assert means["bull"] == means.max() and means["bear"] == means.min()
    assert sorted(det.labels.values()) == ["bear", "bull", "sideways", "volatile"]


def test_probabilities_sum_to_one(fitted):
    det, feats = fitted
    for probs in (det.filtered(feats), det.smoothed(feats)):
        assert list(probs.columns) == ["p_bull", "p_bear", "p_sideways", "p_volatile"]
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert probs.ge(0).all().all()


def test_filtered_equals_smoothed_on_last_day(fitted):
    """At the final observation there is no future, so both must agree."""
    det, feats = fitted
    np.testing.assert_allclose(
        det.filtered(feats).iloc[-1], det.smoothed(feats).iloc[-1], atol=1e-6
    )


def test_filtered_probabilities_do_not_look_ahead(fitted):
    det, feats = fitted
    cut = len(feats) - 300
    early = det.filtered(feats.iloc[:cut])
    full = det.filtered(feats)
    pd.testing.assert_frame_equal(early, full.iloc[:cut])


def test_smoothed_probabilities_do_look_ahead(fitted):
    """Documents why tfg mode is optimistic: past probabilities change with future data."""
    det, feats = fitted
    cut = len(feats) - 300
    early = det.smoothed(feats.iloc[:cut])
    full = det.smoothed(feats)
    assert not np.allclose(early.to_numpy(), full.iloc[:cut].to_numpy(), atol=1e-6)


def test_fit_end_restricts_training_window(spy):
    feats = regime_features(spy)
    end = feats.index[800]
    det = RegimeDetector(HMMConfig(n_iter=50)).fit(feats, fit_end=end)
    assert det.fit_end == end


def test_profile_durations(fitted):
    det, _ = fitted
    prof = det.profile()
    stay = np.diag(prof["transition"].to_numpy())
    np.testing.assert_allclose(prof["duration"].to_numpy(), 1 / (1 - stay))
    np.testing.assert_allclose(prof["transition"].sum(axis=1), 1.0)


def test_monthly_regimes_lag(fitted):
    det, feats = fitted
    raw, lagged = monthly_regimes(det.filtered(feats))
    pd.testing.assert_series_equal(lagged.iloc[0], raw.iloc[0], check_names=False)
    assert lagged.index[0] == raw.index[1]


def test_unfitted_detector_raises(spy):
    with pytest.raises(RuntimeError):
        RegimeDetector(HMMConfig()).filtered(regime_features(spy))


def test_restarts_never_worse_than_single_fit(spy):
    feats = regime_features(spy)
    single = RegimeDetector(HMMConfig(n_iter=50)).fit(feats, n_restarts=1)
    multi = RegimeDetector(HMMConfig(n_iter=50)).fit(feats, n_restarts=4)
    assert multi.log_likelihood >= single.log_likelihood - 1e-6
    assert multi.occupancy.sum() == pytest.approx(1.0)


def test_tfg_mode_uses_single_fit_on_full_sample(spy):
    from regime_rl.regimes.hmm import detect_regimes

    det, probs = detect_regimes(spy, HMMConfig(n_iter=50), tfg=True, train_end="2014-12-31")
    assert det.fit_end == probs.index[-1]
    single = RegimeDetector(HMMConfig(n_iter=50)).fit(regime_features(spy), n_restarts=1)
    assert det.log_likelihood == pytest.approx(single.log_likelihood)
