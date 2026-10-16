import numpy as np
import pandas as pd
import pytest

from regime_rl.agent.train import annualised_alpha, softmax_weights
from regime_rl.evaluation import metrics as M
from regime_rl.evaluation.robustness import block_bootstrap, block_indices, monte_carlo


def test_metrics_on_constant_returns():
    r = pd.Series([0.01] * 12)
    b = pd.Series([0.005] * 12)
    m = M.compute_metrics(r, b)
    assert m["ann_ret"] == pytest.approx(1.01**12 - 1)
    assert m["alpha"] == pytest.approx(1.01**12 - 1.005**12)
    assert m["ann_vol"] == pytest.approx(0.0)
    assert m["max_dd"] == 0.0
    assert np.isnan(m["calmar"])


def test_max_drawdown():
    assert M.max_drawdown(pd.Series([0.10, -0.20, 0.05, -0.10])) == pytest.approx(
        (1.1 * 0.8 * 1.05 * 0.9) / 1.1 - 1
    )


def test_metrics_need_three_months():
    assert M.compute_metrics([0.01, 0.02], [0.0, 0.0]) is None


def test_information_ratio_annualised():
    exc = np.array([0.01, 0.03, -0.01, 0.02])
    m = M.compute_metrics(exc, np.zeros(4))
    assert m["ir"] == pytest.approx(exc.mean() / (exc.std(ddof=1) + 1e-6) * np.sqrt(12))


def test_block_indices_are_contiguous_blocks():
    rng = np.random.RandomState(0)
    idx = block_indices(39, 6, rng)
    assert len(idx) == 39 and idx.min() >= 0 and idx.max() < 39
    for start in range(0, 36, 6):
        block = idx[start : start + 6]
        assert np.all(np.diff(block) == 1)


def test_block_indices_match_legacy_global_rng():
    """RandomState(42) reproduces the original script's np.random.seed(42) draws."""
    np.random.seed(42)  # noqa: NPY002 - deliberately the legacy global RNG
    legacy = []
    while len(legacy) < 39:
        s = np.random.randint(0, 39 - 6 + 1)  # noqa: NPY002
        legacy.extend(range(s, min(s + 6, 39)))
    np.testing.assert_array_equal(block_indices(39, 6, np.random.RandomState(42)), legacy[:39])


def test_block_too_long():
    with pytest.raises(ValueError):
        block_indices(4, 6, np.random.RandomState(0))


def test_bootstrap_is_reproducible_and_brackets_point():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.015, 0.05, 39))
    b = pd.Series(rng.normal(0.01, 0.03, 39))
    a = block_bootstrap(r, b, reps=300, seed=7)
    pd.testing.assert_frame_equal(a, block_bootstrap(r, b, reps=300, seed=7))
    assert set(a.index) == set(M.METRICS)
    assert (a["ci_lower"] <= a["ci_upper"]).all()
    assert a.loc["ann_ret", "ci_lower"] < a.loc["ann_ret", "point"] < a.loc["ann_ret", "ci_upper"]


def test_monte_carlo():
    r = pd.Series([0.01, 0.02, 0.03])
    mc = monte_carlo(r, pd.Series([0.01] * 3), paths=100, horizon=24, seed=0)
    assert mc["p_profit"] == 1.0
    assert mc["paths"].shape == (100, 25)
    assert mc["bench_total"] == pytest.approx(1.01**24 - 1)
    legacy = monte_carlo(r, pd.Series([0.01] * 3), paths=10, compound_benchmark=False)
    assert legacy["bench_total"] == pytest.approx(0.24)


def test_annualised_alpha_and_softmax():
    assert annualised_alpha([0.01] * 12, [0.0] * 12) == pytest.approx(1.01**12 - 1)
    w = softmax_weights([0.15, 0.12, 0.10], 0.5)
    assert sum(w) == pytest.approx(1.0) and w[0] > w[1] > w[2]
    assert max(w) < 0.4  # decimal alphas with T=0.5 give near-equal weights


def test_rolling_ir_window():
    s = pd.Series(np.r_[np.zeros(11), 0.01])
    assert M.rolling_ir(s).iloc[:11].isna().all()
