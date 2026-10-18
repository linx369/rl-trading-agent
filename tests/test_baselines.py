import numpy as np
import pandas as pd
import pytest

from regime_rl.config import Config
from regime_rl.env.trading_env import EpisodeData
from regime_rl.evaluation.baselines import (
    BOND_ACTION,
    MOMENTUM_ACTION,
    baseline_returns,
    baseline_table,
    choose_best_fixed,
    format_table,
)


def episode(n: int, returns: np.ndarray, p_bear: np.ndarray, start: str) -> EpisodeData:
    return EpisodeData(
        dates=list(pd.date_range(start, periods=n, freq="ME")),
        states=np.zeros((n, 3), dtype=np.float32),
        returns=returns,
        bench=np.full(n, 0.01),
        gate_probs=np.column_stack([1 - p_bear, p_bear, np.zeros(n)]),
        signal_probs=np.zeros((n, 3)),
        idx_dispersion=0,
        idx_accel=1,
        idx_hi52=2,
    )


@pytest.fixture
def history() -> EpisodeData:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.01, 0.03, (40, 8))
    rets[:, 3] = 0.01 + 0.004 + rng.normal(0, 0.002, 40)  # action 3: steady excess -> best IR
    return episode(40, rets, np.zeros(40), "2017-01-31")


@pytest.fixture
def test_block() -> EpisodeData:
    n = 12
    rets = np.tile(np.arange(8) / 100.0, (n, 1))  # action a earns a% each month
    p_bear = np.zeros(n)
    p_bear[[2, 5]] = 0.9  # two bearish months
    return episode(n, rets, p_bear, "2023-01-31")


def test_best_fixed_uses_information_ratio(history):
    assert choose_best_fixed(history) == 3


def test_baseline_returns(test_block, history):
    frame, notes = baseline_returns(test_block, history, Config())
    assert np.allclose(frame["spy"], 0.01)
    assert np.allclose(frame["momentum_30"], MOMENTUM_ACTION / 100)
    assert np.allclose(frame["best_fixed"], 3 / 100)
    assert np.allclose(frame["equal_mix"], np.mean(np.arange(8)) / 100)
    expected = np.full(12, MOMENTUM_ACTION / 100)
    expected[[2, 5]] = BOND_ACTION / 100
    assert np.allclose(frame["hmm_switch"], expected)
    assert notes["hmm_switch"] == "bonds in 2 of 12 months"


def test_tfg_mode_drops_untraded_last_month(test_block, history):
    frame, _ = baseline_returns(test_block, history, Config(fidelity="tfg"))
    assert len(frame) == 11


def test_best_fixed_ignores_test_data(test_block, history):
    """Choosing the fixed strategy must not depend on test-period returns."""
    shocked = test_block.slice(np.ones(12, bool))
    shocked.returns = shocked.returns.copy()
    shocked.returns[:, 7] += 1.0  # action 7 is spectacular in the test period only
    f1, _ = baseline_returns(test_block, history, Config())
    f2, _ = baseline_returns(shocked, history, Config())
    assert np.allclose(f1["best_fixed"], f2["best_fixed"])


def test_baseline_table_includes_agent(test_block, history):
    record = pd.DataFrame(
        {"ret": np.full(12, 0.02)}, index=pd.DatetimeIndex(test_block.dates, name="date")
    )
    table, rets = baseline_table(test_block, history, record, Config(), agent_detail="seed 1")
    assert list(rets.columns)[-1] == "agent" and len(rets) == 12
    assert list(table.index) == [
        "spy",
        "momentum_30",
        "best_fixed",
        "equal_mix",
        "hmm_switch",
        "agent",
    ]
    assert table.loc["spy", "alpha"] == pytest.approx(0.0)
    assert table.loc["agent", "alpha"] == pytest.approx(1.02**12 - 1.01**12)
    assert table.loc["agent", "detail"] == "seed 1"
    assert "RL agent" in format_table(table)


def test_baseline_table_rejects_misaligned_agent(test_block, history):
    record = pd.DataFrame({"ret": [0.0]}, index=pd.DatetimeIndex(["1999-01-31"], name="date"))
    with pytest.raises(ValueError, match="same months"):
        baseline_table(test_block, history, record, Config())
