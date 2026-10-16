import pytest

from regime_rl.config import GateConfig, RewardConfig
from regime_rl.env import reward as R
from regime_rl.env.gates import ALL_ACTIONS, allowed_actions, resolve_action

G = GateConfig()
NEUTRAL = R.Signals(bull=0.5, bear=0.5, volatile=0.5, dispersion=0.5, accel=0.5, hi52=0.5)


# ----------------------------------------------------------------- gates
def test_strong_bull_allows_everything_but_bonds():
    assert allowed_actions(p_bull=0.9, p_bear=0.0, p_volatile=0.0, gates=G) == ALL_ACTIONS - {6}


def test_bear_blocks_concentrated():
    a = allowed_actions(p_bull=0.1, p_bear=0.6, p_volatile=0.0, gates=G)
    assert not a & {0, 1, 5}
    assert 6 in a and 2 in a


def test_volatile_blocks_top5_and_top3():
    a = allowed_actions(p_bull=0.8, p_bear=0.0, p_volatile=0.6, gates=G)
    assert not a & {0, 5} and 1 in a


def test_top3_needs_strong_bull():
    assert 5 not in allowed_actions(0.6, 0.0, 0.0, G)
    assert 5 in allowed_actions(0.7, 0.0, 0.0, G)


def test_thresholds_are_strict():
    assert 0 in allowed_actions(0.7, 0.40, 0.0, G)  # > 0.40 required to block
    assert 0 not in allowed_actions(0.7, 0.4001, 0.0, G)


def test_resolve_action():
    allowed = frozenset({2, 3, 4, 6, 7})
    assert resolve_action(3, allowed, tfg=True) == 3
    assert resolve_action(5, allowed, tfg=True) == 4  # nearest number
    assert resolve_action(1, allowed, tfg=True) == 2
    assert resolve_action(5, allowed, tfg=False) == 2  # safe default
    assert resolve_action(0, frozenset({1, 3}), tfg=True) == 1


# ----------------------------------------------------------------- reward terms
def test_information_ratio():
    assert R.information_ratio([0.01, 0.01, 0.01]) == 3.0  # zero std -> clipped
    assert R.information_ratio([0.02]) == pytest.approx(0.24)
    assert R.information_ratio([0.5]) == 3.0
    assert R.information_ratio([]) == 0.0


def test_calmar_signal():
    assert R.calmar_signal([0.01, 0.02, 0.02], drawdown=-0.04) == pytest.approx(
        (1.01 * 1.02 * 1.02 - 1) / 0.05
    )
    assert R.calmar_signal([0.01], drawdown=0) == pytest.approx(0.2)


def test_crash_penalty():
    cfg = RewardConfig()
    assert R.crash_penalty(-0.04, cfg) == 0.0
    assert R.crash_penalty(-0.05, cfg) == 0.0
    assert R.crash_penalty(-0.10, cfg) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("action", "signals", "expected"),
    [
        (2, NEUTRAL, 0.0),
        (0, R.Signals(0.8, 0, 0, 0.8, 0.5, 0.5), 0.4 * 0.8 * 0.8),
        (1, R.Signals(0.8, 0, 0, 0.8, 0.7, 0.7), 0.4 * 0.64 + 0.2 * 0.49),
        (5, R.Signals(0.7, 0, 0, 0.9, 0.5, 0.5), 0.0),  # bull below 0.75
        (5, R.Signals(0.8, 0, 0, 0.8, 0.8, 0.8), 0.6 * 0.64 + 0.3 * 0.64),
        (6, R.Signals(0.2, 0.7, 0, 0, 0, 0), 0.4 * 0.7 + 0.2 * 0.8),
        (7, R.Signals(0.3, 0.2, 0.5, 0, 0, 0), 0.3 * 0.5 + 0.1 * 0.5),
    ],
)
def test_regime_bonus(action, signals, expected):
    assert R.regime_bonus(action, signals) == pytest.approx(expected)


def test_worked_example_from_the_guide():
    """Top 30, -7% month, 6-month excess history, 4% below peak -> reward ~ 0.55."""
    excess = [0.01, -0.02, 0.02, 0.0, 0.01, -0.01]
    rets = [0.02, 0.01, 0.03, 0.01, 0.06, -0.07]
    reward, ir = R.step_reward(
        2, -0.07, excess, rets, drawdown=-0.04, signals=NEUTRAL, repeated=False, cfg=RewardConfig()
    )
    calmar = R.calmar_signal(rets, -0.04)
    assert ir == pytest.approx(R.information_ratio(excess))
    assert reward == pytest.approx(0.4 * ir + 0.4 * calmar - 0.2 * 0.1)


def test_repeat_penalty_applied():
    a, _ = R.step_reward(2, 0.0, [0.0], [0.0], 0.0, NEUTRAL, False, RewardConfig())
    b, _ = R.step_reward(2, 0.0, [0.0], [0.0], 0.0, NEUTRAL, True, RewardConfig())
    assert a - b == pytest.approx(0.05)


def test_zscore_to_unit():
    assert R.zscore_to_unit(0) == 0.5
    assert R.zscore_to_unit(1.5) == 0.75
    assert R.zscore_to_unit(9) == 1.0
