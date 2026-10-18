import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from regime_rl.config import Config
from regime_rl.env.trading_env import EpisodeData, TradingEnv


def make_data(n: int = 12, gate=(0.9, 0.0, 0.0)) -> EpisodeData:
    rng = np.random.default_rng(0)
    rets = np.tile(np.arange(8) / 100.0, (n, 1))  # action a earns a% every month
    return EpisodeData(
        dates=list(pd.date_range("2020-01-31", periods=n, freq="ME")),
        states=rng.uniform(-1, 1, (n, 5)).astype(np.float32),
        returns=rets,
        bench=np.full(n, 0.01),
        gate_probs=np.tile(gate, (n, 1)),
        signal_probs=np.tile(gate, (n, 1)),
        idx_dispersion=0,
        idx_accel=1,
        idx_hi52=2,
    )


@pytest.mark.parametrize("fidelity", ["tfg", "corrected"])
def test_gymnasium_api(fidelity):
    check_env(TradingEnv(make_data(), Config(fidelity=fidelity)), skip_render_check=True)


def test_observation_layout_and_bounds():
    env = TradingEnv(make_data(), Config())
    obs, _ = env.reset()
    assert obs.shape == (5 + 8,)
    assert np.all(obs[5:] == 0)  # no strategy history yet
    obs, *_ = env.step(2)
    assert env.observation_space.contains(obs)
    # strategy-alpha features: (a% - 1%) * 20, clipped to ±3
    np.testing.assert_allclose(obs[5:], np.clip((np.arange(8) / 100 - 0.01) * 20, -3, 3), rtol=1e-6)


@pytest.mark.parametrize(("fidelity", "steps"), [("tfg", 11), ("corrected", 12)])
def test_episode_length(fidelity, steps):
    env = TradingEnv(make_data(12), Config(fidelity=fidelity))
    env.reset()
    n, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(2)
        n += 1
    assert n == steps


def test_blocked_action_is_replaced():
    bear = (0.1, 0.8, 0.0)  # bear regime: 0, 1, 5 blocked
    env_t = TradingEnv(make_data(gate=bear), Config(fidelity="tfg"))
    env_c = TradingEnv(make_data(gate=bear), Config())
    for env, expected in ((env_t, 4), (env_c, 2)):
        env.reset()
        *_, info = env.step(5)
        assert info["requested"] == 5 and info["action"] == expected
        assert info["ret"] == pytest.approx(expected / 100)


def test_portfolio_value_compounds():
    env = TradingEnv(make_data(), Config())
    env.reset()
    for _ in range(3):
        *_, info = env.step(3)
    assert info["port_val"] == pytest.approx(1.03**3)
    assert info["drawdown"] == 0.0


def test_repeat_penalty_only_in_corrected_mode():
    rewards = {}
    for fid in ("tfg", "corrected"):
        env = TradingEnv(make_data(), Config(fidelity=fid))
        env.reset()
        rewards[fid] = [env.step(2)[1] for _ in range(4)]
    assert rewards["tfg"][3] - rewards["corrected"][3] == pytest.approx(0.05)
    assert rewards["tfg"][:3] == pytest.approx(rewards["corrected"][:3])


def test_mismatched_lengths_rejected():
    d = make_data()
    with pytest.raises(ValueError, match="bench"):
        EpisodeData(
            d.dates, d.states, d.returns, d.bench[:-1], d.gate_probs, d.signal_probs, 0, 1, 2
        )
