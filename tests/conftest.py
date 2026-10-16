from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_rl.config import Config, PPOConfig, RobustnessConfig, SelectionConfig, UniverseConfig
from regime_rl.data.ingest import ingest
from regime_rl.data.sources import SyntheticSource
from regime_rl.data.store import MarketStore


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def tfg_cfg() -> Config:
    return Config(fidelity="tfg")


def fast_config(tmp_path, fidelity: str = "corrected") -> Config:
    """Tiny PPO + robustness settings so the full pipeline runs in seconds."""
    return Config(
        fidelity=fidelity,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "out"),
        universe=UniverseConfig(top_n_liquidity=60),
        ppo=PPOConfig(passes=1, n_steps=64, batch_size=32, n_epochs=1, net_arch=(16, 16)),
        selection=SelectionConfig(seeds=(1, 2)),
        robustness=RobustnessConfig(bootstrap_reps=50, mc_paths=50),
    )


@pytest.fixture(scope="session")
def synthetic_store(tmp_path_factory) -> MarketStore:
    root = tmp_path_factory.mktemp("store")
    c = Config(data_dir=str(root))
    store = MarketStore(root, "US")
    ingest(c, SyntheticSource(sector_map=c.market.sector_map), store)
    return store


@pytest.fixture
def monthly_prices() -> pd.DataFrame:
    """Hand-built month-end prices for selection tests (index = 24 month ends)."""
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    base = np.linspace(100, 100, 24)
    df = pd.DataFrame(
        {
            # steady winner over the past year
            "WIN": np.linspace(100, 200, 24),
            # flat for a year, then +60% in the final month only (should NOT rank top on 12-1)
            "JUMP": np.r_[base[:-1], 160.0],
            # steady loser
            "LOSE": np.linspace(100, 60, 24),
            # flat
            "FLAT": base,
            "MID": np.linspace(100, 140, 24),
            "MID2": np.linspace(100, 130, 24),
        },
        index=idx,
    )
    # Listed only in the last 14 months, strong uptrend
    late = np.full(24, np.nan)
    late[-14:] = np.linspace(50, 120, 14)
    df["LATE"] = late
    return df
