from pathlib import Path

import pytest

from regime_rl.config import Config, config_from_dict, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["us.yaml", "eu.yaml", "cn.yaml", "smoke.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.market.code == cfg.market_code
    assert cfg.start < cfg.split.val_start < cfg.split.test_start < cfg.end


def test_us_config_matches_thesis_settings():
    cfg = load_config(CONFIGS / "us.yaml")
    assert cfg.market.benchmark == "SPY"
    assert cfg.transaction_cost == pytest.approx(0.002)
    assert len(cfg.selection.seeds) == 10
    assert cfg.split.val_start == "2021-01-01" and cfg.split.test_start == "2023-01-01"


def test_override_fidelity():
    cfg = load_config(CONFIGS / "us.yaml", fidelity="tfg")
    assert cfg.tfg and cfg.fidelity == "tfg"
    assert cfg.run_dir.name == "us_tfg"


def test_invalid_fidelity_rejected():
    with pytest.raises(ValueError, match="fidelity"):
        Config(fidelity="optimistic")


def test_unknown_market_rejected():
    with pytest.raises(ValueError, match="Unknown market"):
        Config(market_code="JP")


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"market": "US", "splt": {}})
    with pytest.raises(ValueError, match="SplitConfig"):
        config_from_dict({"market": "US", "split": {"val_begin": "2021-01-01"}})


def test_dates_must_be_ordered():
    with pytest.raises(ValueError, match="start < val_start"):
        config_from_dict({"split": {"val_start": "2024-01-01", "test_start": "2023-01-01"}})


def test_market_etf_lists_are_deduplicated():
    eu = Config(market_code="EU").market
    assert len(eu.macro_etfs) == len(set(eu.macro_etfs))
    assert eu.all_etfs[: len(eu.macro_etfs)] == eu.macro_etfs
