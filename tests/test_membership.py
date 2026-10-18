import numpy as np
import pandas as pd
import pytest

from regime_rl.config import Config, UniverseConfig, config_from_dict
from regime_rl.data.membership import (
    apply_renames,
    load_membership,
    members_since,
    membership_matrix,
    parse_spells,
    verified_renames,
)
from regime_rl.features.panel import liquidity_rank

CSV = """ticker,start_date,end_date
AAPL,1996-01-02,
FB,2013-12-23,2022-06-09
META,2022-06-09,
BF.B,1996-01-02,
TSLA,2020-12-21,
LEH,1996-01-02,2008-09-16
OLD,1996-01-02,2004-01-01
XX,2010-01-01,2012-01-01
YY,2012-01-01,
"""


@pytest.fixture
def spells() -> pd.DataFrame:
    return parse_spells(CSV)


def test_parse_normalises_share_classes(spells):
    assert "BF-B" in set(spells.ticker)
    assert spells.loc[spells.ticker == "AAPL", "end"].isna().all()


def test_only_confirmed_renames_are_applied(spells):
    ok = verified_renames(spells, {"FB": "META", "XX": "TSLA", "NOPE": "AAPL"})
    assert ok == {"FB": "META"}  # XX ended 2012, TSLA started 2020 -> not a rename


def test_renamed_company_keeps_its_full_membership(spells):
    out = apply_renames(spells, {"FB": "META"})
    meta = out[out.ticker == "META"]
    assert len(meta) == 1
    assert meta.start.iloc[0] == pd.Timestamp("2013-12-23") and pd.isna(meta.end.iloc[0])
    assert "FB" not in set(out.ticker)


def test_load_membership_uses_the_shipped_rename_list():
    out = load_membership(CSV)
    assert out.loc[out.ticker == "META", "start"].iloc[0] == pd.Timestamp("2013-12-23")


def test_members_since(spells):
    names = members_since(apply_renames(spells, {"FB": "META"}), "2006-01-01")
    assert "LEH" in names  # left in 2008, still relevant to a 2006 backtest
    assert "OLD" not in names  # left before 2006
    assert {"META", "TSLA", "AAPL"} <= set(names)


def test_membership_matrix_dates(spells):
    dates = pd.date_range("2008-07-31", "2021-01-31", freq="ME")
    m = membership_matrix(spells, dates, ["LEH", "TSLA", "AAPL", "UNKNOWN"])
    assert m.loc["2008-08-31", "LEH"] and not m.loc["2008-09-30", "LEH"]  # left 16 Sep
    assert not m.loc["2020-11-30", "TSLA"] and m.loc["2020-12-31", "TSLA"]  # joined 21 Dec
    assert m["AAPL"].all() and not m["UNKNOWN"].any()


def test_liquidity_rank_only_ranks_members():
    idx = pd.date_range("2015-01-31", periods=24, freq="ME")
    dv = pd.DataFrame({"BIG": 1000.0, "MID": 100.0, "SMALL": 10.0}, index=idx)
    members = pd.DataFrame(True, index=idx, columns=dv.columns)
    members.loc[idx[:12], "BIG"] = False  # BIG joins the index after a year
    r = liquidity_rank(dv, Config(universe=UniverseConfig(liquidity_window_months=12)), members)
    assert np.isnan(r.loc[idx[11], "BIG"]) and r.loc[idx[11], "MID"] == 1
    assert r.loc[idx[12], "BIG"] == 1 and r.loc[idx[12], "MID"] == 2


def test_point_in_time_config():
    cfg = config_from_dict({"universe": {"point_in_time": True}})
    assert cfg.tag == "corrected_pit" and cfg.run_dir.name == "us_corrected_pit"
    assert "a2430f2" in cfg.universe.membership_url  # pinned commit
    with pytest.raises(ValueError, match="only available for the US"):
        config_from_dict({"market": "EU", "universe": {"point_in_time": True}})
