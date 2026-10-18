"""Integration tests on the synthetic store: dataset assembly, leakage guards, full run."""

import json

import numpy as np
import pandas as pd
import pytest

from regime_rl.cli import main as cli_main
from regime_rl.config import Config, UniverseConfig
from regime_rl.pipeline import build_dataset, run
from tests.conftest import fast_config


@pytest.fixture(scope="module", params=["tfg", "corrected"])
def dataset(request, synthetic_store):
    cfg = Config(fidelity=request.param, universe=UniverseConfig(top_n_liquidity=60))
    return build_dataset(cfg, synthetic_store)


def test_state_has_expected_columns(dataset):
    cols = dataset.state.columns
    assert cols[:12] == [
        "rsi_mean",
        "rsi_top_pct",
        "macd_pos_pct",
        "mom_12m_mean",
        "mom_dispersion",
        "mom_3m_mean",
        "mom_6m_mean",
        "mom_accel",
        "atr_mean",
        "beta_mean",
        "hi52_pct",
        "hi52_mean",
    ]
    assert {"p_bull", "p_bear", "p_sideways", "p_volatile"} <= set(cols)
    assert len(cols) == 12 + 4 + 4 + 7 + 3  # 30; +8 strategy-alpha features in the env = 38
    assert dataset.state.normalised.abs().max().max() <= 3.0


def test_split_is_chronological_and_disjoint(dataset):
    train, val, test = dataset.split()
    assert train.dates[-1] < val.dates[0] and val.dates[-1] < test.dates[0]
    assert val.dates[0] >= pd.Timestamp("2021-01-01") > train.dates[-1]
    assert test.dates[0] >= pd.Timestamp("2023-01-01") > val.dates[-1]
    n = len(train.dates) + len(val.dates) + len(test.dates)
    assert n == len(dataset.episode.dates)


def test_universe_limited_by_liquidity(dataset):
    per_month = dataset.panel.groupby(level="date").size()
    if not dataset.cfg.tfg:
        assert per_month.max() <= 60


def test_bond_action_return_matches_etf(dataset, synthetic_store):
    """Action 6 = next month's TLT return minus costs; checks date alignment (no leakage)."""
    tlt = synthetic_store.wide("adj_close", "etf_prices", tickers=["TLT"])["TLT"]
    tlt_m = tlt.resample("ME").last().pct_change()
    d = dataset.book.returns.index[50]
    nxt = tlt_m.index[tlt_m.index > d][0]
    assert dataset.book.returns.loc[d, 6] == pytest.approx(tlt_m[nxt] - 0.002)


def test_strategy_returns_are_next_month(dataset):
    """Top-30 return at date d must equal the holdings' realised return in month d+1."""
    book, monthly = dataset.book, dataset.monthly
    d = book.returns.index[60]
    h = book.holdings[(book.holdings.date == d) & (book.holdings.action == 2)]
    nd = monthly.ret.index[monthly.ret.index > d][0]
    r = (monthly.ret.loc[nd, h.ticker].to_numpy() * h.weight.to_numpy()).sum() / h.weight.sum()
    assert book.returns.loc[d, 2] == pytest.approx(r - 0.002)


def test_corrected_regimes_fit_on_training_window_only(dataset):
    if dataset.cfg.tfg:
        assert dataset.detector.fit_end > pd.Timestamp("2025-01-01")
    else:
        assert dataset.detector.fit_end < pd.Timestamp(dataset.cfg.split.val_start)


def test_holdings_weights_sum_to_one(dataset):
    sums = dataset.book.holdings.groupby(["date", "action"])["weight"].sum()
    assert np.allclose(sums, 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("fidelity", ["tfg", "corrected"])
def test_end_to_end_run(tmp_path, synthetic_store, fidelity):
    cfg = fast_config(tmp_path, fidelity)
    res = run(cfg, synthetic_store, plots=True)
    assert set(res.metrics) >= {"alpha", "sharpe", "ir", "max_dd"}
    assert len(res.seed_scores) == 2 and res.seed_scores["chosen"].any()
    assert len(res.cross_seed) == 2
    summary = json.loads((res.out_dir / "summary.json").read_text())
    assert summary["fidelity"] == fidelity
    assert len(list((res.out_dir / "figures").glob("*.png"))) == 7
    assert res.baselines.loc["agent", "alpha"] == pytest.approx(res.metrics["alpha"])
    assert synthetic_store.has(f"baselines_{fidelity}")
    runs = synthetic_store.sql(f"SELECT * FROM runs WHERE run_id = '{res.run_id}'")
    assert len(runs) == 1
    assert synthetic_store.has(f"holdings_{fidelity}")


@pytest.mark.slow
def test_cli_ingest_and_query(tmp_path, capsys):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "market: US\nstart: '2018-01-01'\nend: '2019-12-31'\n"
        "split: {val_start: '2019-01-01', test_start: '2019-06-01'}\n"
    )
    data = str(tmp_path / "d")
    assert (
        cli_main(["ingest", "-c", str(cfg_file), "--data-dir", data, "--source", "synthetic"]) == 0
    )
    assert (
        cli_main(
            [
                "query",
                "-c",
                str(cfg_file),
                "--data-dir",
                data,
                "SELECT count(DISTINCT ticker) AS n FROM etf_prices",
            ]
        )
        == 0
    )
    assert "12" in capsys.readouterr().out


@pytest.mark.slow
def test_training_is_reproducible(synthetic_store, tmp_path):
    """Two trainings with the same seed must give the same policy."""
    from regime_rl.agent.train import rollout, train_agent

    cfg = fast_config(tmp_path)
    ds = build_dataset(cfg, synthetic_store)
    train, val, _ = ds.split()
    a = rollout(train_agent(train, cfg, seed=7), val, cfg)
    b = rollout(train_agent(train, cfg, seed=7), val, cfg)
    pd.testing.assert_series_equal(a["action"], b["action"])
