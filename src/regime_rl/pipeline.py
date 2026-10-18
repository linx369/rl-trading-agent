"""End-to-end orchestration: stored prices -> features -> regimes -> agent -> evaluation.

Each stage is a plain function so it can be run (and tested) on its own; ``run`` chains
them and persists every intermediate table to the store for SQL inspection.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from regime_rl.agent.train import SelectionResult, rollout, select_policy
from regime_rl.config import Config
from regime_rl.data.store import MarketStore
from regime_rl.env.reward import zscore_to_unit
from regime_rl.env.trading_env import EpisodeData
from regime_rl.evaluation.baselines import baseline_table
from regime_rl.evaluation.metrics import benchmark_metrics, compute_metrics
from regime_rl.evaluation.robustness import block_bootstrap, monte_carlo
from regime_rl.features.panel import (
    MONTH_END,
    MonthlyData,
    PriceFrames,
    build_panel,
    cross_section,
    monthly_data,
)
from regime_rl.features.state import StateMatrix, build_state
from regime_rl.portfolio.selection import SelectionContext
from regime_rl.portfolio.strategies import StrategyBook, precompute_returns
from regime_rl.regimes.hmm import RegimeDetector, detect_regimes, monthly_regimes

log = logging.getLogger(__name__)

REGIME_COLS = ["p_bull", "p_bear", "p_volatile"]


def markdown_table(frame: pd.DataFrame | pd.Series, fmt: str = "{:.5f}") -> str:
    df = frame.to_frame() if isinstance(frame, pd.Series) else frame
    head = "| | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = [
        "| " + str(i) + " | " + " | ".join(fmt.format(v) for v in row) + " |"
        for i, row in zip(df.index, df.to_numpy(), strict=True)
    ]
    return "\n".join([head, sep, *body])


@dataclass
class Dataset:
    cfg: Config
    monthly: MonthlyData
    panel: pd.DataFrame
    detector: RegimeDetector
    regimes_daily: pd.DataFrame
    regimes_raw: pd.DataFrame
    regimes_lagged: pd.DataFrame
    state: StateMatrix
    book: StrategyBook
    episode: EpisodeData

    def split(self) -> tuple[EpisodeData, EpisodeData, EpisodeData]:
        d = pd.DatetimeIndex(self.episode.dates)
        val, test = pd.Timestamp(self.cfg.split.val_start), pd.Timestamp(self.cfg.split.test_start)
        parts = (d < val, (d >= val) & (d < test), d >= test)
        names = ("train", "validation", "test")
        for name, m in zip(names, parts, strict=True):
            if m.sum() < 2:
                raise ValueError(f"{name} block has {int(m.sum())} months; check dates/split")
        return tuple(self.episode.slice(np.asarray(m)) for m in parts)  # type: ignore[return-value]


def build_dataset(cfg: Config, store: MarketStore) -> Dataset:
    market = cfg.market
    log.info("Loading prices from %s", store.dir)
    px = PriceFrames.from_long(store.load_prices("prices", start=cfg.start, end=cfg.end))
    etf = PriceFrames.from_long(store.load_prices("etf_prices", start=cfg.start, end=cfg.end))
    if market.benchmark not in etf.adj_close:
        raise ValueError(f"Benchmark {market.benchmark} not in stored ETF prices")

    bench_daily = etf.adj_close[market.benchmark].dropna()
    bench_m = bench_daily.resample(MONTH_END).last()
    bench_ret = bench_m.pct_change(fill_method=None)
    monthly = monthly_data(px)

    log.info("Building monthly panel (%d tickers)…", len(px.tickers))
    panel = build_panel(px, monthly, bench_ret, cfg)
    cs = cross_section(panel)

    train_end = pd.Timestamp(cfg.split.val_start) - pd.Timedelta(days=1)
    log.info(
        "Fitting HMM (%s)…",
        "full sample, smoothed" if cfg.tfg else f"train only to {train_end.date()}, filtered",
    )
    detector, probs = detect_regimes(bench_daily, cfg.hmm, cfg.tfg, train_end)
    reg_raw, reg_lag = monthly_regimes(probs)

    etf_m = etf.adj_close.resample(MONTH_END).last()
    sectors = [
        s for s in market.sector_etfs if s in etf.adj_close and etf.adj_close[s].notna().sum() > 50
    ]
    sector_1m = etf_m[sectors].pct_change(fill_method=None)
    state = build_state(cs, reg_lag, etf.adj_close, sector_1m, bench_ret, market)

    ctx = SelectionContext(
        cfg=cfg,
        monthly=monthly,
        panel=panel,
        bench_ret_3m=bench_m.pct_change(3, fill_method=None),
        sector_ret_1m=sector_1m,
        sector_ret_3m=etf_m[sectors].pct_change(3, fill_method=None),
        sector_map=market.sector_map,
    )
    dates = sorted(set(state.normalised.index) & set(panel.index.get_level_values("date")))
    log.info("Pre-computing 8 portfolios for %d months…", len(dates))
    book = precompute_returns(ctx, dates, etf_m.pct_change(fill_method=None), bench_ret)

    ep_dates = sorted(set(state.normalised.index) & set(book.returns.index))
    st = state.normalised.loc[ep_dates]
    gate = reg_raw.reindex(ep_dates, method="ffill")[REGIME_COLS].fillna(0.0).to_numpy()
    if cfg.tfg:  # original: z-scored regime features mapped to 0..1
        signal = np.vectorize(zscore_to_unit)(st[REGIME_COLS].to_numpy())
    else:  # corrected: the actual (lagged) probabilities
        signal = reg_lag.reindex(ep_dates)[REGIME_COLS].fillna(0.0).to_numpy()
    episode = EpisodeData(
        dates=list(ep_dates),
        states=st.to_numpy(np.float32),
        returns=book.returns.loc[ep_dates].to_numpy(float),
        bench=book.bench_fwd.reindex(ep_dates).fillna(0.0).to_numpy(float),
        gate_probs=gate,
        signal_probs=signal,
        idx_dispersion=state.index_of("mom_dispersion"),
        idx_accel=state.index_of("mom_accel"),
        idx_hi52=state.index_of("hi52_pct"),
    )
    log.info(
        "Dataset: %d months x %d state features (+8 alpha features)", len(ep_dates), st.shape[1]
    )
    return Dataset(cfg, monthly, panel, detector, probs, reg_raw, reg_lag, state, book, episode)


@dataclass
class RunResult:
    run_id: str
    metrics: dict
    bench: dict
    bootstrap: pd.DataFrame
    monte_carlo: dict
    test_record: pd.DataFrame
    seed_scores: pd.DataFrame
    cross_seed: pd.DataFrame
    out_dir: Path
    baselines: pd.DataFrame | None = None


def evaluate(ds: Dataset, sel: SelectionResult, test: EpisodeData) -> dict:
    cfg = ds.cfg
    rec = sel.policy.run(test, cfg)
    metrics = compute_metrics(rec["ret"], rec["bench_ret"]) or {}
    rb = cfg.robustness
    boot = block_bootstrap(
        rec["ret"],
        rec["bench_ret"],
        rb.bootstrap_reps,
        min(rb.bootstrap_block, len(rec)),
        rb.random_state,
    )
    mc = monte_carlo(
        rec["ret"],
        rec["bench_ret"],
        rb.mc_paths,
        rb.mc_horizon_months,
        rb.random_state,
        compound_benchmark=not cfg.tfg,
    )
    history = ds.episode.slice(
        np.asarray(pd.DatetimeIndex(ds.episode.dates) < pd.Timestamp(cfg.split.test_start))
    )
    seeds = ", ".join(map(str, sel.policy.seeds))
    detail = ("ensemble of seeds " if sel.policy.is_ensemble else "seed ") + seeds
    ladder, ladder_rets = baseline_table(test, history, rec, cfg, agent_detail=detail)
    cross = []
    for seed, agent in sel.agents.items():
        r = rollout(agent, test, cfg)
        cross.append({"seed": seed, **(compute_metrics(r["ret"], r["bench_ret"]) or {})})
    return {
        "record": rec,
        "metrics": metrics,
        "bench": benchmark_metrics(rec["bench_ret"]),
        "bootstrap": boot,
        "mc": mc,
        "cross_seed": pd.DataFrame(cross),
        "baselines": ladder,
        "baseline_returns": ladder_rets,
    }


def run(cfg: Config, store: MarketStore | None = None, plots: bool = True) -> RunResult:
    store = store or MarketStore(cfg.data_dir, cfg.market_code)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = cfg.run_dir
    out.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(cfg, store)
    train, val, test = ds.split()
    log.info(
        "Split: train %s..%s (%d) | val %d | test %s..%s (%d)",
        train.dates[0].date(),
        train.dates[-1].date(),
        len(train.dates),
        len(val.dates),
        test.dates[0].date(),
        test.dates[-1].date(),
        len(test.dates),
    )
    sel = select_policy(train, val, cfg, model_dir=out / "models")
    ev = evaluate(ds, sel, test)
    _persist(cfg, store, ds, sel, ev, run_id, out, (train, val, test))
    if plots:
        from regime_rl.reporting.plots import plot_all, plot_baselines

        label = "Ensemble" if sel.policy.is_ensemble else f"Agent (seed {sel.policy.seeds[0]})"
        plot_all(ev["record"], ds.regimes_raw, ev["mc"], out / "figures", label)
        plot_baselines(ev["baselines"], ev["baseline_returns"], out / "figures")
    return RunResult(
        run_id,
        ev["metrics"],
        ev["bench"],
        ev["bootstrap"],
        ev["mc"],
        ev["record"],
        sel.seed_scores,
        ev["cross_seed"],
        out,
        ev["baselines"],
    )


def _persist(
    cfg: Config,
    store: MarketStore,
    ds: Dataset,
    sel: SelectionResult,
    ev: dict,
    run_id: str,
    out: Path,
    blocks,
) -> None:
    tag = f"{cfg.fidelity}"
    rec = ev["record"].reset_index()
    rec["allowed"] = rec["allowed"].astype(str)
    store.write_table(f"regimes_{tag}", ds.regimes_raw.reset_index(names="date"))
    store.write_table(f"state_{tag}", ds.state.normalised.reset_index(names="date"))
    store.write_table(f"strategy_returns_{tag}", ds.book.returns.rename(columns=str).reset_index())
    store.write_table(f"holdings_{tag}", ds.book.holdings)
    store.write_table(f"test_record_{tag}", rec)

    prof = ds.detector.profile()
    summary = {
        "run_id": run_id,
        "market": cfg.market_code,
        "fidelity": cfg.fidelity,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "months": {k: len(b.dates) for k, b in zip(("train", "val", "test"), blocks, strict=True)},
        "test_period": [str(blocks[2].dates[0].date()), str(blocks[2].dates[-1].date())],
        "policy": {
            "seeds": sel.policy.seeds,
            "weights": sel.policy.weights,
            "ensemble_val_alpha": sel.ensemble_val_alpha,
            "single_val_alpha": sel.single_val_alpha,
        },
        "metrics": ev["metrics"],
        "benchmark": ev["bench"],
        "baselines": {
            k: {"alpha": r["alpha"], "ir": r["ir"], "sharpe": r["sharpe"], "max_dd": r["max_dd"]}
            for k, r in ev["baselines"][["alpha", "ir", "sharpe", "max_dd"]].iterrows()
        },
        "monte_carlo": {k: v for k, v in ev["mc"].items() if k != "paths"},
        "action_share": ev["record"]["action"].value_counts(normalize=True).sort_index().to_dict(),
        "config": cfg.to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    ev["bootstrap"].to_csv(out / "bootstrap_ci.csv")
    ev["cross_seed"].to_csv(out / "cross_seed_test_metrics.csv", index=False)
    ladder = ev["baselines"]
    ladder.to_csv(out / "baselines.csv")
    store.write_table(f"baselines_{tag}", ladder.reset_index().assign(run_id=run_id))
    sel.seed_scores.to_csv(out / "seed_selection.csv", index=False)
    rec.to_csv(out / "test_record.csv", index=False)
    with open(out / "hmm_profile.md", "w") as fh:
        fh.write(
            f"# HMM regime profile ({cfg.market_code}, {cfg.fidelity}, fitted to "
            f"{ds.detector.fit_end.date()})\n\n"
        )
        for name, frame in prof.items():
            fh.write(f"## {name}\n\n{markdown_table(frame)}\n\n")

    row = {
        "run_id": run_id,
        "created_utc": summary["created_utc"],
        "market": cfg.market_code,
        "fidelity": cfg.fidelity,
        "seeds": json.dumps(sel.policy.seeds),
        **{f"test_{k}": v for k, v in ev["metrics"].items()},
        **{f"bench_{k}": v for k, v in ev["bench"].items()},
    }
    store.append_table("runs", pd.DataFrame([row]))
