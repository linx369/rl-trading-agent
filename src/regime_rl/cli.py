"""Command-line interface: ``regime-rl {ingest,run,query,regimes}``."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from regime_rl.config import load_config
from regime_rl.data.store import MarketStore


def _source(name: str, cfg):
    from regime_rl.data.sources import SyntheticSource, YahooSource

    if name == "synthetic":
        return SyntheticSource(sector_map=cfg.market.sector_map)
    return YahooSource()


def cmd_ingest(args) -> int:
    from regime_rl.data.ingest import IngestError, ingest

    cfg = load_config(args.config, data_dir=args.data_dir)
    try:
        log = ingest(cfg, _source(args.source, cfg), full_refresh=args.full_refresh)
    except IngestError as exc:
        print(f"\nIngest incomplete: {exc}", file=sys.stderr)
        return 1
    cols = ["table", "tickers_requested", "tickers_stored", "tickers_missing", "rows_fetched"]
    print(log[cols].to_string(index=False))
    for row in log.itertuples():
        if row.tickers_missing:
            print(f"  {row.table}: no data yet for {row.missing_sample}…")
    return 0


def cmd_run(args) -> int:
    from regime_rl.pipeline import run

    cfg = load_config(args.config, fidelity=args.fidelity, data_dir=args.data_dir)
    if args.seeds:
        from dataclasses import replace

        cfg = cfg.replace(selection=replace(cfg.selection, seeds=tuple(args.seeds)))
    store = MarketStore(cfg.data_dir, cfg.market_code)
    if args.source and not store.has("prices"):
        cmd_ingest(
            argparse.Namespace(
                config=args.config, data_dir=cfg.data_dir, source=args.source, full_refresh=False
            )
        )
    res = run(cfg, store, plots=not args.no_plots)
    m, b = res.metrics, res.bench
    rows = [
        ("Annual return", "ann_ret", True),
        ("Annual volatility", "ann_vol", True),
        ("Sharpe ratio", "sharpe", False),
        ("Max drawdown", "max_dd", True),
        ("Calmar ratio", "calmar", False),
    ]
    universe = "point-in-time members" if cfg.universe.point_in_time else "current members"
    print(
        f"\nOut-of-sample ({cfg.market_code}, fidelity={cfg.fidelity}, {universe}, "
        f"{len(res.test_record)} months)"
    )
    print(f"{'Metric':<20}{'Agent':>12}{'Benchmark':>12}")
    for label, k, pct in rows:
        f = (lambda v: f"{v:>11.2%}") if pct else (lambda v: f"{v:>11.2f}")
        print(f"{label:<20}{f(m[k]):>12}{f(b[k]):>12}")
    print(f"{'Alpha (ann.)':<20}{m['alpha']:>11.2%}")
    print(f"{'Information ratio':<20}{m['ir']:>11.2f}")
    ci = res.bootstrap.loc["alpha"]
    print(f"\nAlpha 90% block-bootstrap CI: [{ci.ci_lower:+.2%}, {ci.ci_upper:+.2%}]")
    if res.baselines is not None:
        from regime_rl.evaluation.baselines import format_table

        print(f"\nSimpler strategies over the same {len(res.test_record)} months")
        print(format_table(res.baselines))
    print(f"\nOutputs: {res.out_dir}")
    return 0


def cmd_query(args) -> int:
    cfg = load_config(args.config, data_dir=args.data_dir)
    store = MarketStore(cfg.data_dir, cfg.market_code)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(store.sql(args.sql).to_string(index=False))
    return 0


def cmd_regimes(args) -> int:
    from regime_rl.pipeline import markdown_table
    from regime_rl.regimes.hmm import detect_regimes

    cfg = load_config(args.config, fidelity=args.fidelity, data_dir=args.data_dir)
    store = MarketStore(cfg.data_dir, cfg.market_code)
    prices = store.wide(
        "adj_close", "etf_prices", tickers=[cfg.market.benchmark], start=cfg.start, end=cfg.end
    )[cfg.market.benchmark].dropna()
    train_end = pd.Timestamp(cfg.split.val_start) - pd.Timedelta(days=1)
    det, _ = detect_regimes(prices, cfg.hmm, cfg.tfg, train_end)
    for name, frame in det.profile().items():
        print(f"\n## {name}\n{markdown_table(frame)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="regime-rl", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("-c", "--config", default="configs/us.yaml")
        sp.add_argument("--data-dir", default=None, help="override data_dir from the config")

    sp = sub.add_parser("ingest", help="download / top up prices into the Parquet store")
    common(sp)
    sp.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    sp.add_argument("--full-refresh", action="store_true")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("run", help="build features, train, select and evaluate")
    common(sp)
    sp.add_argument("--fidelity", choices=["tfg", "corrected"], default=None)
    sp.add_argument("--seeds", type=int, nargs="+", help="override the seed list")
    sp.add_argument(
        "--source",
        choices=["yahoo", "synthetic"],
        default=None,
        help="ingest first with this source if the store is empty",
    )
    sp.add_argument("--no-plots", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("query", help="run SQL against the store (DuckDB)")
    common(sp)
    sp.add_argument("sql")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("regimes", help="fit the HMM and print the regime profiles")
    common(sp)
    sp.add_argument("--fidelity", choices=["tfg", "corrected"], default=None)
    sp.set_defaults(func=cmd_regimes)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
