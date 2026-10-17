"""Incremental ingestion: source -> validation -> Parquet store.

Re-running ``ingest`` only downloads what is missing: tickers never seen are fetched from
``cfg.start``; tickers already stored are fetched from the day after their last stored date.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone

import pandas as pd

from regime_rl.config import Config
from regime_rl.data.sources import PriceSource
from regime_rl.data.store import MarketStore
from regime_rl.data.universe import resolve_universe

log = logging.getLogger(__name__)

MIN_COVERAGE = 0.9  # warn when fewer than 90% of the universe has prices


class IngestError(RuntimeError):
    """Ingestion finished without the data needed to run the pipeline."""


def plan_fetches(
    tickers: Sequence[str],
    last_dates: pd.Series,
    start: str,
    end: str,
) -> dict[str, list[str]]:
    """Group tickers by the first date that still needs downloading.

    Returns ``{fetch_start: [tickers]}``; tickers already up to ``end`` are omitted.
    """
    end_ts = pd.Timestamp(end)
    plan: dict[str, list[str]] = {}
    for t in tickers:
        last = last_dates.get(t)
        if last is None or pd.isna(last):
            first = pd.Timestamp(start)
        else:
            first = pd.Timestamp(last) + pd.Timedelta(days=1)
            if first > end_ts:
                continue
        plan.setdefault(first.strftime("%Y-%m-%d"), []).append(t)
    return plan


def _ingest_table(
    store: MarketStore,
    table: str,
    source: PriceSource,
    tickers: Sequence[str],
    start: str,
    end: str,
    full_refresh: bool,
) -> dict:
    last = pd.Series(dtype="datetime64[ns]") if full_refresh else store.last_dates(table)
    plan = plan_fetches(tickers, last, start, end)
    fetched = dropped = 0
    for first, group in sorted(plan.items()):
        log.info("%s: fetching %d tickers from %s to %s", table, len(group), first, end)
        df = source.fetch(group, first, end)
        if df.empty:
            continue
        report = store.upsert_prices(df, table)
        fetched += report.rows_in
        dropped += report.dropped
    stored = set(store.last_dates(table).index) if store.has(table) else set()
    missing = sorted(set(tickers) - stored)
    return {
        "table": table,
        "tickers_requested": len(tickers),
        "tickers_fetched": sum(len(g) for g in plan.values()),
        "tickers_stored": len(set(tickers) & stored),
        "tickers_missing": len(missing),
        "missing_sample": ", ".join(missing[:10]),
        "rows_fetched": fetched,
        "rows_dropped": dropped,
    }


def ingest(
    cfg: Config, source: PriceSource, store: MarketStore | None = None, full_refresh: bool = False
) -> pd.DataFrame:
    """Download (or top up) stock and ETF prices for ``cfg.market`` into the store."""
    store = store or MarketStore(cfg.data_dir, cfg.market_code)
    market = cfg.market
    tickers, uni_source = resolve_universe(market, synthetic=source.name == "synthetic")
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    store.append_table(
        "universe",
        pd.DataFrame(
            {
                "snapshot_date": pd.Timestamp(now.date()),
                "ticker": tickers,
                "source": uni_source,
            }
        ),
    )
    results = [
        _ingest_table(store, "prices", source, tickers, cfg.start, cfg.end, full_refresh),
        _ingest_table(
            store, "etf_prices", source, list(market.all_etfs), cfg.start, cfg.end, full_refresh
        ),
    ]
    log_rows = pd.DataFrame(results).assign(
        ingested_at=pd.Timestamp(now), source=source.name, start=cfg.start, end=cfg.end
    )
    store.append_table("ingest_log", log_rows)

    stocks, etfs = results
    etf_stored = set(store.last_dates("etf_prices").index) if store.has("etf_prices") else set()
    if market.benchmark not in etf_stored:
        raise IngestError(
            f"The benchmark {market.benchmark} could not be downloaded, so nothing can run. "
            "This is usually a network or Yahoo rate-limit problem: wait a few minutes and "
            "run ingest again (it only fetches what is missing)."
        )
    if stocks["tickers_stored"] < MIN_COVERAGE * stocks["tickers_requested"]:
        log.warning(
            "Only %d of %d stocks have data. Run ingest again to fill the gaps before `run`.",
            stocks["tickers_stored"],
            stocks["tickers_requested"],
        )
    return log_rows
