"""Parquet-backed market data store with a DuckDB SQL layer.

Layout (one directory per market)::

    data/<market>/
        prices.parquet      daily OHLCV for stocks       (long: date, ticker, ...)
        etf_prices.parquet  daily OHLCV for macro/sector ETFs
        universe.parquet    constituent snapshots        (snapshot_date, ticker, source)
        ingest_log.parquet  one row per ingestion run
        <derived>.parquet   anything written with ``write_table`` (features, regimes, runs…)

Every Parquet file is exposed as a DuckDB view of the same name, so the data can be
explored with plain SQL::

    store.sql("SELECT ticker, max(date) FROM prices GROUP BY 1")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from regime_rl.data.sources import PRICE_COLUMNS

log = logging.getLogger(__name__)

PRICE_TABLES = ("prices", "etf_prices")
_KEY = ["date", "ticker"]


@dataclass
class ValidationReport:
    rows_in: int
    rows_out: int
    duplicates: int
    non_positive_price: int
    high_below_low: int
    negative_volume: int

    @property
    def dropped(self) -> int:
        return self.rows_in - self.rows_out


def validate_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Drop rows that cannot be real quotes and de-duplicate on (date, ticker).

    Rules: prices must be positive, high >= low, volume >= 0. Duplicates keep the last
    occurrence (the most recent download wins).
    """
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"price frame missing columns: {sorted(missing)}")
    n_in = len(df)
    df = df[PRICE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["ticker"] = df["ticker"].astype("string")
    px = df[["open", "high", "low", "close", "adj_close"]]
    bad_px = (px.le(0) & px.notna()).any(axis=1) | df["adj_close"].isna()
    bad_hl = df["high"].notna() & df["low"].notna() & (df["high"] < df["low"])
    bad_vol = df["volume"].notna() & (df["volume"] < 0)
    df = df.loc[~(bad_px | bad_hl | bad_vol)]
    n_before_dedup = len(df)
    df = df.drop_duplicates(_KEY, keep="last").sort_values(_KEY).reset_index(drop=True)
    report = ValidationReport(
        rows_in=n_in,
        rows_out=len(df),
        duplicates=n_before_dedup - len(df),
        non_positive_price=int(bad_px.sum()),
        high_below_low=int(bad_hl.sum()),
        negative_volume=int(bad_vol.sum()),
    )
    return df, report


class MarketStore:
    """Read/write access to one market's Parquet files plus a DuckDB connection."""

    def __init__(self, root: str | Path, market: str) -> None:
        self.dir = Path(root) / market.upper()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------ paths
    def path(self, table: str) -> Path:
        return self.dir / f"{table}.parquet"

    def has(self, table: str) -> bool:
        return self.path(table).exists()

    def tables(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.parquet"))

    # ------------------------------------------------------------------ writes
    def write_table(self, table: str, df: pd.DataFrame) -> Path:
        """Atomically (re)write ``table``."""
        path = self.path(table)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(path)
        self._refresh_views()
        return path

    def append_table(self, table: str, df: pd.DataFrame) -> Path:
        if self.has(table):
            df = pd.concat([self.read_table(table), df], ignore_index=True)
        return self.write_table(table, df)

    def upsert_prices(self, df: pd.DataFrame, table: str = "prices") -> ValidationReport:
        """Validate ``df`` and merge it into ``table`` (new rows win on (date, ticker))."""
        if table not in PRICE_TABLES:
            raise ValueError(f"{table!r} is not a price table")
        clean, report = validate_prices(df)
        if report.dropped:
            log.warning("%s: dropped %d invalid/duplicate rows", table, report.dropped)
        if self.has(table):
            clean, _ = validate_prices(pd.concat([self.read_table(table), clean]))
        self.write_table(table, clean)
        return report

    # ------------------------------------------------------------------ reads
    def read_table(self, table: str) -> pd.DataFrame:
        if not self.has(table):
            raise FileNotFoundError(f"No table {table!r} in {self.dir}. Run `regime-rl ingest`.")
        return pd.read_parquet(self.path(table))

    def last_dates(self, table: str = "prices") -> pd.Series:
        """Most recent stored date per ticker (empty if the table does not exist)."""
        if not self.has(table):
            return pd.Series(dtype="datetime64[ns]")
        out = self.sql(f"SELECT ticker, max(date) AS last FROM {table} GROUP BY ticker")
        return out.set_index("ticker")["last"]

    def load_prices(
        self,
        table: str = "prices",
        tickers: Sequence[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Long OHLCV, filtered in DuckDB (predicate push-down into Parquet)."""
        where, params = [], []
        if start:
            where.append("date >= ?")
            params.append(pd.Timestamp(start))
        if end:
            where.append("date <= ?")
            params.append(pd.Timestamp(end))
        if tickers is not None:
            where.append("list_contains(?, ticker)")
            params.append(list(tickers))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        return self.sql(f"SELECT * FROM {table} {clause} ORDER BY date, ticker", params)

    def wide(self, field: str, table: str = "prices", **kw) -> pd.DataFrame:
        """Pivot one field to a (date x ticker) frame."""
        long = self.load_prices(table, **kw)
        wide = long.pivot(index="date", columns="ticker", values=field).sort_index()
        wide.columns = wide.columns.astype(str)
        wide.columns.name = None
        return wide.astype(float)

    # ------------------------------------------------------------------ SQL
    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect()
            self._refresh_views()
        return self._con

    def _refresh_views(self) -> None:
        if self._con is None:
            return
        for p in self.dir.glob("*.parquet"):
            self._con.execute(
                f"CREATE OR REPLACE VIEW {p.stem} AS SELECT * FROM read_parquet('{p.as_posix()}')"
            )

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(query, params or []).df()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
