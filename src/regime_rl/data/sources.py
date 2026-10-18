"""Price sources. Every source returns the same *long* schema::

    date (datetime64[ns]) | ticker (str) | open | high | low | close | adj_close | volume

Two implementations ship: :class:`YahooSource` (live, via yfinance) and
:class:`SyntheticSource` (deterministic, offline — used by the tests and CI).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
_YF_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class PriceSource(Protocol):
    name: str

    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        """Return daily OHLCV in the long schema for ``start <= date <= end``."""
        ...


def empty_prices() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS})
    df["date"] = pd.Series(dtype="datetime64[ns]")
    df["ticker"] = pd.Series(dtype="string")
    return df[PRICE_COLUMNS]


def _batched(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def yf_frame_to_long(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Convert a ``yf.download`` result (single or multi-ticker) to the long schema."""
    if raw is None or raw.empty:
        return empty_prices()
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance >= 0.2 returns (field, ticker) columns
        stacked = raw.stack(level=1, future_stack=True)
        stacked.index.names = ["date", "ticker"]
        df = stacked.reset_index()
    else:
        df = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
        df["ticker"] = tickers[0]
    df = df.rename(columns=_YF_FIELDS)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[PRICE_COLUMNS].dropna(subset=["adj_close"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    df["ticker"] = df["ticker"].astype("string")
    return df.reset_index(drop=True)


def raise_open_file_limit(target: int = 4096) -> int:
    """Raise this process's soft limit on open files (macOS defaults to 256).

    yfinance opens a network connection and a SQLite cache handle per download thread;
    with the low macOS default this runs out mid-ingest and every request then fails with
    "unable to open database file" or "Could not resolve host". Returns the new soft limit.
    """
    try:
        import resource
    except ImportError:  # Windows
        return -1
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if soft != resource.RLIM_INFINITY and soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            soft = want
        except (ValueError, OSError) as exc:
            log.warning("could not raise open-file limit from %d: %s", soft, exc)
    return soft


class YahooSource:
    """Batched yfinance downloader with throttled, retried per-ticker fallbacks.

    Network use is deliberately conservative: a few download threads, a pause between
    batches and between single-ticker retries, and a long back-off on rate limiting.
    """

    name = "yahoo"

    def __init__(
        self,
        batch_size: int = 50,
        pause: float = 2.0,
        threads: int = 4,
        retry_pause: float = 0.3,
        rate_limit_wait: float = 60.0,
        downloader=None,
    ) -> None:
        self.batch_size = batch_size
        self.pause = pause
        self.retry_pause = retry_pause
        self.rate_limit_wait = rate_limit_wait
        if downloader is None:
            import yfinance as yf

            raise_open_file_limit()

            def downloader(tks, start, end):
                return yf.download(
                    tks,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    group_by="column",
                    threads=threads,
                    timeout=30,
                )

        self._download = downloader

    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        # yfinance's `end` is exclusive
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        parts: list[pd.DataFrame] = []
        batches = list(_batched(list(tickers), self.batch_size))
        failed: list[str] = []
        for i, batch in enumerate(batches, 1):
            log.info("Downloading batch %d/%d (%d tickers)", i, len(batches), len(batch))
            got = self._safe(batch, start, end_excl)
            missing = [t for t in batch if t not in set(got["ticker"])]
            if missing:
                log.info("  retrying %d missing tickers one by one", len(missing))
                for t in missing:
                    if self.retry_pause:
                        time.sleep(self.retry_pause)
                    single = self._safe([t], start, end_excl)
                    if single.empty:
                        failed.append(t)
                    else:
                        got = pd.concat([got, single], ignore_index=True)
            parts.append(got)
            if i < len(batches) and self.pause:
                time.sleep(self.pause)
        if failed:
            log.warning(
                "%d of %d tickers returned no data (e.g. %s). Re-running ingest retries only "
                "these.",
                len(failed),
                len(tickers),
                ", ".join(failed[:8]),
            )
        return pd.concat(parts, ignore_index=True) if parts else empty_prices()

    def _safe(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """One download call; never raises. Waits and retries once if rate limited."""
        for attempt in (1, 2):
            try:
                raw = self._download(tickers if len(tickers) > 1 else tickers[0], start, end)
                return yf_frame_to_long(raw, tickers)
            except Exception as exc:  # network / parsing failures must not kill the batch
                rate_limited = "ratelimit" in type(exc).__name__.lower() or "Too Many" in str(exc)
                if rate_limited and attempt == 1:
                    log.warning("rate limited by Yahoo; waiting %.0f s", self.rate_limit_wait)
                    time.sleep(self.rate_limit_wait)
                    continue
                log.warning("download failed for %s…: %s", tickers[:3], exc)
                break
        return empty_prices()


class SyntheticSource:
    """Deterministic regime-switching market for offline tests and demos.

    The market follows a 4-state Markov chain (bull, sideways, volatile, bear) with
    state-dependent drift and volatility. Each stock = beta * market + sector factor +
    a slowly-varying idiosyncratic drift (which creates momentum) + noise. About 20% of
    stocks list part-way through the sample, as real ones do.

    The numbers are **not** real market data; they exist so that every stage of the
    pipeline can be exercised without network access.
    """

    name = "synthetic"
    CALENDAR_START = "2000-01-03"
    CALENDAR_END = "2035-12-31"

    _REGIMES = {  # daily drift, daily vol
        "bull": (0.0007, 0.008),
        "sideways": (0.0000, 0.007),
        "volatile": (0.0002, 0.016),
        "bear": (-0.0012, 0.025),
    }
    _TRANS = np.array(
        [
            [0.990, 0.006, 0.003, 0.001],
            [0.008, 0.985, 0.005, 0.002],
            [0.006, 0.006, 0.980, 0.008],
            [0.002, 0.004, 0.020, 0.974],
        ]
    )

    def __init__(
        self,
        seed: int = 7,
        sector_map: dict[str, Sequence[str]] | None = None,
        late_listing_share: float = 0.2,
    ) -> None:
        self.seed = seed
        self.sector_map = {k: tuple(v) for k, v in (sector_map or {}).items()}
        self.late_listing_share = late_listing_share

    def _calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        # Generate on a fixed calendar so the same date always gets the same value,
        # regardless of the requested window (makes incremental ingestion consistent).
        if pd.Timestamp(end) > pd.Timestamp(self.CALENDAR_END):
            raise ValueError(f"SyntheticSource covers dates up to {self.CALENDAR_END}")
        return pd.bdate_range(self.CALENDAR_START, self.CALENDAR_END)

    def _market(self, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        n = len(dates)
        states = np.empty(n, dtype=int)
        states[0] = 0
        u = rng.random(n)
        cum = self._TRANS.cumsum(axis=1)
        for t in range(1, n):
            states[t] = int(np.searchsorted(cum[states[t - 1]], u[t]))
        params = np.array(list(self._REGIMES.values()))
        mu, sigma = params[states, 0], params[states, 1]
        ret = mu + sigma * rng.standard_normal(n)
        return ret, states

    def fetch(self, tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
        dates = self._calendar(start, end)
        mkt, states = self._market(dates)
        sector_of = {t: s for s, members in self.sector_map.items() for t in members}
        sectors = sorted(set(sector_of.values()) | {"_other"})
        rng_f = np.random.default_rng(self.seed + 1)
        sector_ret = {
            s: 0.0004 * rng_f.standard_normal() / 10 + 0.006 * rng_f.standard_normal(len(dates))
            for s in sectors
        }
        frames = []
        for t in tickers:
            if t.upper() in {"TLT"}:
                r = (
                    -0.25 * mkt
                    + 0.0001
                    + 0.009 * _rng_for(self.seed, t).standard_normal(len(dates))
                )
            elif t.upper() == "HYG":
                r = 0.35 * mkt + 0.0001 + 0.004 * _rng_for(self.seed, t).standard_normal(len(dates))
            elif t in sector_ret or t in self.sector_map:
                r = (
                    mkt
                    + sector_ret.get(t, 0)
                    + 0.002 * _rng_for(self.seed, t).standard_normal(len(dates))
                )
            elif t.upper() in {"SPY", "EZU", "FXI"}:
                r = mkt.copy()
            else:
                r = self._stock_returns(t, mkt, sector_ret[sector_of.get(t, "_other")])
            frames.append(self._ohlcv(t, dates, r))
        out = pd.concat(frames, ignore_index=True)
        mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
        return out.loc[mask].reset_index(drop=True)

    def _stock_returns(self, ticker: str, mkt: np.ndarray, sector: np.ndarray) -> np.ndarray:
        rng = _rng_for(self.seed, ticker)
        n = len(mkt)
        beta = rng.uniform(0.6, 1.5)
        # AR(1) idiosyncratic drift -> persistent winners/losers -> momentum
        drift = np.empty(n)
        drift[0] = rng.normal(0, 0.0004)
        shocks = rng.normal(0, 0.00004, n)
        for i in range(1, n):
            drift[i] = 0.997 * drift[i - 1] + shocks[i]
        noise = rng.normal(0, rng.uniform(0.010, 0.022), n)
        r = beta * mkt + sector + drift + noise
        if rng.random() < self.late_listing_share:
            r[: int(rng.uniform(0.15, 0.6) * n)] = np.nan
        return r

    def _ohlcv(self, ticker: str, dates: pd.DatetimeIndex, r: np.ndarray) -> pd.DataFrame:
        rng = _rng_for(self.seed + 99, ticker)
        r = np.clip(r, -0.5, 0.5)
        listed = ~np.isnan(r)
        log_px = np.cumsum(np.where(listed, np.log1p(np.nan_to_num(r)), 0.0))
        close = rng.uniform(20, 300) * np.exp(log_px)
        rng_hl = np.abs(rng.normal(0, 0.008, len(r))) * close
        open_ = close * (1 + rng.normal(0, 0.003, len(r)))
        high = np.maximum(open_, close) + rng_hl
        low = np.maximum(np.minimum(open_, close) - rng_hl, 0.01)
        volume = rng.lognormal(mean=rng.uniform(13, 17), sigma=0.3, size=len(r)).round()
        df = pd.DataFrame(
            {
                "date": dates.astype("datetime64[ns]"),
                "ticker": ticker,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": close,
                "volume": volume,
            }
        )
        df["ticker"] = df["ticker"].astype("string")
        return df.loc[listed].reset_index(drop=True)


def _rng_for(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, *[ord(c) for c in key]])
