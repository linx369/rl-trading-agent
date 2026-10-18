"""Investment-universe resolution.

For the US market the *current* S&P 500 constituents are fetched from the first source that
works (yfinance -> Wikipedia -> a GitHub CSV). Note: this is a present-day snapshot, so any
backtest built on it carries survivorship bias; the snapshot date is stored alongside the list.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Sequence

import pandas as pd

from regime_rl.markets import Market

log = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB_CSV = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
MIN_CONSTITUENTS = 400


def normalise_ticker(symbol: str) -> str:
    """Yahoo uses '-' for share classes (BRK.B -> BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def _from_yfinance() -> list[str]:
    import yfinance as yf

    fn = getattr(yf, "tickers_sp500", None)
    return list(fn()) if callable(fn) else []


def _from_wikipedia() -> list[str]:
    import requests

    r = requests.get(WIKI_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0 regime-rl"})
    r.raise_for_status()
    for table in pd.read_html(io.StringIO(r.text)):
        col = next(
            (c for c in table.columns if "symbol" in str(c).lower() or "ticker" in str(c).lower()),
            None,
        )
        if col is not None:
            return table[col].astype(str).tolist()
    return []


def _from_github() -> list[str]:
    import requests

    r = requests.get(GITHUB_CSV, timeout=30, headers={"User-Agent": "Mozilla/5.0 regime-rl"})
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))["Symbol"].astype(str).tolist()


DEFAULT_SP500_SOURCES: tuple[tuple[str, Callable[[], list[str]]], ...] = (
    ("yfinance", _from_yfinance),
    ("wikipedia", _from_wikipedia),
    ("github", _from_github),
)


def fetch_sp500(
    sources: Sequence[tuple[str, Callable[[], list[str]]]] = DEFAULT_SP500_SOURCES,
) -> tuple[list[str], str]:
    """Return ``(tickers, source_name)`` from the first source with >= 400 symbols."""
    errors = []
    for name, fn in sources:
        try:
            syms = sorted({normalise_ticker(s) for s in fn() if isinstance(s, str) and s})
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if len(syms) >= MIN_CONSTITUENTS:
            log.info("S&P 500 constituents from %s (%d tickers)", name, len(syms))
            return syms, name
        errors.append(f"{name}: only {len(syms)} symbols")
    raise RuntimeError("Could not resolve S&P 500 constituents: " + "; ".join(errors))


def resolve_universe(market: Market, synthetic: bool = False) -> tuple[list[str], str]:
    """Tickers to download for ``market``.

    With ``synthetic=True`` the US universe is the sector-map members (no network needed).
    """
    if market.tickers is not None:
        return list(market.tickers), "curated"
    if synthetic:
        members = sorted({t for ts in market.sector_map.values() for t in ts})
        return members, "sector_map"
    return fetch_sp500()
