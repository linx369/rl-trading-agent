"""Point-in-time S&P 500 membership, to reduce survivorship bias.

Source: https://github.com/fja05680/sp500 (MIT licence), file ``sp500_ticker_start_end.csv``,
one row per membership spell: ``ticker, start_date, end_date`` (empty end = still a member).
The URL in the config is pinned to a commit so results are reproducible.

Two adjustments are needed before the spells can be matched to Yahoo prices:

* **Share-class tickers** use a dot in the source (``BF.B``) and a dash on Yahoo (``BF-B``).
* **Ticker changes.** When a company changes symbol, the source ends the old spell and starts
  a new one on the same day, while Yahoo files the whole price history under the new symbol.
  Without a fix, Meta (FB → META in June 2022) would look like it joined the index in 2022.
  :data:`RENAMES` maps old → new symbols; a pair is only applied if the source confirms it,
  i.e. the old spell ends on exactly the day the new one starts.

What this does **not** fix: companies that were delisted (bankruptcies, acquisitions) have no
Yahoo prices, so they still cannot be bought in the backtest. That needs a paid database such
as CRSP.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from regime_rl.data.universe import normalise_ticker

log = logging.getLogger(__name__)

# Old symbol -> current symbol, for ticker changes of companies in the index since 2006.
RENAMES: dict[str, str] = {
    "FB": "META",  # Facebook -> Meta, 2022
    "ANTM": "ELV",  # Anthem -> Elevance Health, 2022
    "ABC": "COR",  # AmerisourceBergen -> Cencora, 2023
    "FLT": "CPAY",  # FleetCor -> Corpay, 2024
    "PEAK": "DOC",  # Healthpeak, 2024
    "RE": "EG",  # Everest Re -> Everest Group, 2023
    "NLOK": "GEN",  # NortonLifeLock -> Gen Digital, 2022
    "TMK": "GL",  # Torchmark -> Globe Life, 2019
    "JEC": "J",  # Jacobs, 2019
    "HRS": "LHX",  # Harris -> L3Harris, 2019
    "PKI": "RVTY",  # PerkinElmer -> Revvity, 2023
    "WLTW": "WTW",  # Willis Towers Watson, 2022
    "BLL": "BALL",  # Ball Corp, 2022
    "BHGE": "BKR",  # Baker Hughes, 2019
    "BK": "BNY",  # BNY Mellon, 2026
    "MMC": "MRSH",  # Marsh McLennan, 2026
    "DISCA": "WBD",  # Discovery -> Warner Bros. Discovery, 2022
    "PARA": "PSKY",  # Paramount -> Paramount Skydance, 2025
    "KRFT": "KHC",  # Kraft -> Kraft Heinz, 2015
}

COLUMNS = ["ticker", "start", "end"]


def parse_spells(text: str) -> pd.DataFrame:
    """Parse the raw CSV into ``ticker, start, end`` (end is NaT for current members)."""
    raw = pd.read_csv(io.StringIO(text))
    expected = {"ticker", "start_date", "end_date"}
    if not expected <= set(raw.columns):
        raise ValueError(f"membership file must have columns {sorted(expected)}")
    return pd.DataFrame(
        {
            "ticker": raw["ticker"].map(normalise_ticker),
            "start": pd.to_datetime(raw["start_date"]),
            "end": pd.to_datetime(raw["end_date"]),
        }
    )


def verified_renames(spells: pd.DataFrame, renames: dict[str, str] = RENAMES) -> dict[str, str]:
    """Keep only renames the data confirms: old spell ends the day the new one starts."""
    ok: dict[str, str] = {}
    for old, new in renames.items():
        o, n = normalise_ticker(old), normalise_ticker(new)
        old_end = spells.loc[spells.ticker == o, "end"].max()
        new_start = spells.loc[spells.ticker == n, "start"].min()
        if pd.notna(old_end) and pd.notna(new_start) and old_end == new_start:
            ok[o] = n
        else:
            log.warning("rename %s -> %s not confirmed by the membership data; ignored", o, n)
    return ok


def apply_renames(spells: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    """Relabel old symbols and merge back-to-back spells of the same company."""
    out = spells.assign(ticker=spells["ticker"].replace(renames)).sort_values(["ticker", "start"])
    merged = []
    for ticker, g in out.groupby("ticker", sort=True):
        cur_start, cur_end = None, None
        for row in g.itertuples():
            if cur_start is None:
                cur_start, cur_end = row.start, row.end
            elif pd.notna(cur_end) and row.start <= cur_end:
                cur_end = row.end if pd.isna(row.end) else max(cur_end, row.end)
            else:
                merged.append((ticker, cur_start, cur_end))
                cur_start, cur_end = row.start, row.end
        merged.append((ticker, cur_start, cur_end))
    return pd.DataFrame(merged, columns=COLUMNS)


def load_membership(text: str) -> pd.DataFrame:
    spells = parse_spells(text)
    return apply_renames(spells, verified_renames(spells))


def fetch_membership(url: str) -> pd.DataFrame:
    import requests

    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 regime-rl"})
    r.raise_for_status()
    return load_membership(r.text)


def members_since(spells: pd.DataFrame, start: str) -> list[str]:
    """Every ticker that was in the index at any time on or after ``start``."""
    live = spells["end"].isna() | (spells["end"] >= pd.Timestamp(start))
    return sorted(spells.loc[live, "ticker"].unique())


def membership_matrix(spells: pd.DataFrame, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Boolean (date x ticker): True if the ticker was an index member on that date.

    A spell covers ``start <= date < end`` (the end date is the day it left).
    """
    tickers = list(tickers)
    out = pd.DataFrame(False, index=dates, columns=tickers)
    for row in spells[spells.ticker.isin(tickers)].itertuples():
        after = dates >= row.start
        before = dates < row.end if pd.notna(row.end) else True
        out.loc[after & before, row.ticker] = True
    return out
