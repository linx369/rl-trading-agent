import numpy as np
import pandas as pd
import pytest

from regime_rl.config import Config
from regime_rl.data.ingest import ingest, plan_fetches
from regime_rl.data.sources import (
    PRICE_COLUMNS,
    SyntheticSource,
    YahooSource,
    yf_frame_to_long,
)
from regime_rl.data.store import MarketStore, validate_prices
from regime_rl.data.universe import fetch_sp500, normalise_ticker, resolve_universe


def rows(**overrides) -> dict:
    base = {
        "date": pd.Timestamp("2024-01-02"),
        "ticker": "AAA",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.4,
        "volume": 1000.0,
    }
    return {**base, **overrides}


# ----------------------------------------------------------------- validation
def test_validation_drops_impossible_quotes():
    df = pd.DataFrame(
        [
            rows(),
            rows(ticker="NEG", close=-1.0),
            rows(ticker="HL", high=8.0),
            rows(ticker="VOL", volume=-5.0),
            rows(ticker="NAN", adj_close=np.nan),
        ]
    )
    clean, rep = validate_prices(df)
    assert clean["ticker"].tolist() == ["AAA"]
    assert (rep.non_positive_price, rep.high_below_low, rep.negative_volume) == (2, 1, 1)
    assert rep.dropped == 4


def test_validation_deduplicates_keeping_latest():
    df = pd.DataFrame([rows(close=1.0), rows(close=2.0)])
    clean, rep = validate_prices(df)
    assert len(clean) == 1 and clean["close"].iloc[0] == 2.0 and rep.duplicates == 1


def test_validation_requires_schema():
    with pytest.raises(ValueError, match="missing"):
        validate_prices(pd.DataFrame({"date": [], "ticker": []}))


# ----------------------------------------------------------------- store
def test_upsert_and_sql(tmp_path):
    store = MarketStore(tmp_path, "us")
    store.upsert_prices(pd.DataFrame([rows(), rows(ticker="BBB")]))
    store.upsert_prices(pd.DataFrame([rows(close=99.0), rows(date=pd.Timestamp("2024-01-03"))]))
    df = store.read_table("prices")
    assert len(df) == 3
    assert df.loc[(df.ticker == "AAA") & (df.date == "2024-01-02"), "close"].item() == 99.0
    n = store.sql("SELECT count(DISTINCT ticker) AS n FROM prices")["n"].item()
    assert n == 2
    assert store.last_dates()["AAA"] == pd.Timestamp("2024-01-03")
    assert store.dir.name == "US"


def test_load_prices_filters_in_sql(tmp_path):
    store = MarketStore(tmp_path, "US")
    store.upsert_prices(
        pd.DataFrame(
            [
                rows(date=pd.Timestamp(d), ticker=t)
                for d in ("2024-01-02", "2024-01-03", "2024-01-04")
                for t in ("A", "B", "C")
            ]
        )
    )
    out = store.load_prices(tickers=["A", "C"], start="2024-01-03", end="2024-01-03")
    assert sorted(out["ticker"]) == ["A", "C"] and out["date"].nunique() == 1
    wide = store.wide("close", tickers=["B"])
    assert list(wide.columns) == ["B"] and len(wide) == 3


def test_missing_table_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="regime-rl ingest"):
        MarketStore(tmp_path, "US").read_table("prices")


def test_write_table_is_atomic(tmp_path):
    store = MarketStore(tmp_path, "US")
    store.write_table("runs", pd.DataFrame({"x": [1]}))
    store.append_table("runs", pd.DataFrame({"x": [2]}))
    assert store.read_table("runs")["x"].tolist() == [1, 2]
    assert not list(store.dir.glob("*.tmp"))


# ----------------------------------------------------------------- ingestion
def test_plan_fetches():
    last = pd.Series({"OLD": pd.Timestamp("2024-06-28"), "DONE": pd.Timestamp("2024-12-31")})
    plan = plan_fetches(["NEW", "OLD", "DONE"], last, "2020-01-01", "2024-12-31")
    assert plan == {"2020-01-01": ["NEW"], "2024-06-29": ["OLD"]}


def test_ingest_is_incremental(tmp_path):
    cfg = Config(data_dir=str(tmp_path), start="2015-01-01", end="2026-03-31")
    src = SyntheticSource(sector_map=cfg.market.sector_map)
    store = MarketStore(tmp_path, "US")
    universe, _ = resolve_universe(cfg.market, synthetic=True)
    first = ingest(cfg.replace(end="2026-02-27"), src, store)
    assert first["tickers_fetched"].tolist() == [len(universe), len(cfg.market.all_etfs)]
    second = ingest(cfg, src, store)
    assert second.loc[0, "rows_fetched"] < first.loc[0, "rows_fetched"] / 50  # only a month
    third = ingest(cfg, src, store)
    assert third["tickers_fetched"].sum() == 0
    assert len(store.read_table("ingest_log")) == 6
    # incremental result identical to a single full download
    full = SyntheticSource(sector_map=cfg.market.sector_map).fetch(universe, cfg.start, cfg.end)
    stored = store.read_table("prices")
    pd.testing.assert_frame_equal(
        stored.sort_values(["date", "ticker"]).reset_index(drop=True)[PRICE_COLUMNS],
        full.sort_values(["date", "ticker"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_synthetic_source_is_deterministic_and_valid():
    a = SyntheticSource(seed=1).fetch(["X", "SPY"], "2020-01-01", "2021-01-01")
    b = SyntheticSource(seed=1).fetch(["X", "SPY"], "2020-01-01", "2021-01-01")
    pd.testing.assert_frame_equal(a, b)
    _, rep = validate_prices(a)
    assert rep.dropped == 0


# ----------------------------------------------------------------- yahoo adapter
def test_yf_multiindex_to_long():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    cols = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["AAA", "BBB"]]
    )
    raw = pd.DataFrame(np.arange(24, dtype=float).reshape(2, 12) + 1, index=idx, columns=cols)
    raw.loc[idx[1], ("Adj Close", "BBB")] = np.nan
    out = yf_frame_to_long(raw, ["AAA", "BBB"])
    assert list(out.columns) == PRICE_COLUMNS
    assert len(out) == 3  # missing adj_close row dropped


def test_yahoo_source_retries_missing_tickers():
    calls = []

    def fake(tickers, start, end):
        calls.append(tickers)
        tks = [tickers] if isinstance(tickers, str) else tickers
        idx = pd.to_datetime(["2024-01-02"])
        present = [t for t in tks if t != "FLAKY" or isinstance(tickers, str)]
        cols = pd.MultiIndex.from_product(
            [["Adj Close", "Close", "High", "Low", "Open", "Volume"], present]
        )
        return pd.DataFrame(1.0, index=idx, columns=cols)

    out = YahooSource(pause=0, downloader=fake).fetch(["AAA", "FLAKY"], "2024-01-01", "2024-01-02")
    assert sorted(out["ticker"]) == ["AAA", "FLAKY"]
    assert calls[-1] == "FLAKY"


def test_yahoo_source_survives_exceptions():
    def boom(*_):
        raise ConnectionError("no network")

    assert YahooSource(pause=0, downloader=boom).fetch(["A"], "2024-01-01", "2024-01-02").empty


# ----------------------------------------------------------------- universe
def test_normalise_ticker():
    assert normalise_ticker(" brk.b ") == "BRK-B"


def test_fetch_sp500_falls_back():
    def broken():
        raise OSError("down")

    names = [f"T{i}" for i in range(450)]
    syms, src = fetch_sp500((("a", broken), ("b", lambda: names[:10]), ("c", lambda: names)))
    assert src == "c" and len(syms) == 450
    with pytest.raises(RuntimeError, match="a: down"):
        fetch_sp500((("a", broken),))
