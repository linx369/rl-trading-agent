"""Technical indicators, vectorised over (date x ticker) frames.

All functions accept a Series or a wide DataFrame and are column-wise, so they can be
tested on a single series and applied to the full universe in one call. The smoothing
conventions follow pandas_ta (which the original script used), so values match it:

* RSI and ATR use Wilder's moving average ``ewm(alpha=1/n, adjust=True, min_periods=n)``.
* EMA (for MACD) is seeded with the simple average of the first ``n`` observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Frame = pd.Series | pd.DataFrame

TRADING_DAYS_MONTH = 21
TRADING_DAYS_QUARTER = 63
TRADING_DAYS_HALF = 126
TRADING_DAYS_YEAR = 252


def _rma(x: Frame, n: int) -> Frame:
    """Wilder's moving average (as in pandas_ta)."""
    return x.ewm(alpha=1.0 / n, min_periods=n).mean()


def _ema_one(s: pd.Series, n: int) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if len(valid) < n:
        return out
    seeded = valid.copy()
    seeded.iloc[: n - 1] = np.nan
    seeded.iloc[n - 1] = valid.iloc[:n].mean()
    out.loc[valid.index] = seeded.ewm(span=n, adjust=False).mean()
    return out


def ema(x: Frame, n: int) -> Frame:
    """EMA seeded with the SMA of the first ``n`` points (pandas_ta ``presma`` behaviour)."""
    if isinstance(x, pd.Series):
        return _ema_one(x, n)
    return x.apply(_ema_one, n=n)


def rsi(close: Frame, n: int = 20) -> Frame:
    """Relative Strength Index on a 0–100 scale."""
    delta = close.diff()
    gain = _rma(delta.clip(lower=0), n)
    loss = _rma((-delta).clip(lower=0), n)
    return 100.0 * gain / (gain + loss)


def true_range(high: Frame, low: Frame, close: Frame) -> Frame:
    prev = close.shift(1)
    ranges = [high - low, (high - prev).abs(), (low - prev).abs()]
    if isinstance(high, pd.Series):
        tr = pd.concat(ranges, axis=1).max(axis=1, skipna=False)
    else:
        tr = np.maximum(np.maximum(ranges[0], ranges[1]), ranges[2])
    return tr.where(prev.notna())


def atr(high: Frame, low: Frame, close: Frame, n: int = 14) -> Frame:
    """Average True Range (Wilder)."""
    return _rma(true_range(high, low, close), n)


def macd(close: Frame, fast: int = 12, slow: int = 26) -> Frame:
    """MACD line: EMA(fast) - EMA(slow). The signal line is not used by the strategy."""
    return ema(close, fast) - ema(close, slow)


def skip1_momentum(
    close: Frame, lookback: int = TRADING_DAYS_YEAR, skip: int = TRADING_DAYS_MONTH
) -> Frame:
    """Jegadeesh–Titman 12-1 momentum: return from t-lookback to t-skip."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def high52_proximity(close: Frame, window: int = TRADING_DAYS_YEAR) -> Frame:
    """Price / rolling 52-week high, in [0, 1]."""
    hi = close.rolling(window).max()
    return (close / hi.replace(0, np.nan)).clip(0, 1)


def zscore_full(x: Frame) -> Frame:
    """Z-score using the *whole-sample* mean and std (look-ahead; ``tfg`` mode only)."""
    sd = x.std()
    sd = sd.replace(0, np.nan) if isinstance(sd, pd.Series) else (sd or np.nan)
    return (x - x.mean()) / sd


def zscore_rolling(x: Frame, window: int = TRADING_DAYS_YEAR, min_periods: int = 63) -> Frame:
    """Z-score using only past data (trailing window)."""
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return (x - mu) / sd


def rolling_normalise(x: Frame, window: int = 36, min_periods: int = 6, clip: float = 3.0) -> Frame:
    """Trailing z-score clipped to ±``clip``; std of 0 is treated as 1 (as in the original)."""
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std().replace(0, 1)
    return ((x - mu) / sd).clip(-clip, clip)
