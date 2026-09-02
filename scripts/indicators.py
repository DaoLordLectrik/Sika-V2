"""
Technical indicators module using pandas/numpy.
Pure pandas implementation - no exotic dependencies.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Uses pandas ewm with adjust=False for Wilder smoothing.
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index with Wilder smoothing.
    
    Args:
        series: Price series (typically close)
        period: RSI period (default 14)
    
    Returns:
        RSI values between 0 and 100
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder smoothing uses alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    This is the first indicator that produces NaN values (from rolling).
    
    Args:
        series: Price series
        period: MA period
        num_std: Number of standard deviations
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return middle, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range with Wilder smoothing.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        ATR values
    """
    # Calculate true range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder smoothing (alpha = 1/period)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    
    return atr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index.
    Measures trend strength regardless of direction.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
    
    Returns:
        ADX values (0-100, higher = stronger trend)
    """
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = pd.Series(index=high.index, dtype=float)
    minus_dm = pd.Series(index=high.index, dtype=float)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    plus_dm.fillna(0, inplace=True)
    minus_dm.fillna(0, inplace=True)
    
    # Smooth with Wilder's method
    atr_period = atr(high, low, close, period)
    
    # Avoid division by zero
    atr_period = atr_period.replace(0, 0.001)
    
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_period
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_period
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and enrich the DataFrame.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        DataFrame with all indicators added as columns
    
    Note:
        This function assumes the DataFrame is indexed by time (datetime).
        The Bollinger Band rolling window produces NaN rows that need dropping.
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # EMAs
    df['ema_9'] = ema(close, 9)
    df['ema_21'] = ema(close, 21)
    df['ema_50'] = ema(close, 50)
    
    # RSI
    df['rsi_14'] = rsi(close, 14)
    
    # MACD
    df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(close, 12, 26, 9)
    
    # Bollinger Bands (this produces NaN rows)
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = bollinger_bands(close, 20, 2.0)
    
    # ATR
    df['atr_14'] = atr(high, low, close, 14)
    
    # ADX (trend strength)
    df['adx_14'] = adx(high, low, close, 14)
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 0.001)
    
    # ATR Percentile (rolling window)
    df['atr_percentile'] = df['atr_14'].rolling(window=50, min_periods=20).rank(pct=True)
    df['atr_percentile'] = df['atr_percentile'] * 100  # Convert to percentage
    
    return df