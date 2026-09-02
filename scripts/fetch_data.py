"""
Data fetching module for Kraken (live) and Binance (historical).
Implements dual-source strategy per specification section 2.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional


def get_klines(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch live OHLC data from Kraken public API.
    
    Args:
        symbol: Trading pair (e.g., "XBTUSD" - note Kraken uses XBT not BTC)
        interval: Timeframe (e.g., "15m", "1h")
        limit: Number of candles to fetch (max 720)
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Indexed by close_time (UTC)
    
    Note:
        Kraken caps at ~720 recent candles regardless of since parameter.
        This is intentional for live use; see get_historical_klines_binance()
        for deep historical data.
    """
    # Kraken OHLC endpoint
    url = "https://api.kraken.com/0/public/OHLC"
    
    # Map interval to Kraken's format
    interval_map = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
    }
    
    params = {
        "pair": symbol,
        "interval": interval_map.get(interval, 15)
    }
    
    # If limit is specified, calculate since timestamp
    if limit:
        # Estimate time range based on interval
        interval_minutes = interval_map.get(interval, 15)
        # Add a buffer of 10 extra candles to ensure we get enough
        params["since"] = int((datetime.now() - timedelta(minutes=(limit + 10) * interval_minutes)).timestamp())
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data and data["error"]:
            raise ValueError(f"Kraken API error: {data['error']}")
        
        # Kraken returns OHLC data in the 'result' key with the pair name
        result_key = list(data["result"].keys())[0]  # Usually the pair name
        ohlc_data = data["result"][result_key]
        
        df = pd.DataFrame(ohlc_data, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        
        # Convert timestamp to datetime and set as index
        df["close_time"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("close_time", inplace=True)
        
        # Convert string prices to float
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Keep only needed columns
        df = df[["open", "high", "low", "close", "volume"]]
        
        # Sort by time ascending
        df.sort_index(inplace=True)
        
        # If limit is specified, only return the most recent N candles
        if limit and len(df) > limit:
            df = df.tail(limit)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Kraken: {e}")
        # Return empty DataFrame on error
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def get_historical_klines_kraken(symbol: str, interval: str = "15m", lookback_days: int = 180) -> pd.DataFrame:
    """
    Fetch historical data from Kraken with pagination.
    
    WARNING: This caps at ~720 candles regardless of lookback_days.
    The 720-candle limit is a hard Kraken API constraint.
    For deep historical data, use get_historical_klines_binance() instead.
    
    Args:
        symbol: Trading pair
        interval: Timeframe
        lookback_days: Requested days (will be truncated to ~720 candles max)
    
    Returns:
        DataFrame with same format as get_klines()
    """
    # Calculate max candles Kraken can return
    interval_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
    }.get(interval, 15)
    
    max_candles = 720
    max_minutes = max_candles * interval_minutes
    max_days = max_minutes / (24 * 60)
    
    print(f"Warning: Kraken historical data capped at {max_candles} candles (~{max_days:.1f} days)")
    
    # Just use the regular get_klines with a large limit
    return get_klines(symbol, interval, limit=max_candles)


def get_historical_klines_binance(symbol: str, interval: str = "15m", lookback_days: int = 180) -> pd.DataFrame:
    """
    Fetch deep historical data from Binance public API.
    
    This paginates through history cleanly. For local use only due to Binance
    US-IP restrictions on GitHub Actions (see spec section 2.2).
    
    Args:
        symbol: Trading pair (Binance uses "BTCUSDT", not "XBTUSD")
        interval: Timeframe
        lookback_days: Number of days of historical data to fetch
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Indexed by close_time (UTC)
    
    Note:
        This function is intended for local backtesting only.
        It will NOT work from GitHub Actions US IP addresses.
    """
    url = "https://api.binance.com/api/v3/klines"
    
    # Map interval to Binance format
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
    }
    
    all_klines = []
    start_time = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval_map.get(interval, "15m"),
            "startTime": start_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_klines.extend(data)
            
            # Get the last timestamp for the next request
            last_timestamp = data[-1][0]
            start_time = last_timestamp + 1  # Avoid duplicate
            
            # If we got fewer than 1000, we've reached the end
            if len(data) < 1000:
                break
                
            time.sleep(0.5)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Binance: {e}")
            break
    
    if not all_klines:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    
    # Parse the data
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # Convert timestamps
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("close_time", inplace=True)
    
    # Convert to float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Keep only needed columns
    df = df[["open", "high", "low", "close", "volume"]]
    
    # Sort by time ascending
    df.sort_index(inplace=True)
    
    return df


def get_current_price(symbol: str) -> Optional[float]:
    """
    Fetch current price from Kraken ticker endpoint.
    
    Args:
        symbol: Trading pair (e.g., "XBTUSD")
    
    Returns:
        Current price as float, or None if error
    """
    url = "https://api.kraken.com/0/public/Ticker"
    
    params = {"pair": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data and data["error"]:
            raise ValueError(f"Kraken API error: {data['error']}")
        
        result_key = list(data["result"].keys())[0]
        ticker = data["result"][result_key]
        
        return float(ticker["c"][0])  # 'c' is last trade price
        
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        print(f"Error fetching current price: {e}")
        return None