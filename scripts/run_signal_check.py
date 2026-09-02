"""
Main entry point - runs signal checks for all configured pairs.
"""

import os
import sys
import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Get the absolute path to the project root (Sika-V2)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Add the project root to Python path so we can import scripts modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import from scripts
from scripts.fetch_data import get_klines, get_current_price
from scripts.indicators import compute_all_indicators
from scripts.signal_engine import generate_signal, get_trend_bias, format_signal_telegram
from scripts.telegram_alert import send_telegram_message


# Constants
DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"
LATEST_SIGNALS = DATA_DIR / "latest_signals.json"
ALERT_STATE = DATA_DIR / "alert_state.json"


def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    config_path = PROJECT_ROOT / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_alert_state() -> Dict[str, Dict[str, Any]]:
    """Load alert deduplication state."""
    if ALERT_STATE.exists():
        with open(ALERT_STATE, 'r') as f:
            return json.load(f)
    return {}


def save_alert_state(state: Dict[str, Dict[str, Any]]):
    """Save alert deduplication state."""
    with open(ALERT_STATE, 'w') as f:
        json.dump(state, f, indent=2)


def log_signal(signal: Dict[str, Any], alerted: bool = False):
    """Log signal to CSV and latest JSON."""
    ensure_data_dir()
    
    # Add alerted flag and timestamp
    signal['alerted'] = alerted
    signal['log_time'] = datetime.now().isoformat()
    
    # Append to CSV
    file_exists = SIGNALS_LOG.exists()
    
    with open(SIGNALS_LOG, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=signal.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(signal)
    
    # Update latest signals JSON (overwrite)
    with open(LATEST_SIGNALS, 'w') as f:
        json.dump(signal, f, indent=2)


def check_market_hours(pair_config: Dict[str, Any], now_utc: datetime) -> bool:
    """
    Check if a market is open for trading.
    
    Args:
        pair_config: Pair configuration
        now_utc: Current UTC time
    
    Returns:
        True if market is open, False otherwise
    """
    # Only check if market hours are respected
    if not pair_config.get('respect_market_hours', False):
        return True
    
    # Gold market hours (example: closes weekends, with buffer)
    # Sunday 10pm UTC close - Friday 10pm UTC open
    # With 1-hour buffer on each side
    weekday = now_utc.weekday()  # Monday=0, Sunday=6
    hour = now_utc.hour
    minute = now_utc.minute
    
    # Market closes at 22:00 UTC on Friday, opens at 22:00 UTC on Sunday
    # With buffer: closes at 21:00 UTC Friday, opens at 23:00 UTC Sunday
    
    # Friday after 21:00 UTC -> closed
    if weekday == 4 and (hour > 21 or (hour == 21 and minute >= 0)):
        return False
    
    # Saturday -> closed all day
    if weekday == 5:
        return False
    
    # Sunday before 23:00 UTC -> closed
    if weekday == 6 and (hour < 23 or (hour == 23 and minute == 0)):
        return False
    
    return True


def get_higher_timeframe_data(pair: str, interval: str, lookback: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch higher timeframe data for trend filter.
    
    Args:
        pair: Trading pair
        interval: Higher timeframe interval
        lookback: Number of candles to fetch
    
    Returns:
        DataFrame with indicators, or None if error
    """
    try:
        df = get_klines(pair, interval, limit=lookback)
        if df.empty:
            return None
        df = compute_all_indicators(df)
        return df
    except Exception as e:
        print(f"Error fetching higher timeframe data for {pair}: {e}")
        return None


def run_signal_check():
    """Main signal check routine."""
    print(f"{'='*60}")
    print(f"Signal Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")
    
    # Load configuration
    config = load_config()
    pairs = config.get('pairs', [])
    min_confidence = config.get('min_confidence_to_alert', 75)
    sl_mult = config.get('sl_atr_multiplier', 1.5)
    tp_mult = config.get('tp_atr_multiplier', 4.0)
    trend_filter_enabled = config.get('trend_filter_enabled', True)
    trend_interval = config.get('trend_filter_interval', '1h')
    
    if not pairs:
        print("No pairs configured. Exiting.")
        return
    
    # Load alert state for deduplication
    alert_state = load_alert_state()
    
    now_utc = datetime.now()
    results = []
    alerts_sent = 0
    
    for pair_config in pairs:
        symbol = pair_config['symbol']
        label = pair_config.get('label', symbol)
        interval = pair_config.get('interval', '15m')
        
        print(f"\n--- {label} ({symbol}) ---")
        
        # Check market hours (if applicable)
        if not check_market_hours(pair_config, now_utc):
            print(f"  Market closed - skipping analysis (will still log if data available)")
            market_open = False
        else:
            market_open = True
        
        # Fetch live data
        try:
            df = get_klines(symbol, interval, limit=200)
            if df.empty:
                print(f"  No data received for {symbol}")
                continue
            
            # Compute indicators
            df = compute_all_indicators(df)
            
            # Drop NaN rows from Bollinger Bands
            df.dropna(inplace=True)
            
            if df.empty:
                print(f"  Not enough data to compute indicators")
                continue
            
            # Get trend bias from higher timeframe if enabled
            trend_bias = None
            if trend_filter_enabled:
                df_htf = get_higher_timeframe_data(symbol, trend_interval, lookback=100)
                if df_htf is not None and not df_htf.empty:
                    # Use the last row for trend bias
                    last_row = df_htf.iloc[-1]
                    trend_bias = get_trend_bias(last_row)
                    print(f"  Trend filter: {trend_bias}")
                else:
                    print(f"  Trend filter: unavailable (using no filter)")
            
            # Generate signal
            signal = generate_signal(
                df=df,
                pair=symbol,
                label=label,
                timeframe=interval,
                sl_atr_multiplier=sl_mult,
                tp_atr_multiplier=tp_mult,
                trend_bias=trend_bias
            )
            
            print(f"  Direction: {signal.direction}")
            print(f"  Confidence: {signal.confidence}%")
            print(f"  Price: ${signal.price:.2f}")
            
            # Log the signal (always)
            signal_dict = signal.to_dict()
            log_signal(signal_dict, alerted=False)
            
            # Determine if we should send an alert
            should_alert = False
            alert_reason = ""
            
            if signal.direction == "HOLD":
                should_alert = False
                alert_reason = "HOLD signal"
            elif signal.confidence < min_confidence:
                should_alert = False
                alert_reason = f"Confidence {signal.confidence}% < {min_confidence}%"
            elif not market_open:
                should_alert = False
                alert_reason = "Market closed"
            else:
                # Check deduplication
                last_alerted_dir = alert_state.get(symbol, {}).get('last_alerted_direction')
                last_alert_time = alert_state.get(symbol, {}).get('last_alert_time')
                
                if last_alerted_dir is None:
                    # No previous alert, send it
                    should_alert = True
                    alert_reason = "First alert for this pair"
                elif last_alerted_dir != signal.direction:
                    # Direction changed, send it
                    should_alert = True
                    alert_reason = f"Direction changed from {last_alerted_dir} to {signal.direction}"
                else:
                    # Same direction as last alert, don't send
                    should_alert = False
                    alert_reason = f"Duplicate {signal.direction} (last alerted {last_alert_time})"
            
            # Send alert if eligible
            if should_alert:
                # Format and send Telegram message
                telegram_text = format_signal_telegram(signal)
                success = send_telegram_message(telegram_text, parse_mode="Markdown")
                
                if success:
                    alerts_sent += 1
                    print(f"  ✅ Alert sent - {alert_reason}")
                    
                    # Update alert state
                    alert_state[symbol] = {
                        'last_alerted_direction': signal.direction,
                        'last_alert_time': now_utc.isoformat()
                    }
                    save_alert_state(alert_state)
                    
                    # Update log with alerted flag
                    signal_dict['alerted'] = True
                    signal_dict['alert_time'] = now_utc.isoformat()
                    log_signal(signal_dict, alerted=True)
                else:
                    print(f"  ❌ Alert failed to send")
            else:
                print(f"  ⏭️  Alert suppressed: {alert_reason}")
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Run complete - {alerts_sent} alert(s) sent")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_signal_check()