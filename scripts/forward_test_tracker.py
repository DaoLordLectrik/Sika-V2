"""
Forward-testing tracker - evaluates real signals against actual outcomes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_data import get_historical_klines_binance
from scripts.backtest import map_kraken_to_binance


DATA_DIR = Path(__file__).parent.parent / "data"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"
FORWARD_TEST_RESULTS = DATA_DIR / "forward_test_results.csv"
CHECKED_SIGNALS = DATA_DIR / "checked_signals.json"


def load_checked_signals() -> set:
    """Load set of already-checked signal timestamps."""
    if CHECKED_SIGNALS.exists():
        with open(CHECKED_SIGNALS, 'r') as f:
            return set(json.load(f))
    return set()


def save_checked_signals(checked: set):
    """Save checked signal timestamps."""
    with open(CHECKED_SIGNALS, 'w') as f:
        json.dump(list(checked), f)


def forward_test(pair: str, max_hold_hours: int = 24):
    """
    Evaluate real signals against actual forward outcomes.
    
    Args:
        pair: Trading pair to evaluate
        max_hold_hours: Maximum hours to hold a trade
    """
    print(f"\n{'='*70}")
    print(f"FORWARD TEST: {pair}")
    print(f"{'='*70}\n")
    
    # Load signal log
    if not SIGNALS_LOG.exists():
        print("No signal log found.")
        return
    
    df_signals = pd.read_csv(SIGNALS_LOG)
    df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
    df_signals = df_signals[df_signals['pair'] == pair]
    df_signals = df_signals[df_signals['direction'] != 'HOLD']
    df_signals = df_signals[df_signals['alerted'] == True]
    
    if df_signals.empty:
        print(f"No BUY/SELL signals found for {pair}")
        return
    
    # Load already checked
    checked = load_checked_signals()
    
    # Filter to unprocessed signals
    df_signals['checked'] = df_signals['timestamp'].astype(str).isin(checked)
    df_unchecked = df_signals[~df_signals['checked']]
    
    if df_unchecked.empty:
        print("All signals already checked.")
        return
    
    print(f"Found {len(df_unchecked)} new signals to check")
    
    # Fetch historical data for evaluation
    binance_pair = map_kraken_to_binance(pair)
    lookback_days = 30  # Enough to cover recent signals
    df_hist = get_historical_klines_binance(binance_pair, "15m", lookback_days)
    
    if df_hist.empty:
        print("Error: No historical data available")
        return
    
    # Evaluate each signal
    results = []
    max_hold_candles = int(max_hold_hours * 4)  # 15m candles
    
    for _, signal in df_unchecked.iterrows():
        signal_time = signal['timestamp']
        entry_price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        direction = signal['direction']
        
        # Find future candles
        future = df_hist[df_hist.index > signal_time]
        
        if future.empty:
            print(f"  No future data for signal at {signal_time}")
            continue
        
        # Check outcome
        outcome = None
        exit_price = None
        exit_reason = None
        hold_candles = 0
        
        for idx, (time, row) in enumerate(future.iterrows()):
            if idx >= max_hold_candles:
                exit_price = row['close']
                exit_reason = "timeout"
                break
            
            high = row['high']
            low = row['low']
            hold_candles += 1
            
            if direction == "BUY":
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    break
                if high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                    break
            else:  # SELL
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    break
                if low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                    break
        
        # If we didn't exit (shouldn't happen with timeout), use last price
        if exit_price is None:
            exit_price = future.iloc[-1]['close']
            exit_reason = "timeout"
        
        # Calculate result
        risk = abs(entry_price - stop_loss)
        if risk > 0:
            if direction == "BUY":
                r = (exit_price - entry_price) / risk
            else:
                r = (entry_price - exit_price) / risk
        else:
            r = 0
        
        win = (direction == "BUY" and exit_price > entry_price) or \
              (direction == "SELL" and exit_price < entry_price)
        
        # Record result
        results.append({
            'timestamp': signal_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'r_multiple': r,
            'win': win,
            'exit_reason': exit_reason,
            'hold_candles': hold_candles
        })
        
        checked.add(signal_time.isoformat())
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        
        # Append to CSV
        file_exists = FORWARD_TEST_RESULTS.exists()
        df_results.to_csv(FORWARD_TEST_RESULTS, mode='a', header=not file_exists, index=False)
        
        # Save checked signals
        save_checked_signals(checked)
        
        # Print summary
        print("\n" + "="*70)
        print("FORWARD TEST RESULTS")
        print("="*70)
        print(f"Signals evaluated: {len(results)}")
        print(f"Wins: {sum(r['win'] for r in results)}")
        print(f"Win Rate: {sum(r['win'] for r in results) / len(results) * 100:.1f}%")
        print(f"Total R: {sum(r['r_multiple'] for r in results):.2f}")
        print(f"Avg R: {np.mean([r['r_multiple'] for r in results]):.2f}")
        
        exit_reasons = {}
        for r in results:
            exit_reasons[r['exit_reason']] = exit_reasons.get(r['exit_reason'], 0) + 1
        print(f"\nExit reasons: {exit_reasons}")
    else:
        print("No signals could be evaluated")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Forward-test signals")
    parser.add_argument("--pair", default="XBTUSD", help="Trading pair")
    parser.add_argument("--max-hold-hours", type=int, default=24, help="Max hold time")
    args = parser.parse_args()
    forward_test(args.pair, args.max_hold_hours)


if __name__ == "__main__":
    main()