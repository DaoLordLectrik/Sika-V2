"""
Backtesting engine - evaluates strategy performance on historical data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_data import get_historical_klines_binance
from scripts.indicators import compute_all_indicators
from scripts.signal_engine import generate_signal_from_row, get_trend_bias


def map_kraken_to_binance(symbol: str) -> str:
    """Map Kraken pair names to Binance pair names."""
    mapping = {
        "XBTUSD": "BTCUSDT",
        "PAXGUSD": "PAXGUSDT"
    }
    return mapping.get(symbol, symbol)


def run_backtest(df: pd.DataFrame, 
                sl_mult: float = 1.5, 
                tp_mult: float = 3.0,
                max_hold_candles: int = 96,  # 24 hours at 15m
                use_trend_filter: bool = False,
                pair: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Run backtest on historical data.
    
    Args:
        df: DataFrame with OHLC data
        sl_mult: Stop-loss ATR multiplier
        tp_mult: Take-profit ATR multiplier
        max_hold_candles: Maximum candles to hold a trade
        use_trend_filter: Whether to apply trend filter
        pair: Trading pair name
    
    Returns:
        Dictionary of backtest results
    """
    # Compute indicators
    df = compute_all_indicators(df)
    df.dropna(inplace=True)
    
    if df.empty:
        return {
            'trades': 0,
            'win_rate': 0,
            'avg_r': 0,
            'total_r': 0,
            'breakeven_win_rate': 0,
            'profit_factor': 0
        }
    
    # Backtest state
    in_trade = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    trade_count = 0
    wins = 0
    total_r = 0
    r_values = []
    
    # Iterate row by row (skip last row since we need future data for exit)
    for idx in range(len(df) - 1):
        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]
        
        # Get trend bias if using filter
        trend_bias = None
        if use_trend_filter:
            trend_bias = get_trend_bias(current_row)
        
        # Generate signal if not in a trade
        if not in_trade:
            signal = generate_signal_from_row(
                current_row,
                pair=pair,
                label=pair,
                timeframe="15m",
                sl_atr_multiplier=sl_mult,
                tp_atr_multiplier=tp_mult,
                trend_bias=trend_bias
            )
            
            if signal.direction != "HOLD":
                # Enter trade at close price
                entry_price = current_row['close']
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_time = idx
                in_trade = True
                trade_count += 1
        
        # Check exit conditions if in a trade
        if in_trade:
            high = next_row['high']
            low = next_row['low']
            
            # Check if TP or SL was hit
            hit_tp = False
            hit_sl = False
            
            if signal.direction == "BUY":
                # For BUY: TP above, SL below
                if high >= take_profit:
                    hit_tp = True
                if low <= stop_loss:
                    hit_sl = True
            else:  # SELL
                # For SELL: TP below, SL above
                if low <= take_profit:
                    hit_tp = True
                if high >= stop_loss:
                    hit_sl = True
            
            # Check timeout
            hold_candles = idx - entry_time
            timeout = hold_candles >= max_hold_candles
            
            if hit_tp or hit_sl or timeout:
                # Determine exit price
                if hit_tp and hit_sl:
                    # Both within same candle - conservative: assume loss first
                    exit_price = stop_loss
                    hit_tp = False  # Mark as loss
                elif hit_tp:
                    exit_price = take_profit
                elif hit_sl:
                    exit_price = stop_loss
                else:  # timeout
                    exit_price = next_row['close']  # Close at market
                
                # Calculate R-multiple
                risk = abs(entry_price - stop_loss)
                if risk > 0:
                    if signal.direction == "BUY":
                        r = (exit_price - entry_price) / risk
                    else:  # SELL
                        r = (entry_price - exit_price) / risk
                else:
                    r = 0
                
                # Record result
                r_values.append(r)
                total_r += r
                
                if (signal.direction == "BUY" and exit_price > entry_price) or \
                   (signal.direction == "SELL" and exit_price < entry_price):
                    wins += 1
                
                in_trade = False
    
    # Calculate metrics
    trades = len(r_values)
    win_rate = wins / trades if trades > 0 else 0
    
    # Calculate breakeven win rate
    # Breakeven when (win_rate * avg_win) - (loss_rate * avg_loss) = 0
    # For 1:1 R:R, breakeven = 0.5
    # For 1:N R:R, breakeven = 1/(N+1)
    avg_win = np.mean([r for r in r_values if r > 0]) if any(r > 0 for r in r_values) else 0
    avg_loss = abs(np.mean([r for r in r_values if r < 0])) if any(r < 0 for r in r_values) else 0
    
    if avg_win > 0 and avg_loss > 0:
        breakeven_win_rate = avg_loss / (avg_win + avg_loss)
    else:
        breakeven_win_rate = 0.5
    
    # Profit factor
    total_wins = sum(r for r in r_values if r > 0)
    total_losses = abs(sum(r for r in r_values if r < 0))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    return {
        'trades': trades,
        'win_rate': win_rate,
        'avg_r': np.mean(r_values) if r_values else 0,
        'total_r': total_r,
        'breakeven_win_rate': breakeven_win_rate,
        'profit_factor': profit_factor
    }


def backtest_with_compare(pair: str, lookback_days: int = 180, 
                         sl_mult: float = 1.5, tp_mult: float = 3.0):
    """
    Run backtest with and without trend filter for comparison.
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {pair} | {lookback_days} days | SL={sl_mult}xATR | TP={tp_mult}xATR")
    print(f"{'='*70}")
    print("\nDISCLAIMER: No fees/slippage modeled. SL wins same-candle ties.")
    print("This is a sanity check on the rules, not a promise of live performance.\n")
    
    # Fetch data
    binance_pair = map_kraken_to_binance(pair)
    print(f"Fetching {lookback_days} days of historical data for {binance_pair}...")
    df = get_historical_klines_binance(binance_pair, "15m", lookback_days)
    
    if df.empty:
        print(f"Error: No data returned for {binance_pair}")
        return
    
    print(f"Got {len(df)} candles\n")
    
    # Run backtest without trend filter
    print("Without trend filter:")
    results_no_filter = run_backtest(df, sl_mult, tp_mult, use_trend_filter=False, pair=pair)
    print_results(results_no_filter)
    
    # Run backtest with trend filter
    print("\nWith trend filter:")
    results_with_filter = run_backtest(df, sl_mult, tp_mult, use_trend_filter=True, pair=pair)
    print_results(results_with_filter)
    
    # Side-by-side comparison
    print("\n" + "-"*70)
    print("COMPARISON:")
    print("-"*70)
    print(f"Trades:           {results_no_filter['trades']:>6} vs {results_with_filter['trades']:>6}")
    print(f"Win Rate:         {results_no_filter['win_rate']*100:>5.1f}% vs {results_with_filter['win_rate']*100:>5.1f}%")
    print(f"Avg R-multiple:   {results_no_filter['avg_r']:>6.2f} vs {results_with_filter['avg_r']:>6.2f}")
    print(f"Total R:          {results_no_filter['total_r']:>6.2f} vs {results_with_filter['total_r']:>6.2f}")
    print(f"Profit Factor:    {results_no_filter['profit_factor']:>6.2f} vs {results_with_filter['profit_factor']:>6.2f}")
    
    # Improvement
    improvement = results_with_filter['total_r'] - results_no_filter['total_r']
    print(f"\nTrend filter delta: {improvement:+.2f} R")
    
    if improvement > 0:
        print("✅ Trend filter improves performance")
    else:
        print("❌ Trend filter degrades performance")


def print_results(results: Dict[str, Any]):
    """Pretty print backtest results."""
    print(f"  Trades: {results['trades']}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Breakeven Win Rate: {results['breakeven_win_rate']*100:.1f}%")
    print(f"  Avg R-multiple: {results['avg_r']:.2f}")
    print(f"  Total R: {results['total_r']:.2f}")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument("--pair", default="XBTUSD", help="Trading pair (Kraken symbol)")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    parser.add_argument("--sl", type=float, default=1.5, help="Stop-loss ATR multiplier")
    parser.add_argument("--tp", type=float, default=3.0, help="Take-profit ATR multiplier")
    parser.add_argument("--compare", action="store_true", help="Compare with/without trend filter")
    
    args = parser.parse_args()
    
    if args.compare:
        backtest_with_compare(args.pair, args.days, args.sl, args.tp)
    else:
        binance_pair = map_kraken_to_binance(args.pair)
        df = get_historical_klines_binance(binance_pair, "15m", args.days)
        if df.empty:
            print(f"Error: No data returned")
            return
        results = run_backtest(df, args.sl, args.tp, pair=args.pair)
        print_results(results)


if __name__ == "__main__":
    main()