"""
Optimize stop-loss and take-profit multipliers using grid search.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import argparse
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_data import get_historical_klines_binance
from scripts.indicators import compute_all_indicators
from scripts.backtest import run_backtest, map_kraken_to_binance


def optimize_tp_sl(pair: str, lookback_days: int = 180,
                   sl_range: List[float] = None, tp_range: List[float] = None,
                   fee_pct: float = 0.0, validate_split: float = None):
    """
    Grid search over SL/TP multiplier combinations.
    
    Args:
        pair: Trading pair
        lookback_days: Lookback days
        sl_range: List of SL multipliers to test
        tp_range: List of TP multipliers to test
        fee_pct: Round-trip fee as percentage of entry
        validate_split: Fraction to use for tuning vs validation
    """
    if sl_range is None:
        sl_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    if tp_range is None:
        tp_range = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    binance_pair = map_kraken_to_binance(pair)
    print(f"Fetching {lookback_days} days of {binance_pair} data...")
    
    # Fetch data once
    df = get_historical_klines_binance(binance_pair, "15m", lookback_days)
    if df.empty:
        print("Error: No data returned")
        return
    
    df = compute_all_indicators(df)
    df.dropna(inplace=True)
    print(f"Got {len(df)} candles\n")
    
    # Prepare for split validation
    if validate_split and 0 < validate_split < 1:
        split_idx = int(len(df) * validate_split)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        print(f"Split: {len(df_train)} train, {len(df_test)} test")
    else:
        df_train = df
        df_test = None
    
    # Grid search on training data
    print("Running grid search...")
    results = []
    
    for sl_mult in sl_range:
        for tp_mult in tp_range:
            result = run_backtest(df_train, sl_mult, tp_mult, pair=pair)
            
            # Apply fee adjustment
            if fee_pct > 0:
                result['total_r'] = result['total_r'] - (fee_pct / 100 * result['trades'])
                result['avg_r'] = result['total_r'] / result['trades'] if result['trades'] > 0 else 0
            
            results.append({
                'sl': sl_mult,
                'tp': tp_mult,
                'trades': result['trades'],
                'win_rate': result['win_rate'],
                'avg_r': result['avg_r'],
                'total_r': result['total_r'],
                'breakeven_win_rate': result['breakeven_win_rate'],
                'profit_factor': result['profit_factor']
            })
    
    # Sort by total R
    results.sort(key=lambda x: x['total_r'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (Training Data)")
    print("="*80)
    print(f"SL Multiplier | TP Multiplier | Trades | Win Rate | Avg R | Total R | Breakeven")
    print("-"*80)
    
    for r in results[:10]:  # Top 10
        print(f"{r['sl']:>13.1f} | {r['tp']:>13.1f} | {r['trades']:>6} | {r['win_rate']*100:>8.1f}% | {r['avg_r']:>6.2f} | {r['total_r']:>8.2f} | {r['breakeven_win_rate']*100:>8.1f}%")
    
    # Validate best on test data
    if df_test is not None and results:
        best = results[0]
        print("\n" + "="*80)
        print("VALIDATION (Test Data)")
        print("="*80)
        print(f"Testing best combination: SL={best['sl']:.1f}, TP={best['tp']:.1f}")
        
        test_result = run_backtest(df_test, best['sl'], best['tp'], pair=pair)
        if fee_pct > 0:
            test_result['total_r'] = test_result['total_r'] - (fee_pct / 100 * test_result['trades'])
            test_result['avg_r'] = test_result['total_r'] / test_result['trades'] if test_result['trades'] > 0 else 0
        
        print(f"  Trades: {test_result['trades']}")
        print(f"  Win Rate: {test_result['win_rate']*100:.1f}%")
        print(f"  Avg R: {test_result['avg_r']:.2f}")
        print(f"  Total R: {test_result['total_r']:.2f}")
        print(f"  Breakeven: {test_result['breakeven_win_rate']*100:.1f}%")
        
        # Compare to training
        train_result = results[0]
        print(f"\nTraining vs Validation:")
        print(f"  Total R: {train_result['total_r']:.2f} → {test_result['total_r']:.2f}")
        print(f"  Delta: {test_result['total_r'] - train_result['total_r']:+.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize SL/TP multipliers")
    parser.add_argument("--pair", default="XBTUSD", help="Trading pair")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    parser.add_argument("--fee-pct", type=float, default=0.0, help="Round-trip fee %")
    parser.add_argument("--validate-split", type=float, default=None, help="Fraction for validation")
    parser.add_argument("--sl-range", type=str, default="1.0,1.5,2.0,2.5,3.0", help="SL multipliers")
    parser.add_argument("--tp-range", type=str, default="2.0,3.0,4.0,5.0,6.0", help="TP multipliers")
    
    args = parser.parse_args()
    
    sl_range = [float(x) for x in args.sl_range.split(",")]
    tp_range = [float(x) for x in args.tp_range.split(",")]
    
    optimize_tp_sl(
        args.pair,
        args.days,
        sl_range,
        tp_range,
        args.fee_pct,
        args.validate_split
    )


if __name__ == "__main__":
    main()