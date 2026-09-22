"""
Train the ML model for hybrid bot
Run this ONCE locally, then commit the model file
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from scripts.fetch_data import get_klines
from scripts.indicators import compute_all_indicators


def create_training_data():
    """Create training data from historical prices."""
    all_data = []
    
    for symbol in ['XBTUSD', 'ETHUSD', 'PAXGUSD']:
        print(f"Fetching {symbol}...")
        df = get_klines(symbol, "15m", limit=720)
        if df.empty:
            print(f"  No data for {symbol}")
            continue
        
        df = compute_all_indicators(df)
        df.dropna(inplace=True)
        
        if len(df) < 50:
            print(f"  Not enough data: {len(df)} rows")
            continue
        
        features = pd.DataFrame({
            'rsi': df['rsi_14'],
            'macd': df['macd_line'],
            'macd_signal': df['macd_signal'],
            'atr': df['atr_14'],
            'ema9': df['ema_9'],
            'ema21': df['ema_21'],
            'ema50': df['ema_50'],
            'bb_position': (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6),
        })
        
        future_return = df['close'].shift(-5) / df['close'] - 1
        target = np.where(
            future_return > 0.005, 1,
            np.where(future_return < -0.005, 2, 0)
        )
        
        features['target'] = target
        features['symbol'] = symbol
        all_data.append(features)
        print(f"  Added {len(features)} samples")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True).dropna()


def train_model():
    """Train and save the model."""
    print("\n" + "=" * 60)
    print("Training ML Model")
    print("=" * 60)
    
    df = create_training_data()
    
    if len(df) < 100:
        print(f"ERROR: Insufficient data: {len(df)} samples")
        return False
    
    feature_cols = ['rsi', 'macd', 'macd_signal', 'atr', 'ema9', 'ema21', 'ema50', 'bb_position']
    X = df[feature_cols].values
    y = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nTraining on {len(X)} samples...")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    model_path = Path(__file__).parent.parent / "models" / "rf_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    
    accuracy = model.score(X_scaled, y)
    print(f"\n[OK] Model saved to: {model_path}")
    print(f"     Accuracy: {accuracy:.2%}")
    print(f"     File size: {model_path.stat().st_size / 1024:.1f} KB")
    print("\nIMPORTANT: Commit this file to git!")
    print(f"   git add {model_path}")
    print(f"   git commit -m 'Add trained ML model'")
    print(f"   git push")
    
    return True


if __name__ == "__main__":
    train_model()