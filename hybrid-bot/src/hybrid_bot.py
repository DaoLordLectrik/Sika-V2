"""
Hybrid Trading Bot - Simplified Version
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing modules
from scripts.fetch_data import get_klines
from scripts.indicators import compute_all_indicators
from scripts.signal_engine import generate_signal as rule_generate_signal, get_trend_bias
from scripts.telegram_alert import send_telegram_message

# Try to import ML modules
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class HybridBot:
    def __init__(self):
        self.config = {
            "pairs": [
                {"symbol": "XBTUSD", "label": "BTC/USD"},
                {"symbol": "ETHUSD", "label": "ETH/USD"},
                {"symbol": "PAXGUSD", "label": "Gold (PAXG/USD)"}
            ],
            "rule_weight": 0.60,
            "ml_weight": 0.40,
            "min_confidence": 0.60
        }
        
        self.model = None
        self.scaler = None
        self.ml_loaded = False
        
        if ML_AVAILABLE:
            self._load_model()
        
        print("="*60)
        print("🧠 HYBRID BOT STARTED")
        print(f"   ML Available: {ML_AVAILABLE}")
        print(f"   ML Loaded: {self.ml_loaded}")
        print("="*60)
    
    def _load_model(self):
        """Load ML model if exists."""
        model_path = Path(__file__).parent.parent / "models" / "rf_model.pkl"
        if model_path.exists():
            try:
                data = joblib.load(model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.ml_loaded = True
                print("✅ ML model loaded")
            except:
                pass
    
    def _get_ml_signal(self, df):
        """Get ML signal."""
        if not self.ml_loaded or not ML_AVAILABLE:
            return None
        
        try:
            # Simple features
            features = []
            for col in ['rsi_14', 'macd_line', 'macd_signal', 'atr_14']:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    features.append(df[col].iloc[-1])
                else:
                    features.append(0)
            
            features = [features]
            if self.scaler:
                features = self.scaler.transform(features)
            
            probs = self.model.predict_proba(features)[0]
            pred = self.model.predict(features)[0]
            
            direction_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            return {
                'direction': direction_map.get(pred, 'HOLD'),
                'confidence': max(probs),
                'source': 'ml'
            }
        except:
            return None
    
    def run_check(self):
        """Run one signal check."""
        print(f"\n{'='*60}")
        print(f"Signal Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        alerts_sent = 0
        
        for pair in self.config['pairs']:
            symbol = pair['symbol']
            label = pair['label']
            
            print(f"\n--- {label} ({symbol}) ---")
            
            try:
                # Get data
                df = get_klines(symbol, "15m", limit=300)
                if df.empty:
                    print("  No data")
                    continue
                
                df = compute_all_indicators(df)
                df.dropna(inplace=True)
                if df.empty:
                    print("  Not enough data")
                    continue
                
                # Rule signal
                rule_signal = rule_generate_signal(
                    df=df,
                    pair=symbol,
                    label=label,
                    sl_atr_multiplier=1.5,
                    tp_atr_multiplier=4.0,
                    trend_bias=get_trend_bias(df.iloc[-1])
                )
                
                rule_direction = rule_signal.direction
                rule_confidence = rule_signal.confidence / 100.0
                
                print(f"  Rules: {rule_direction} ({rule_confidence:.0%})")
                
                # ML signal
                ml_signal = self._get_ml_signal(df)
                ml_direction = 'HOLD'
                ml_confidence = 0
                if ml_signal:
                    ml_direction = ml_signal['direction']
                    ml_confidence = ml_signal['confidence']
                    print(f"  ML:    {ml_direction} ({ml_confidence:.0%})")
                
                # Combine signals
                final_direction = 'HOLD'
                final_confidence = 0
                
                if rule_direction != 'HOLD' and ml_direction != 'HOLD':
                    if rule_direction == ml_direction:
                        # Both agree
                        final_direction = rule_direction
                        final_confidence = (rule_confidence * 0.6 + ml_confidence * 0.4)
                    else:
                        # Disagree - trust rules more
                        if rule_confidence > 0.65:
                            final_direction = rule_direction
                            final_confidence = rule_confidence * 0.7
                elif rule_direction != 'HOLD':
                    # Only rules
                    final_direction = rule_direction
                    final_confidence = rule_confidence * 0.6
                elif ml_direction != 'HOLD' and ml_confidence > 0.7:
                    # Only ML (high confidence)
                    final_direction = ml_direction
                    final_confidence = ml_confidence * 0.5
                
                print(f"  → Final: {final_direction} ({final_confidence:.0%})")
                
                # Send alert if confident
                if final_direction != 'HOLD' and final_confidence >= 0.60:
                    price = df['close'].iloc[-1]
                    atr = df['atr_14'].iloc[-1]
                    
                    if final_direction == 'BUY':
                        sl = price - (1.5 * atr)
                        tp = price + (4.0 * atr)
                    else:
                        sl = price + (1.5 * atr)
                        tp = price - (4.0 * atr)
                    
                    msg = f"""
{pair_emoji(symbol)} *{final_direction} {label}*
🧠 Hybrid Signal
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

💰 Price: ${price:.2f}
📊 Confidence: {final_confidence:.0%}

📊 *Components:*
   🔹 Rules: {rule_direction} ({rule_confidence:.0%})
   🔹 ML:    {ml_direction} ({ml_confidence:.0%})

🎯 TP: ${tp:.2f}
🛑 SL: ${sl:.2f}
📈 R/R: 1:2.67
---
"""
                    send_telegram_message(msg, parse_mode="Markdown")
                    alerts_sent += 1
                    print(f"  ✅ ALERT SENT")
                else:
                    print(f"  ⏭️ No alert")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"\nAlerts sent: {alerts_sent}\n")
        return alerts_sent

def pair_emoji(symbol):
    emojis = {"XBTUSD": "₿", "ETHUSD": "⟠", "PAXGUSD": "🏅"}
    return emojis.get(symbol, "📊")

if __name__ == "__main__":
    bot = HybridBot()
    bot.run_check()