"""
Hybrid Trading Bot - With Deduplication
"""

import sys
import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

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

# Data files
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "hybrid_signals_log.csv"
ALERT_STATE_FILE = DATA_DIR / "hybrid_alert_state.json"  # ⭐ Separate state file

# Deduplication settings
DEDUP_WINDOW_HOURS = 2       # Minimum time between same-direction alerts
MIN_CONFIDENCE = 0.55         # Minimum confidence to alert


class HybridBot:
    def __init__(self):
        self.config = {
            "pairs": [
                {"symbol": "XBTUSD", "label": "BTC/USD"},
                {"symbol": "ETHUSD", "label": "ETH/USD"},
                {"symbol": "PAXGUSD", "label": "Gold (PAXG/USD)"}
            ],
        }
        
        self.model = None
        self.scaler = None
        self.ml_loaded = False
        
        if ML_AVAILABLE:
            self._load_model()
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # ⭐ Load alert state
        self.alert_state = self._load_alert_state()
        
        print("="*60)
        print("🧠 HYBRID BOT STARTED")
        print(f"   ML Available: {ML_AVAILABLE}")
        print(f"   ML Loaded: {self.ml_loaded}")
        print(f"   CSV Log: {CSV_FILE}")
        print(f"   Alert State: {ALERT_STATE_FILE}")
        print(f"   Dedup Window: {DEDUP_WINDOW_HOURS} hours")
        print("="*60)
    
    def _load_alert_state(self):
        """Load deduplication state."""
        if ALERT_STATE_FILE.exists():
            try:
                with open(ALERT_STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_alert_state(self):
        """Save deduplication state."""
        with open(ALERT_STATE_FILE, 'w') as f:
            json.dump(self.alert_state, f, indent=2)
    
    def _should_alert(self, symbol, direction):
        """
        Check if we should send an alert.
        Returns (should_alert: bool, reason: str)
        """
        last = self.alert_state.get(symbol, {})
        last_direction = last.get('direction')
        last_time_str = last.get('timestamp')
        
        if not last_direction or not last_time_str:
            return True, "first_alert"
        
        try:
            last_time = datetime.fromisoformat(last_time_str)
        except:
            return True, "invalid_state"
        
        hours_since = (datetime.now() - last_time).total_seconds() / 3600
        
        # Rule 1: Direction changed → always alert
        if last_direction != direction:
            return True, f"direction_change_{last_direction}_to_{direction}"
        
        # Rule 2: Same direction but enough time passed → alert
        if hours_since >= DEDUP_WINDOW_HOURS:
            return True, f"same_direction_after_{hours_since:.1f}h"
        
        # Rule 3: Same direction too soon → suppress
        return False, f"duplicate_{direction}_{hours_since:.1f}h_ago"
    
    def _record_alert(self, symbol, direction, confidence):
        """Record that an alert was sent."""
        self.alert_state[symbol] = {
            'direction': direction,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence
        }
        self._save_alert_state()
    
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
            except Exception as e:
                print(f"⚠️ ML model failed to load: {e}")
        else:
            print(f"⚠️ ML model not found at {model_path}")
    
    def _get_ml_signal(self, df):
        """Get ML signal."""
        if not self.ml_loaded or not ML_AVAILABLE:
            return None
        
        try:
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
        except Exception as e:
            return None
    
    def _log_signal(self, symbol, label, direction, confidence, price, alerted, source='hybrid'):
        """Log signal to CSV."""
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'label': label,
                'direction': direction,
                'confidence': round(confidence * 100, 1),
                'price': price,
                'alerted': alerted,
                'source': source
            }
            
            file_exists = CSV_FILE.exists()
            
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
            
        except Exception as e:
            print(f"  ⚠️ Log error: {e}")
    
    def run_check(self):
        """Run one signal check."""
        print(f"\n{'='*60}")
        print(f"Signal Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        alerts_sent = 0
        signals_logged = 0
        alerts_suppressed = 0
        
        for pair in self.config['pairs']:
            symbol = pair['symbol']
            label = pair['label']
            
            print(f"\n--- {label} ({symbol}) ---")
            
            try:
                df = get_klines(symbol, "15m", limit=300)
                if df.empty:
                    print("  No data")
                    continue
                
                df = compute_all_indicators(df)
                df.dropna(inplace=True)
                if df.empty:
                    print("  Not enough data")
                    continue
                
                price = df['close'].iloc[-1]
                
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
                if ml_signal:
                    ml_direction = ml_signal['direction']
                    ml_confidence = ml_signal['confidence']
                    print(f"  ML:    {ml_direction} ({ml_confidence:.0%})")
                else:
                    ml_direction = None
                    ml_confidence = 0
                    print(f"  ML:    Not available")
                
                # Combine signals
                final_direction = 'HOLD'
                final_confidence = 0
                final_source = 'rules'
                
                if ml_direction is None:
                    if rule_direction != 'HOLD':
                        final_direction = rule_direction
                        final_confidence = rule_confidence
                        final_source = 'rules_only'
                    else:
                        final_direction = 'HOLD'
                        final_confidence = 0
                        final_source = 'rules'
                else:
                    if rule_direction != 'HOLD' and ml_direction != 'HOLD':
                        if rule_direction == ml_direction:
                            final_direction = rule_direction
                            final_confidence = (rule_confidence * 0.6 + ml_confidence * 0.4)
                            final_source = 'ensemble_agree'
                        else:
                            if rule_confidence >= 0.65:
                                final_direction = rule_direction
                                final_confidence = rule_confidence
                                final_source = 'rules_dominant'
                    elif rule_direction != 'HOLD':
                        final_direction = rule_direction
                        final_confidence = rule_confidence
                        final_source = 'rules_only'
                    elif ml_direction != 'HOLD' and ml_confidence > 0.7:
                        final_direction = ml_direction
                        final_confidence = ml_confidence * 0.7
                        final_source = 'ml_only'
                
                print(f"  → Final: {final_direction} ({final_confidence:.0%}) Source: {final_source}")
                
                # Log to CSV (always)
                self._log_signal(
                    symbol=symbol,
                    label=label,
                    direction=final_direction,
                    confidence=final_confidence,
                    price=price,
                    alerted=False,
                    source=final_source
                )
                signals_logged += 1
                
                # ⭐ DEDUPLICATION CHECK
                if final_direction != 'HOLD' and final_confidence >= MIN_CONFIDENCE:
                    should_alert, reason = self._should_alert(symbol, final_direction)
                    
                    if should_alert:
                        atr = df['atr_14'].iloc[-1]
                        
                        if final_direction == 'BUY':
                            sl = price - (1.5 * atr)
                            tp = price + (4.0 * atr)
                        else:
                            sl = price + (1.5 * atr)
                            tp = price - (4.0 * atr)
                        
                        msg = self._format_alert(
                            symbol, label, final_direction, final_confidence,
                            price, sl, tp, rule_direction, rule_confidence,
                            ml_direction, ml_confidence, final_source
                        )
                        
                        success = send_telegram_message(msg, parse_mode="Markdown")
                        
                        if success:
                            alerts_sent += 1
                            # ⭐ Record the alert for deduplication
                            self._record_alert(symbol, final_direction, final_confidence)
                            
                            self._log_signal(
                                symbol=symbol,
                                label=label,
                                direction=final_direction,
                                confidence=final_confidence,
                                price=price,
                                alerted=True,
                                source=final_source + '_alerted'
                            )
                            
                            print(f"  ✅ ALERT SENT ({reason})")
                        else:
                            print(f"  ❌ ALERT FAILED")
                    else:
                        alerts_suppressed += 1
                        print(f"  ⏭️ Suppressed: {reason}")
                else:
                    print(f"  ⏭️ Below threshold ({final_confidence:.0%} < {MIN_CONFIDENCE:.0%})")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 Summary:")
        print(f"   Signals logged: {signals_logged}")
        print(f"   Alerts sent: {alerts_sent}")
        print(f"   Alerts suppressed: {alerts_suppressed}")
        print(f"{'='*60}\n")
        
        return alerts_sent
    
    def _format_alert(self, symbol, label, direction, confidence, price, sl, tp,
                      rule_dir, rule_conf, ml_dir, ml_conf, source):
        """Format alert message."""
        direction_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}.get(direction, "")
        pair_emoji = {"XBTUSD": "₿", "ETHUSD": "⟠", "PAXGUSD": "🏅"}.get(symbol, "📊")
        
        lines = []
        lines.append(f"{pair_emoji} {direction_emoji} *{direction} {label} ({symbol})*")
        lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("")
        lines.append(f"💰 *Price:* ${price:.2f}")
        lines.append(f"📊 *Confidence:* {confidence:.0%}")
        lines.append("")
        lines.append("*Signal Components:*")
        lines.append(f"  🔹 Rules: {rule_dir} ({rule_conf:.0%})")
        if ml_dir:
            lines.append(f"  🔹 ML: {ml_dir} ({ml_conf:.0%})")
        else:
            lines.append(f"  🔹 ML: not available")
        lines.append("")
        lines.append(f"🎯 *Take Profit:* ${tp:.2f}")
        lines.append(f"🛑 *Stop Loss:* ${sl:.2f}")
        
        risk = abs(price - sl)
        reward = abs(price - tp)
        rr_ratio = reward / risk if risk > 0 else 0
        lines.append(f"📈 *Risk/Reward:* 1:{rr_ratio:.2f}")
        lines.append("")
        lines.append("---")
        
        return "\n".join(lines)


if __name__ == "__main__":
    bot = HybridBot()
    bot.run_check()