"""
Trading Signal Dashboard - Flask Backend
Serves signal data, performance metrics, and charts
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Initialize Flask app FIRST
app = Flask(__name__)
CORS(app)

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import data modules
try:
    from scripts.fetch_data import get_historical_klines_binance
    from scripts.indicators import compute_all_indicators
    from scripts.signal_engine import generate_signal_from_row, get_trend_bias
    DATA_AVAILABLE = True
except ImportError as e:
    DATA_AVAILABLE = False
    print(f"Warning: Some data modules not available: {e}")

DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# If no signals_log.csv exists, create an empty one
if not SIGNALS_LOG.exists():
    print("⚠️ No signals_log.csv found. Creating empty file...")
    with open(SIGNALS_LOG, 'w') as f:
        f.write("pair,label,timeframe,timestamp,price,direction,confidence,stop_loss,take_profit,risk_reward_ratio,alerted,log_time\n")

def load_signals() -> pd.DataFrame:
    """Load signal log data."""
    if not SIGNALS_LOG.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(SIGNALS_LOG)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading signals: {e}")
        return pd.DataFrame()


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate performance metrics from signal data."""
    if df.empty:
        return {
            "total_alerts": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "avg_confidence": 0,
            "reason_counts": {},
            "hourly_activity": {},
            "pair_counts": {},
            "latest_signal": {}
        }
    
    # Filter to alerted signals only
    df_alerts = df[df['alerted'] == True]
    
    total_signals = len(df_alerts)
    buy_signals = len(df_alerts[df_alerts['direction'] == 'BUY'])
    sell_signals = len(df_alerts[df_alerts['direction'] == 'SELL'])
    
    # Confidence analysis
    avg_confidence = df_alerts['confidence'].mean() if 'confidence' in df_alerts else 0
    
    # Reasons analysis
    all_reasons = []
    for reasons in df_alerts['reasons'].dropna():
        all_reasons.extend([r.strip() for r in str(reasons).split(';')])
    
    reason_counts = pd.Series(all_reasons).value_counts().head(10).to_dict()
    
    # Time analysis
    try:
        df_alerts['hour'] = df_alerts['timestamp'].dt.hour
        hourly_activity = df_alerts['hour'].value_counts().sort_index().to_dict()
    except:
        hourly_activity = {}
    
    # Pair analysis
    pair_counts = df_alerts['pair'].value_counts().to_dict()
    
    # Latest signal
    latest_signal = {}
    if not df_alerts.empty:
        latest = df_alerts.iloc[-1]
        if isinstance(latest, pd.Series):
            latest_signal = latest.to_dict()
            if 'timestamp' in latest_signal and hasattr(latest_signal['timestamp'], 'isoformat'):
                latest_signal['timestamp'] = latest_signal['timestamp'].isoformat()
    
    return {
        "total_alerts": int(total_signals),
        "buy_signals": int(buy_signals),
        "sell_signals": int(sell_signals),
        "avg_confidence": round(float(avg_confidence), 1),
        "reason_counts": reason_counts,
        "hourly_activity": hourly_activity,
        "pair_counts": pair_counts,
        "latest_signal": latest_signal
    }


@app.route('/')
def index():
    """Serve the dashboard HTML."""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        return f"Error loading template: {e}"


@app.route('/api/metrics')
def get_metrics():
    """Get performance metrics."""
    df = load_signals()
    metrics = calculate_metrics(df)
    return jsonify(metrics)


@app.route('/api/signals')
def get_signals():
    """Get recent signals."""
    df = load_signals()
    limit = request.args.get('limit', 50, type=int)
    
    if df.empty:
        return jsonify([])
    
    # Sort by timestamp descending
    df = df.sort_values('timestamp', ascending=False).head(limit)
    
    # Convert to dict with proper serialization
    records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        if 'timestamp' in record and hasattr(record['timestamp'], 'isoformat'):
            record['timestamp'] = record['timestamp'].isoformat()
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
        records.append(record)
    
    return jsonify(records)


@app.route('/api/signals/<pair>')
def get_pair_signals(pair):
    """Get signals for a specific pair."""
    df = load_signals()
    
    if df.empty:
        return jsonify([])
    
    df_pair = df[df['pair'] == pair].sort_values('timestamp', ascending=False).head(50)
    
    records = []
    for _, row in df_pair.iterrows():
        record = row.to_dict()
        if 'timestamp' in record and hasattr(record['timestamp'], 'isoformat'):
            record['timestamp'] = record['timestamp'].isoformat()
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
        records.append(record)
    
    return jsonify(records)


@app.route('/api/performance/<pair>')
def get_pair_performance(pair):
    """Get performance metrics for a specific pair."""
    df = load_signals()
    
    if df.empty:
        return jsonify({"message": "No data available"})
    
    df_pair = df[df['pair'] == pair]
    df_alerts = df_pair[df_pair['alerted'] == True]
    
    if df_alerts.empty:
        return jsonify({"message": f"No alerts for {pair}"})
    
    # Calculate win rate from forward test results
    forward_results = DATA_DIR / "forward_test_results.csv"
    win_rate = 0
    if forward_results.exists():
        try:
            fwd_df = pd.read_csv(forward_results)
            if not fwd_df.empty:
                if 'pair' in fwd_df.columns:
                    pair_fwd = fwd_df[fwd_df['pair'] == pair]
                else:
                    pair_fwd = fwd_df
                if 'win' in pair_fwd.columns and not pair_fwd.empty:
                    wins = len(pair_fwd[pair_fwd['win'] == True])
                    total = len(pair_fwd)
                    win_rate = wins / total * 100 if total > 0 else 0
        except:
            pass
    
    return jsonify({
        "pair": pair,
        "total_signals": int(len(df_alerts)),
        "buy_vs_sell": {
            "buy": int(len(df_alerts[df_alerts['direction'] == 'BUY'])),
            "sell": int(len(df_alerts[df_alerts['direction'] == 'SELL']))
        },
        "avg_confidence": float(df_alerts['confidence'].mean()) if 'confidence' in df_alerts else 0,
        "win_rate": round(win_rate, 1),
        "latest_timestamp": df_alerts.iloc[-1]['timestamp'].isoformat() if not df_alerts.empty else None
    })


@app.route('/api/latest')
def get_latest():
    """Get the latest signal."""
    latest_file = DATA_DIR / "latest_signals.json"
    if latest_file.exists():
        try:
            with open(latest_file, 'r') as f:
                return jsonify(json.load(f))
        except:
            pass
    return jsonify({})


@app.route('/api/health')
def get_health():
    """Health check endpoint."""
    latest_file = DATA_DIR / "latest_signals.json"
    signals_log = DATA_DIR / "signals_log.csv"
    
    health = {
        "status": "healthy",
        "last_run": None,
        "minutes_since_last_run": None,
        "heartbeat_ok": True,
        "files": {}
    }
    
    if latest_file.exists():
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                health["last_run"] = data.get('log_time')
        except:
            pass
    
    health["files"] = {
        "signals_log": signals_log.exists(),
        "latest_signals": latest_file.exists(),
        "alert_state": (DATA_DIR / "alert_state.json").exists()
    }
    
    if health["last_run"]:
        try:
            last_time = datetime.fromisoformat(health["last_run"])
            gap = (datetime.now() - last_time).total_seconds() / 60
            health["minutes_since_last_run"] = round(gap, 1)
            health["heartbeat_ok"] = gap < 40
            health["status"] = "healthy" if gap < 40 else "warning"
        except:
            pass
    
    return jsonify(health)


@app.route('/api/test')
def test():
    """Test endpoint to verify API is working."""
    return jsonify({
        "status": "ok",
        "message": "Dashboard API is running",
        "data_dir": str(DATA_DIR),
        "signals_log_exists": SIGNALS_LOG.exists()
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("📊 Sika-V2 Dashboard Starting...")
    print("="*60)
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📄 Signals log exists: {SIGNALS_LOG.exists()}")
    print(f"🌐 Open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)

    