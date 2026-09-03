"""
Sika-V2 Trading Dashboard - Single File Deployment
For Render.com deployment
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create empty CSV if it doesn't exist
if not SIGNALS_LOG.exists():
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
    
    df_alerts = df[df['alerted'] == True]
    
    total_signals = len(df_alerts)
    buy_signals = len(df_alerts[df_alerts['direction'] == 'BUY'])
    sell_signals = len(df_alerts[df_alerts['direction'] == 'SELL'])
    
    avg_confidence = df_alerts['confidence'].mean() if 'confidence' in df_alerts else 0
    
    all_reasons = []
    for reasons in df_alerts['reasons'].dropna():
        all_reasons.extend([r.strip() for r in str(reasons).split(';')])
    
    reason_counts = pd.Series(all_reasons).value_counts().head(10).to_dict()
    
    try:
        df_alerts['hour'] = df_alerts['timestamp'].dt.hour
        hourly_activity = df_alerts['hour'].value_counts().sort_index().to_dict()
    except:
        hourly_activity = {}
    
    pair_counts = df_alerts['pair'].value_counts().to_dict()
    
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


# Dashboard HTML template (condensed for deployment)
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sika-V2 Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0e17; color: #e0e0e0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid #1a2332; margin-bottom: 30px; }
        .header h1 { font-size: 28px; font-weight: 700; background: linear-gradient(135deg, #f7931a, #ff6b6b); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
        .header .status { display: flex; align-items: center; gap: 10px; padding: 8px 16px; background: #1a2332; border-radius: 20px; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .status-dot.online { background: #00ff88; box-shadow: 0 0 10px #00ff8844; }
        .status-dot.offline { background: #ff4444; box-shadow: 0 0 10px #ff444444; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #111927; border: 1px solid #1a2332; border-radius: 12px; padding: 20px; transition: all 0.3s ease; }
        .card:hover { border-color: #2a3a55; transform: translateY(-2px); }
        .card .label { font-size: 12px; text-transform: uppercase; color: #8899aa; letter-spacing: 1px; }
        .card .value { font-size: 28px; font-weight: 700; margin-top: 8px; }
        .card .value.gold { color: #f7931a; }
        .card .value.green { color: #00ff88; }
        .card .value.blue { color: #4a9eff; }
        .card .value.red { color: #ff6b6b; }
        .charts { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 30px; }
        .chart-box { background: #111927; border: 1px solid #1a2332; border-radius: 12px; padding: 20px; }
        .chart-box h3 { font-size: 14px; color: #8899aa; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; }
        .chart-container { height: 200px; position: relative; }
        .bar-chart { display: flex; align-items: flex-end; height: 100%; gap: 4px; }
        .bar { flex: 1; background: linear-gradient(180deg, #4a9eff, #1a3a6a); border-radius: 4px 4px 0 0; transition: all 0.3s ease; min-height: 4px; position: relative; }
        .bar .bar-label { position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #667788; }
        .table-section { background: #111927; border: 1px solid #1a2332; border-radius: 12px; overflow: hidden; margin-bottom: 30px; }
        .table-section h3 { padding: 20px 20px 0; font-size: 14px; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 12px 20px; font-size: 12px; color: #667788; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #1a2332; }
        td { padding: 12px 20px; border-bottom: 1px solid #0a0e17; font-size: 14px; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
        .badge.buy { background: #00ff8822; color: #00ff88; }
        .badge.sell { background: #ff6b6b22; color: #ff6b6b; }
        .badge.hold { background: #8899aa22; color: #8899aa; }
        .badge.alerts-on { background: #00ff8822; color: #00ff88; }
        .badge.alerts-off { background: #ff6b6b22; color: #ff6b6b; }
        .pair-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; background: #1a2332; color: #4a9eff; }
        .reason-tags { display: flex; flex-wrap: wrap; gap: 4px; }
        .reason-tag { background: #1a2332; padding: 2px 8px; border-radius: 12px; font-size: 11px; color: #8899aa; }
        @media (max-width: 768px) { .charts { grid-template-columns: 1fr; } .cards { grid-template-columns: repeat(2, 1fr); } }
        .loading { text-align: center; padding: 40px; color: #667788; }
        .loading .spinner { display: inline-block; width: 30px; height: 30px; border: 3px solid #1a2332; border-top-color: #4a9eff; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Sika-V2 Dashboard</h1>
            <div class="status" id="healthStatus">
                <span class="status-dot online" id="statusDot"></span>
                <span id="statusText">Loading...</span>
            </div>
        </div>
        
        <div class="cards" id="metricsCards">
            <div class="card"><div class="label">Total Signals</div><div class="value blue" id="totalSignals">-</div></div>
            <div class="card"><div class="label">BUY Signals</div><div class="value green" id="buySignals">-</div></div>
            <div class="card"><div class="label">SELL Signals</div><div class="value red" id="sellSignals">-</div></div>
            <div class="card"><div class="label">Avg Confidence</div><div class="value gold" id="avgConfidence">-%</div></div>
            <div class="card"><div class="label">Last Run</div><div class="value" style="font-size:16px;" id="lastRun">-</div></div>
            <div class="card"><div class="label">Heartbeat Status</div><div class="value" style="font-size:16px;" id="heartbeatStatus">-</div></div>
        </div>
        
        <div class="charts">
            <div class="chart-box">
                <h3>📈 Hourly Signal Activity</h3>
                <div class="chart-container"><div class="bar-chart" id="hourlyChart"><div class="loading"><div class="spinner"></div></div></div></div>
            </div>
            <div class="chart-box">
                <h3>🎯 Signal Distribution</h3>
                <div class="chart-container" id="pairDistribution"><div class="loading"><div class="spinner"></div></div></div>
            </div>
        </div>
        
        <div class="table-section">
            <h3>📋 Recent Signals</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead><tr><th>Time</th><th>Pair</th><th>Direction</th><th>Price</th><th>Confidence</th><th>Reasons</th><th>Alert</th></tr></thead>
                    <tbody id="signalTable"><tr><td colspan="7" class="loading">Loading signals...</td></tr></tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        setInterval(loadData, 30000);
        function loadData() { loadMetrics(); loadSignals(); loadHealth(); }
        
        async function loadMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                document.getElementById('totalSignals').textContent = data.total_alerts || 0;
                document.getElementById('buySignals').textContent = data.buy_signals || 0;
                document.getElementById('sellSignals').textContent = data.sell_signals || 0;
                document.getElementById('avgConfidence').textContent = (data.avg_confidence || 0) + '%';
                if (data.hourly_activity) updateHourlyChart(data.hourly_activity);
                if (data.pair_counts) updatePairDistribution(data.pair_counts);
                if (data.latest_signal && data.latest_signal.log_time) {
                    document.getElementById('lastRun').textContent = new Date(data.latest_signal.log_time).toLocaleString();
                }
            } catch (e) { console.error('Error loading metrics:', e); }
        }
        
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals?limit=20');
                const signals = await response.json();
                const tbody = document.getElementById('signalTable');
                if (!signals || signals.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:20px;color:#667788;">No signals yet</td></tr>';
                    return;
                }
                tbody.innerHTML = signals.map(s => {
                    const direction = s.direction || 'HOLD';
                    const dirClass = direction.toLowerCase();
                    const time = new Date(s.timestamp);
                    const reasons = s.reasons ? s.reasons.split(';').slice(0, 3) : [];
                    const alerted = s.alerted === true || s.alerted === 'True';
                    return `<tr>
                        <td style="font-size:13px;color:#8899aa;">${time.toLocaleString()}</td>
                        <td><span class="pair-badge">${s.label || s.pair}</span></td>
                        <td><span class="badge ${dirClass}">${direction}</span></td>
                        <td>$${parseFloat(s.price).toFixed(2)}</td>
                        <td>${s.confidence || 0}%</td>
                        <td><div class="reason-tags">${reasons.map(r => `<span class="reason-tag">${r.trim()}</span>`).join('')}</div></td>
                        <td><span class="badge ${alerted ? 'alerts-on' : 'alerts-off'}">${alerted ? '✅ Sent' : '⏸️ Suppressed'}</span></td>
                    </tr>`;
                }).join('');
            } catch (e) { console.error('Error loading signals:', e); }
        }
        
        async function loadHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                const heartbeat = document.getElementById('heartbeatStatus');
                if (data.status === 'healthy' && data.heartbeat_ok) {
                    dot.className = 'status-dot online';
                    text.textContent = 'Online';
                    heartbeat.textContent = '✅ Healthy';
                    heartbeat.style.color = '#00ff88';
                } else {
                    dot.className = 'status-dot offline';
                    text.textContent = data.minutes_since_last_run ? `⚠️ ${data.minutes_since_last_run}m ago` : 'Offline';
                    heartbeat.textContent = '⚠️ Check bot';
                    heartbeat.style.color = '#ff6b6b';
                }
            } catch (e) { console.error('Error loading health:', e); }
        }
        
        function updateHourlyChart(activity) {
            const container = document.getElementById('hourlyChart');
            const maxVal = Math.max(...Object.values(activity), 1);
            container.innerHTML = Object.entries(activity).map(([hour, count]) => {
                const height = (count / maxVal) * 100;
                return `<div class="bar" style="height: ${Math.max(height, 10)}%;"><div class="bar-label">${hour}h</div></div>`;
            }).join('') || '<div style="text-align:center;color:#667788;padding:20px;">No hourly data</div>';
        }
        
        function updatePairDistribution(pairCounts) {
            const container = document.getElementById('pairDistribution');
            const total = Object.values(pairCounts).reduce((a, b) => a + b, 0);
            if (total === 0) {
                container.innerHTML = '<div style="text-align:center;color:#667788;padding:20px;">No pair data</div>';
                return;
            }
            const colors = ['#4a9eff', '#f7931a', '#00ff88', '#a855f7', '#ff6b6b'];
            let idx = 0;
            container.innerHTML = Object.entries(pairCounts).map(([pair, count]) => {
                const pct = (count / total * 100).toFixed(1);
                const color = colors[idx++ % colors.length];
                return `<div style="display:flex;align-items:center;gap:10px;padding:4px 0;">
                    <span style="width:60px;font-size:13px;color:#8899aa;">${pair.replace('USD', '')}</span>
                    <div style="flex:1;height:20px;background:#1a2332;border-radius:10px;overflow:hidden;">
                        <div style="width:${pct}%;height:100%;background:${color};border-radius:10px;transition:width 0.5s;"></div>
                    </div>
                    <span style="font-size:13px;font-weight:600;color:#e0e0e0;">${pct}%</span>
                    <span style="font-size:12px;color:#667788;">(${count})</span>
                </div>`;
            }).join('');
        }
        
        loadData();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return render_template_string(DASHBOARD_TEMPLATE)


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
    
    df = df.sort_values('timestamp', ascending=False).head(limit)
    
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


@app.route('/api/health')
def get_health():
    """Health check endpoint."""
    latest_file = DATA_DIR / "latest_signals.json"
    
    health = {
        "status": "healthy",
        "last_run": None,
        "minutes_since_last_run": None,
        "heartbeat_ok": True,
        "files": {
            "signals_log": SIGNALS_LOG.exists(),
            "latest_signals": latest_file.exists()
        }
    }
    
    if latest_file.exists():
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                health["last_run"] = data.get('log_time')
        except:
            pass
    
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
    """Test endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Dashboard API is running",
        "data_dir": str(DATA_DIR),
        "signals_log_exists": SIGNALS_LOG.exists()
    })


if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("📊 Sika-V2 Trading Dashboard")
    print("="*60)
    print(f"📁 Project: {PROJECT_ROOT}")
    print(f"📄 Signals log exists: {SIGNALS_LOG.exists()}")
    print(f"🌐 Server starting on port: {PORT}")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=PORT)