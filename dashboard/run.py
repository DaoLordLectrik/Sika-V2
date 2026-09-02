#!/usr/bin/env python
"""
Run the Sika-V2 Trading Dashboard
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment
os.environ['FLASK_APP'] = 'dashboard.dashboard'
os.environ['FLASK_ENV'] = 'development'

print("\n" + "="*60)
print("📊 Sika-V2 Trading Dashboard")
print("="*60)
print(f"📁 Project: {PROJECT_ROOT}")
print("🌐 Server starting at: http://127.0.0.1:5000")
print("👉 Open your browser and go to: http://127.0.0.1:5000")
print("="*60)
print("")

try:
    from dashboard.dashboard import app
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
except KeyboardInterrupt:
    print("\n\n👋 Dashboard stopped")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error starting dashboard: {e}")
    sys.exit(1)