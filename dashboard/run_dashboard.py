#!/usr/bin/env python
"""
Run the Sika-V2 Trading Dashboard
Production-ready for Render deployment
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment
os.environ['FLASK_APP'] = 'dashboard.dashboard'

# Get port from environment (Render sets this)
PORT = int(os.environ.get('PORT', 5000))

print("\n" + "="*60)
print("📊 Sika-V2 Trading Dashboard")
print("="*60)
print(f"📁 Project: {PROJECT_ROOT}")
print(f"🌐 Server starting on port: {PORT}")
print("="*60 + "\n")

try:
    from dashboard.dashboard import app
    
    # Run with production settings
    app.run(
        debug=False,
        host='0.0.0.0',  # Listen on all interfaces (required for Render)
        port=PORT,
        threaded=True
    )
except KeyboardInterrupt:
    print("\n\n👋 Dashboard stopped")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error starting dashboard: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)