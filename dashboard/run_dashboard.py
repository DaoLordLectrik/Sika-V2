#!/usr/bin/env python
"""
Run the Sika-V2 Trading Dashboard
"""

import sys
from pathlib import Path
import os

# Add dashboard directory to path
DASHBOARD_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(DASHBOARD_DIR))

# Set environment variable for Flask
os.environ['FLASK_APP'] = 'dashboard.py'
os.environ['FLASK_ENV'] = 'development'

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   📊 Sika-V2 Trading Dashboard                              ║
    ║                                                              ║
    ║   Starting server...                                       ║
    ║                                                              ║
    ║   Open in your browser:  http://127.0.0.1:5000             ║
    ║                                                              ║
    ║   Press Ctrl+C to stop                                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    from dashboard import app
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)