"""
Heartbeat check - monitors main bot's health and alerts if it goes silent.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.telegram_alert import send_telegram_message


DATA_DIR = Path(__file__).parent.parent / "data"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"


def check_heartbeat():
    """Check if the main bot has been running recently."""
    
    # Load config to get heartbeat threshold
    config_path = Path(__file__).parent.parent / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        max_gap = config.get('heartbeat_max_gap_minutes', 40)
    else:
        max_gap = 40
    
    # Check if signal log exists
    if not SIGNALS_LOG.exists():
        message = (
            "⚠️ *HEARTBEAT WARNING*\n\n"
            "The bot hasn't logged any signals yet.\n"
            "Check if the main workflow is running.\n\n"
            f"Gap threshold: {max_gap} minutes"
        )
        send_telegram_message(message, parse_mode="Markdown")
        return
    
    # Read the last signal
    try:
        with open(SIGNALS_LOG, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                message = (
                    "⚠️ *HEARTBEAT WARNING*\n\n"
                    "Signal log exists but is empty.\n"
                    "Check if the main workflow is running.\n\n"
                    f"Gap threshold: {max_gap} minutes"
                )
                send_telegram_message(message, parse_mode="Markdown")
                return
            
            # Get the last signal's timestamp
            last_row = rows[-1]
            last_log_time = datetime.fromisoformat(last_row['log_time'])
            now = datetime.now()
            
            gap_minutes = (now - last_log_time).total_seconds() / 60
            
            if gap_minutes > max_gap:
                message = (
                    "🚨 *HEARTBEAT ALERT*\n\n"
                    f"Last signal: {last_log_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    f"Gap: {gap_minutes:.0f} minutes\n"
                    f"Threshold: {max_gap} minutes\n\n"
                    "_The bot may be silent. Please investigate._"
                )
                send_telegram_message(message, parse_mode="Markdown")
                print(f"Heartbeat alert: gap of {gap_minutes:.0f} minutes detected")
            else:
                print(f"Heartbeat OK: last signal {gap_minutes:.0f} minutes ago")
                
    except Exception as e:
        print(f"Error in heartbeat check: {e}")
        message = (
            "⚠️ *HEARTBEAT ERROR*\n\n"
            f"Could not read signal log: {e}\n\n"
            "_Please check the bot's health._"
        )
        send_telegram_message(message, parse_mode="Markdown")


if __name__ == "__main__":
    check_heartbeat()