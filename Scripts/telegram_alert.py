"""
Telegram alert module - sends notifications via Bot API.
"""

import os
import requests
from typing import Optional


def send_telegram_message(text: str, parse_mode: Optional[str] = None) -> bool:
    """
    Send a Telegram message using bot token and chat ID from environment.
    
    Args:
        text: Message text to send
        parse_mode: Optional parse mode ('Markdown', 'HTML', or None)
    
    Returns:
        True if sent successfully, False otherwise
    
    Note:
        If credentials are missing, prints to console instead of crashing.
        This is intentional for local testing.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    # Fallback for local testing
    if not token or not chat_id:
        print("\n" + "="*50)
        print("TELEGRAM ALERT (credentials not set - printing to console)")
        print("="*50)
        print(text)
        print("="*50 + "\n")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        if result.get("ok"):
            print(f"Telegram message sent successfully to chat_id: {chat_id}")
            return True
        else:
            print(f"Telegram API error: {result}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram message: {e}")
        return False