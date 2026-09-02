"""
Signal engine - core trading logic with 4 scoring functions.
Simplified version - filters removed based on backtest results.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


@dataclass
class Signal:
    """Signal data structure."""
    pair: str
    label: str
    timeframe: str
    timestamp: datetime
    price: float
    direction: str  # 'BUY', 'SELL', or 'HOLD'
    confidence: int  # 0-100
    reasons: List[str]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pair': self.pair,
            'label': self.label,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'direction': self.direction,
            'confidence': self.confidence,
            'reasons': '; '.join(self.reasons),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio
        }


def get_trend_bias(row: pd.Series) -> str:
    """
    Determine trend bias from EMA stacking.
    
    Args:
        row: Series with EMA columns
    
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    ema_9 = row.get('ema_9')
    ema_21 = row.get('ema_21')
    ema_50 = row.get('ema_50')
    
    if pd.isna(ema_9) or pd.isna(ema_21) or pd.isna(ema_50):
        return 'neutral'
    
    # Check for stacked EMAs
    if ema_9 > ema_21 > ema_50:
        return 'bullish'
    elif ema_9 < ema_21 < ema_50:
        return 'bearish'
    else:
        return 'neutral'


def score_trend(row: pd.Series) -> Tuple[int, str]:
    """
    Trend scoring: stacked EMA alignment.
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    bias = get_trend_bias(row)
    
    if bias == 'bullish':
        return 1, "EMA stack bullish (9>21>50)"
    elif bias == 'bearish':
        return -1, "EMA stack bearish (9<21<50)"
    else:
        return 0, "EMA stack neutral"


def score_momentum(row: pd.Series) -> Tuple[int, str]:
    """
    Momentum scoring: RSI zones.
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    rsi = row.get('rsi_14')
    
    if pd.isna(rsi):
        return 0, "RSI unavailable"
    
    if rsi < 30:
        return 1, f"Oversold bounce potential (RSI={rsi:.1f})"
    elif rsi > 70:
        return -1, f"Overbought reversal potential (RSI={rsi:.1f})"
    elif 55 <= rsi <= 70:
        return 1, f"Bullish momentum (RSI={rsi:.1f})"
    elif 30 <= rsi < 45:
        return -1, f"Bearish momentum (RSI={rsi:.1f})"
    else:
        return 0, f"Neutral momentum (RSI={rsi:.1f})"


def score_macd(row: pd.Series) -> Tuple[int, str]:
    """
    MACD scoring: line vs signal with histogram sign.
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    macd_line = row.get('macd_line')
    macd_signal = row.get('macd_signal')
    macd_hist = row.get('macd_hist')
    
    if pd.isna(macd_line) or pd.isna(macd_signal) or pd.isna(macd_hist):
        return 0, "MACD unavailable"
    
    if macd_line > macd_signal and macd_hist > 0:
        return 1, f"MACD bullish (line>{macd_signal:.2f}, hist={macd_hist:.2f})"
    elif macd_line < macd_signal and macd_hist < 0:
        return -1, f"MACD bearish (line<{macd_signal:.2f}, hist={macd_hist:.2f})"
    else:
        return 0, f"MACD neutral (line={macd_line:.2f}, hist={macd_hist:.2f})"


def score_mean_reversion(row: pd.Series) -> Tuple[int, str]:
    """
    Mean reversion scoring: Bollinger Band extremes.
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    close = row.get('close')
    bb_lower = row.get('bb_lower')
    bb_upper = row.get('bb_upper')
    bb_middle = row.get('bb_middle')
    
    if pd.isna(close) or pd.isna(bb_lower) or pd.isna(bb_upper):
        return 0, "Bollinger Bands unavailable"
    
    if close <= bb_lower:
        return 1, f"Price at lower band (support bounce potential)"
    elif close >= bb_upper:
        return -1, f"Price at upper band (resistance rejection potential)"
    elif close < bb_middle:
        return 0, f"Price below middle band, near support"
    else:
        return 0, f"Price above middle band, near resistance"


def generate_signal_from_row(row: pd.Series, 
                            pair: str = "Unknown",
                            label: str = "Unknown",
                            timeframe: str = "15m",
                            sl_atr_multiplier: float = 1.5,
                            tp_atr_multiplier: float = 4.0,
                            trend_bias: Optional[str] = None) -> Signal:
    """
    Generate a signal from a single row of indicator data.
    
    This is the core logic function that backtesters should call iteratively.
    Uses 4 scoring functions (trend, momentum, MACD, mean reversion).
    
    Args:
        row: Series with all indicator columns
        pair: Trading pair symbol
        label: Human-readable pair label
        timeframe: Timeframe string
        sl_atr_multiplier: Stop-loss multiplier for ATR
        tp_atr_multiplier: Take-profit multiplier for ATR
        trend_bias: Optional trend bias from higher timeframe
    
    Returns:
        Signal object
    """
    # Calculate all four scores
    trend_score, trend_reason = score_trend(row)
    momentum_score, momentum_reason = score_momentum(row)
    macd_score, macd_reason = score_macd(row)
    meanrev_score, meanrev_reason = score_mean_reversion(row)
    
    # Sum votes
    total_score = trend_score + momentum_score + macd_score + meanrev_score
    
    # Collect reasons
    reasons = []
    if trend_score != 0:
        reasons.append(trend_reason)
    if momentum_score != 0:
        reasons.append(momentum_reason)
    if macd_score != 0:
        reasons.append(macd_reason)
    if meanrev_score != 0:
        reasons.append(meanrev_reason)
    
    # Calculate confidence (0, 25, 50, 75, 100)
    confidence = int(abs(total_score) / 4 * 100)
    
    # Determine direction
    if confidence >= 50:  # At least 2 votes
        if total_score >= 2:
            direction = "BUY"
        elif total_score <= -2:
            direction = "SELL"
        else:
            direction = "HOLD"
    else:
        direction = "HOLD"
    
    # Apply trend filter if provided
    if direction != "HOLD" and trend_bias is not None:
        if trend_bias == "bullish" and direction == "SELL":
            direction = "HOLD"
            reasons.append(f"Suppressed by bullish trend filter")
        elif trend_bias == "bearish" and direction == "BUY":
            direction = "HOLD"
            reasons.append(f"Suppressed by bearish trend filter")
    
    # Calculate TP/SL if BUY or SELL
    price = row.get('close')
    atr_val = row.get('atr_14')
    stop_loss = None
    take_profit = None
    risk_reward_ratio = None
    
    if direction != "HOLD" and price is not None and atr_val is not None and not pd.isna(atr_val):
        if direction == "BUY":
            stop_loss = price - (sl_atr_multiplier * atr_val)
            take_profit = price + (tp_atr_multiplier * atr_val)
        else:  # SELL
            stop_loss = price + (sl_atr_multiplier * atr_val)
            take_profit = price - (tp_atr_multiplier * atr_val)
        
        # Calculate risk-reward ratio
        if stop_loss is not None and take_profit is not None:
            risk = abs(price - stop_loss)
            reward = abs(price - take_profit)
            if risk > 0:
                risk_reward_ratio = reward / risk
    
    # Get timestamp from index if available
    timestamp = row.name if hasattr(row, 'name') else datetime.now()
    if not isinstance(timestamp, datetime):
        timestamp = datetime.now()
    
    return Signal(
        pair=pair,
        label=label,
        timeframe=timeframe,
        timestamp=timestamp,
        price=price if price is not None else 0.0,
        direction=direction,
        confidence=confidence,
        reasons=reasons if reasons else ["No strong signal"],
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=risk_reward_ratio
    )


def generate_signal(df: pd.DataFrame,
                   pair: str = "Unknown",
                   label: str = "Unknown",
                   timeframe: str = "15m",
                   sl_atr_multiplier: float = 1.5,
                   tp_atr_multiplier: float = 4.0,
                   trend_bias: Optional[str] = None) -> Signal:
    """
    Convenience wrapper that generates a signal from the last row.
    
    Args:
        df: DataFrame with indicator columns
        pair: Trading pair symbol
        label: Human-readable pair label
        timeframe: Timeframe string
        sl_atr_multiplier: Stop-loss multiplier for ATR
        tp_atr_multiplier: Take-profit multiplier for ATR
        trend_bias: Optional trend bias from higher timeframe
    
    Returns:
        Signal from the last row
    """
    if df.empty:
        return Signal(
            pair=pair,
            label=label,
            timeframe=timeframe,
            timestamp=datetime.now(),
            price=0.0,
            direction="HOLD",
            confidence=0,
            reasons=["No data available"]
        )
    
    row = df.iloc[-1].copy()
    return generate_signal_from_row(
        row, pair, label, timeframe,
        sl_atr_multiplier, tp_atr_multiplier, trend_bias
    )


def format_signal(signal: Signal, include_emoji: bool = False) -> str:
    """
    Format signal as plain text for console/logs.
    
    Args:
        signal: Signal object to format
        include_emoji: Whether to include emoji symbols
    
    Returns:
        Formatted text string
    """
    lines = []
    
    # Header
    lines.append(f"{signal.label} ({signal.pair}) - {signal.timeframe}")
    lines.append(f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Direction with emoji
    if include_emoji:
        emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}
        lines.append(f"Direction: {emoji_map.get(signal.direction, '')} {signal.direction}")
    else:
        lines.append(f"Direction: {signal.direction}")
    
    lines.append(f"Confidence: {signal.confidence}%")
    lines.append(f"Price: ${signal.price:.2f}")
    
    if signal.stop_loss is not None and signal.take_profit is not None:
        lines.append(f"Stop Loss: ${signal.stop_loss:.2f}")
        lines.append(f"Take Profit: ${signal.take_profit:.2f}")
        if signal.risk_reward_ratio is not None:
            lines.append(f"Risk/Reward: 1:{signal.risk_reward_ratio:.2f}")
    
    lines.append("Reasons:")
    for reason in signal.reasons:
        lines.append(f"  • {reason}")
    
    return "\n".join(lines)


def format_signal_telegram(signal: Signal) -> str:
    """
    Format signal for Telegram with Markdown styling.
    
    Args:
        signal: Signal object to format
    
    Returns:
        Markdown-formatted text string
    """
    emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}
    emoji = emoji_map.get(signal.direction, "")
    
    lines = []
    
    # Leading blank line for breathing room
    lines.append("")
    
    # Divider top
    lines.append("---")
    lines.append("")
    
    # Header
    lines.append(f"*{signal.label} ({signal.pair})*")
    lines.append(f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    
    # Direction with emoji (bold)
    lines.append(f"*{emoji} {signal.direction}*")
    lines.append("")
    
    # Price and confidence
    lines.append(f"💰 *Price:* ${signal.price:.2f}")
    lines.append(f"📊 *Confidence:* {signal.confidence}%")
    
    # TP/SL if available
    if signal.stop_loss is not None and signal.take_profit is not None:
        lines.append("")
        lines.append(f"🛑 *Stop Loss:* ${signal.stop_loss:.2f}")
        lines.append(f"🎯 *Take Profit:* ${signal.take_profit:.2f}")
        if signal.risk_reward_ratio is not None:
            lines.append(f"📈 *Risk/Reward:* 1:{signal.risk_reward_ratio:.2f}")
    
    # Reasons
    if signal.reasons:
        lines.append("")
        lines.append("*Reasons:*")
        for reason in signal.reasons:
            lines.append(f"• {reason}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)