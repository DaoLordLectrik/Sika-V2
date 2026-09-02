"""
Signal engine - core trading logic with 7 scoring functions.
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
    
    if rsi < 25:
        return 1, f"Strong oversold bounce (RSI={rsi:.1f})"
    elif rsi > 75:
        return -1, f"Strong overbought reversal (RSI={rsi:.1f})"
    elif 60 <= rsi <= 75:
        return 1, f"Strong bullish momentum (RSI={rsi:.1f})"
    elif 25 <= rsi < 40:
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


def score_volume(row: pd.Series, threshold: float = 1.2) -> Tuple[int, str]:
    """
    Volume confirmation filter.
    
    Args:
        row: Data row with volume_ratio
        threshold: Minimum volume ratio for confirmation
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    volume_ratio = row.get('volume_ratio', 0)
    
    if pd.isna(volume_ratio):
        return 0, "Volume data unavailable"
    
    if volume_ratio > threshold:
        return 1, f"Volume confirmed ({volume_ratio:.2f}x avg)"
    else:
        return -1, f"Low volume ({volume_ratio:.2f}x avg)"


def score_volatility(row: pd.Series, min_percentile: int = 30, max_percentile: int = 90) -> Tuple[int, str]:
    """
    Volatility filter using ATR percentile.
    
    Args:
        row: Data row with atr_percentile
        min_percentile: Minimum percentile for optimal volatility
        max_percentile: Maximum percentile for optimal volatility
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    atr_percentile = row.get('atr_percentile', 50)
    
    if pd.isna(atr_percentile):
        return 0, "ATR percentile unavailable"
    
    if atr_percentile < min_percentile:
        return -1, f"Low volatility - choppy market ({atr_percentile:.0f}th percentile)"
    elif atr_percentile > max_percentile:
        return -1, f"High volatility - risky ({atr_percentile:.0f}th percentile)"
    else:
        return 1, f"Optimal volatility ({atr_percentile:.0f}th percentile)"


def score_trend_strength(row: pd.Series, threshold: float = 25) -> Tuple[int, str]:
    """
    Trend strength filter using ADX.
    
    Args:
        row: Data row with adx_14
        threshold: Minimum ADX for strong trend
    
    Returns:
        (vote, reason) where vote is -1, 0, or +1
    """
    adx_val = row.get('adx_14', 0)
    
    if pd.isna(adx_val):
        return 0, "ADX unavailable"
    
    if adx_val > threshold:
        return 1, f"Strong trend (ADX={adx_val:.1f})"
    elif adx_val > 20:
        return 0, f"Moderate trend (ADX={adx_val:.1f})"
    else:
        return -1, f"Weak/choppy trend - avoid (ADX={adx_val:.1f})"


def generate_signal_from_row(row: pd.Series, 
                            pair: str = "Unknown",
                            label: str = "Unknown",
                            timeframe: str = "15m",
                            sl_atr_multiplier: float = 1.5,
                            tp_atr_multiplier: float = 4.0,
                            trend_bias: Optional[str] = None,
                            volume_threshold: float = 1.2,
                            min_atr_percentile: int = 30,
                            max_atr_percentile: int = 90,
                            adx_threshold: float = 25) -> Signal:
    """
    Generate a signal from a single row of indicator data.
    
    This is the core logic function that backtesters should call iteratively.
    
    Args:
        row: Series with all indicator columns
        pair: Trading pair symbol
        label: Human-readable pair label
        timeframe: Timeframe string
        sl_atr_multiplier: Stop-loss multiplier for ATR
        tp_atr_multiplier: Take-profit multiplier for ATR
        trend_bias: Optional trend bias from higher timeframe
        volume_threshold: Minimum volume ratio for confirmation
        min_atr_percentile: Minimum ATR percentile for trading
        max_atr_percentile: Maximum ATR percentile for trading
        adx_threshold: Minimum ADX for trend strength
    
    Returns:
        Signal object
    """
    # Primary scoring (4 indicators)
    trend_score, trend_reason = score_trend(row)
    momentum_score, momentum_reason = score_momentum(row)
    macd_score, macd_reason = score_macd(row)
    meanrev_score, meanrev_reason = score_mean_reversion(row)
    
    # Quality filters (3 indicators)
    volume_score, volume_reason = score_volume(row, volume_threshold)
    volatility_score, volatility_reason = score_volatility(row, min_atr_percentile, max_atr_percentile)
    trend_strength_score, trend_strength_reason = score_trend_strength(row, adx_threshold)
    
    # Sum all scores (7 indicators now)
    total_score = (trend_score + momentum_score + macd_score + meanrev_score + 
                   volume_score + volatility_score + trend_strength_score)
    
    # Collect reasons
    reasons = []
    for score, reason in [
        (trend_score, trend_reason),
        (momentum_score, momentum_reason),
        (macd_score, macd_reason),
        (meanrev_score, meanrev_reason),
        (volume_score, volume_reason),
        (volatility_score, volatility_reason),
        (trend_strength_score, trend_strength_reason)
    ]:
        if score != 0:
            reasons.append(reason)
    
    # Calculate confidence (out of 7)
    max_score = 7
    confidence = int(abs(total_score) / max_score * 100)
    
    # Determine direction (at least 3 positive out of 7)
    if total_score >= 3:
        direction = "BUY"
    elif total_score <= -3:
        direction = "SELL"
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
                   trend_bias: Optional[str] = None,
                   volume_threshold: float = 1.2,
                   min_atr_percentile: int = 30,
                   max_atr_percentile: int = 90,
                   adx_threshold: float = 25) -> Signal:
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
        volume_threshold: Minimum volume ratio for confirmation
        min_atr_percentile: Minimum ATR percentile for trading
        max_atr_percentile: Maximum ATR percentile for trading
        adx_threshold: Minimum ADX for trend strength
    
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
        sl_atr_multiplier, tp_atr_multiplier, trend_bias,
        volume_threshold, min_atr_percentile, max_atr_percentile, adx_threshold
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