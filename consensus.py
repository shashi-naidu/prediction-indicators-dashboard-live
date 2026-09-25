"""
Consensus Engine & Overall Average Output Predictor.
Aggregates and normalizes outputs from multiple famous technical indicators,
computes individual normalized signals [-1.0 to +1.0], and predicts the overall average output.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


# Default Indicator Weights in the Overall Consensus Engine
DEFAULT_WEIGHTS = {
    'RSI': 1.5,
    'MACD': 1.8,
    'MA_Trend': 2.0,
    'BollingerBands': 1.2,
    'Stochastic': 1.2,
    'ADX': 1.5,
    'SuperTrend': 2.0,
    'VWAP': 1.2,
    'WilliamsR': 1.0,
    'CCI': 1.0
}


def evaluate_rsi(rsi_val: float) -> tuple[float, str]:
    """Evaluates RSI into [-1.0, +1.0] signal."""
    if np.isnan(rsi_val):
        return 0.0, "N/A"
    if rsi_val < 30:
        score = 0.8 + (30 - rsi_val) / 30 * 0.2  # Oversold (Bullish Reversal)
        return min(score, 1.0), f"Oversold ({rsi_val:.1f}) - Bullish Reversal"
    elif rsi_val > 70:
        score = -0.8 - (rsi_val - 70) / 30 * 0.2  # Overbought (Bearish Reversal)
        return max(score, -1.0), f"Overbought ({rsi_val:.1f}) - Bearish Reversal"
    elif rsi_val >= 50:
        score = (rsi_val - 50) / 20 * 0.6
        return score, f"Bullish Zone ({rsi_val:.1f})"
    else:
        score = (rsi_val - 50) / 20 * 0.6
        return score, f"Bearish Zone ({rsi_val:.1f})"


def evaluate_macd(macd_line: float, signal_line: float, hist: float) -> tuple[float, str]:
    """Evaluates MACD line, signal, and histogram into [-1.0, +1.0] signal."""
    if np.isnan(macd_line) or np.isnan(signal_line):
        return 0.0, "N/A"
    
    if macd_line > signal_line:
        if hist > 0:
            score = 0.6 + min(abs(hist) * 10, 0.4)
            desc = "Bullish Crossover & Growing Histogram"
        else:
            score = 0.3
            desc = "Bullish Alignment (Weak Momentum)"
    else:
        if hist < 0:
            score = -0.6 - min(abs(hist) * 10, 0.4)
            desc = "Bearish Crossover & Growing Histogram"
        else:
            score = -0.3
            desc = "Bearish Alignment (Weak Momentum)"
    return score, desc


def evaluate_ma_trend(price: float, ema20: float, ema50: float, ema200: float) -> tuple[float, str]:
    """Evaluates Moving Average trend alignment."""
    if np.isnan(ema20) or np.isnan(ema50) or np.isnan(ema200):
        return 0.0, "N/A"
    
    score = 0.0
    # Price position relative to EMAs
    if price > ema20: score += 0.3
    else: score -= 0.3
    
    if price > ema50: score += 0.3
    else: score -= 0.3
    
    if price > ema200: score += 0.4
    else: score -= 0.4

    # Moving Average Stacking
    if ema20 > ema50 > ema200:
        desc = "Perfect Bullish Stack (Golden Trend)"
    elif ema20 < ema50 < ema200:
        desc = "Perfect Bearish Stack (Death Trend)"
    elif price > ema200:
        desc = "Above 200 EMA (Long-term Bullish)"
    else:
        desc = "Below 200 EMA (Long-term Bearish)"
        
    return score, desc


def evaluate_bollinger(pct_b: float, close: float, upper: float, lower: float) -> tuple[float, str]:
    """Evaluates Bollinger Bands %B into signal."""
    if np.isnan(pct_b):
        return 0.0, "N/A"
    
    if pct_b < 0:
        return 0.9, f"Below Lower Band (${lower:.2f}) - Oversold"
    elif pct_b > 1.0:
        return -0.9, f"Above Upper Band (${upper:.2f}) - Overbought"
    elif pct_b > 0.7:
        return 0.4, f"Upper Range (%B: {pct_b:.2f})"
    elif pct_b < 0.3:
        return -0.4, f"Lower Range (%B: {pct_b:.2f})"
    else:
        return 0.0, f"Neutral Middle (%B: {pct_b:.2f})"


def evaluate_stochastic(stoch_k: float, stoch_d: float) -> tuple[float, str]:
    """Evaluates Stochastic Oscillator."""
    if np.isnan(stoch_k) or np.isnan(stoch_d):
        return 0.0, "N/A"
    
    if stoch_k < 20:
        if stoch_k > stoch_d:
            return 0.9, f"Oversold Bullish Crossover (%K: {stoch_k:.1f})"
        return 0.7, f"Oversold (%K: {stoch_k:.1f})"
    elif stoch_k > 80:
        if stoch_k < stoch_d:
            return -0.9, f"Overbought Bearish Crossover (%K: {stoch_k:.1f})"
        return -0.7, f"Overbought (%K: {stoch_k:.1f})"
    elif stoch_k > stoch_d:
        return 0.3, f"Bullish Cross (%K: {stoch_k:.1f})"
    else:
        return -0.3, f"Bearish Cross (%K: {stoch_k:.1f})"


def evaluate_adx(adx: float, plus_di: float, minus_di: float) -> tuple[float, str]:
    """Evaluates ADX & DI indicators."""
    if np.isnan(adx) or np.isnan(plus_di) or np.isnan(minus_di):
        return 0.0, "N/A"
    
    is_trending = adx >= 25
    trend_str = "Strong Trend" if is_trending else "Weak/Ranging Trend"
    
    if plus_di > minus_di:
        multiplier = 1.0 if is_trending else 0.5
        score = 0.8 * multiplier
        desc = f"{trend_str}: +DI > -DI (ADX: {adx:.1f})"
    else:
        multiplier = 1.0 if is_trending else 0.5
        score = -0.8 * multiplier
        desc = f"{trend_str}: -DI > +DI (ADX: {adx:.1f})"
    return score, desc


def evaluate_supertrend(supertrend_dir: float, supertrend_val: float) -> tuple[float, str]:
    """Evaluates SuperTrend direction."""
    if np.isnan(supertrend_dir):
        return 0.0, "N/A"
    
    if supertrend_dir == 1:
        return 1.0, f"Bullish (Support: ${supertrend_val:.2f})"
    else:
        return -1.0, f"Bearish (Resistance: ${supertrend_val:.2f})"


def evaluate_vwap(close: float, vwap: float) -> tuple[float, str]:
    """Evaluates Volume Weighted Average Price."""
    if np.isnan(vwap) or vwap == 0:
        return 0.0, "N/A"
    
    diff_pct = (close - vwap) / vwap * 100
    if close > vwap:
        score = min(0.3 + diff_pct * 0.1, 0.8)
        return score, f"Above VWAP (+{diff_pct:.2f}%)"
    else:
        score = max(-0.3 + diff_pct * 0.1, -0.8)
        return score, f"Below VWAP ({diff_pct:.2f}%)"


def evaluate_williams_r(w_r: float) -> tuple[float, str]:
    """Evaluates Williams %R."""
    if np.isnan(w_r):
        return 0.0, "N/A"
    
    if w_r < -80:
        return 0.8, f"Oversold ({w_r:.1f})"
    elif w_r > -20:
        return -0.8, f"Overbought ({w_r:.1f})"
    else:
        score = (w_r + 50) / 30 * 0.5
        return score, f"Mid Range ({w_r:.1f})"


def evaluate_cci(cci: float) -> tuple[float, str]:
    """Evaluates Commodity Channel Index (CCI)."""
    if np.isnan(cci):
        return 0.0, "N/A"
    
    if cci > 100:
        return 0.8, f"Strong Uptrend Momentum (CCI: {cci:.1f})"
    elif cci < -100:
        return -0.8, f"Strong Downtrend Momentum (CCI: {cci:.1f})"
    else:
        score = cci / 100.0 * 0.5
        return score, f"Neutral Channel (CCI: {cci:.1f})"


def calculate_consensus_prediction(df: pd.DataFrame, custom_weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Computes overall average output score and signal prediction for the latest candle in df.
    """
    weights = custom_weights or DEFAULT_WEIGHTS
    row = df.iloc[-1]
    price = float(row['Close'])
    atr = float(row['ATR']) if 'ATR' in df.columns and not np.isnan(row['ATR']) else price * 0.02

    indicators_eval = []

    # 1. RSI
    rsi_score, rsi_desc = evaluate_rsi(row.get('RSI', np.nan))
    indicators_eval.append({
        'name': 'RSI (14)',
        'key': 'RSI',
        'raw_value': f"{row.get('RSI', 0):.2f}",
        'score': rsi_score,
        'weight': weights.get('RSI', 1.0),
        'signal': 'BULLISH' if rsi_score > 0.15 else ('BEARISH' if rsi_score < -0.15 else 'NEUTRAL'),
        'description': rsi_desc
    })

    # 2. MACD
    macd_score, macd_desc = evaluate_macd(row.get('MACD', np.nan), row.get('MACD_Signal', np.nan), row.get('MACD_Hist', np.nan))
    indicators_eval.append({
        'name': 'MACD (12,26,9)',
        'key': 'MACD',
        'raw_value': f"Hist: {row.get('MACD_Hist', 0):.2f}",
        'score': macd_score,
        'weight': weights.get('MACD', 1.0),
        'signal': 'BULLISH' if macd_score > 0.15 else ('BEARISH' if macd_score < -0.15 else 'NEUTRAL'),
        'description': macd_desc
    })

    # 3. MA Trend
    ma_score, ma_desc = evaluate_ma_trend(price, row.get('EMA_20', np.nan), row.get('EMA_50', np.nan), row.get('EMA_200', np.nan))
    indicators_eval.append({
        'name': 'MA Trend (EMA 20/50/200)',
        'key': 'MA_Trend',
        'raw_value': f"EMA20: ${row.get('EMA_20', 0):.2f}",
        'score': ma_score,
        'weight': weights.get('MA_Trend', 1.0),
        'signal': 'BULLISH' if ma_score > 0.15 else ('BEARISH' if ma_score < -0.15 else 'NEUTRAL'),
        'description': ma_desc
    })

    # 4. Bollinger Bands
    bb_score, bb_desc = evaluate_bollinger(row.get('BB_PctB', np.nan), price, row.get('BB_Upper', 0), row.get('BB_Lower', 0))
    indicators_eval.append({
        'name': 'Bollinger Bands (%B)',
        'key': 'BollingerBands',
        'raw_value': f"%B: {row.get('BB_PctB', 0):.2f}",
        'score': bb_score,
        'weight': weights.get('BollingerBands', 1.0),
        'signal': 'BULLISH' if bb_score > 0.15 else ('BEARISH' if bb_score < -0.15 else 'NEUTRAL'),
        'description': bb_desc
    })

    # 5. Stochastic
    stoch_score, stoch_desc = evaluate_stochastic(row.get('Stoch_K', np.nan), row.get('Stoch_D', np.nan))
    indicators_eval.append({
        'name': 'Stochastic Oscillator',
        'key': 'Stochastic',
        'raw_value': f"%K: {row.get('Stoch_K', 0):.1f}",
        'score': stoch_score,
        'weight': weights.get('Stochastic', 1.0),
        'signal': 'BULLISH' if stoch_score > 0.15 else ('BEARISH' if stoch_score < -0.15 else 'NEUTRAL'),
        'description': stoch_desc
    })

    # 6. ADX
    adx_score, adx_desc = evaluate_adx(row.get('ADX', np.nan), row.get('Plus_DI', np.nan), row.get('Minus_DI', np.nan))
    indicators_eval.append({
        'name': 'ADX Trend Strength',
        'key': 'ADX',
        'raw_value': f"ADX: {row.get('ADX', 0):.1f}",
        'score': adx_score,
        'weight': weights.get('ADX', 1.0),
        'signal': 'BULLISH' if adx_score > 0.15 else ('BEARISH' if adx_score < -0.15 else 'NEUTRAL'),
        'description': adx_desc
    })

    # 7. SuperTrend
    st_score, st_desc = evaluate_supertrend(row.get('SuperTrend_Dir', np.nan), row.get('SuperTrend', price))
    indicators_eval.append({
        'name': 'SuperTrend (10, 3.0)',
        'key': 'SuperTrend',
        'raw_value': f"${row.get('SuperTrend', price):.2f}",
        'score': st_score,
        'weight': weights.get('SuperTrend', 1.0),
        'signal': 'BULLISH' if st_score > 0.15 else ('BEARISH' if st_score < -0.15 else 'NEUTRAL'),
        'description': st_desc
    })

    # 8. VWAP
    vwap_score, vwap_desc = evaluate_vwap(price, row.get('VWAP', np.nan))
    indicators_eval.append({
        'name': 'VWAP',
        'key': 'VWAP',
        'raw_value': f"${row.get('VWAP', price):.2f}",
        'score': vwap_score,
        'weight': weights.get('VWAP', 1.0),
        'signal': 'BULLISH' if vwap_score > 0.15 else ('BEARISH' if vwap_score < -0.15 else 'NEUTRAL'),
        'description': vwap_desc
    })

    # 9. Williams %R
    wr_score, wr_desc = evaluate_williams_r(row.get('Williams_R', np.nan))
    indicators_eval.append({
        'name': 'Williams %R',
        'key': 'WilliamsR',
        'raw_value': f"{row.get('Williams_R', 0):.1f}",
        'score': wr_score,
        'weight': weights.get('WilliamsR', 1.0),
        'signal': 'BULLISH' if wr_score > 0.15 else ('BEARISH' if wr_score < -0.15 else 'NEUTRAL'),
        'description': wr_desc
    })

    # 10. CCI
    cci_score, cci_desc = evaluate_cci(row.get('CCI', np.nan))
    indicators_eval.append({
        'name': 'CCI (Commodity Channel Index)',
        'key': 'CCI',
        'raw_value': f"{row.get('CCI', 0):.1f}",
        'score': cci_score,
        'weight': weights.get('CCI', 1.0),
        'signal': 'BULLISH' if cci_score > 0.15 else ('BEARISH' if cci_score < -0.15 else 'NEUTRAL'),
        'description': cci_desc
    })

    # Compute Overall Weighted Average Score [-1.0 to +1.0]
    total_weighted_score = sum(item['score'] * item['weight'] for item in indicators_eval)
    total_weight = sum(item['weight'] for item in indicators_eval)
    
    composite_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
    bullish_percentage = ((composite_score + 1.0) / 2.0) * 100.0

    # Determine Action Recommendation & Color
    if composite_score >= 0.5:
        action = "STRONG BUY"
        color = "#00C853"
    elif composite_score >= 0.2:
        action = "BUY"
        color = "#64DD17"
    elif composite_score <= -0.5:
        action = "STRONG SELL"
        color = "#D50000"
    elif composite_score <= -0.2:
        action = "SELL"
        color = "#FF1744"
    else:
        action = "NEUTRAL"
        color = "#FFC107"

    # Compute Support, Resistance, Target Price & Stop Loss
    stop_loss_long = price - (1.5 * atr)
    take_profit_long = price + (3.0 * atr)
    stop_loss_short = price + (1.5 * atr)
    take_profit_short = price - (3.0 * atr)

    bullish_count = sum(1 for item in indicators_eval if item['signal'] == 'BULLISH')
    bearish_count = sum(1 for item in indicators_eval if item['signal'] == 'BEARISH')
    neutral_count = sum(1 for item in indicators_eval if item['signal'] == 'NEUTRAL')

    return {
        'price': price,
        'composite_score': composite_score,
        'bullish_percentage': bullish_percentage,
        'action': action,
        'color': color,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count,
        'atr': atr,
        'stop_loss_long': stop_loss_long,
        'take_profit_long': take_profit_long,
        'stop_loss_short': stop_loss_short,
        'take_profit_short': take_profit_short,
        'indicators': indicators_eval
    }


def compute_historical_consensus_series(df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.Series:
    """
    Computes historical composite consensus score for every row in df (for backtesting & charting).
    """
    scores = []
    # For speed, compute vectorized indicator components
    r_rsi, _ = np.vectorize(evaluate_rsi)(df['RSI'].values)
    r_st, _ = np.vectorize(evaluate_supertrend)(df['SuperTrend_Dir'].values, df['SuperTrend'].values)
    r_wr, _ = np.vectorize(evaluate_williams_r)(df['Williams_R'].values)
    r_cci, _ = np.vectorize(evaluate_cci)(df['CCI'].values)
    
    # Simple vector aggregation for backtesting
    w = weights or DEFAULT_WEIGHTS
    
    # Vectorized MACD score
    macd_hist = df['MACD_Hist'].values
    r_macd = np.where(macd_hist > 0, 0.7, -0.7)
    
    # Vectorized MA score
    c = df['Close'].values
    e20 = df['EMA_20'].values
    e50 = df['EMA_50'].values
    e200 = df['EMA_200'].values
    r_ma = np.where((c > e20) & (e20 > e50), 0.8, np.where((c < e20) & (e20 < e50), -0.8, 0.0))
    
    # Vectorized BB score
    pct_b = df['BB_PctB'].values
    r_bb = np.where(pct_b < 0, 0.9, np.where(pct_b > 1.0, -0.9, (pct_b - 0.5) * 0.8))
    
    total_score = (
        r_rsi * w.get('RSI', 1.0) +
        r_macd * w.get('MACD', 1.0) +
        r_ma * w.get('MA_Trend', 1.0) +
        r_bb * w.get('BollingerBands', 1.0) +
        r_st * w.get('SuperTrend', 1.0) +
        r_wr * w.get('WilliamsR', 1.0) +
        r_cci * w.get('CCI', 1.0)
    )
    total_w = (
        w.get('RSI', 1.0) + w.get('MACD', 1.0) + w.get('MA_Trend', 1.0) +
        w.get('BollingerBands', 1.0) + w.get('SuperTrend', 1.0) +
        w.get('WilliamsR', 1.0) + w.get('CCI', 1.0)
    )
    
    comp_series = pd.Series(total_score / total_w, index=df.index).fillna(0.0)
    return comp_series
