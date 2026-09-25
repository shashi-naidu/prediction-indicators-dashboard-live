"""
Technical Indicators Engine for Live Trading Analysis & Prediction.
Computes famous technical indicators with vectorized Pandas/Numpy implementations:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA / EMA Crossovers (20, 50, 200)
- Bollinger Bands (BB Upper, Lower, %B, Bandwidth)
- Stochastic Oscillator (%K, %D)
- ADX (Average Directional Index) & DI+/DI-
- ATR (Average True Range)
- SuperTrend (Trend & Dynamic Trailing Stop)
- VWAP (Volume Weighted Average Price)
- Williams %R
- CCI (Commodity Channel Index)
"""

import pandas as pd
import numpy as np


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Hist': macd_hist
    }, index=df.index)


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate SMA and EMA for standard periods (20, 50, 200)."""
    res = pd.DataFrame(index=df.index)
    res['SMA_20'] = df['Close'].rolling(window=20).mean()
    res['SMA_50'] = df['Close'].rolling(window=50).mean()
    res['SMA_200'] = df['Close'].rolling(window=200).mean()
    res['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    res['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    res['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    res['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    res['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return res


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands, %B and Bandwidth."""
    sma = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()

    upper = sma + (std_mult * rolling_std)
    lower = sma - (std_mult * rolling_std)
    pct_b = (df['Close'] - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / (sma + 1e-10)

    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': sma,
        'BB_Lower': lower,
        'BB_PctB': pct_b,
        'BB_Bandwidth': bandwidth
    }, index=df.index)


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator (%K, %D)."""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()

    pct_k = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    pct_d = pct_k.rolling(window=d_period).mean()

    return pd.DataFrame({
        'Stoch_K': pct_k,
        'Stoch_D': pct_d
    }, index=df.index)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX) along with +DI and -DI."""
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = compute_atr(df, period=period)

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({
        'ADX': adx,
        'Plus_DI': plus_di,
        'Minus_DI': minus_di
    }, index=df.index)


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate SuperTrend indicator line and direction (+1 for Bull, -1 for Bear)."""
    atr = compute_atr(df, period=period)
    hl2 = (df['High'] + df['Low']) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    n = len(df)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)  # 1 = Bullish, -1 = Bearish

    close = df['Close'].values

    for i in range(1, n):
        # Upper Band
        if basic_upper.iloc[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            upper_band[i] = basic_upper.iloc[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # Lower Band
        if basic_lower.iloc[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            lower_band[i] = basic_lower.iloc[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # SuperTrend Direction
        if direction[i - 1] == 1:
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    return pd.DataFrame({
        'SuperTrend': supertrend,
        'SuperTrend_Dir': direction
    }, index=df.index)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].replace(0, 1e-5)
    cum_tp_vol = (typical_price * vol).cumsum()
    cum_vol = vol.cumsum()
    return cum_tp_vol / cum_vol


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    high_max = df['High'].rolling(window=period).max()
    low_min = df['Low'].rolling(window=period).min()
    w_r = -100 * ((high_max - df['Close']) / (high_max - low_min + 1e-10))
    return w_r


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index (CCI)."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    return cci


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical indicators and attaches them as columns to the dataframe.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute individual indicators
    df['RSI'] = compute_rsi(df)

    macd_df = compute_macd(df)
    df['MACD'] = macd_df['MACD']
    df['MACD_Signal'] = macd_df['MACD_Signal']
    df['MACD_Hist'] = macd_df['MACD_Hist']

    ma_df = compute_moving_averages(df)
    for col in ma_df.columns:
        df[col] = ma_df[col]

    bb_df = compute_bollinger_bands(df)
    for col in bb_df.columns:
        df[col] = bb_df[col]

    stoch_df = compute_stochastic(df)
    df['Stoch_K'] = stoch_df['Stoch_K']
    df['Stoch_D'] = stoch_df['Stoch_D']

    df['ATR'] = compute_atr(df)

    adx_df = compute_adx(df)
    df['ADX'] = adx_df['ADX']
    df['Plus_DI'] = adx_df['Plus_DI']
    df['Minus_DI'] = adx_df['Minus_DI']

    st_df = compute_supertrend(df)
    df['SuperTrend'] = st_df['SuperTrend']
    df['SuperTrend_Dir'] = st_df['SuperTrend_Dir']

    df['VWAP'] = compute_vwap(df)
    df['Williams_R'] = compute_williams_r(df)
    df['CCI'] = compute_cci(df)

    return df
