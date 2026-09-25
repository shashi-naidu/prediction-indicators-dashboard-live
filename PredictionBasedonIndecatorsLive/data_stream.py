"""
Live & Historical Data Stream Fetcher.
Fetches live price data and candlestick history via yfinance, with fallback to synthetic live ticker simulation.
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataStream")


# Popular preset assets for live trading & backtesting
PRESET_ASSETS = {
    'BTC-USD': 'Bitcoin / USD',
    'ETH-USD': 'Ethereum / USD',
    'SOL-USD': 'Solana / USD',
    'AAPL': 'Apple Inc.',
    'NVDA': 'NVIDIA Corp.',
    'TSLA': 'Tesla Inc.',
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Invesco QQQ (Nasdaq)',
    'EURUSD=X': 'EUR / USD Forex',
    'GC=F': 'Gold Futures'
}


def generate_synthetic_candles(symbol: str = "SIM-BTC", num_bars: int = 300, interval: str = "5m") -> pd.DataFrame:
    """Generates realistic synthetic OHLCV candle data for offline testing or demo mode."""
    end_time = datetime.now()
    if "m" in interval:
        mins = int(interval.replace("m", ""))
        times = [end_time - timedelta(minutes=mins * i) for i in range(num_bars)][::-1]
    elif "h" in interval:
        hrs = int(interval.replace("h", ""))
        times = [end_time - timedelta(hours=hrs * i) for i in range(num_bars)][::-1]
    else:
        times = [end_time - timedelta(days=i) for i in range(num_bars)][::-1]

    base_price = 50000.0 if "BTC" in symbol else (250.0 if "AAPL" in symbol or "NVDA" in symbol else 100.0)
    np.random.seed(42)
    
    # Drift and volatility
    returns = np.random.normal(loc=0.0002, scale=0.008, size=num_bars)
    price_path = base_price * np.exp(np.cumsum(returns))

    opens = price_path * (1 + np.random.normal(0, 0.001, num_bars))
    closes = price_path
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, num_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, num_bars)))
    volumes = np.random.randint(1000, 50000, size=num_bars)

    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=pd.DatetimeIndex(times))
    
    df.index.name = 'Date'
    return df


def fetch_market_data(symbol: str = "BTC-USD", period: str = "7d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetches historical & live candlestick market data for a symbol.
    Falls back to synthetic generator if download fails or returns empty data.
    """
    logger.info(f"Fetching data for {symbol} (period={period}, interval={interval})...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df is not None and len(df) >= 20:
            # Clean up columns
            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[cols].dropna()
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}.")
            return df
        else:
            logger.warning(f"Empty or insufficient data returned for {symbol}. Using synthetic simulation.")
            return generate_synthetic_candles(symbol=symbol, num_bars=300, interval=interval)
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}. Falling back to synthetic candles.")
        return generate_synthetic_candles(symbol=symbol, num_bars=300, interval=interval)
