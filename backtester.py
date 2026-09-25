"""
Backtesting & Strategy Performance Engine.
Simulates trading based on indicator consensus and individual indicator strategies.
Computes return, win rate, drawdown, Sharpe ratio, and profit factor.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from consensus import compute_historical_consensus_series


def run_backtest(
    df: pd.DataFrame,
    strategy_type: str = "Consensus",
    initial_capital: float = 10000.0,
    buy_threshold: float = 0.2,
    sell_threshold: float = -0.2,
    fee_pct: float = 0.001
) -> Dict[str, Any]:
    """
    Executes a backtest on the provided dataset containing calculated indicators.
    
    strategy_type: 'Consensus', 'RSI', 'MACD', 'SuperTrend', or 'EMA_Cross'
    """
    df = df.copy()

    # Generate signals based on strategy
    if strategy_type == "Consensus":
        df['Signal_Score'] = compute_historical_consensus_series(df)
    elif strategy_type == "RSI":
        rsi = df['RSI']
        df['Signal_Score'] = np.where(rsi < 30, 0.8, np.where(rsi > 70, -0.8, 0.0))
    elif strategy_type == "MACD":
        hist = df['MACD_Hist']
        df['Signal_Score'] = np.where(hist > 0, 0.7, -0.7)
    elif strategy_type == "SuperTrend":
        df['Signal_Score'] = df['SuperTrend_Dir'].astype(float)
    elif strategy_type == "EMA_Cross":
        df['Signal_Score'] = np.where(df['EMA_12'] > df['EMA_26'], 0.8, -0.8)
    else:
        df['Signal_Score'] = compute_historical_consensus_series(df)

    # Trading Logic
    position = 0  # 1 = Long, -1 = Short, 0 = Cash
    capital = initial_capital
    entry_price = 0.0
    entry_time = None

    equity_curve = []
    trades = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        score = df['Signal_Score'].iloc[i]

        # Check exit or position change
        if position == 1:
            if score <= sell_threshold:  # Close Long
                pnl_pct = (price - entry_price) / entry_price - (2 * fee_pct)
                pnl_dollar = capital * pnl_pct
                capital += pnl_dollar
                trades.append({
                    'Type': 'LONG',
                    'EntryTime': entry_time,
                    'ExitTime': date,
                    'EntryPrice': entry_price,
                    'ExitPrice': price,
                    'PnL_%': pnl_pct * 100,
                    'PnL_$': pnl_dollar,
                    'Capital': capital
                })
                position = 0
                if score <= sell_threshold:  # Open Short
                    position = -1
                    entry_price = price
                    entry_time = date

        elif position == -1:
            if score >= buy_threshold:  # Close Short
                pnl_pct = (entry_price - price) / entry_price - (2 * fee_pct)
                pnl_dollar = capital * pnl_pct
                capital += pnl_dollar
                trades.append({
                    'Type': 'SHORT',
                    'EntryTime': entry_time,
                    'ExitTime': date,
                    'EntryPrice': entry_price,
                    'ExitPrice': price,
                    'PnL_%': pnl_pct * 100,
                    'PnL_$': pnl_dollar,
                    'Capital': capital
                })
                position = 0
                if score >= buy_threshold:  # Open Long
                    position = 1
                    entry_price = price
                    entry_time = date

        elif position == 0:
            if score >= buy_threshold:
                position = 1
                entry_price = price
                entry_time = date
            elif score <= sell_threshold:
                position = -1
                entry_price = price
                entry_time = date

        # Track unrealized equity
        if position == 1:
            unrealized = capital * (1 + (price - entry_price) / entry_price)
        elif position == -1:
            unrealized = capital * (1 + (entry_price - price) / entry_price)
        else:
            unrealized = capital

        equity_curve.append(unrealized)

    df_equity = pd.DataFrame({'Equity': equity_curve}, index=df.index)

    # Buy & Hold return
    bh_return_pct = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    # Calculate Performance Statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['PnL_$'] > 0]
    losing_trades = [t for t in trades if t['PnL_$'] < 0]

    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0
    total_return_pct = (capital - initial_capital) / initial_capital * 100

    gross_profit = sum(t['PnL_$'] for t in winning_trades)
    gross_loss = abs(sum(t['PnL_$'] for t in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 1.0)

    # Max Drawdown
    peak = df_equity['Equity'].cummax()
    drawdown = (df_equity['Equity'] - peak) / peak
    max_drawdown_pct = abs(drawdown.min() * 100) if not np.isnan(drawdown.min()) else 0.0

    # Sharpe Ratio
    daily_returns = df_equity['Equity'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * np.sqrt(252) if len(daily_returns) > 1 else 0.0

    return {
        'strategy_name': strategy_type,
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return_pct': total_return_pct,
        'buy_and_hold_return_pct': bh_return_pct,
        'total_trades': total_trades,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown_pct,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': df_equity,
        'trades_list': trades
    }
