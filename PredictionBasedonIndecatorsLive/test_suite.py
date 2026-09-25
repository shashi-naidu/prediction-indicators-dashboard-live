"""
Comprehensive Automated Test Suite.
Tests indicators calculation, consensus scoring engine, data fetcher, backtesting system, paper trading bot, and chat assistant.
"""

import sys
import unittest
import pandas as pd
import numpy as np

from indicators import calculate_all_indicators, compute_rsi, compute_macd, compute_supertrend
from consensus import calculate_consensus_prediction, compute_historical_consensus_series
from data_stream import generate_synthetic_candles, fetch_market_data
from backtester import run_backtest
from trading_bot import PaperTradingBot
from chat_bot import generate_chat_response


class TestTradingSystem(unittest.TestCase):

    def setUp(self):
        # Create reproducible sample candle dataframe
        self.df_raw = generate_synthetic_candles("BTC-USD", num_bars=150, interval="5m")
        self.df_ind = calculate_all_indicators(self.df_raw)

    def test_indicator_calculations(self):
        """Test that all technical indicators are correctly calculated with expected columns."""
        expected_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA_20', 'EMA_20', 'EMA_50', 'EMA_200',
            'BB_Upper', 'BB_Lower', 'BB_PctB', 'Stoch_K', 'Stoch_D', 'ATR', 'ADX', 'Plus_DI', 'Minus_DI',
            'SuperTrend', 'SuperTrend_Dir', 'VWAP', 'Williams_R', 'CCI'
        ]
        for col in expected_cols:
            self.assertIn(col, self.df_ind.columns, f"Missing indicator column {col}")

        # Check bounds for RSI (0 to 100)
        valid_rsi = self.df_ind['RSI'].dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI values out of [0, 100] bounds")

        # Check SuperTrend direction (+1 or -1)
        st_dirs = self.df_ind['SuperTrend_Dir'].dropna()
        self.assertTrue(set(st_dirs.unique()).issubset({1, -1}), "SuperTrend direction must be 1 or -1")

    def test_consensus_prediction_engine(self):
        """Test aggregate consensus scoring engine output structure and math."""
        consensus = calculate_consensus_prediction(self.df_ind)
        
        self.assertIn('composite_score', consensus)
        self.assertIn('action', consensus)
        self.assertIn('bullish_percentage', consensus)

        score = consensus['composite_score']
        self.assertGreaterEqual(score, -1.0, "Composite score < -1.0")
        self.assertLessEqual(score, 1.0, "Composite score > 1.0")

        pct = consensus['bullish_percentage']
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)

        action = consensus['action']
        self.assertIn(action, ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"])

        self.assertEqual(len(consensus['indicators']), 10, "Should evaluate 10 indicators")

    def test_historical_consensus_series(self):
        """Test calculation of historical consensus series for backtesting."""
        series = compute_historical_consensus_series(self.df_ind)
        self.assertEqual(len(series), len(self.df_ind))
        self.assertFalse(series.isna().any(), "Consensus series contains NaNs")

    def test_backtester(self):
        """Test backtester execution and metrics calculation."""
        res = run_backtest(self.df_ind, strategy_type="Consensus", initial_capital=10000.0)
        
        self.assertEqual(res['strategy_name'], "Consensus")
        self.assertIn('total_return_pct', res)
        self.assertIn('win_rate_pct', res)
        self.assertIn('equity_curve', res)
        self.assertGreater(len(res['equity_curve']), 0)

    def test_paper_trading_bot(self):
        """Test paper trading bot tick processing and order execution."""
        bot = PaperTradingBot(symbol="BTC-USD", initial_balance=10000.0)
        bot.start()
        self.assertTrue(bot.is_active)

        tick_result = bot.process_tick(self.df_ind)
        self.assertIn('portfolio_value', tick_result)
        self.assertIn('consensus', tick_result)

    def test_chat_bot(self):
        """Test AI trading chat assistant response generation."""
        resp_rsi = generate_chat_response("What is RSI saying?", self.df_ind, "BTC-USD")
        self.assertIn("RSI", resp_rsi)

        resp_overall = generate_chat_response("predict overall average output", self.df_ind, "BTC-USD")
        self.assertIn("Consensus Prediction", resp_overall)


if __name__ == '__main__':
    unittest.main()
