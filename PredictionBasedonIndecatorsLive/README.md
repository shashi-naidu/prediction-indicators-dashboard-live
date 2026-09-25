# 📈 Live Trading Bot & Technical Indicators Consensus Predictor

An end-to-end Python trading system that aggregates famous technical indicators in real-time, calculates normalized signal outputs, predicts overall average output scores (-1.0 to +1.0 / 0% to 100% Bullish confidence), and provides an interactive Streamlit dashboard with a Live AI Trading Chat, strategy backtesting suite, and paper trading bot simulation.

---

## 🌟 Key Features

1. **Multiple Famous Technical Indicators Engine (`indicators.py`)**:
   - **RSI (14)**: Relative Strength Index (Oversold < 30, Overbought > 70)
   - **MACD (12, 26, 9)**: Convergence/Divergence line, signal line, & histogram momentum
   - **EMA Trend Crossovers**: EMA 20, 50, 200 Golden Cross / Death Cross alignment
   - **Bollinger Bands (20, 2.0)**: %B position & Volatility Bandwidth
   - **Stochastic Oscillator (%K, %D)**: 14-period momentum oscillator
   - **ADX & DI+/DI-**: Average Directional Index trend strength filter
   - **SuperTrend (10, 3.0)**: Dynamic ATR trailing trend direction (+1 Bull / -1 Bear)
   - **VWAP**: Volume Weighted Average Price benchmark
   - **Williams %R**: Reversal momentum oscillator
   - **CCI**: Commodity Channel Index trend momentum

2. **Consensus Engine & Overall Average Predictor (`consensus.py`)**:
   - Normalizes all 10+ technical indicators into a continuous signal score in `[-1.0, +1.0]`.
   - Computes weighted overall average composite score & percentage bullish confidence (`0%` to `100%`).
   - Categorizes market actions into: `STRONG BUY`, `BUY`, `NEUTRAL`, `SELL`, `STRONG SELL`.
   - Dynamic ATR-based Stop Loss & Take Profit price level targets.

3. **Live AI Trading Chat Assistant (`chat_bot.py`)**:
   - Responds to questions on live market output, specific indicators (RSI, MACD, SuperTrend), prediction scores, and target prices.

4. **Strategy Backtester (`backtester.py`)**:
   - Evaluates the aggregate consensus model against individual indicator strategies.
   - Calculates Total Return %, Win Rate %, Max Drawdown %, Profit Factor, Sharpe Ratio, and trade logs.

5. **Simulated Live Paper Trading Bot (`trading_bot.py`)**:
   - Simulates order execution, wallet balances, open position tracking, trailing stop losses, take profit triggers, and execution logs.

6. **Streamlit Multi-Tab Dashboard (`app.py`)**:
   - Interactive Plotly Candlestick charts with technical overlays & MACD/RSI/Consensus subcharts.
   - Real-time prediction gauge & indicator matrix table.
   - Customizable indicator weights slider panel.

---

## 🚀 How to Run the System

### 1. Run the Unit Test Suite
To verify indicator calculations, backtesting logic, consensus scoring, and bot execution:
```bash
python test_suite.py
```

### 2. Launch the Interactive Streamlit Web App
To start the live trading dashboard and AI chat interface:
```bash
python -m streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 📁 Repository File Structure

- `indicators.py` - Vectorized calculations for 10+ technical indicators.
- `consensus.py` - Indicator output normalization, weighted overall average calculation, and prediction engine.
- `data_stream.py` - Real-time market candle fetcher (yfinance) with fallback synthetic ticker stream.
- `backtester.py` - Historical strategy backtesting and performance analytics.
- `trading_bot.py` - Live paper trading bot execution and risk management engine.
- `chat_bot.py` - Natural language AI trading chat assistant logic.
- `app.py` - Streamlit multi-tab UI dashboard & interactive chat app.
- `test_suite.py` - Automated unit tests for all components.
