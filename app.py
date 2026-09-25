"""
Live Trading Dashboard, Technical Indicators Engine, Consensus Predictor, Backtester & AI Chat Bot App.
Built with Streamlit and Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from indicators import calculate_all_indicators
from consensus import calculate_consensus_prediction, compute_historical_consensus_series, DEFAULT_WEIGHTS
from data_stream import fetch_market_data, PRESET_ASSETS
from backtester import run_backtest
from trading_bot import PaperTradingBot
from chat_bot import generate_chat_response

# Streamlit Page Config
st.set_page_config(
    page_title="Live Trading Bot & Indicator Consensus Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2A2E39;
        text-align: center;
    }
    .stApp {
        background-color: #131722;
        color: #D1D4DC;
    }
    .badge-buy {
        background-color: #00C853;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-sell {
        background-color: #D50000;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-neutral {
        background-color: #FFC107;
        color: black;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if 'paper_bot' not in st.session_state:
    st.session_state['paper_bot'] = PaperTradingBot()

if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = [
        {"role": "assistant", "content": "👋 Welcome to the Live Trading & Indicators Chat Bot! Select your asset and ask me anything about live indicator outputs, predictions, signals, or price targets."}
    ]


# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚡ Trading Parameters")

selected_asset_label = st.sidebar.selectbox(
    "Select Trading Pair / Stock",
    options=list(PRESET_ASSETS.keys()),
    format_func=lambda x: f"{x} ({PRESET_ASSETS[x]})"
)

custom_asset = st.sidebar.text_input("Or enter custom ticker symbol (e.g. MSFT, SOL-USD)", "")
asset_symbol = custom_asset.strip().upper() if custom_asset.strip() else selected_asset_label

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    timeframe = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
with col_sb2:
    period = st.selectbox("Period", ["1d", "7d", "1mo", "6mo", "1y"], index=1)

refresh_btn = st.sidebar.button("🔄 Refresh Market Data", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Indicator Weights in Consensus")

weights = {}
weights['RSI'] = st.sidebar.slider("RSI Weight", 0.0, 3.0, DEFAULT_WEIGHTS['RSI'], 0.1)
weights['MACD'] = st.sidebar.slider("MACD Weight", 0.0, 3.0, DEFAULT_WEIGHTS['MACD'], 0.1)
weights['MA_Trend'] = st.sidebar.slider("Moving Averages Weight", 0.0, 3.0, DEFAULT_WEIGHTS['MA_Trend'], 0.1)
weights['SuperTrend'] = st.sidebar.slider("SuperTrend Weight", 0.0, 3.0, DEFAULT_WEIGHTS['SuperTrend'], 0.1)
weights['BollingerBands'] = st.sidebar.slider("Bollinger Bands Weight", 0.0, 3.0, DEFAULT_WEIGHTS['BollingerBands'], 0.1)
weights['Stochastic'] = st.sidebar.slider("Stochastic Weight", 0.0, 3.0, DEFAULT_WEIGHTS['Stochastic'], 0.1)
weights['ADX'] = st.sidebar.slider("ADX Weight", 0.0, 3.0, DEFAULT_WEIGHTS['ADX'], 0.1)
weights['VWAP'] = st.sidebar.slider("VWAP Weight", 0.0, 3.0, DEFAULT_WEIGHTS['VWAP'], 0.1)
weights['WilliamsR'] = st.sidebar.slider("Williams %R Weight", 0.0, 3.0, DEFAULT_WEIGHTS['WilliamsR'], 0.1)
weights['CCI'] = st.sidebar.slider("CCI Weight", 0.0, 3.0, DEFAULT_WEIGHTS['CCI'], 0.1)


# --- FETCH DATA & COMPUTE INDICATORS ---
@st.cache_data(ttl=15, show_spinner=False)
def load_data(symbol, p, tf):
    df_raw = fetch_market_data(symbol, period=p, interval=tf)
    df_calculated = calculate_all_indicators(df_raw)
    return df_calculated

with st.spinner(f"Loading live market data for {asset_symbol}..."):
    df_data = load_data(asset_symbol, period, timeframe)

consensus = calculate_consensus_prediction(df_data, custom_weights=weights)


# APP HEADER
st.title(f"📊 Live Trading Bot & Consensus Output Predictor - `{asset_symbol}`")
st.caption(f"Real-time technical indicators aggregate engine & automated live bot | Last Update: {datetime.now().strftime('%H:%M:%S')}")

# CREATING MAIN NAVIGATION TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Market & Prediction Gauge",
    "💬 Live AI Trading Chat",
    "🔬 Backtesting & Indicator Suite",
    "🤖 Live Paper Trading Bot"
])


# ==============================================================================
# TAB 1: LIVE MARKET & PREDICTION GAUGE
# ==============================================================================
with tab1:
    # Metric Summary Row
    col1, col2, col3, col4, col5 = st.columns(5)

    price_diff = df_data['Close'].iloc[-1] - df_data['Close'].iloc[-2]
    price_diff_pct = (price_diff / df_data['Close'].iloc[-2]) * 100

    col1.metric("Current Price", f"${consensus['price']:,.2f}", f"{price_diff_pct:+.2f}%")
    col2.metric("Overall Prediction", consensus['action'], f"Score: {consensus['composite_score']:+.2f}")
    col3.metric("Bullish Confidence", f"{consensus['bullish_percentage']:.1f}%")
    col4.metric("Consensus Split", f"🟢 {consensus['bullish_count']} | 🔴 {consensus['bearish_count']} | 🟡 {consensus['neutral_count']}")
    col5.metric("Volatility (ATR)", f"${consensus['atr']:,.2f}")

    st.markdown("---")

    # Candlestick & Technical Indicators Chart
    st.subheader("🕯️ Interactive Technical Analysis & Consensus Signal Chart")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(f"{asset_symbol} Price Chart with Overlays", "MACD & RSI Subchart", "Overall Composite Average Output Score (-1.0 to +1.0)")
    )

    # Candlestick Plot
    fig.add_trace(go.Candlestick(
        x=df_data.index,
        open=df_data['Open'], high=df_data['High'],
        low=df_data['Low'], close=df_data['Close'],
        name='Price'
    ), row=1, col=1)

    # EMA Overlays
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['EMA_20'], line=dict(color='#00E5FF', width=1.2), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['EMA_50'], line=dict(color='#FFEA00', width=1.2), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['EMA_200'], line=dict(color='#FF2975', width=1.5), name='EMA 200'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['BB_Upper'], line=dict(color='rgba(255, 255, 255, 0.3)', dash='dot'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['BB_Lower'], line=dict(color='rgba(255, 255, 255, 0.3)', dash='dot'), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.03)', name='BB Lower'), row=1, col=1)

    # SuperTrend Line
    st_color = np.where(df_data['SuperTrend_Dir'] == 1, '#00E676', '#FF1744')
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['SuperTrend'], line=dict(color='#7C4DFF', width=1.8), name='SuperTrend'), row=1, col=1)

    # MACD Subchart
    colors_macd = np.where(df_data['MACD_Hist'] >= 0, '#00C853', '#D50000')
    fig.add_trace(go.Bar(x=df_data.index, y=df_data['MACD_Hist'], marker_color=colors_macd, name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['MACD'], line=dict(color='#29B6F6', width=1), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_data.index, y=df_data['MACD_Signal'], line=dict(color='#FFA726', width=1), name='Signal'), row=2, col=1)

    # Consensus Historical Score Line
    consensus_series = compute_historical_consensus_series(df_data, weights)
    score_colors = np.where(consensus_series >= 0.2, '#00C853', np.where(consensus_series <= -0.2, '#D50000', '#FFC107'))
    fig.add_trace(go.Scatter(x=df_data.index, y=consensus_series, line=dict(color='#00E5FF', width=2), name='Consensus Score'), row=3, col=1)

    # Add threshold lines
    fig.add_hline(y=0.2, line_dash="dash", line_color="#00C853", row=3, col=1)
    fig.add_hline(y=-0.2, line_dash="dash", line_color="#D50000", row=3, col=1)
    fig.add_hline(y=0.0, line_color="#787B86", row=3, col=1)

    fig.update_layout(
        height=750,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed Indicators Table
    st.subheader("📋 Famous Technical Indicators Output & Prediction Matrix")

    df_ind_matrix = pd.DataFrame(consensus['indicators'])
    df_ind_matrix = df_ind_matrix[['name', 'raw_value', 'signal', 'score', 'weight', 'description']]
    df_ind_matrix.columns = ['Indicator', 'Live Output Value', 'Individual Signal', 'Score (-1 to +1)', 'Weight', 'Analysis & Explanation']

    st.dataframe(
        df_ind_matrix,
        use_container_width=True,
        hide_index=True
    )

    # Risk Targets Box
    st.subheader("🎯 Calculated ATR Target & Risk Management Levels")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.info(f"🟢 **Bullish Take Profit (3x ATR)**\n\n`${consensus['take_profit_long']:,.2f}`")
    tc2.error(f"🔴 **Bullish Stop Loss (1.5x ATR)**\n\n`${consensus['stop_loss_long']:,.2f}`")
    tc3.info(f"🔴 **Bearish Take Profit (3x ATR)**\n\n`${consensus['take_profit_short']:,.2f}`")
    tc4.error(f"🟢 **Bearish Stop Loss (1.5x ATR)**\n\n`${consensus['stop_loss_short']:,.2f}`")


# ==============================================================================
# TAB 2: LIVE AI TRADING CHAT
# ==============================================================================
with tab2:
    st.subheader(f"💬 Live Trading Chat Assistant & Prediction Feed ({asset_symbol})")
    st.caption("Ask questions about live indicators, predictions, target levels, or market signals.")

    # Quick prompt buttons
    qp_cols = st.columns(4)
    qp1 = qp_cols[0].button("🔮 Predict overall average output", use_container_width=True)
    qp2 = qp_cols[1].button("📊 RSI & MACD Analysis", use_container_width=True)
    qp3 = qp_cols[2].button("🎯 Stop Loss & Targets", use_container_width=True)
    qp4 = qp_cols[3].button("🛡️ SuperTrend Status", use_container_width=True)

    prompt_to_send = None
    if qp1: prompt_to_send = "predict overall average output"
    elif qp2: prompt_to_send = "RSI and MACD status"
    elif qp3: prompt_to_send = "give me stop loss and take profit targets"
    elif qp4: prompt_to_send = "SuperTrend status"

    # Display chat history
    for msg in st.session_state['chat_messages']:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # User Input Box
    user_input = st.chat_input("Type your question about live indicators or trading prediction...")
    if prompt_to_send:
        user_input = prompt_to_send

    if user_input:
        st.session_state['chat_messages'].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = generate_chat_response(user_input, df_data, asset_symbol, consensus)
        st.session_state['chat_messages'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


# ==============================================================================
# TAB 3: BACKTESTING & STRATEGY EVALUATION
# ==============================================================================
with tab3:
    st.subheader("🔬 Historical Backtesting & Strategy Performance Comparison")
    st.caption("Test how the overall consensus indicator prediction model performs against individual indicators.")

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        initial_cap = st.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
    with b_col2:
        buy_thresh = st.slider("Consensus Buy Threshold", 0.05, 0.8, 0.20, 0.05)
    with b_col3:
        sell_thresh = st.slider("Consensus Sell Threshold", -0.8, -0.05, -0.20, 0.05)
    with b_col4:
        trading_fee = st.number_input("Trading Fee (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100.0

    run_bt_btn = st.button("🚀 Run Comprehensive Backtest", use_container_width=True)

    # Run backtests for Consensus and benchmark strategies
    strategies_to_test = ["Consensus", "RSI", "MACD", "SuperTrend", "EMA_Cross"]
    results_list = []

    for strat in strategies_to_test:
        res = run_backtest(
            df_data,
            strategy_type=strat,
            initial_capital=initial_cap,
            buy_threshold=buy_thresh,
            sell_threshold=sell_thresh,
            fee_pct=trading_fee
        )
        results_list.append(res)

    # Performance Comparison Table
    st.markdown("### 🏆 Strategy Comparison Results")
    df_res_summary = pd.DataFrame([{
        'Strategy': r['strategy_name'],
        'Final Capital': f"${r['final_capital']:,.2f}",
        'Total Return (%)': f"{r['total_return_pct']:+.2f}%",
        'Buy & Hold Return': f"{r['buy_and_hold_return_pct']:+.2f}%",
        'Win Rate (%)': f"{r['win_rate_pct']:.1f}%",
        'Total Trades': r['total_trades'],
        'Profit Factor': f"{r['profit_factor']:.2f}",
        'Max Drawdown (%)': f"{r['max_drawdown_pct']:.2f}%",
        'Sharpe Ratio': f"{r['sharpe_ratio']:.2f}"
    } for r in results_list])

    st.dataframe(df_res_summary, use_container_width=True, hide_index=True)

    # Equity Curve Comparison Chart
    fig_equity = go.Figure()
    for r in results_list:
        fig_equity.add_trace(go.Scatter(
            x=r['equity_curve'].index,
            y=r['equity_curve']['Equity'],
            mode='lines',
            name=f"{r['strategy_name']} Strategy"
        ))

    fig_equity.update_layout(
        title="Equity Growth Comparison Over Time",
        xaxis_title="Date",
        yaxis_title="Account Equity ($)",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Detailed Trades Log for Consensus Strategy
    st.markdown("### 📜 Consensus Strategy Trade Log")
    consensus_res = results_list[0]
    if len(consensus_res['trades_list']) > 0:
        df_trades = pd.DataFrame(consensus_res['trades_list'])
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed during this backtest window for the given thresholds.")


# ==============================================================================
# TAB 4: LIVE PAPER TRADING BOT CONTROL
# ==============================================================================
with tab4:
    st.subheader(f"🤖 Live Paper Trading Bot Simulation ({asset_symbol})")
    st.caption("Automated bot executes trades based on live consensus indicators signal.")

    bot = st.session_state['paper_bot']

    bot_col1, bot_col2, bot_col3 = st.columns(3)
    with bot_col1:
        if not bot.is_active:
            if st.button("▶️ START BOT", use_container_width=True, type="primary"):
                bot.start()
                st.rerun()
        else:
            if st.button("⏹️ STOP BOT", use_container_width=True):
                bot.stop()
                st.rerun()

    with bot_col2:
        st.write(f"**Bot Status**: {'🟢 ACTIVE' if bot.is_active else '🔴 INACTIVE'}")

    with bot_col3:
        # Process tick
        tick_data = bot.process_tick(df_data)

    st.markdown("---")

    # Bot Account Summary Cards
    p1, p2, p3 = st.columns(3)
    p1.metric("Available Cash Balance", f"${bot.cash:,.2f}")
    p2.metric("Portfolio Value", f"${tick_data['portfolio_value']:,.2f}")
    
    net_pnl = tick_data['portfolio_value'] - bot.initial_balance
    net_pnl_pct = (net_pnl / bot.initial_balance) * 100
    p3.metric("Total Net PnL", f"${net_pnl:+,.2f}", f"{net_pnl_pct:+.2f}%")

    st.markdown("---")

    # Open Position Details
    st.subheader("📌 Current Open Position")
    if bot.position:
        pos = bot.position
        p_type = pos['type']
        p_entry = pos['entry_price']
        curr_price = consensus['price']

        if p_type == 'LONG':
            unrealized_pnl = pos['size'] * (curr_price - p_entry)
            unrealized_pnl_pct = (curr_price - p_entry) / p_entry * 100
        else:
            unrealized_pnl = pos['size'] * (p_entry - curr_price)
            unrealized_pnl_pct = (p_entry - curr_price) / p_entry * 100

        pos_col1, pos_col2, pos_col3, pos_col4 = st.columns(4)
        pos_col1.metric("Position Type", p_type)
        pos_col2.metric("Entry Price", f"${p_entry:,.2f}")
        pos_col3.metric("Current Price", f"${curr_price:,.2f}")
        pos_col4.metric("Unrealized PnL", f"${unrealized_pnl:+,.2f}", f"{unrealized_pnl_pct:+.2f}%")
    else:
        st.info("No open positions currently. Bot is waiting for consensus entry signal.")

    # Execution Logs & Trade History
    st.markdown("---")
    l_col1, l_col2 = st.columns(2)

    with l_col1:
        st.subheader("📜 Bot Execution Activity Logs")
        if bot.logs:
            for log in reversed(bot.logs[-10:]):
                st.text(f"[{log['time']}] [{log['level']}] {log['message']}")
        else:
            st.text("No log events yet.")

    with l_col2:
        st.subheader("🏁 Closed Trades History")
        if bot.trades_history:
            st.dataframe(pd.DataFrame(bot.trades_history), use_container_width=True)
        else:
            st.text("No trades closed yet.")
