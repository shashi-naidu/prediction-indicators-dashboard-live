"""
Live Trading Chat Bot Assistant.
Answers user queries, provides technical indicator breakdowns, predicts trend consensus, and gives AI trading advice.
"""

from typing import Dict, Any, List
import pandas as pd
from consensus import calculate_consensus_prediction


def generate_chat_response(user_input: str, df: pd.DataFrame, symbol: str, consensus_data: Dict[str, Any] = None) -> str:
    """
    Generates intelligent responses for the trading chat based on live technical indicators and user query.
    """
    if consensus_data is None:
        consensus_data = calculate_consensus_prediction(df)

    user_msg = user_input.lower().strip()
    price = consensus_data['price']
    action = consensus_data['action']
    score = consensus_data['composite_score']
    pct = consensus_data['bullish_percentage']
    color_emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "🟡")

    # 1. RSI query
    if "rsi" in user_msg:
        rsi_item = next((item for item in consensus_data['indicators'] if item['key'] == 'RSI'), None)
        if rsi_item:
            return (
                f"📊 **RSI (14) Status for {symbol}**:\n"
                f"- **Value**: `{rsi_item['raw_value']}`\n"
                f"- **Signal**: {rsi_item['signal']}\n"
                f"- **Analysis**: {rsi_item['description']}\n\n"
                f"💡 *Rule*: RSI < 30 indicates oversold (buy opportunity), while RSI > 70 indicates overbought (sell signal)."
            )

    # 2. MACD query
    if "macd" in user_msg:
        macd_item = next((item for item in consensus_data['indicators'] if item['key'] == 'MACD'), None)
        if macd_item:
            return (
                f"📈 **MACD Analysis for {symbol}**:\n"
                f"- **Histogram / Status**: `{macd_item['raw_value']}`\n"
                f"- **Signal**: {macd_item['signal']}\n"
                f"- **Detail**: {macd_item['description']}\n\n"
                f"💡 *Rule*: Bullish when MACD line crosses above Signal line with green histogram expansion."
            )

    # 3. SuperTrend query
    if "supertrend" in user_msg:
        st_item = next((item for item in consensus_data['indicators'] if item['key'] == 'SuperTrend'), None)
        if st_item:
            return (
                f"🛡️ **SuperTrend Indicator for {symbol}**:\n"
                f"- **Value**: `{st_item['raw_value']}`\n"
                f"- **Trend Direction**: {st_item['signal']}\n"
                f"- **Detail**: {st_item['description']}"
            )

    # 4. Target / Stop Loss query
    if "target" in user_msg or "stop loss" in user_msg or "level" in user_msg:
        return (
            f"🎯 **Calculated Risk & Target Levels for {symbol} (Price: ${price:,.2f})**:\n"
            f"- **Bullish Target (Take Profit)**: `${consensus_data['take_profit_long']:,.2f}` (+3x ATR)\n"
            f"- **Bullish Stop Loss**: `${consensus_data['stop_loss_long']:,.2f}` (-1.5x ATR)\n"
            f"- **Bearish Target (Take Profit)**: `${consensus_data['take_profit_short']:,.2f}` (-3x ATR)\n"
            f"- **Bearish Stop Loss**: `${consensus_data['stop_loss_short']:,.2f}` (+1.5x ATR)\n"
            f"- **Average True Range (ATR)**: `${consensus_data['atr']:,.2f}`"
        )

    # 5. Overall Average / Prediction / Consensus query
    if any(k in user_msg for k in ["predict", "average", "overall", "consensus", "summary", "should i buy", "signal"]):
        ind_summary = "\n".join([f"- **{ind['name']}**: `{ind['raw_value']}` -> **{ind['signal']}** ({ind['description']})" for ind in consensus_data['indicators']])
        return (
            f"{color_emoji} **Live Overall Consensus Prediction for {symbol}**\n\n"
            f"- **Current Price**: `${price:,.2f}`\n"
            f"- **Overall Action**: **{action}**\n"
            f"- **Bullish Confidence**: `{pct:.1f}%` (Score: `{score:+.2f}` / `+1.00`)\n"
            f"- **Indicators Consensus Count**: 🟢 {consensus_data['bullish_count']} Bullish | 🔴 {consensus_data['bearish_count']} Bearish | 🟡 {consensus_data['neutral_count']} Neutral\n\n"
            f"📋 **Detailed Indicator Breakdown**:\n{ind_summary}\n\n"
            f"⚡ **Trading Recommendation**: Strategy calls for **{action}** with ATR Stop Loss at `${consensus_data['stop_loss_long']:,.2f}`."
        )

    # Default general chat response
    return (
        f"🤖 **Trading Assistant Bot ({symbol})**:\n\n"
        f"Current Price: `${price:,.2f}` | Prediction: {color_emoji} **{action}** (`{pct:.1f}%` Bullish Score)\n\n"
        f"You can ask me questions like:\n"
        f"• *'Predict overall average output'* or *'Summary'* \n"
        f"• *'What is RSI saying?'*\n"
        f"• *'MACD status'*\n"
        f"• *'Give me stop loss and take profit targets'*\n"
        f"• *'SuperTrend status'*"
    )
