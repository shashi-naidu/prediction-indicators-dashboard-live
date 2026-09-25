"""
Live Paper Trading Bot Engine.
Simulates live position management, risk management (Stop Loss & Take Profit), order execution, and performance logging.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from consensus import calculate_consensus_prediction


class PaperTradingBot:
    def __init__(
        self,
        symbol: str = "BTC-USD",
        initial_balance: float = 10000.0,
        trade_size_pct: float = 0.20,  # 20% capital per trade
        stop_loss_pct: float = 0.02,   # 2% Stop Loss
        take_profit_pct: float = 0.04,  # 4% Take Profit
        buy_consensus_threshold: float = 0.25,
        sell_consensus_threshold: float = -0.25
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.trade_size_pct = trade_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.buy_threshold = buy_consensus_threshold
        self.sell_threshold = sell_consensus_threshold
        
        self.is_active = False
        self.position: Optional[Dict[str, Any]] = None  # Current open trade
        self.trades_history: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []

    def start(self):
        self.is_active = True
        self._add_log("INFO", f"Bot started paper trading on {self.symbol} with initial balance ${self.initial_balance:,.2f}.")

    def stop(self):
        self.is_active = False
        self._add_log("INFO", f"Bot stopped.")

    def _add_log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            'time': timestamp,
            'level': level,
            'message': message
        })

    def process_tick(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Processes latest candle tick and executes paper trade orders if criteria are met."""
        if len(df) == 0:
            return {'status': 'NO_DATA'}

        latest_price = float(df['Close'].iloc[-1])
        latest_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S") if hasattr(df.index[-1], 'strftime') else str(df.index[-1])
        
        # Calculate live consensus
        consensus = calculate_consensus_prediction(df)
        composite_score = consensus['composite_score']
        action = consensus['action']

        # 1. Check existing position for Stop Loss or Take Profit
        if self.position is not None:
            pos_type = self.position['type']
            entry = self.position['entry_price']
            size = self.position['size']
            
            if pos_type == 'LONG':
                pnl_pct = (latest_price - entry) / entry
                # Check Stop Loss
                if pnl_pct <= -self.stop_loss_pct:
                    self._close_position(latest_price, latest_time, f"STOP LOSS (-{self.stop_loss_pct*100:.1f}%)")
                # Check Take Profit
                elif pnl_pct >= self.take_profit_pct:
                    self._close_position(latest_price, latest_time, f"TAKE PROFIT (+{self.take_profit_pct*100:.1f}%)")
                # Check Sell Consensus reversal signal
                elif composite_score <= self.sell_threshold:
                    self._close_position(latest_price, latest_time, f"SIGNAL REVERSAL ({consensus['action']})")

            elif pos_type == 'SHORT':
                pnl_pct = (entry - latest_price) / entry
                # Check Stop Loss
                if pnl_pct <= -self.stop_loss_pct:
                    self._close_position(latest_price, latest_time, f"STOP LOSS (-{self.stop_loss_pct*100:.1f}%)")
                # Check Take Profit
                elif pnl_pct >= self.take_profit_pct:
                    self._close_position(latest_price, latest_time, f"TAKE PROFIT (+{self.take_profit_pct*100:.1f}%)")
                # Check Buy Consensus reversal signal
                elif composite_score >= self.buy_threshold:
                    self._close_position(latest_price, latest_time, f"SIGNAL REVERSAL ({consensus['action']})")

        # 2. Open new position if no active trade and bot is active
        if self.position is None and self.is_active:
            if composite_score >= self.buy_threshold:
                self._open_position('LONG', latest_price, latest_time, consensus)
            elif composite_score <= self.sell_threshold:
                self._open_position('SHORT', latest_price, latest_time, consensus)

        # Total portfolio equity value
        portfolio_value = self.cash
        if self.position:
            entry = self.position['entry_price']
            size = self.position['size']
            if self.position['type'] == 'LONG':
                portfolio_value += size * (latest_price - entry)
            else:
                portfolio_value += size * (entry - latest_price)

        return {
            'timestamp': latest_time,
            'price': latest_price,
            'consensus': consensus,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'open_position': self.position,
            'recent_logs': self.logs[-5:]
        }

    def _open_position(self, pos_type: str, price: float, timestamp: str, consensus: Dict[str, Any]):
        trade_capital = self.cash * self.trade_size_pct
        size = trade_capital / price
        
        self.position = {
            'type': pos_type,
            'entry_price': price,
            'entry_time': timestamp,
            'size': size,
            'capital': trade_capital,
            'consensus_score': consensus['composite_score'],
            'action': consensus['action']
        }
        self._add_log("EXECUTION", f"BUY ORDER EXECUTED [{pos_type}]: {size:.4f} units @ ${price:,.2f} (Score: {consensus['composite_score']:.2f})")

    def _close_position(self, price: float, timestamp: str, reason: str):
        if not self.position:
            return

        pos_type = self.position['type']
        entry = self.position['entry_price']
        size = self.position['size']

        if pos_type == 'LONG':
            pnl_dollar = size * (price - entry)
            pnl_pct = (price - entry) / entry * 100
        else:
            pnl_dollar = size * (entry - price)
            pnl_pct = (entry - price) / entry * 100

        self.cash += pnl_dollar

        trade_record = {
            'type': pos_type,
            'entry_price': entry,
            'exit_price': price,
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'reason': reason,
            'final_cash': self.cash
        }

        self.trades_history.append(trade_record)
        self.position = None

        log_type = "PROFIT" if pnl_dollar > 0 else "LOSS"
        self._add_log("EXECUTION", f"CLOSED [{pos_type}] via {reason} @ ${price:,.2f} | PnL: ${pnl_dollar:+,.2f} ({pnl_pct:+.2f}%)")
