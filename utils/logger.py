"""
utils/logger.py

Logging utilities for tracking portfolio performance and trades.
"""

import json
import csv
from datetime import datetime
from pathlib import Path


class TradingLogger:
    """
    Logs portfolio state and events to files for dashboard visualization.
    """
    
    def __init__(self, log_dir='logs'):
        """Initialize logger with log directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.portfolio_history_file = self.log_dir / 'portfolio_history.csv'
        self.trade_history_file = self.log_dir / 'trade_history.json'
        
        # Initialize portfolio history CSV if it doesn't exist
        if not self.portfolio_history_file.exists():
            with open(self.portfolio_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'portfolio_value'])
    
    def log_portfolio_value(self, date, portfolio_value):
        """
        Log portfolio value for a given date.
        
        Args:
            date: datetime or str (YYYY-MM-DD)
            portfolio_value: float
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Append to CSV
        with open(self.portfolio_history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([date_str, portfolio_value])
        
        print(f"Logged portfolio value: {date_str} = ${portfolio_value:,.2f}")
    
    def log_event(self, event_text):
        """
        Log a trading event (rebalance, trade, stop-loss, etc.).
        
        Args:
            event_text: str describing the event
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Load existing events
        if self.trade_history_file.exists():
            with open(self.trade_history_file, 'r') as f:
                try:
                    events = json.load(f)
                except json.JSONDecodeError:
                    events = []
        else:
            events = []
        
        # Add new event
        events.append({
            'timestamp': timestamp,
            'event': event_text
        })
        
        # Keep only last 100 events
        events = events[-100:]
        
        # Save
        with open(self.trade_history_file, 'w') as f:
            json.dump(events, f, indent=2)
        
        print(f"Logged event: {event_text}")
    
    def log_rebalance(self, n_trades):
        """Log a portfolio rebalancing event."""
        self.log_event(f"Portfolio rebalanced ({n_trades} orders)")
    
    def log_trade(self, symbol, side, qty):
        """Log an individual trade."""
        action = "Bought" if side.lower() == "buy" else "Sold"
        self.log_event(f"{action} {qty} shares of {symbol}")
    
    def log_stop_loss(self, symbol, loss_pct):
        """Log a stop-loss trigger."""
        self.log_event(f"Stop-loss triggered: {symbol} ({loss_pct:.2%})")
    
    def log_drawdown_protection(self, active, drawdown_pct):
        """Log drawdown protection state changes."""
        if active:
            self.log_event(f"Drawdown protection ACTIVATED ({drawdown_pct:.1%})")
        else:
            self.log_event(f"Drawdown protection DEACTIVATED ({drawdown_pct:.1%})")


# Global logger instance
_logger = None

def get_logger(log_dir='logs'):
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = TradingLogger(log_dir)
    return _logger