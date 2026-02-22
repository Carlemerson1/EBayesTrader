"""
dashboard/data_loader.py

Loads live and historical data for the dashboard.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

from execution.trader import AlpacaTrader
from config.settings import AlpacaConfig, StrategyConfig
from data.fetcher import fetch_daily_bars
from data.processor import compute_log_returns, clean_returns
from model.prior import estimate_prior
from model.posterior import update_all_posteriors
from model.signals import compute_all_signals
from backtest.metrics import compute_metrics, compute_drawdowns


def load_live_portfolio():
    """Load current portfolio state from Alpaca."""
    try:
        config = AlpacaConfig()
        trader = AlpacaTrader(config)
        
        account = trader.get_account()
        positions = trader.get_positions()
        
        return {
            'portfolio_value': float(account.equity),
            'cash': float(account.cash),
            'positions': positions,
            'buying_power': float(account.buying_power),
            'timestamp': datetime.now()
        }
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return None


def load_backtest_results(results_file='backtest_results.pkl'):
    """
    Load saved backtest results.
    
    Expected to find a pickle file with BacktestResult object.
    """
    results_path = Path(__file__).parent.parent / results_file
    
    if results_path.exists():
        import pickle
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    else:
        # If no saved results, return None
        return None


def get_live_signals(config: StrategyConfig = None):
    """
    Run the model on current data to get live signals.
    
    Returns dict of {symbol: {'prob': float, 'action': str}}
    """
    if config is None:
        config = StrategyConfig()
    
    try:
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.window + 10)
        
        raw = fetch_daily_bars(
            symbols=config.symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        log_returns = compute_log_returns(raw)
        log_returns = clean_returns(log_returns)
        
        # Use last window days
        window = log_returns.tail(config.window)
        
        # Run model
        prior = estimate_prior(window)
        posteriors = update_all_posteriors(window, prior)
        signals = compute_all_signals(posteriors, config.min_prob_threshold)
        
        # Format for dashboard
        signal_data = {}
        for symbol, sig in signals.items():
            signal_data[symbol] = {
                'prob': sig.prob_positive,
                'action': sig.action,
                'expected_return': sig.expected_return,
            }
        
        return signal_data
    
    except Exception as e:
        print(f"Error computing signals: {e}")
        return {}


def load_trade_history(log_file='logs/trade_history.json'):
    """
    Load recent trade history from log file.
    
    Expected format:
    [
        {"timestamp": "2026-02-21 09:35:00", "event": "Rebalanced portfolio"},
        {"timestamp": "2026-02-21 09:36:00", "event": "Bought 247 XOM"},
        ...
    ]
    """
    log_path = Path(__file__).parent.parent / log_file
    
    if log_path.exists():
        with open(log_path, 'r') as f:
            events = json.load(f)
        return events[-20:]  # Last 20 events
    else:
        return []


def compute_live_metrics(portfolio_value_history):
    """
    Compute performance metrics from portfolio value time series.
    
    Args:
        portfolio_value_history: Series of portfolio values indexed by date
        
    Returns:
        dict of metrics
    """
    if portfolio_value_history is None or len(portfolio_value_history) < 2:
        return None
    
    # Compute returns
    returns = np.diff(np.log(portfolio_value_history.values))
    returns_series = pd.Series(returns, index=portfolio_value_history.index[1:])
    
    # Basic metrics
    total_return = (portfolio_value_history.iloc[-1] / portfolio_value_history.iloc[0]) - 1
    n_days = len(portfolio_value_history)
    n_years = n_days / 252
    
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = returns_series.std() * np.sqrt(252)
    
    # Sharpe
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Sortino
    negative_returns = returns_series[returns_series < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino = annual_return / downside_vol if downside_vol > 0 else 0
    
    # Drawdown
    drawdown_series = compute_drawdowns(portfolio_value_history)
    max_dd = drawdown_series.min()
    
    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate
    wins = returns_series[returns_series > 0]
    losses = returns_series[returns_series < 0]
    win_rate = len(wins) / len(returns_series) if len(returns_series) > 0 else 0
    
    # Average win/loss
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'drawdown_series': drawdown_series,
    }


def get_portfolio_history_from_alpaca(days_back=30):
    """
    Fetch portfolio value history from Alpaca.
    
    Note: Alpaca only provides daily snapshots, not continuous equity curve.
    For continuous tracking, you need to log values yourself.
    """
    # Placeholder - Alpaca API doesn't provide historical portfolio values easily
    # You need to log these yourself in a CSV or database
    
    # Check if local log exists
    history_file = Path(__file__).parent.parent / 'logs' / 'portfolio_history.csv'
    
    if history_file.exists():
        df = pd.read_csv(history_file, index_col=0, parse_dates=True)
        return df['portfolio_value']
    else:
        return None