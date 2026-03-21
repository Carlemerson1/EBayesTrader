"""
dashboard/data_loader.py

Loads live and historical data for the dashboard.
"""

# =============== QUERY SUBMODULE ===============
import streamlit as st
import subprocess
import os
import sys
import importlib

if not os.path.exists("model/model/prior.py"):
    deploy_key = st.secrets.get("ssh", {}).get("deploy_key", None)
    if deploy_key:
        key_path = "/tmp/deploy_key"
        with open(key_path, "w") as f:
            f.write(deploy_key)
        os.chmod(key_path, 0o600)
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = f"ssh -i {key_path} -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            check=False, capture_output=True, text=True, env=env
        )

# 2. Clear any cached failed model imports
for key in list(sys.modules.keys()):
    if key.startswith('model.'):
        del sys.modules[key]

# =============== END QUERY SUBMODULE ===============

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import csv

from execution.trader import AlpacaTrader
from model.config.settings import AlpacaConfig, StrategyConfig, AGGRESSIVE_GROWTH_CONFIG
from data.fetcher import fetch_daily_bars
from data.processor import compute_log_returns, clean_returns
from model.model.prior import estimate_prior
from model.model.posterior import update_all_posteriors
from model.model.signals import compute_all_signals
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
    if config is None:
        config = AGGRESSIVE_GROWTH_CONFIG  # Default to aggressive config for live signals
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(config.window * 1.5) + 10)
        
        raw, sector_map = fetch_daily_bars(
            symbols=config.symbols,
            start_date=start_date,
            end_date=end_date
        )
        log_returns = compute_log_returns(raw)
        log_returns = clean_returns(log_returns)
        window = log_returns.tail(config.window)
        prior = estimate_prior(window)
        posteriors = update_all_posteriors(window, prior)
        signals = compute_all_signals(posteriors, config.min_prob_threshold)
        
        signal_data = {}
        for symbol, sig in signals.items():
            signal_data[symbol] = {
                'prob': sig.prob_positive,
                'action': sig.action,
                'expected_return': sig.expected_return,
            }
        return signal_data

    except Exception as e:
        import traceback
        import streamlit as st
        st.error(f"Signal error: {e}")
        st.code(traceback.format_exc())
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
    # Try Alpaca portfolio history endpoint directly
    try:
        import requests
        config = AlpacaConfig()
        
        url = f"{config.base_url}/v2/account/portfolio/history"
        headers = {
            'APCA-API-KEY-ID':     config.api_key,
            'APCA-API-SECRET-KEY': config.secret_key,
        }
        params = {
            'period':         f'{days_back}D',
            'timeframe':      '1D',
            'extended_hours': 'false',
        }
        
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('equity') and data.get('timestamp'):
            timestamps = pd.to_datetime(data['timestamp'], unit='s', utc=True)
            series = pd.Series(data['equity'], index=timestamps)
            series = series[series > 0].dropna()
            return series

    except Exception as e:
        print(f"Alpaca portfolio history API failed: {e}")

    # Fallback to local CSV
    history_file = Path(__file__).parent.parent / 'logs' / 'portfolio_history.csv'
    if history_file.exists():
        df = pd.read_csv(history_file, index_col=0)
        if 'portfolio_value' not in df.columns:
            return None
        df.index = pd.to_datetime(df.index, format='mixed')
        df.index = pd.DatetimeIndex(df.index)
        series = df['portfolio_value'].sort_index()
        series = series.resample('D').last().dropna()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        return series[series.index >= cutoff] if len(series) > 0 else series

    return None
    
def snapshot_portfolio_value():
    """
    Fetch current portfolio value from Alpaca and append to history CSV.
    
    Called on every dashboard refresh so the equity curve grows continuously,
    not just on trade days. Deduplicates by timestamp — won't write a second
    entry if one already exists for the current minute.
    """
    try:
        config  = AlpacaConfig()
        trader  = AlpacaTrader(config)
        account = trader.get_account()
        current_value = float(account.equity)
        now = datetime.now()

        history_file = Path(__file__).parent.parent / 'logs' / 'portfolio_history.csv'
        history_file.parent.mkdir(exist_ok=True)

        # Read existing entries to check for duplicates
        existing = []
        if history_file.exists():
            df = pd.read_csv(history_file)
            existing = df['date'].tolist() if 'date' in df.columns else []

        # Write header if file is new
        write_header = not history_file.exists() or history_file.stat().st_size == 0

        # Deduplicate on minute-level timestamp
        timestamp_str = now.strftime('%Y-%m-%d %H:%M')
        if timestamp_str not in existing:
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['date', 'portfolio_value'])
                writer.writerow([timestamp_str, current_value])

        return current_value

    except Exception as e:
        print(f"Error snapshotting portfolio value: {e}")
        return None