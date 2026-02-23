"""
Docstring for backtest/engine.py

Walk-forward backtesting engine

Simulated strategy as if trading it in real time, re-running model daily and recording results.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np, pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from model.prior import estimate_prior
from model.posterior import update_all_posteriors
from model.signals import compute_all_signals
from risk.manager import compute_portfolio_weights

@dataclass
class BacktestConfig:
    """
    Configuration for backtest parms
    """
    window: int = 60 #rolling window for model (trading days)
    min_prob_threshold: float = 0.60 #signal threshold
    target_vol: float = 0.15 #annualized portfolio vol target
    max_position: float = 0.20 #max weight per stock
    initial_capital: float = 100000.0 #starting portfolio value

    # Drawdown protection
    drawdown_threshold: float = 0.15 # If drawdown exceeds this, we reduce risk
    drawdown_scaling: float = 0.5 # How much to reduce risk when drawdown threshold is breached (e.g. 0.5 = cut position sizes in half)
    drawdown_recovery: float = 0.10 # If drawdown recovers to this level, we can restore full risk (e.g. 0.10 = recover at -10% drawdown)
    stop_loss_threshold: float = -0.01 # If an individual position loses more than this in a day, we close it (e.g. -0.02 = 2% loss)

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError("Window must be positive")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes: 
        equity_curve: series of portfolio values indexed by date
        returns: series of daily portfolio returns
        positions: DataFrame of portfolio weights over time (dates x symbols)
        trades: DataFrame of position changes (entry/exit points)
        config: BacktestConfig used
        symbols: list of symbols in the universe
    """
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    config: BacktestConfig
    symbols: Optional[list[str]] = None #added for easier access to universe in metrics
    
def run_backtest(
    log_returns: pd.DataFrame,
    config: BacktestConfig,
    benchmark_returns: Optional[pd.Series] = None
) -> BacktestResult:
    """
    Runs a walk-forward backtest of the strategy.

    Args:
        log_returns: DataFrame of log returns (dates x symbols)
        config: BacktestConfig with backtest parameters
        benchmark_returns: Optional series of benchmark returns for comparison

    Returns:
        BacktestResult with equity curve, returns, positions, trades, and config
    """
    dates =log_returns.index[config.window:] # start after we have enough data
    symbols = log_returns.columns.tolist()

    #storage for results
    portfolio_values = []
    portfolio_returns = []
    position_history = []
    trade_history = []

    portfolio_value = config.initial_capital
    current_weights = {s: 0.0 for s in symbols}
    breaker_active = False  #track state of drawdown protection circuit breaker
    prev_breaker_active = False #track previous state to log only on changes

    stop_loss_threshold = config.stop_loss_threshold # stop-loss threshold for individual positions (e.g. -0.02 = 2% loss)

    print(f"Running backtest from {dates[0]} to {dates[-1]}")
    print(f"Universe: {len(symbols)} stocks, Window: {config.window} days\n")

    for i, date in enumerate(dates):
        # get rolling window of data up to (but not incl) this date
        # this simulates "looking back" at data we would have had at this time
        # so model is always trained on past data, never future data
        if i % 5 != 0: #only rebalance every N days (weekly=5 trading days)
            #don't rebalancetoday, just apply yesterday's returns
            day_returns = log_returns.loc[date]

            portfolio_return = 0.0
            for symbol in symbols:
                weight = current_weights.get(symbol, 0.0)
                asset_return = day_returns[symbol] if not pd.isna(day_returns[symbol]) else 0.0
                portfolio_return += weight * asset_return

            portfolio_value *= np.exp(portfolio_return) #update value with today's return

            portfolio_values.append(portfolio_value)
            portfolio_returns.append(portfolio_return)
            position_history.append({"date": date, **current_weights})
            continue # skip to next day

        #if we reach here, it is a rebalance day, so run the model
        window_end_idx = log_returns.index.get_loc(date) - 1
        window_start_idx = window_end_idx - config.window + 1

        if window_start_idx < 0:
            continue

        window = log_returns.iloc[window_start_idx:window_end_idx + 1]

        #run model
        prior = estimate_prior(window)
        posteriors = update_all_posteriors(window, prior)
        signals = compute_all_signals(
            posteriors,
            min_prob_threshold=config.min_prob_threshold
        )

        portfolio_weights = compute_portfolio_weights(
            signals,
            returns_window=window,
            target_vol=config.target_vol,
            max_position=config.max_position,
            target_gross_exposure=2.0,
            market_neutral=True
        )

        from risk.manager import apply_drawdown_protection

        # portfolio peak value for drawdown calc - we can use current portfolio value as proxy for peak, since we only reduce risk after we have already taken the drawdown hit. This is a simplification but should be fine for our purposes.
        peak_value = max(portfolio_values)  if portfolio_values else config.initial_capital

        # asset weights after applying drawdown protection (if needed)
        new_weights, breaker_active = apply_drawdown_protection(
            weights=portfolio_weights.weights,
            current_value=portfolio_value,
            peak_value=peak_value,
            drawdown_trigger=config.drawdown_threshold,
            drawdown_recovery=0.10,  # Recover at -10%
            scaling_factor=config.drawdown_scaling,
            breaker_was_active=breaker_active,  # ← Pass previous state
        )

        # Log when state CHANGES (not every time it's active)
        if i % 5 == 0 and breaker_active != prev_breaker_active:
            current_dd = (portfolio_value - peak_value) / peak_value
            if breaker_active and current_dd <= -config.drawdown_threshold:
                print(f"  🔴 Drawdown protection ACTIVE at {date.date()}: "
                    f"{current_dd:.1%} drawdown, exposure at 50%")
            elif not breaker_active and current_dd > -0.10:
                if (portfolio_value - peak_value) / peak_value > -config.drawdown_threshold:
                    print(f"  🟢 Drawdown protection OFF at {date.date()}: "
                        f"{current_dd:.1%} drawdown, full exposure restored")
                    
        prev_breaker_active = breaker_active #update previous state for next iteration


        #record trades (position changes)
        for symbol in symbols:
            old_w = current_weights.get(symbol, 0.0)
            new_w = new_weights.get(symbol, 0.0)

            if abs(new_w - old_w) > 1e-4: #only record meaningful changes
                trade_history.append({
                    "date": date,
                    "symbol": symbol,
                    "old_weight": old_w,
                    "new_weight": new_w,
                    "change": new_w - old_w
                })

        #calc today's portfolio return using today's actual returns
        day_returns = log_returns.loc[date]

        for symbol in symbols:
            if symbol not in current_weights or current_weights[symbol] == 0:
                continue  # No position, no stop-loss
            
            weight = current_weights[symbol]
            asset_return = day_returns[symbol] if not pd.isna(day_returns[symbol]) else 0.0
            position_return = weight * asset_return
            
            # If this position lost >2% of portfolio value, close it
            if position_return < stop_loss_threshold:
                new_weights[symbol] = 0.0
                trade_history.append({
                    "date": date,
                    "symbol": symbol,
                    "old_weight": weight,
                    "new_weight": 0.0,
                    "change": -weight,
                })

        #now calc portfolio return with new weights (after stop-loss adjustments (if applied))
        portfolio_return = 0.0
        for symbol in symbols:
            weight = new_weights.get(symbol, 0.0)
            asset_return = day_returns[symbol] if not pd.isna(day_returns[symbol]) else 0.0
            portfolio_return += weight * asset_return

        # add transaction costs - we can estimate this as proportional to the turnover (sum of absolute weight changes) multiplied by a fixed cost per dollar traded (e.g. 5 bps = 0.0005)
        total_turnover = 0.0
        for symbol in symbols:
            old_w = current_weights.get(symbol, 0.0)
            new_w = new_weights.get(symbol, 0.0)
            total_turnover += abs(new_w - old_w)

        transaction_cost = total_turnover * 0.0005  # 5 bps per dollar traded
        portfolio_return -= transaction_cost

        #update portfolio value (exponential for log returns)
        portfolio_value *= np.exp(portfolio_return)

        #record
        portfolio_values.append(portfolio_value)
        portfolio_returns.append(portfolio_return)
        position_history.append({"date": date, **new_weights})

        #update current weights for next iteration
        current_weights = new_weights

        #progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dates)} days..."
                    f"Portfolio Value: ${portfolio_value:,.2f}")
            
    print(f"\nBacktest complete! Final Portfolio Value: ${portfolio_value:,.2f}")

    #convert to dataframes/series
    equity_curve = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
    returns_series = pd.Series(portfolio_returns, index=dates[:len(portfolio_returns)])
    positions_df = pd.DataFrame(position_history).set_index("date")
    trades_df = pd.DataFrame(trade_history)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns_series,
        positions=positions_df,
        trades=trades_df,
        config=config,
        symbols=symbols
    )


#This block only runs when you execute file directly:
#   python3 backtest/engine.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns
    
    print("Fetching historical data for backtest...\n")
    raw = fetch_daily_bars(
        symbols=["PLTR", "VOO", "SPY", "QQQ", "DIA"],
        start_date=datetime(2020, 1, 1),  # 2 years of data
        end_date=datetime(2025, 12, 31),
    )
    
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    
    print(f"Data range: {log_returns.index[0]} to {log_returns.index[-1]}")
    print(f"Total trading days: {len(log_returns)}\n")
    
    # Run backtest
    config = BacktestConfig(
        window=60,
        min_prob_threshold=0.75,
        target_vol=0.15,
        max_position=0.20,
        initial_capital=100000.0,
    )
    
    result = run_backtest(log_returns, config)
    
    # Quick summary
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    
    total_return = (result.equity_curve.iloc[-1] / config.initial_capital - 1)
    print(f"Total Return: {total_return:.2%}")
    print(f"Starting Value: ${config.initial_capital:,.2f}")
    print(f"Ending Value: ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"Number of Trades: {len(result.trades)}")
    
    # Annualized metrics (quick estimates)
    annual_return = (1 + total_return) ** (252 / len(result.returns)) - 1
    annual_vol = result.returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    print(f"\nAnnualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    print(f"\nMax Drawdown: (calculated in metrics module)")
    print(f"Calmar Ratio: (calculated in metrics module)")