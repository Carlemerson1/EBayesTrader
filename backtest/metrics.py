"""
Docstring for backtest/metrics.py

Performance metrics and diagnostics for backtest resutls

Incl Sharpe ratio, max drawdown, Calmar, win rate,
and posterior calibration checks.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.engine import BacktestConfig, BacktestResult


@dataclass
class PerformanceMetrics:
    """
    Container for all performance metrics of a backtest run
    """
    # returns
    total_return: float             # total return over the backtest period  
    annualized_return: float        # total return and annualized return of the strategy over the backtest period

    #risk
    annualized_volatility: float    # total volatility of portfolio returns, annualized
    downside_volatility: float      # volatility computed only on negative return days, used for Sortino ratio
    max_drawdown: float             # max peak-to-trough decline in equity curve
    max_drawdown_duration: int      # days

    #risk-adjusted returns
    sharpe_ratio: float             # annualized return / annualized volatility ------ how much return we get per unit of risk taken
    calmar_ratio: float             # annualized return / abs(max drawdown) ------- how much return we get per unit of drawdown risk
    sortino_ratio: float            # annualized return / downside volatility ------- how much return we get per unit of downside risk (ignoring upside volatility)

    #trading activity
    win_rate: float                 # % of trades that were profitable
    avg_win: float                  # average return of winning trades
    avg_loss: float                 # average return of losing trades
    total_trades: int               # number of trades executed
    avg_turnover: float             # daily portfolio turnover

    #drawdown details
    drawdown_series: pd.Series      # full drawdown time series
    underwater_periods: pd.DataFrame # all drawdown episodes


def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Computes performance metrics from backtest results
    
    Args:
        returns: Daily portfolio log returns
        equity_curve: Daily portfolio value over time
        positions: DataFrame of portfolio weights over time
        trades: DataFrame of position changes (entry/exit points)
        risk_free_rate: Daily risk-free rate (default 0.0)

    Returns:
        PerformanceMetrics object
    """

    n_days = len(returns)
    n_years = n_days / 252

    #=== Returns ===
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    annualized_return = (1 + total_return) ** (1 / n_years) - 1

    #=== Volatility ===
    annualized_vol = returns.std() * np.sqrt(252)

    # Downside volatility (only negative returns)
    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0

    #=== Drawdown ===
    drawdown_series = compute_drawdowns(equity_curve)
    max_dd = drawdown_series.min()

    #find max drawdown duration
    underwater = drawdown_series[drawdown_series < 0]
    if len(underwater) > 0:
        dd_periods = []
        current_start = underwater.index[0]
        prev_date = underwater.index[0]

        for date in underwater.index[1:]:
            if (date - prev_date).days > 5: # gap in underwater periods
                dd_periods.append((current_start, prev_date))
                current_start = date
            prev_date = date

        dd_periods.append((current_start, underwater.index[-1]))

        #find longest duration
        max_dd_duration = max((end - start).days for start, end in dd_periods)

        #create df of all drawdown episodes
        underwater_df = pd.DataFrame([
            {'start': start, 
             'end': end, 
             'duration_days': (end - start).days,
             'depth': drawdown_series[start:end].min()
             }
             for start, end in dd_periods
        ]).sort_values(by='depth')
    else:
        max_dd_duration = 0
        underwater_df = pd.DataFrame()

    #=== Risk-adjusted returns ===
    rf_daily = risk_free_rate / 252

    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0
    sortino = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

    #=== WIN RATE ===
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    #=== Turnover ===
    #Turnover = sum of absolute weight changes per day
    if len(positions) > 1:
        position_changes = positions.diff().abs().sum(axis=1)
        avg_turnover = position_changes.mean()
    else:
        avg_turnover = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        downside_volatility=downside_vol,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        sortino_ratio=sortino,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=len(trades),
        avg_turnover=avg_turnover,
        drawdown_series=drawdown_series,
        underwater_periods=underwater_df
    )

def compute_drawdowns(equity_curve: pd.Series) -> pd.Series:
    """
    Computes drawdown series: (current_value - peak_value) / peak_value

    Drawdown is always <=0. A value of -0.2 means portfolio is 20% below its previous peak.
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown

def print_metrics_report(
        metrics: PerformanceMetrics, 
        config: Optional['BacktestConfig'] = None,
        result: Optional['BacktestResult'] = None,
        benchmark_metrics: Optional[PerformanceMetrics] = None):
    """
    Petty-print a performance report
    """
    print("\n" + "="*70)
    print(" "*20 + "PERFORMANCE REPORT")
    print("="*70)

    print("\n--- CONFIGURATION ---")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"List of Symbols: {result.symbols}")
    print(f"Window Size: {config.window} days")
    print(f"Min Prob Threshold: {config.min_prob_threshold:.2f}")
    print(f"Target Volatility: {config.target_vol:.2%}")
    print(f"Max Position Size: {config.max_position:.2%}")
    print(f"Drawdown Threshold: {config.drawdown_threshold:.2%}")
    print(f"Drawdown Scaling: {config.drawdown_scaling:.2f}")
    print(f"Drawdown Recovery: {config.drawdown_recovery:.2%}")
    print(f"Backtest Period: {metrics.drawdown_series.index[0].date()} to {metrics.drawdown_series.index[-1].date()}")

    print("\n--- RETURNS ---")
    print(f"Total Return: {metrics.total_return:>10.2%}")
    print(f"Annualized Return: {metrics.annualized_return:>10.2%}")

    if benchmark_metrics:
        print(f"Benchmark Return: {benchmark_metrics.annualized_return:>10.2%}")
        alpha = metrics.annualized_return - benchmark_metrics.annualized_return
        print(f"Alpha vs Benchmark: {alpha:>10.2%}")

    print("\n--- RISK ---")
    print(f"Annualized Volatility: {metrics.annualized_volatility:>10.2%}")
    print(f"Downside Volatility: {metrics.downside_volatility:>10.2%}")
    print(f"Max Drawdown: {metrics.max_drawdown:>10.2%}")
    print(f"Max Drawdown Duration: {metrics.max_drawdown_duration:>10} days")

    print("\n--- RISK-ADJUSTED RETURNS ---")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:>10.2f}")

    if benchmark_metrics:
        print(f"Benchmark Sharpe: {benchmark_metrics.sharpe_ratio:>10.2f}")
        print(f"Benchmark Sortino: {benchmark_metrics.sortino_ratio:>10.2f}")
        print(f"Benchmark Calmar: {benchmark_metrics.calmar_ratio:>10.2f}")

    print("\n--- TRADING ACTIVITY ---")
    print(f"Win Rate: {metrics.win_rate:>10.2%}")
    print(f"Average Win: {metrics.avg_win:>10.2%}")
    print(f"Average Loss: {metrics.avg_loss:>10.2%}")
    print(f"Total Trades: {metrics.total_trades:>10}")
    print(f"Average Daily Turnover: {metrics.avg_turnover:>10.2%}")

    print("\n--- TOP 5 DRAWDOWN PERIODS ---")
    if len(metrics.underwater_periods) > 0:
        top5 = metrics.underwater_periods.head(5)
        print(f"{'Start':<12} {'End':<12} {'Duration':<10} {'Depth':<10}")
        print("-"*50)
        for _, row in top5.iterrows():
            print(f"{str(row['start'])[:10]:<12} {str(row['end'])[:10]:<12}"
                  f"{row['duration_days']:>10} d {row['depth']:>9.2%}")
    else:
        print("No drawdowns! (lucky you)")

    print("\n" + "="*70)


#This block only runs when you execute file directly:
#   python3 backtest/metrics.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns
    from backtest.engine import run_backtest, BacktestConfig
    
    print("Running backtest for metrics calculation...\n")
    
    from config.settings import AGGRESSIVE_GROWTH_CONFIG

    raw = fetch_daily_bars(
        symbols=AGGRESSIVE_GROWTH_CONFIG.symbols,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
    )
    
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    
    config = BacktestConfig(
        window=60,
        min_prob_threshold=0.69,
        target_vol=0.15,
        max_position=0.20,
        initial_capital=100000.0,
        drawdown_threshold=0.15,
        drawdown_scaling=0.75,
        drawdown_recovery=0.13,
        stop_loss_threshold=-0.005
    )
    
    result = run_backtest(log_returns, config)
    
    # Compute metrics
    metrics = compute_metrics(
        returns=result.returns,
        equity_curve=result.equity_curve,
        positions=result.positions,
        trades=result.trades,
        risk_free_rate=0.03678,  # 3.678% risk-free rate
    )
    
    # Print report
    print_metrics_report(metrics, config=config, result=result)
    
    # Plot drawdown curve
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Equity curve
    ax1.plot(result.equity_curve.index, result.equity_curve.values, linewidth=2, color='navy')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(config.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.legend()
    
    # Drawdown
    ax2.fill_between(metrics.drawdown_series.index, 0, metrics.drawdown_series.values * 100, 
                     color='red', alpha=0.3)
    ax2.plot(metrics.drawdown_series.index, metrics.drawdown_series.values * 100, 
             color='darkred', linewidth=1.5)
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


