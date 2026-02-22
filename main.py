"""
main.py

Main entry point for the Bayesian trading system.

Usage:
    python main.py --backtest              # Run historical backtest
    python main.py --paper                 # Run live in paper trading mode
    python main.py --live                  # Run live with real money (use with caution!)
    python main.py --backtest --config conservative  # Use different config
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import sys
from datetime import datetime, timedelta

# Data & modeling
from data.fetcher import fetch_daily_bars
from data.processor import compute_log_returns, clean_returns, get_rolling_window
from model.prior import estimate_prior
from model.posterior import update_all_posteriors
from model.signals import compute_all_signals
from risk.manager import compute_portfolio_weights

# Execution & config
from execution.trader import AlpacaTrader
from config.settings import (
    StrategyConfig, AlpacaConfig,
    DEFAULT_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
)

# Backtesting
from backtest.engine import run_backtest, BacktestConfig
from backtest.metrics import compute_metrics, print_metrics_report


def run_backtest_mode(config: StrategyConfig):
    """
    Runs a full historical backtest.
    """
    print("\n" + "="*70)
    print(" "*20 + "BACKTEST MODE")
    print("="*70 + "\n")
    
    # Fetch historical data
    print("Fetching historical data...")
    end_date = datetime.now()  # today
    start_date = end_date - timedelta(days=365 * 5)  # 5 years
    
    raw = fetch_daily_bars(
        symbols=config.symbols,
        start_date=start_date,
        end_date=end_date,
    )
    
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    
    print(f"Data loaded: {log_returns.index[0]} to {log_returns.index[-1]}")
    print(f"Total trading days: {len(log_returns)}\n")
    
    # Convert to BacktestConfig
    backtest_config = BacktestConfig(
        window=config.window,
        min_prob_threshold=config.min_prob_threshold,
        target_vol=config.target_vol,
        max_position=config.max_position,
        initial_capital=config.initial_capital,
        drawdown_threshold=config.drawdown_threshold,
        drawdown_scaling=config.drawdown_scaling,
        drawdown_recovery=config.drawdown_recovery,
        stop_loss_threshold=config.stop_loss_threshold
    )
    
    # Run backtest
    result = run_backtest(log_returns, backtest_config)
    
    # Compute metrics
    metrics = compute_metrics(
        returns=result.returns,
        equity_curve=result.equity_curve,
        positions=result.positions,
        trades=result.trades,
        risk_free_rate=0.03678,
    )
    
    # Print report
    print_metrics_report(metrics, config=backtest_config, result=result)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Equity curve
    ax1.plot(result.equity_curve.index, result.equity_curve.values, 
             linewidth=2, color='navy', label='Strategy')
    ax1.axhline(config.initial_capital, color='red', linestyle='--', 
                alpha=0.5, label='Initial Capital')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2.fill_between(metrics.drawdown_series.index, 0, 
                     metrics.drawdown_series.values * 100, 
                     color='red', alpha=0.3)
    ax2.plot(metrics.drawdown_series.index, 
             metrics.drawdown_series.values * 100, 
             color='darkred', linewidth=1.5)
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_paper_mode(config: StrategyConfig):
    """
    Runs the strategy live in paper trading mode.
    """
    print("\n" + "="*70)
    print(" "*20 + "PAPER TRADING MODE")
    print("="*70 + "\n")
    
    # Initialize trader
    alpaca_config = AlpacaConfig()
    trader = AlpacaTrader(alpaca_config)
    
    # Get account info
    account = trader.get_account()
    print(f"Account Status: {account.status}")
    print(f"Portfolio Value: ${float(account.equity):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}\n")
    
    # Fetch recent data
    print("Fetching recent market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.window + 10)  # Extra buffer
    
    raw = fetch_daily_bars(
        symbols=config.symbols,
        start_date=start_date,
        end_date=end_date,
    )
    
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    
    # Get rolling window
    window = get_rolling_window(
        log_returns,
        end_date=log_returns.index[-1],
        window=config.window
    )
    
    print(f"Using data from {window.index[0]} to {window.index[-1]}\n")
    
    # Run the model
    print("Running Bayesian model...")
    prior = estimate_prior(window)
    posteriors = update_all_posteriors(window, prior)
    signals = compute_all_signals(posteriors, min_prob_threshold=config.min_prob_threshold)
    
    portfolio_weights_obj = compute_portfolio_weights(
        signals,
        returns_window=window,
        target_vol=config.target_vol,
        max_position=config.max_position,
        target_gross_exposure=2.0,
        market_neutral=True,
    )
    
    target_weights = portfolio_weights_obj.weights
    
    print(f"\n--- Target Portfolio Weights ---")
    print(f"Target Vol: {portfolio_weights_obj.target_vol:.1%}")
    print(f"Actual Vol: {portfolio_weights_obj.actual_vol:.1%}")
    print(f"Total Exposure: {portfolio_weights_obj.total_exposure:.1%}\n")
    
    for symbol in sorted(target_weights.keys()):
        weight = target_weights[symbol]
        sig = signals[symbol]
        if abs(weight) > 0.001:
            print(f"{symbol:<6} {weight:>7.2%}  [{sig.action:<6}]  P(mu>0)={sig.prob_positive:.1%}")
    
    # Get current prices for order sizing
    print("\n--- Fetching current prices ---")
    current_prices = {}

    # Get the most recent timestamp
    latest_timestamp = raw.index.get_level_values('timestamp')[-1]

    for symbol in config.symbols:
        try:
            # Use tuple indexing to get exact row
            price = float(raw.loc[(symbol, latest_timestamp), 'close'])
            current_prices[symbol] = price
            print(f"{symbol}: ${price:.2f}")
        except KeyError:
            print(f"No price data for {symbol}")
    
    # Submit orders
    print("\n--- Submitting Orders to Alpaca ---")
    response = input("Submit these orders? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        orders = trader.submit_orders(
            target_weights=target_weights,
            current_prices=current_prices,
            dry_run=False
        )
        
        successful = [o for o in orders if o.success]
        failed = [o for o in orders if not o.success]
        
        print(f"\n{len(successful)} orders submitted successfully")
        if failed:
            print(f"{len(failed)} orders failed")
    else:
        print("Orders cancelled.")


def run_live_mode(config: StrategyConfig):
    """
    Runs the strategy with real money.
    
    USE WITH EXTREME CAUTION
    """
    print("\n" + "="*70)
    print(" "*20 + "LIVE TRADING MODE")
    print("="*70 + "\n")
    
    print("WARNING: This will trade with REAL MONEY.")
    print("Make sure you have:")
    print("  1. Thoroughly backtested this strategy")
    print("  2. Run it in paper trading for at least 30 days")
    print("  3. Reviewed all positions and risk limits")
    print("  4. Set ALPACA_BASE_URL to the live API endpoint\n")
    
    response = input("Are you ABSOLUTELY SURE you want to proceed? (type 'LIVE' to confirm): ")
    
    if response == 'LIVE':
        run_paper_mode(config)  # Same logic as paper, just with live credentials
    else:
        print("Live trading cancelled.")


def main():
    parser = argparse.ArgumentParser(description='Bayesian Trading System')
    
    # Mode selection
    parser.add_argument('--backtest', action='store_true', help='Run historical backtest')
    parser.add_argument('--paper', action='store_true', help='Run live in paper trading')
    parser.add_argument('--live', action='store_true', help='Run live with real money')
    
    # Config selection
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'conservative', 'aggressive'],
                       help='Strategy configuration to use')
    
    args = parser.parse_args()
    
    # Load config
    if args.config == 'conservative':
        config = CONSERVATIVE_CONFIG
    elif args.config == 'aggressive':
        config = AGGRESSIVE_CONFIG
    else:
        config = DEFAULT_CONFIG
    
    # Run appropriate mode
    if args.backtest:
        run_backtest_mode(config)
    elif args.paper:
        run_paper_mode(config)
    elif args.live:
        run_live_mode(config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()