"""
Docstring for model/signals.py

For each stock, compute:
-expected risk-adj return (ERAR) = E[mu_i|data] / sigma_hat_i
-P(mu_i > 0 | data) using normal cdf
- action: long/short/flat based on probability threshold
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np, scipy.stats as stats
from dataclasses import dataclass
from model.posterior import AssetPosterior

@dataclass
class TradingSignal:
    """
    Trading signal for a single asset
    """
    symbol: str
    expected_return: float # E[mu_i|data] -- posterior mean
    signal_strength: float # risk-adj E[mu]/sigma
    prob_positive: float # P(mu_i > 0 | data)
    posterior_std: float # uncertainty about mu_i
    sigma_hat: float # return volatility (for position sizing)
    action: str # "long", "short", or "flat"

def compute_signal(
    posterior: AssetPosterior,
    risk_free_daily: float=0.0,
    min_prob_threshold: float=0.6
) -> TradingSignal:
    """
    Computes a trading signal for a posterior distribution
    
    :param posterior: Posterior distribution of the asset
    :type posterior: AssetPosterior
    :param risk_free_daily: Daily risk-free rate
    :type risk_free_daily: float
    :param min_prob_threshold: Minimum probability threshold to go long (default 60%)
    :type min_prob_threshold: float
    :return: Trading signal dataclass
    :rtype: TradingSignal
    """

    # Risk adj expected return (like a daily sharpe ratio)
    if posterior.sigma_hat > 0:
        erar = posterior.post_mean / posterior.sigma_hat
    else:
        erar = 0.0

    #P(mu_i < r_f | data)
    # Posterior on mu_i is Normal(post_mean, post_std^2)
    # We want P(mu_i > r_f) = 1 - phi(r_f)
    prob_positive=float(stats.norm.sf(
        risk_free_daily,
        loc=posterior.post_mean,
        scale=posterior.post_std
    ))

    # Determine action based on probability threshold
    if prob_positive >= min_prob_threshold:
        action = "long"
    elif prob_positive <= (1 - min_prob_threshold):
        action = "short"
    else:
        action = "flat"

    return TradingSignal(
        symbol=posterior.symbol,
        expected_return=posterior.post_mean,
        signal_strength=erar,
        prob_positive=prob_positive,
        posterior_std=posterior.post_std,
        sigma_hat=posterior.sigma_hat,
        action=action
    )

def compute_all_signals(
    posteriors: dict[str, AssetPosterior],
    risk_free_daily: float=0.0,
    min_prob_threshold: float=0.6,
) -> dict[str, TradingSignal]:
    """
    Compute trading signals for all assets given their posteriors
    
    :param posteriors: Dictionary of asset symbol to posterior distribution
    :type posteriors: dict[str, AssetPosterior]
    :param risk_free_daily: Daily risk-free rate
    :type risk_free_daily: float
    :param min_prob_threshold: Minimum probability threshold to go long (default 60%)
    :type min_prob_threshold: float
    :return: {symbol: TradingSignal}
    :rtype: dict[str, TradingSignal]
    """
    signals = {}
    for symbol, posterior in posteriors.items():
        signals[symbol] = compute_signal(
            posterior,
            risk_free_daily=risk_free_daily,
            min_prob_threshold=min_prob_threshold
        )
    return signals

#This block only runs when you execute file directly:
#   python3 model/signals.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns, get_rolling_window
    from prior import estimate_prior
    from posterior import update_all_posteriors

    print("Fetching data...\n")
    raw = fetch_daily_bars(
        symbols=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"],
        start_date=datetime(2022, 7, 1),
        end_date=datetime(2022, 10, 31),
    )
    
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    window = get_rolling_window(log_returns, end_date=log_returns.index[-1], window=60)
    
    prior = estimate_prior(window)
    posteriors = update_all_posteriors(window, prior)
    signals = compute_all_signals(posteriors, min_prob_threshold=0.95)

    print("\n--- Trading Signals ---")
    print(f"{'Symbol':<8} {'Action':<8} {'P(mu>0)':<10} {'ERAR':<10} {'Post Mean':<12} {'Post Std':<12}")
    print("-" * 60)

    for symbol, sig in signals.items():    
        print(f"{sig.symbol:<8} {sig.action:<8} {sig.prob_positive:>9.1%} {sig.signal_strength:>9.3f} {sig.expected_return:>11.6f} {sig.posterior_std:>11.6f}")      

    print("\n--- Summary ---    ")
    longs = [s for s in signals.values() if s.action == "long"]
    shorts = [s for s in signals.values() if s.action == "short"]
    flats = [s for s in signals.values() if s.action == "flat"]

    print(f"Long: {len(longs)}")
    print(f"Short: {len(shorts)}")
    print(f"Flat: {len(flats)}")

    if longs:
        print(f"\nStrongest Long: {max(longs, key=lambda x: x.prob_positive).symbol} "
              f"(P={max(longs, key=lambda x: x.prob_positive).prob_positive:.1%})")