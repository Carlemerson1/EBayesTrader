"""
risk/manager.py

Position sizing, volatility targeting, risk constraints

Converts trading signals into portfolio weights, subject to risk limits
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np, pandas as pd
from dataclasses import dataclass
from model.signals import TradingSignal
from sklearn.covariance import LedoitWolf

@dataclass
class PortfolioWeights:
    """
    Target weights for the portfolio

    Attributes:
        weights: dict {symbol: weight}, where weights sum to target exposure
        total_exposure: scalar in [0,1], fraction of capital to deploy (after risk constraints)
        target_vol: annualized target volatility (e.g. 0.1 for 10%)
        actual_vol: realized portfolio volatility (ex-ante estimate)
    """
    weights: dict[str, float]
    total_exposure: float
    target_vol: float
    actual_vol: float

def compute_kelly_weights(
    signals: dict[str, TradingSignal],
    max_position: float = 0.20,
) -> dict[str, float]:
    """
    Computes raw position weights using a Kelly-like criterion

    Kelly logic (simplified): weight_i propto expected_return_i / sigma_i^2
        i.e., weight_i propto (post_mean / post_var) * direction
        where direction = +1 for long, -1 for short, 0 for flat
    
    :param signals: dict of TradingSignal objects
    :type signals: dict[str, TradingSignal]
    :param max_position: max weifght for any single stock (default 20%)
    :type max_position: float
    :return: dict {symbol: raw_weight}, not normalized to sum to 1
    :rtype: dict[str, float]
    """
    raw_weights={}
    for symbol, sig in signals.items():
        #skip flat positions
        if sig.action == "flat":
            raw_weights[symbol] = 0.0
            continue

        #direction: +1 for long, -1 for short
        direction = 1.0 if sig.action == "long" else -1.0

        #Kelly weight \propto expected_return / variance
        #higher expected return -> bigger position
        #higher variance -> smaller position
        if sig.posterior_std > 0:
            kelly = (sig.expected_return / sig.sigma_hat) * sig.prob_positive
        else:
            kelly = 0.0

        # Apply direction and cap at max_position
        weight = direction * kelly
        weight = np.clip(weight, -max_position, max_position)

        raw_weights[symbol] = weight

    return raw_weights

def normalize_weights(
    raw_weights: dict[str, float],
    target_gross_exposure: float = 2.0,
    market_neutral: bool=True
) -> dict[str, float]:
    """
    Normalizes weights to sum to target gross exposure

    Gross exposure = sum of absolute values of all weights.

    If market_neutral=True:
      - Longs sum to +1.0
      - Shorts sum to -1.0
      - Gross exposure = 2.0 (fully invested long and short)
      - Net exposure = 0.0 (market neutral)

    If market_neutral=False:
      - Just scales to target_gross_exposure
      - Could be 1.5 long, 0.5 short (net +1.0)

    For a long-only portfolio with target gross exposure of 1.0,
    all weights sum to 1.0 (fully invested).

    For a long-short with target=2.0, longs sum to 1.0 and shorts sum to -1.0
    """
    if market_neutral:
        # Seperate longs and shorts
        longs = {s: w for s, w in raw_weights.items() if w > 0}
        shorts = {s: w for s, w in raw_weights.items() if w < 0}
        
        long_total = sum(longs.values())
        short_total = abs(sum(shorts.values()))

        # edge case: no longs or no shorts
        if long_total == 0 and short_total == 0:
            return {s: 0.0 for s in raw_weights}
        
        if long_total ==0:
            #only shorts - scale shorts to -1.0, no longs
            scale_short = 1.0 / short_total
            result = {s: 0.0 for s in raw_weights} #start with all zeros for longs
            for s, w in shorts.items():
                result[s] = w * scale_short
            return result
        
        if short_total == 0:
            #only longs - scale longs to +1.0, no shorts
            scale_long = 1.0 / long_total
            result = {s: 0.0 for s in raw_weights}
            for s, w in longs.items():
                result[s] = w * scale_long
            return result
        
        #normal case: scale longs to +1.0 and shorts to -1.0
        scale_long = 1.0 / long_total
        scale_short = 1.0 / short_total

        normalized = {}
        for symbol, weight in raw_weights.items():
            if weight > 0:
                normalized[symbol] = weight * scale_long
            elif weight < 0:
                normalized[symbol] = weight * scale_short
            else:
                normalized[symbol] = 0.0

        for symbol in raw_weights.keys():
            if symbol not in normalized:
                normalized[symbol] = 0.0

        return normalized
    
    else:
        # non-market neutral: just scale to target gross exposure
        gross = sum(abs(w) for w in raw_weights.values())

        if gross == 0:
            return {symbol: 0.0 for symbol in raw_weights}
        
        scale_factor = target_gross_exposure / gross
        return {symbol: w * scale_factor for symbol, w in raw_weights.items()}

def apply_volatility_target(
        weights: dict[str, float],
        signals: dict[str, TradingSignal],
        returns_window: pd.DataFrame,
        target_vol: float = 0.15 # target annualized volatility (e.g. 0.15 for 15%)
) -> tuple[dict[str, float], float]:
    """
    Scales all weights to hit a target annualized portfolio volatility
    
    Uses full cov matrix with Ledoit-Wolf shrinkage to account for multicollinearity of assets

    :param weights: dict of raw weights
    :type weights: dict[str, float]
    :param signals: dict of TradingSignal objects
    :type signals: dict[str, TradingSignal]
    :param returns_window: DataFrame of log returns for covariance estimation
    :type returns_window: pd.DataFrame
    :param target_vol: target annualized volatility
    :type target_vol: float
    :return: (scaled_weights, realized_vol)
    :rtype: tuple[dict[str, float], float]
    """

    symbols = list(weights.keys())

    # filter returns to only include symbols we're trading
    # (in case returns_window has extra columns)
    returns_subset = returns_window[symbols].dropna()

    # CRITICAL: Update symbols to match what's actually in returns_subset after dropping NaNs
    symbols = returns_subset.columns.tolist()

    if len(returns_subset) < 20:
        # not enough data to estimate cov
        # fallback to zero correlation assumption (diagonal cov matrix)
        vols = np.array([signals[s].sigma_hat for s in symbols])
        w = np.array([weights[s] for s in symbols])
        port_variance = np.sum((w * vols) ** 2)
        port_vol_daily = np.sqrt(port_variance)
        port_vol_annual = port_vol_daily * np.sqrt(252)

        if port_vol_annual == 0:
            return weights, 0.0
        
        scale_factor = target_vol / port_vol_annual

        all_symbols = list(weights.keys())
        scaled_weights = {}
        for s in all_symbols:
            if s in symbols:
                scaled_weights[s] = weights[s] * scale_factor
            else:
                scaled_weights[s] = 0.0 # stock had missing data, keep weight at 0

        return scaled_weights, port_vol_annual
    
    #estimate cov matrix with Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    lw.fit(returns_subset)
    cov_matrix = lw.covariance_ #shrunk covariance matrix

    #portfolio variance: w^T * Sigma * w
    w = np.array([weights[s] for s in symbols])
    port_variance = w.T @ cov_matrix @ w
    port_vol_daily = np.sqrt(port_variance)
    port_vol_annual = port_vol_daily * np.sqrt(252)

    if port_vol_annual == 0:
        return weights, 0.0
    
    #scale to target
    scale_factor = target_vol / port_vol_annual
    all_symbols = list(weights.keys())
    scaled_weights = {}
    for s in all_symbols:
        if s in symbols:
            scaled_weights[s] = weights[s] * scale_factor
        else:
            scaled_weights[s] = 0.0 # stock had missing data, keep weight at 0

    return scaled_weights, port_vol_annual

def compute_portfolio_weights(
    signals: dict[str, TradingSignal],
    returns_window: pd.DataFrame,
    target_vol: float = 0.15,
    max_position: float = 0.20,
    target_gross_exposure: float = 2.0,
    market_neutral: bool = True # NEW parm
) -> PortfolioWeights:
    """
    Master function: converts signals into portfolio weights
    
    For market-neutral long/short:
        - target_gross_exposure = 2.0
        - market_neutral = True
        - Result: 1.0 long, 1.0 short, 2.0 gross, 0.0 net
    
    Pipeline:
    i. compute kelly weights (conviction based sizing)
    ii. normalize to target gross explosure
    iii. apply volatility targeting with full cov matrix

    Args:
        signals: Trading signals for each asset
        returns_window: DataFrame of log returns for covariance estimation
        target_vol: annualized portfolio volatility target (default 15%)
        max_position: max weight per stock (default 20%)
        target_gross_exposure: sum of weights (default 2.0 = fully invested)

    Returns: 
        PortfolioWeights object with final weights and diagnostics
    """

    # Step 1: Compute raw Kelly weights
    raw = compute_kelly_weights(signals, max_position=max_position)

    # Step 2: Normalize to target gross exposure
    normalized = normalize_weights(raw, target_gross_exposure=target_gross_exposure, market_neutral=market_neutral)

    #Step 3 Vol targeting (NOW WITH FULL COV MATRIX)
    final_weights, actual_vol = apply_volatility_target(
        normalized, signals, returns_window, target_vol=target_vol
    )

    #compute total exposure
    total_exposure = sum(abs(w) for w in final_weights.values())

    return PortfolioWeights(
        weights=final_weights,
        total_exposure=total_exposure,
        target_vol=target_vol,
        actual_vol=actual_vol
    )

def apply_drawdown_protection(
    weights: dict[str, float],
    current_value: float,
    peak_value: float,
    drawdown_trigger: float = 0.15,   # Activate at -15%
    drawdown_recovery: float = 0.10,  # Deactivate at -10%
    scaling_factor: float = 0.50,
    breaker_was_active: bool = False, # Track state from previous period
) -> tuple[dict[str, float], bool]:
    """
    Scales down positions if portfolio is in significant drawdown.
    
    Uses hysteresis to avoid flip-flopping:
      - Trigger at -15% → cut to 50%
      - Recovery at -10% → restore to 100%
    
    Args:
        weights: Target portfolio weights
        current_value: Current portfolio value
        peak_value: All-time high portfolio value
        drawdown_trigger: Drawdown level that activates protection
        drawdown_recovery: Drawdown level that deactivates protection
        scaling_factor: How much to scale positions when active
        breaker_was_active: Was protection active last period?
        
    Returns:
        (adjusted_weights, circuit_breaker_active)
    """
    drawdown = (current_value - peak_value) / peak_value
    
    # State machine with hysteresis
    if breaker_was_active:
        # Already active - only turn off if we recover above recovery threshold
        if drawdown > -drawdown_recovery:
            return weights, False  # Recovery! Turn off protection
        else:
            # Still underwater - keep protection active
            scaled_weights = {s: w * scaling_factor for s, w in weights.items()}
            return scaled_weights, True
    else:
        # Not active - only turn on if we drop below trigger threshold
        if drawdown <= -drawdown_trigger:
            scaled_weights = {s: w * scaling_factor for s, w in weights.items()}
            return scaled_weights, True  # Activate protection
        else:
            return weights, False  # No protection needed


#This block only runs when you execute file directly:
#   python3 risk/manager.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns, get_rolling_window
    from model.prior import estimate_prior
    from model.posterior import update_all_posteriors
    from model.signals import compute_all_signals
    
    print("Fetching data...\n")
    raw = fetch_daily_bars(
        symbols=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "HL"],
        start_date=datetime(2022, 7, 1),
        end_date=datetime(2023, 10, 31),
    )
    
    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)
    window = get_rolling_window(log_returns, end_date=log_returns.index[-1], window=60)
    
    prior = estimate_prior(window)
    posteriors = update_all_posteriors(window, prior)
    signals = compute_all_signals(posteriors, min_prob_threshold=0.75)
    
    # Compute portfolio weights
    portfolio = compute_portfolio_weights(
        signals,
        returns_window=window,  # Pass the 60-day window
        target_vol=0.15,
        max_position=0.20,
        target_gross_exposure=2.0
    )
    
    print("--- Portfolio Weights ---")
    print(f"Target Vol: {portfolio.target_vol:.1%} annual")
    print(f"Actual Vol: {portfolio.actual_vol:.1%} annual")
    print(f"Total Exposure: {portfolio.total_exposure:.1%}\n")
    
    print(f"{'Symbol':<8} {'Weight':<10} {'Action':<8} {'P(mu>0)':<10}")
    print("-" * 50)
    
    for symbol in sorted(portfolio.weights.keys()):
        weight = portfolio.weights[symbol]
        sig = signals[symbol]
        print(f"{symbol:<8} {weight:>9.2%}  {sig.action:<8} {sig.prob_positive:>9.1%}")
    
    print("\n--- Interpretation ---")
    print("Weights are sized based on:")
    print("  1. Expected return (higher = bigger position)")
    print("  2. Posterior uncertainty (more uncertain = smaller position)")
    print("  3. Volatility targeting (all positions scaled to hit 15% portfolio vol)")
    print("  4. Max position cap (no single stock > 20%)")