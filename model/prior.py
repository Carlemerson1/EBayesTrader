"""
Docstring for model/prior.py
Empirical Bayes prior estimation

Given a window of log-returns for N stocks we compute:
-mu_0: mean of all sample means (i.e., group-level expected return)
-tau_0: std of all sample means (how much stocks vary around mu_0)
-sigma_bar: mean of all sample stds (typical volatility)

These become hyperparamaters for asset-level priors
"""
# group level prior because we use the cross-section of stocks to fit a single "group" prior that applies to all stocks.
# We could extend this to have multiple groups (e.g. tech vs non-tech) but for simplicity we just fit one prior to the whole universe.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from dataclasses import dataclass

@dataclass
class GroupPrior:
    """
    Docstring for GroupPrior

    Container for group-level prior parms
    """
    mu_0:float #prior mean for each asset's expected return
    tau_0: float #prior std (controls shrikage strength)
    sigma_bar: float # typical cross-sectional volatility

def estimate_prior(log_returns:pd.DataFrame) -> GroupPrior:
    """
    Fits group-level prior from a cross-section of returns

    Args: log_returns: Shape (n_dates, n_symbols). Each column is one stock
    
    :param log_returns: df of log returns for stock(s)
    :type log_returns: pd.DataFrame
    :return: GroupPrior w fitted hyperparms
    :rtype: GroupPrior

    If all stocks had identical expected returns, tau_0 would be near zero. 
    If stocks are wildly different, tau_0 would be large.
    tau_0 controls how much we shrink individual estimates toward mu_0
    """

    sample_means=log_returns.mean(axis=0)
    sample_stds=log_returns.std(axis=0)

    #group level
    mu_0=float(sample_means.mean())
    tau_0=float(sample_means.std())

    #typical volatility (treated as "known sigma" in conjugate update)
    sigma_bar=float(sample_stds.mean())

    #Guard: tau_0 can be near zero if all stocks look identical.
    #We use a floor of 0.01% daily, or ~2.5% annual std across stocks
    tau_0=max(tau_0, 1e-4)

    return GroupPrior(mu_0=mu_0, tau_0=tau_0, sigma_bar=sigma_bar)

def estimate_sector_priors(
    returns_window: pd.DataFrame,
    sector_map: dict[str, str],
    verbose: bool=False
) -> dict[str, GroupPrior]:
    """
    Estimate separate priors for each sector.
    
    For sectors with only 1 stock, we use that stock's sample mean
    but borrow the global tau_0 for shrinkage strength.
    
    Args:
        returns_window: DataFrame of returns (dates x symbols)
        sector_map: Dict mapping {symbol: sector}
        
    Returns:
        Dict mapping {sector: GroupPrior}
    """
    from data.sector_mapping import get_symbols_by_sector
    
    symbols = returns_window.columns.tolist()
    sector_groups = get_symbols_by_sector(symbols)
    sector_priors = {}
    
    # First, estimate global prior for fallback
    global_prior = estimate_prior(returns_window)
    
    if verbose:
        print("\nEstimating sector-specific priors:")
        print("="*60)
    
    for sector, sector_symbols in sorted(sector_groups.items()):
        # Only use stocks actually in returns_window
        available = [s for s in sector_symbols if s in returns_window.columns]
        
        if not available:
            continue
        
        sector_returns = returns_window[available]
        
        # Handle single-stock sectors specially
        if len(available) == 1:
            # Use stock's mean but borrow global tau_0
            stock_mean = sector_returns[available[0]].mean()
            stock_std = sector_returns[available[0]].std()
            
            prior = GroupPrior(
                mu_0=stock_mean,
                tau_0=global_prior.tau_0,  # ← Borrow from global
                sigma_bar=stock_std if stock_std > 0 else global_prior.sigma_bar
            )
            if verbose:
                print(f"  [{sector.upper():<12}] μ₀={prior.mu_0:>8.5f}, τ₀={prior.tau_0:>8.6f} ({len(available)} stock, using global τ₀)")
        
        else:
            # Multi-stock sector: estimate normally
            prior = estimate_prior(sector_returns)
            if verbose:
                print(f"  [{sector.upper():<12}] μ₀={prior.mu_0:>8.5f}, τ₀={prior.tau_0:>8.6f} ({len(available)} stocks)")
        
        sector_priors[sector] = prior
    
    if verbose:
        print("="*60 + "\n")
    return sector_priors


#This block only runs when you execute file directly:
#   python3 model/prior.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns
    from scipy import stats
    import matplotlib.pyplot as plt

    print("="*70)
    print("PRIOR ESTIMATION SANITY CHECK")
    print("="*70)

    # Fetch diverse stocks across sectors
    symbols = ["AAPL", "MSFT", "NVDA", "XOM", "CVX", "JPM", "BAC", "TLT"]
    
    print("\nFetching data for prior estimation...\n")
    raw, sector_map = fetch_daily_bars(
        symbols=symbols,
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2024, 10, 31)
    )

    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)

    # TEST 1: Global prior (old way)
    print("\n" + "="*70)
    print("TEST 1: GLOBAL PRIOR (treats all stocks the same)")
    print("="*70)
    
    global_prior = estimate_prior(log_returns)

    print(f"μ₀ (group mean):        {global_prior.mu_0:.6f}  ({global_prior.mu_0 * 252:.2%} annualized)")
    print(f"τ₀ (group std):         {global_prior.tau_0:.6f}")
    print(f"σ̄ (typical vol):       {global_prior.sigma_bar:.6f}  ({global_prior.sigma_bar * (252**0.5):.2%} annualized)")

    # TEST 2: Sector-specific priors (new way)
    print("\n" + "="*70)
    print("TEST 2: SECTOR-SPECIFIC PRIORS (each sector gets its own)")
    print("="*70)
    
    sector_priors = estimate_sector_priors(log_returns, sector_map)

    # Compare sectors
    print("\n" + "="*70)
    print("COMPARISON: Why sector priors matter")
    print("="*70)
    
    for sector, prior in sorted(sector_priors.items()):
        sector_stocks = [s for s, sec in sector_map.items() if sec == sector]
        print(f"\n{sector.upper()}: {', '.join(sector_stocks)}")
        print(f"  μ₀ = {prior.mu_0:>8.6f} ({prior.mu_0 * 252:>7.1%} annual)")
        print(f"  τ₀ = {prior.tau_0:>8.6f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("With GLOBAL prior: All stocks shrink toward same mean")
    print("With SECTOR prior: Each sector preserves its own characteristics ✓")
    
    # TEST 3: Normality checks
    print("\n" + "="*70)
    print("TEST 3: NORMALITY ASSUMPTION CHECKS")
    print("="*70)
    
    sample_means = log_returns.mean(axis=0)
    
    # QQ Plot
    print("\nGenerating QQ-Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Global QQ plot
    stats.probplot(sample_means, dist="norm", plot=axes[0])
    axes[0].set_title("QQ Plot: All Sample Means vs Normal", fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Sector-colored scatter
    for sector in sector_priors.keys():
        sector_stocks = [s for s, sec in sector_map.items() if sec == sector]
        sector_means = log_returns[sector_stocks].mean(axis=0)
        axes[1].scatter(sector_means.index, sector_means.values, 
                       label=sector, s=100, alpha=0.7)
    
    axes[1].axhline(global_prior.mu_0, color='red', linestyle='--', 
                   label='Global μ₀', linewidth=2)
    axes[1].set_xlabel('Stock', fontweight='bold')
    axes[1].set_ylabel('Mean Return', fontweight='bold')
    axes[1].set_title('Sample Means by Sector', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Shapiro-Wilk test
    print("\nShapiro-Wilk Normality Test:")
    stat, p_val = stats.shapiro(sample_means)
    print(f"  Test statistic: {stat:.4f}")
    print(f"  P-value: {p_val:.4f}")
    
    if p_val > 0.05:
        print("  ✓ Fail to reject normality assumption (p > 0.05)")
        print("  → Normal prior is reasonable")
    else: 
        print("  ✗ Reject normality assumption (p < 0.05)")
        print("  → Consider robust prior or larger universe")
    
    print("\n" + "="*70)
    print("SANITY CHECK COMPLETE")
    print("="*70)