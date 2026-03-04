"""
Docstring for model/posterior.py

Conjugate posterior updates for each asset

Takes a prior and observed returns, returns a posterior dist on mu_i for each
stock using normal-normal conjugate formula
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np, pandas as pd
from dataclasses import dataclass
from model.prior import GroupPrior

@dataclass
class AssetPosterior:
    """
    Posterior parms for a single asset
    """
    symbol:str
    post_mean: float # E[mu_i|data]
    post_var: float # Var[mu_i|data]
    post_std: float # sqrt(post_var)
    n_obs: int # number of return obs for this asset
    sigma_hat: float # estimated volatility of returns (for position sizing)


    @property
    def shrinkage_factor(self) -> float:
        """
        How much weight did we put on the prior vs the data?
        0.0 -> all prior, 1.0 -> all data (ignoring prior)

        With 5 days of data, may get 0.2-0.3 shrinkage, meaning we put 20-30% weight on the data and 70-80% on the prior.
        With 60 days of data, may get 0.8-0.9 shrinkage, meaning we put 80-90% weight on the data and only 10-20% on the prior.
        """
        if self.n_obs == 0:
            return 0.0
        
        prior_precision = 1 / self.post_var # precision is inverse of variance
        data_precision = self.n_obs / (self.sigma_hat ** 2) # precision of the likelihood (data)

        return data_precision / (prior_precision + data_precision)
    
def update_posterior(
        symbol: str,
        returns: pd.Series,
        prior: GroupPrior
) ->  AssetPosterior:
    """
    Computes posterior for one asset's expected return given observed returns and group-level prior
    
    Math (Normal-Normal conjugate update):
        Prior: mu_i ~ N(mu_0, tau_0^2)
        Likelihood: r_it ~ N(mu_i, sigma^2) for t=1..n_obs [sigma estimated from data]

        Posterior precision: kappa_N = kappa_0 + N_sigma^2
        Posterior mean: m_N = (kappa_0*m_0 + N*x_bar/sigma^2) / kappa_N
        Posterior variance: v_N^2 = 1 / kappa_N

    :param symbol: stock ticker symbol
    :type symbol: str
    :param returns: observed returns for the asset
    :type returns: pd.Series
    :param prior: group-level prior for the asset
    :type prior: GroupPrior
    :return: posterior distribution parameters for the asset
    :rtype: AssetPosterior
    """
    #drop NaN, (missing data days)
    clean_returns = returns.dropna()
    n=len(clean_returns)

    if n == 0:
        #no data, return prior as posterior
        return AssetPosterior(
            symbol=symbol,
            post_mean=prior.mu_0,
            post_var=prior.tau_0 ** 2,
            post_std=prior.tau_0,
            n_obs=0,
            sigma_hat=prior.sigma_bar
        )
    
    #sample stats
    x_bar=float(clean_returns.mean())
    sigma_hat=float(clean_returns.std())

    #if sample std is 0 (stock didn't move), use prior's typical volatility
    if sigma_hat == 0 or np.isnan(sigma_hat):
        sigma_hat = prior.sigma_bar

    #prior parms
    mu_0=prior.mu_0
    kappa_0=1 / (prior.tau_0 ** 2) # prior precision

    #data precision (treating sigma_hat as known)
    kappa_data = n / (sigma_hat ** 2)

    #posterior (Normal-Normal conjugate formula)
    kappa_N = kappa_0 + kappa_data
    m_N = (kappa_0 * mu_0 + kappa_data * x_bar) / kappa_N
    v_N_sq = 1.0 / kappa_N

    return AssetPosterior(
        symbol=symbol,
        post_mean=m_N,
        post_var=v_N_sq,
        post_std=np.sqrt(v_N_sq),
        n_obs=n,
        sigma_hat=sigma_hat
    )


def update_all_posteriors(
        window_returns: pd.DataFrame,
        prior: GroupPrior
) -> dict[str, AssetPosterior]:
    """
    Update posteriors for all symbols in the universe
    
    :param window_returns: dataframe of log returns for each stock in the window
    :type window_returns: pd.DataFrame
    :param prior: group-level prior for the assets
    :type prior: GroupPrior
    :return: dictionary: {symbol: AssetPosterior}
    :rtype: dict[str, AssetPosterior]
    """
    posteriors = {}
    for symbol in window_returns.columns:
        posteriors[symbol] = update_posterior(
            symbol=symbol,
            returns=window_returns[symbol],
            prior=prior
        )
    return posteriors

def update_all_posteriors_by_sector(
    returns_window: pd.DataFrame,
    sector_priors: dict[str, GroupPrior],
    sector_map: dict[str, str]
) -> dict[str, AssetPosterior]:
    """
    Update posteriors using sector-specific priors.
    
    Each stock's posterior is computed using its sector's prior,
    not a global prior. This preserves sector-specific characteristics.
    
    Args:
        returns_window: DataFrame of returns (dates x symbols)
        sector_priors: Dict mapping {sector: GroupPrior}
        sector_map: Dict mapping {symbol: sector}
        
    Returns:
        Dict mapping {symbol: AssetPosterior}
    """
    posteriors = {}
    
    for symbol in returns_window.columns:
        sector = sector_map.get(symbol, 'other')
        prior = sector_priors.get(sector)
        
        if prior is None:
            print(f"WARNING: No prior for sector '{sector}' (symbol {symbol}), skipping")
            continue
        
        # Use sector-specific prior instead of global prior
        posterior = update_posterior(symbol, returns_window[symbol], prior)
        posteriors[symbol] = posterior
    
    return posteriors
    
#This block only runs when you execute file directly:
#   python3 model/posterior.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns, get_rolling_window
    from prior import estimate_prior, estimate_sector_priors

    print("="*70)
    print("POSTERIOR ESTIMATION SANITY CHECK")
    print("="*70)

    # Use diverse sectors to show the difference
    symbols = ["AAPL", "MSFT", "XOM", "CVX", "JPM", "TLT"]
    
    print("\nFetching data...\n")
    raw, sector_map = fetch_daily_bars(
        symbols=symbols,
        start_date=datetime(2024, 7, 1),
        end_date=datetime(2024, 10, 31)
    )

    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)

    # Use 60 day rolling window
    window = get_rolling_window(log_returns, end_date=log_returns.index[-1], window=60)

    # TEST 1: Global approach (old way)
    print("\n" + "="*70)
    print("TEST 1: GLOBAL POSTERIOR (all stocks use same prior)")
    print("="*70)
    
    global_prior = estimate_prior(window)
    print(f"\nGlobal Prior: μ₀={global_prior.mu_0:.6f}, τ₀={global_prior.tau_0:.6f}\n")
    
    global_posteriors = update_all_posteriors(window, global_prior)

    print(f"{'Symbol':<8} {'Sector':<10} {'Post Mean':<12} {'Post Std':<12} {'Shrinkage':<12}")
    print("-" * 70)
    for symbol in sorted(symbols):
        post = global_posteriors[symbol]
        sector = sector_map[symbol]
        print(f"{symbol:<8} {sector:<10} {post.post_mean:>11.6f} {post.post_std:>11.6f} {post.shrinkage_factor:>11.2%}")

    # TEST 2: Sector approach (new way)
    print("\n" + "="*70)
    print("TEST 2: SECTOR POSTERIOR (each sector uses its own prior)")
    print("="*70)
    
    sector_priors = estimate_sector_priors(window, sector_map, verbose=False)
    sector_posteriors = update_all_posteriors_by_sector(window, sector_priors, sector_map)

    print(f"\n{'Symbol':<8} {'Sector':<10} {'Post Mean':<12} {'Post Std':<12} {'Shrinkage':<12}")
    print("-" * 70)
    for symbol in sorted(symbols):
        post = sector_posteriors[symbol]
        sector = sector_map[symbol]
        print(f"{symbol:<8} {sector:<10} {post.post_mean:>11.6f} {post.post_std:>11.6f} {post.shrinkage_factor:>11.2%}")

    # COMPARISON
    print("\n" + "="*70)
    print("COMPARISON: How posterior means change")
    print("="*70)
    print(f"{'Symbol':<8} {'Sector':<10} {'Global Post':<15} {'Sector Post':<15} {'Difference':<12}")
    print("-" * 70)
    for symbol in sorted(symbols):
        global_mean = global_posteriors[symbol].post_mean
        sector_mean = sector_posteriors[symbol].post_mean
        diff = sector_mean - global_mean
        sector = sector_map[symbol]
        print(f"{symbol:<8} {sector:<10} {global_mean:>14.6f} {sector_mean:>14.6f} {diff:>11.6f}")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("GLOBAL: Tech stocks get pulled DOWN toward global mean")
    print("SECTOR: Tech stocks stay closer to tech mean (preserves the signal!)")
    print("="*70)