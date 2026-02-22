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
    
#This block only runs when you execute file directly:
#   python3 model/posterior.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns, get_rolling_window
    from prior import estimate_prior

    print("Fetching data...\n")
    raw=fetch_daily_bars(
        symbols=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"],
        start_date=datetime(2024,7,1),
        end_date=datetime(2024,10,31)
    )

    log_returns=compute_log_returns(raw)
    log_returns=clean_returns(log_returns)

    #use 60 day rolling window
    window=get_rolling_window(log_returns, end_date=log_returns.index[-1], window=60)

    #estimate prior from window
    prior=estimate_prior(window)

    #update posteriors for all stocks
    posteriors=update_all_posteriors(window, prior)

    print("---Posterior Estimates---")
    print(f"{'Symbol':<8} {'Post Mean':<12} {'Post Std':<12} {'Shrinkage':<12} {'N Obs':<8}")
    print("-" * 60)
    for symbol, post in posteriors.items():
        print(f"{symbol:<8} {post.post_mean:>11.6f} {post.post_std:>11.6f} {post.shrinkage_factor:>11.2%} {post.n_obs:>7}")

    print("\n--- Interpretation ---")
    print("Post Mean: Our updated belief about this stock's expected return")
    print("Post Std: Our uncertainty about that estimate")
    print("Shrinkage: How much we trusted the data vs. pulled toward the prior")
    print("     (Higher=more reliance on likelihood, lower=more reliance on prior)")