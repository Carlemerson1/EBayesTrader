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


#This block only runs when you execute file directly:
#   python3 model/prior.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from data.fetcher import fetch_daily_bars
    from data.processor import compute_log_returns, clean_returns

    print("Fetching data for prior estimation...\n")
    raw=fetch_daily_bars(
        symbols=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"],
        start_date=datetime(2024,7,1),
        end_date=datetime(2024,10,31)
    )

    log_returns = compute_log_returns(raw)
    log_returns = clean_returns(log_returns)

    prior = estimate_prior(log_returns)

    print("--- Estimated Group Prior ---")
    print(f"mu_0 (group mean):        {prior.mu_0:.6f}  ({prior.mu_0 * 252:.2%} annualized)")
    print(f"tau_0 (group std):        {prior.tau_0:.6f}")
    print(f"sigma_bar (typical vol):  {prior.sigma_bar:.6f}  ({prior.sigma_bar * (252**0.5):.2%} annualized)")

    print("\n--- Interpretation ---")
    print(f"The 'average stock' in this universe has a daily return around {prior.mu_0:.4f}.")
    print(f"Individual stocks deviate from this mean with a std of {prior.tau_0:.4f}.")
    print(f"When we have very little data on a stock, we'll assume it looks like this prior.")

    print("\n--- Normality Check: QQ-Plot of Sample Means ---")
    print("If points lie on the diagonal, Normal assumption is reasonable.\n")

    from scipy import stats
    import matplotlib.pyplot as plt

    #QQ
    sample_means=log_returns.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8,6))
    stats.probplot(sample_means,dist="norm",plot=ax)
    ax.set_title("QQ: Sample Means vs Gaussian PDF")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Shapiro-Wilk test
    stat, p_val=stats.shapiro(sample_means)
    print(f"Shapiro-Wilk test:")
    print(f" Test statistic: {stat:.4f}")
    print(f" p-value: {p_val:.4f}")
    if p_val>0.05:
        print("\nFTR Normality assumtion, p>0.05")
    else: 
        print("\nReject normality assumption, p<0.05")