"""
Docstring for data/processor.py

Pivot raw bar df to wide format
compute log returns from adj close prices
handle missing values and stale prices
provide rolling window slice for model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np, pandas as pd

def compute_log_returns(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Docstring for compute_log_returns
    
    :param bars_df: bars data frame
    :type bars_df: pd.DataFrame
    :return: df of daily log returns, given a multiindex (symbol,timestamp) df of ohlcv bars
    :rtype: DataFrame

    Shape:(n_dates, n_symbols)
    """

    # Pivot: rows=dates, cols=symbols, values=close price
    prices=bars_df['close'].unstack(level='symbol')

    #sort dates chronologically
    prices.sort_index(inplace=True)

    #compute log returns r_t=log(P_t/P_{t-1}), drops first row (NaN)
    log_returns=np.log(prices/prices.shift(1)).dropna(how='all')

    return log_returns


def clean_returns(
        log_returns: pd.DataFrame,
        max_abs_return: float=0.25
) -> pd.DataFrame:
    """
    Docstring for clean_returns
    
    Winsorizes extreme returns (+/-25%)
    :param log_returns: df of log returns
    :type log_returns: pd.DataFrame
    :param max_abs_return: extreme return cap
    :type max_abs_return: float
    :return: clipped log_returns df
    :rtype: DataFrame
    """
    return log_returns.clip(-max_abs_return, max_abs_return)


def get_rolling_window(
        log_returns: pd.DataFrame,
        end_date:pd.Timestamp,
        window: int=60
) -> pd.DataFrame:
    """
    Docstring for get_rolling_window
    
    :param log_returns: df of log returns
    :type log_returns: pd.DataFrame
    :param end_date: Should be current date, e.g., log_returns.index[-1]
    :type end_date: pd.Timestamp
    :param window: # days incl
    :type window: int
    :return: most recent window of trading dats up to and incl end_date
            -feeds model at each time setp, e.g., 60 trading days is approx 3 calendar months
    :rtype: DataFrame
    """
    subset=log_returns[log_returns.index<=end_date]
    return subset.tail(window)


#This block only runs when you execute file directly:
#   python3 data/processor.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    from datetime import datetime
    from fetcher import fetch_daily_bars
    
    print("Fetching data...")
    raw=fetch_daily_bars(
        symbols=['AAPL', 'MSFT', 'GOOG'],
        start_date=datetime(2024,7,1),
        end_date=datetime(2024,10,31)
    )

    log_returns=compute_log_returns(raw)
    log_returns=clean_returns(log_returns)

    print("\n--- Log-returns (first 10 rows) ---")
    print(log_returns.head(10))

    print("\n--- Shape ---")
    print(f"{log_returns.shape[0]} trading days, {log_returns.shape[1]} stocks")

    print("\n--- Basic statistics by stock ---")
    print(log_returns.describe().round(4))

    print("\n--- Rolling window (last 60 days) ---")
    window=get_rolling_window(log_returns,end_date=log_returns.index[-1])
    print(f"Window shape: {window.shape}")
    print(window.tail(3))