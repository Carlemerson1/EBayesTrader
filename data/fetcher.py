"""
Docstring for data/fetcher.py

Connect to alpaca API, retreive historical daily price bars for
several symbols. Returns pd df
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv(dotenv_path="config/.env") #usage os.getenv('ALPACA_API_KEY')

def get_alpaca_client() -> StockHistoricalDataClient:
    """
    Docstring for get_alpaca_client
    
    :return: authenticated alpaca data client
    :rtype: StockHistoricalDataClient
    """

    api_key=os.getenv("ALPACA_API_KEY")
    secret_key=os.getenv("ALPACA_SECRET_KEY")

    #auth error if missing keys
    if not api_key or not secret_key:
        raise ValueError(
            "Missing API keys. Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in .env"
        )
    
    return StockHistoricalDataClient(api_key, secret_key)


def fetch_daily_bars(
        symbols:list[str],
        start_date: datetime,
        end_date: datetime,
        client: StockHistoricalDataClient | None=None
) -> pd.DataFrame:
    """
    Docstring for fetch_daily_bars
    Fetch daily OHLCV bars for a list of symbols from Alpaca
    
    :param symbols: e.g. ['AAPL', 'MSFT', 'GOOG']
    :type symbols: list[str]
    :param start_date: First date to include
    :type start_date: datetime
    :param end_date: Last date to include
    :type end_date: datetime
    :param client: pass an existing client to avoid re-auth. If none, a new client is created from .env keys
    :type client: StockHistoricalDataClient | None
    :return: pandas df with MultiIndex of (symbol, timestamp). Columns: open, high, low, close, volume, vwap, trade_count
    :rtype: DataFrame
    """
    if client is None:
        client=get_alpaca_client()

    #build request object
    request=StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day, #daily bars
        start=start_date,
        end=end_date,
        adjustment="all" #adjust for splits and dividends
    )

    #alpaca paginates automatically - req 5yrs of data makes multiple API calls, stiches together
    bars=client.get_stock_bars(request)

    #.df converts response into df, index is multiindex (symbol,timestamp)
    df=bars.df
    df.sort_index(inplace=True)

    print(f"Fetched {len(df)} bars for {len(symbols)} symbol(s).")
    return df

#This block only runs when you execute file directly:
#   python3 data/fetcher.py
#Does NOT run when another file imports from this module

if __name__ == "__main__":
    print("Running fetcher sanity check...\n")

    df=fetch_daily_bars(
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2024,10,1),
        end_date=datetime(2024,10,31)
    )

    print("\n--- Raw DataFrame (first 10 rows) ---")
    print(df.head(10))

    print("\n--- DataFrame shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Index structure ---")
    print(f"Index type: {type(df.index)}")
    print(f"Index names: {df.index.names}")
    print(f"Symbols: {df.index.get_level_values('symbol').unique().tolist()}")

    print("\n--- Column names ---")
    print(df.columns.tolist())

    print("\n--- AAPL close prices only ---")
    aapl_close=df.loc["AAPL", "close"]
    print(aapl_close)