import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import yfinance as yf
import pandas as pd
import numpy as np
from config import SP500_ticker, SP500, horizon, time, method, thresold_data

def get_data():
    
    data: pd.DataFrame = yf.download(tickers = SP500_ticker, period = horizon, interval = time, auto_adjust = False)
        
    if "Adj Close" in data:
        
        prices: pd.DataFrame = data["Adj Close"]
        
    else:
        
        prices: pd.DataFrame = data["Close"]
            
    prices.index = pd.to_datetime(prices.index).astype("datetime64[ns]")
    prices.columns = [SP500]
        
    if prices.isna().sum().sum() > 0:
        
        mask = prices.isna().mean(axis = 0) <= (1 - thresold_data)
        group = prices.loc[:, mask]
        group.ffill(inplace = True)
        group.bfill(inplace = True)

    if method == "log":
        
        returns: pd.DataFrame = np.log(prices).diff().dropna()
        
    else:
    
        returns: pd.DataFrame = prices.pct_change().dropna()
        
    returns.columns = [SP500]
    
    return prices.squeeze(), returns.squeeze()