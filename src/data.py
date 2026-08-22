import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from config import SP500_URL, step, horizon, time, method, thresold_data

def get_data():
    def get_sp500_tickers():
        headers = {"User-Agent": "Mozzilla/5.0"}
        response = requests.get(SP500_URL, headers = headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        rows = table.find_all("tr")[1:]
        tickers = [row.find("td").get_text(strip = True) for row in rows if row.find("td")]
        asset = [ticker.replace(".", "-") for ticker in tickers]
        return asset
    tickers = get_sp500_tickers()
    n = len(tickers)
    chuncks = []
    for i in range(0, n, step):
        data: pd.DataFrame = yf.download(tickers = tickers[i: i + step], period = horizon, interval = time, auto_adjust = False)
        if "Adj Close" in data:
            group: pd.DataFrame = data["Adj Close"]
        else:
            group: pd.DataFrame = data["Close"]
        group.index = pd.to_datetime(group.index).astype("datetime64[ns]")
        if group.isna().sum().sum() > 0:
            mask = group.isna().mean(axis = 0) <= (1 - thresold_data)
            group = group.loc[:, mask]
            group.ffill(inplace = True)
            group.bfill(inplace = True)
        chuncks.append(group)
    prices: pd.DataFrame = pd.concat(chuncks, axis = 1)
    if method == "log":
        returns: pd.DataFrame = np.log(prices).diff().dropna()
    else:
        returns: pd.DataFrame = prices.pct_change().dropna()
    return prices, returns

def get_market_returns(returns: pd.DataFrame):
    def get_capitalization():
        tickers = returns.columns
        market_cap = []
        for asset in tickers:
            try:
                capitalization = yf.Ticker(asset).fast_info["market_cap"]
            except Exception:
                capitalization = None
            market_cap.append(capitalization)
        market_cap = pd.Series(market_cap, index = tickers)
        market_cap.rename("Capitalization", inplace = True)
        market_cap.fillna(0, inplace = True)
        return market_cap
    market_cap = get_capitalization()
    weights = market_cap.values / market_cap.values.sum()
    market_returns = returns.values @ weights
    market_returns = pd.Series(market_returns, index = returns.index, name = "Market Returns")
    weights = pd.Series(weights, index = returns.columns, name = "Weights")
    return market_returns, weights, market_cap