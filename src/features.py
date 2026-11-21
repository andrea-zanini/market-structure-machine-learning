import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import mean_variance_window, skewness_kurtosis_window, rsi_window, division, lenght_validation_test

BASE_DIR = Path(__file__).parent.parent
asset_path = BASE_DIR / "data" / "asset data"

def compute_features(prices: pd.Series, log_returns: pd.Series):
    
    def compute_rsi():
        
        delta = prices.diff()
        gain = delta.clip(lower = 0)
        loss = - delta.clip(upper = 0)
        avg_gain = gain.ewm(com = rsi_window - 1, min_periods = rsi_window, adjust = False).mean()
        avg_loss = loss.ewm(com = rsi_window - 1, min_periods = rsi_window, adjust = False).mean()
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def compute_dataframe():
        
            common_index = (rsi.index.intersection(media.index).intersection(volatilità.index).intersection(skewness.index).intersection(kurtosis.index))
            dataset = pd.DataFrame({"RSI": rsi.loc[common_index].values,
                                   "Media": media.loc[common_index].values,
                                   "Volatilità": volatilità.loc[common_index].values,
                                   "Skewness": skewness.loc[common_index].values,
                                   "Kurtosis": kurtosis.loc[common_index].values},
                                   index = common_index)
            dataset = dataset.dropna()
            
            return dataset

    rsi = compute_rsi()
    media = log_returns.rolling(mean_variance_window).mean().dropna()
    volatilità = log_returns.rolling(mean_variance_window).std().dropna()
    skewness = log_returns.rolling(skewness_kurtosis_window).skew().dropna()
    kurtosis = log_returns.rolling(skewness_kurtosis_window).kurt().dropna()
    
    dataset = compute_dataframe()
    
    return dataset

def pca(dataset: pd.DataFrame):
    
    dates = dataset.index
    length = int(len(dates) * division)
    train_dates = dates[:length]
    validation_test_dates = dates[length:]
    lenght_2 = int(len(validation_test_dates) * lenght_validation_test)
    validation_dates = validation_test_dates[:lenght_2]
    test_dates = validation_test_dates[lenght_2:]
    train_dataset = dataset.loc[train_dates]
    validation_dataset = dataset.loc[validation_dates]
    test_dataset = dataset.loc[test_dates]
    
    x = train_dataset.sort_index()
    indice = x.index
    x_mean = x.mean()
    x_std = x.std()
    x_normalized = (x - x_mean) / x_std
    correlation = np.corrcoef(x_normalized, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    t, n = x.shape
    q = float(n / t)
    variance = 1
    lambda_max = variance * (1 + np.sqrt(q))**2
    number_components = max(1, (eigenvalues > lambda_max).sum())

    clean_eigenvectors = eigenvectors[:, :number_components]
    x_hmm = np.dot(x_normalized, clean_eigenvectors)
    x_hmm = pd.DataFrame(x_hmm, index = indice, columns = [f'PC_{i+1}' for i in range(number_components)])
    train = x_hmm.groupby(level = "Date").mean().sort_index()
    
    z = validation_dataset.sort_index()
    z_normalized = (z - x_mean) / x_std
    z_hmm = np.dot(z_normalized, clean_eigenvectors)
    z_hmm = pd.DataFrame(z_hmm, index = z.index, columns = x_hmm.columns)
    validation = z_hmm.groupby(level = "Date").mean().sort_index()
    
    y = test_dataset.sort_index()
    y_normalized = (y - x_mean) / x_std
    y_hmm = np.dot(y_normalized, clean_eigenvectors)
    y_hmm = pd.DataFrame(y_hmm, index = y.index, columns = x_hmm.columns)
    test = y_hmm.groupby(level = "Date").mean().sort_index()
    
    return test, validation, train