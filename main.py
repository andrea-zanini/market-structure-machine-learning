import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.data import get_data, get_market_returns
from src.features import compute_features, pca
from src.hidden_markov import state_selection, validation_states
from src.hidden_markov import out_of_sample
from src.plot import plot_market_regimes
from config import starting_point, end_point

BASE_DIR = Path(__file__).parent.parent
feature_path = BASE_DIR / "data" / "feature"
market_path = BASE_DIR / "data" / "market data"
asset_path = BASE_DIR / "data" / "asset data"
asset_path.mkdir(parents = True, exist_ok = True)
market_path.mkdir(parents = True, exist_ok = True)
feature_path.mkdir(parents = True, exist_ok = True)

prices, log_returns = get_data()
market_log_returns, weights, market_cap = get_market_returns(log_returns)
dataset = compute_features()
test, validation, train = pca(dataset)

prices.to_parquet(asset_path / "prices.parquet")
log_returns.to_parquet(asset_path / "log_returns.parquet")
market_log_returns.to_frame().to_parquet(market_path / "market_log_returns.parquet")
weights.to_frame().to_parquet(market_path / "weights.parquet")
market_cap.to_frame().to_parquet(asset_path / "market_cap.parquet")
train.to_parquet(feature_path / "train.parquet")
validation.to_parquet(feature_path / "validation.parquet")
test.to_parquet(feature_path / "test.parquet")

modello, stati_predetti_train, metriche_train, state_profile_train = state_selection(train)

for n in range(starting_point, end_point):
   print(f'le metriche del train set riguardanti lo stato {n} sono:')
   print(state_profile_train[n])
print('le metriche del train set sono:')
print(metriche_train)

stati_predetti_validation, metriche_validation, state_profile_validation = validation_states(modello, validation)

for n in range(starting_point, end_point):
   print(f'le metriche del validation set riguardanti lo stato {n} sono:')
   print(state_profile_validation[n])
print('le metriche del validation set sono:')
print(metriche_validation)

stati_out_of_sample = out_of_sample(modello, test)
plot_market_regimes(market_log_returns, train, validation, test, stati_predetti_train, stati_predetti_validation, stati_out_of_sample)