import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.hidden_markov import state_selection, validation_states
from src.hidden_markov import out_of_sample
from src.plot import plot_market_regimes
from config import starting_point, end_point

BASE_DIR = Path(__file__).parent
asset_path = BASE_DIR / "data" / "asset data"
feature_path = BASE_DIR / "data" / "feature"

prices = pd.read_parquet(asset_path / "prices.parquet").squeeze()
log_returns = pd.read_parquet(asset_path / "log_returns.parquet").squeeze()
train = pd.read_parquet(feature_path / "train.parquet")
validation = pd.read_parquet(feature_path / "validation.parquet")
test = pd.read_parquet(feature_path / "test.parquet")

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
plot_market_regimes(log_returns, train, validation, test, stati_predetti_train, stati_predetti_validation, stati_out_of_sample)