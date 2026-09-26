import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import get_data, get_market_returns
from src.features import compute_features, pca

BASE_DIR = Path(__file__).parent.parent
asset_path = BASE_DIR / "data" / "asset data"
asset_path.mkdir(parents = True, exist_ok = True)
market_path = BASE_DIR / "data" / "market data"
market_path.mkdir(parents = True, exist_ok = True)
feature_path = BASE_DIR / "data" / "feature"
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