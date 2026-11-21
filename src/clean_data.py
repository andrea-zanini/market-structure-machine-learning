import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import get_data
from src.features import compute_features, pca

BASE_DIR = Path(__file__).parent.parent
asset_path = BASE_DIR / "data" / "asset data"
feature_path = BASE_DIR / "data" / "feature"
asset_path.mkdir(parents = True, exist_ok = True)
feature_path.mkdir(parents = True, exist_ok = True)

prices, log_returns = get_data()
dataset = compute_features(prices, log_returns)
test, validation, train = pca(dataset)

prices.to_frame().to_parquet(asset_path / "prices.parquet")
log_returns.to_frame().to_parquet(asset_path / "log_returns.parquet")
train.to_parquet(feature_path / "train.parquet")
validation.to_parquet(feature_path / "validation.parquet")
test.to_parquet(feature_path / "test.parquet")