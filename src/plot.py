import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from config import optimal_state, optimal_hidden_state


BASE_DIR = Path(__file__).parent.parent
plot_path = BASE_DIR / "data" / "plot"
plot_path.mkdir(parents = True, exist_ok = True)

STYLE = {"style": "whitegrid",
         "context": "paper",
         "font": "serif",
         "rc": {"figure.figsize": (12, 6),
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.color": "0.85",
                "grid.linewidth": 0.8,
                "grid.alpha": 0,
                "axes.edgecolor": "0.25",
                "axes.labelcolor": "0.15",
                "xtick.color": "0.25",
                "ytick.color": "0.25",
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 11,
                "legend.fontsize": 10}
         }

sns.set_theme(**STYLE)

def plot_market_regimes(market_log_returns: pd.Series, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, stati_predetti_train: np.ndarray, stati_predetti_validation: np.ndarray, stati_out_of_sample: np.ndarray):

    stati_predetti_train = pd.Series(stati_predetti_train[optimal_state], index = pd.to_datetime(train.index), name = "State")
    stati_predetti_validation = pd.Series(stati_predetti_validation[optimal_state], index = pd.to_datetime(validation.index), name = "State")
    test_states = pd.Series(stati_out_of_sample, index = pd.to_datetime(test.index), name = "State")

    common_index_train = market_log_returns.index.intersection(stati_predetti_train.index)
    market_states_train = stati_predetti_train.loc[common_index_train]
    
    common_index_validation = market_log_returns.index.intersection(stati_predetti_validation.index)
    market_states_validation= stati_predetti_validation.loc[common_index_validation]
    
    common_index_test = market_log_returns.index.intersection(test_states.index)
    market_states_test = test_states.loc[common_index_test]
    
    price_not_aligned = np.exp(market_log_returns.cumsum())
    price_train = price_not_aligned.loc[common_index_train]
    price_validation = price_not_aligned.loc[common_index_validation]
    price_test = price_not_aligned.loc[common_index_test]
    price = pd.concat([price_train, price_validation, price_test], axis = 0)

    regime_config = {0: {"name": "Bull",
                         "description": ("Positive return, lower volatility"),
                         "color": "#5a9b68",
                         "alpha": 0.5},
                     1: {"name": "Bear", 
                         "description": ("Negative return, higher volatility"),
                         "color": "#e3242b",
                         "alpha": 0.5},
                     2: {"name": "Tail Event",
                         "description": ("Positive return, lower volatility"),
                         "color": "#f4a261",
                         "alpha": 0.5},
                     3: {"name": "Rally",
                         "description": ("Positive return, lower volatility"),
                         "color": "#2d5c3f",
                         "alpha": 0.5}
                     }

    fig, ax = plt.subplots()
    sns.despine()
    
    def shade_regimes(market_states: pd.Series):
        states = market_states.to_numpy()
        dates = market_states.index
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                state = states[start_idx]
                config = regime_config[state]
                ax.axvspan(dates[start_idx], dates[i], color = config["color"], alpha=config["alpha"], linewidth = 0)
                start_idx = i

        state = states[start_idx]
        config = regime_config[state]
        ax.axvspan(dates[start_idx], dates[-1], color = config["color"], alpha = config["alpha"], linewidth = 0)
    
    shade_regimes(market_states_train)
    shade_regimes(market_states_validation)
    shade_regimes(market_states_test)
    
    sns.lineplot(x = price.index, y = price.values, ax = ax, color = "black", linewidth = 1.5, label = "Market Index", )
    ax.axvline(market_states_train.index[-1], color = "#3377e3", linestyle = "--", linewidth = 1.5, label = "Validation")
    ax.axvline(market_states_validation.index[-1], color = "#E8DFDF", linestyle = "--", linewidth = 1.5, label = "Out Of Sample")

    ax.set_title("Market Regimes — Hidden Markov Model")
    ax.set_xlabel( "Date")
    ax.set_ylabel("Market Index")

    legend = []
    for state in optimal_hidden_state:
        config = regime_config[state]
        legend.append(mpatches.Patch(color = config["color"], label=( f"State {state}: " f"{config['name']}")))

    legend.append(plt.Line2D([0], [0], color = "black", linewidth = 1.25, label = "Market Index"))
    legend.append(plt.Line2D([0], [0], color = "#3377e3", linestyle = "--", linewidth = 1.5, label = "Validation"))
    legend.append(plt.Line2D([0], [0], color = "#E8DFDF", linestyle = "--", linewidth = 1.5, label = "Out Of Sample"))

    ax.legend(handles = legend, loc = "upper left", frameon = True, facecolor = "white")
    plt.tight_layout()
    plt.savefig(plot_path / "regime.png", dpi = 150, bbox_inches = "tight")
    plt.close()