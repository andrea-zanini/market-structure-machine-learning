import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from hmmlearn import hmm
from config import starting_point, end_point, numero_seed, risk_free_rate, annuallizzazione, optimal_state

BASE_DIR = Path(__file__).parent.parent
asset_path = BASE_DIR / "data" / "asset data"

log_returns = pd.read_parquet(asset_path / "log_returns.parquet")

def state_selection(train: pd.DataFrame):
    
    x = train.values
    data_points, feature = x.shape
    common_index = log_returns.index.intersection(train.index)
    rendimenti = log_returns.reindex(index = common_index).values
    
    model = {}
    stati_predetti = {}
    state_profile = {}
    bic = []
    aic = []
    probabilità = []
    stati_nascosti = []
    
    for n in range(starting_point, end_point):
        
        model[n] = None
        best_log_likelihood = - np.inf
        
        for seed in range(numero_seed):
            
            best_model = hmm.GaussianHMM(n_components = n, covariance_type = "diag", random_state = seed).fit(X = x)
            log_likelihood = best_model.score(x)
            
            if best_log_likelihood < log_likelihood:
                
               model[n] = best_model
               best_log_likelihood = log_likelihood
                    
        probabilità.append(round(best_log_likelihood, 4))
        parameter = n**2 + 2 * n * feature - 1
        aic.append(round(2 * parameter - 2 * best_log_likelihood, 4))
        bic.append(round(parameter * np.log(data_points) - 2 * best_log_likelihood, 4))
        stati_nascosti.append(n)
        persistenza = np.diag(model[n].transmat_)
        stati_predetti[n] = model[n].predict(x)
        metriche_per_stato = []
        
        for s in range(n):
            
            rendimenti_stato = rendimenti[stati_predetti[n] == s]
            media = np.mean(rendimenti_stato) * annuallizzazione
            volatilità = np.std(rendimenti_stato) * np.sqrt(annuallizzazione)
            sharpe_ratio = (media - (risk_free_rate * annuallizzazione)) / volatilità
            conteggio = (len(rendimenti_stato) / len(stati_predetti[n]))
            metriche_per_stato.append({"Stato": s,
                                       "Media.": round(media, 4),
                                       "Volatilità": round(volatilità, 4) * 100,
                                       "Sharpe Ratio": round(sharpe_ratio, 4),
                                       "Persistenza": round(persistenza[s], 4),
                                       "Osservazioni %": round(conteggio * 100, 2)
                                       })         
        state_profile[n] = pd.DataFrame(metriche_per_stato).set_index("Stato")
           
    metriche = pd.DataFrame({"Log-Likelihood": probabilità,
                             "AIC": aic,
                             "BIC": bic},
                            index = stati_nascosti)
    metriche.index.name = "Hidden_States"
    
    return model, stati_predetti, metriche, state_profile

def validation_states(model: dict, validation: pd.DataFrame):
    
    x = validation.values
    data_points, feature = x.shape
    common_index = log_returns.index.intersection(validation.index)
    rendimenti = log_returns.reindex(index = common_index).values

    stati_predetti = {}
    state_profile = {}
    bic = []
    aic = []
    probabilità = []
    stati_nascosti = []
    
    for n in range(starting_point, end_point):
        
        log_likelihood = model[n].score(x)
        parameter = n**2 + 2 * n * feature - 1
        aic.append(round(2 * parameter - 2 * log_likelihood, 4))
        bic.append(round(parameter * np.log(data_points) - 2 * log_likelihood, 4))
        probabilità.append(round(log_likelihood, 4))
        stati_nascosti.append(n)
        stati_predetti[n] = model[n].predict(x)
        t_states = pd.Categorical(stati_predetti[n][:-1], categories = range(n))
        t1_states = pd.Categorical(stati_predetti[n][1:], categories = range(n))
        matrice_conteggi = pd.crosstab(t_states, t1_states, dropna = False)
        somme_riga = matrice_conteggi.sum(axis = 1)
        matrice_empirica = matrice_conteggi.div(somme_riga, axis = 0).fillna(0)
        persistenza = np.diag(matrice_empirica.values)
        metriche_per_stato = []
        
        for s in range(n):
            
            rendimenti_stato = rendimenti[stati_predetti[n] == s]
            media = np.mean(rendimenti_stato) * annuallizzazione
            volatilità = np.std(rendimenti_stato) * np.sqrt(annuallizzazione)
            sharpe_ratio = (media - (risk_free_rate * annuallizzazione)) / volatilità
            conteggio = (len(rendimenti_stato) / len(stati_predetti[n]))
            metriche_per_stato.append({"Stato": s,
                                       "Media.": round(media, 4),
                                       "Volatilità": round(volatilità, 4) * 100,
                                       "Sharpe Ratio": round(sharpe_ratio, 4),
                                       "Persistenza": round(persistenza[s], 4),
                                       "Osservazioni %": round(conteggio * 100, 2)
                                       })         
        state_profile[n] = pd.DataFrame(metriche_per_stato).set_index("Stato")
            
    metriche = pd.DataFrame({"Log-Likelihood": probabilità,
                             "AIC": aic,
                             "BIC": bic},
                            index = stati_nascosti)
    metriche.index.name = "Hidden_States"
    
    return stati_predetti, metriche, state_profile

def out_of_sample(model: dict, test: pd.DataFrame):
    
    y = test.values
    stati_out_of_sample = model[optimal_state].predict(y)
    
    return stati_out_of_sample