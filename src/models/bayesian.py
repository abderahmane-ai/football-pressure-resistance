import logging
import pandas as pd
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from config import PROCESSED_DATA_DIR, MODEL_TRACES_DIR, MODEL_SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_pooled_model():
    """
    Fit a Zero-Inflated (Hurdle) model separating Turnover Risk from Value Retention.
    """
    dataset_path = PROCESSED_DATA_DIR / "all_pressure_dataset.parquet"
    if not dataset_path.exists():
        logger.error(f"Dataset {dataset_path} not found.")
        return None
        
    df = pd.read_parquet(dataset_path)
    df = df.dropna()
    
    logger.info(f"Loaded dataset: {len(df)} observations")
    
    feature_cols = [
        'dist_nearest_opp', 'dist_2nd_nearest_opp', 'opps_within_1yd',
        'opps_within_2yd', 'opps_within_4yd', 'angle_nearest_opp',
        'coverage_arc', 'voronoi_area', 'n_free_teammates',
        'max_free_triangle_area', 'dist_nearest_free_teammate',
        'angle_nearest_free_teammate', 'pitch_control', 'opp_density_5yd',
        'has_progressive_option', 'xt_value', 'bc_x', 'bc_y',
        'game_state_diff', 'minutes_elapsed', 'match_period'
    ]
    
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].values
    
    y_success = df['success'].values.astype(int)
    y_value = df['value_preserved'].values
    max_value = y_value.max() if y_value.max() > 0 else 0.15
    
    epsilon = 1e-6
    y_value_scaled = (y_value / max_value) * (1 - 2*epsilon) + epsilon
    y_value_scaled = np.clip(y_value_scaled, epsilon, 1 - epsilon)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    MODEL_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_TRACES_DIR / "pooled_scaler.pkl", "wb") as f:
        pickle.dump({
            'scaler': scaler, 
            'features': available_features,
            'max_value': max_value,
            'epsilon': epsilon
        }, f)
    
    player_cats = df['player_id'].astype('category')
    comp_cats = df['competition'].astype('category')
    opp_team_cats = df['opponent_team_id'].astype('category')
    pos_cats = df['position_group'].astype('category')
    
    player_idx = player_cats.cat.codes.values
    comp_idx = comp_cats.cat.codes.values
    opp_idx = opp_team_cats.cat.codes.values
    pos_idx = pos_cats.cat.codes.values
    
    n_players = len(player_cats.cat.categories)
    n_comp = len(comp_cats.cat.categories)
    n_opp = len(opp_team_cats.cat.categories)
    n_pos = len(pos_cats.cat.categories)
    n_features = len(available_features)
    
    player_mapping = dict(enumerate(player_cats.cat.categories))
    comp_mapping = dict(enumerate(comp_cats.cat.categories))
    opp_team_mapping = dict(enumerate(opp_team_cats.cat.categories))
    pos_mapping = dict(enumerate(pos_cats.cat.categories))
    
    name_lookup = df.drop_duplicates('player_id').set_index('player_id')['player_name'].to_dict()
    pos_lookup = df.drop_duplicates('player_id').set_index('player_id')['position_group'].to_dict()

    with open(MODEL_TRACES_DIR / "pooled_mappings.pkl", "wb") as f:
        pickle.dump({
            'player': player_mapping, 
            'competition': comp_mapping,
            'team': opp_team_mapping,
            'position': pos_mapping,
            'name_lookup': name_lookup,
            'position_lookup': pos_lookup
        }, f)
        
    # Beta model subset: successes only
    mask = y_success == 1
    X_val = X_scaled[mask]
    y_val_scaled = y_value_scaled[mask]
    player_idx_val = player_idx[mask]
    comp_idx_val = comp_idx[mask]
    opp_idx_val = opp_idx[mask]
    pos_idx_val = pos_idx[mask]
    
    with pm.Model():
        # --- TURNOVER RISK MODEL (Logistic) ---
        X_data_succ = pm.Data("X_succ", X_scaled)
        pid_succ = pm.Data("pid_succ", player_idx)
        cid_succ = pm.Data("cid_succ", comp_idx)
        oid_succ = pm.Data("oid_succ", opp_idx)
        posid_succ = pm.Data("posid_succ", pos_idx)
        
        alpha_succ = pm.Normal("alpha_succ", 0, 1.5)
        beta_succ = pm.Normal("beta_succ", 0, 1.0, shape=n_features)
        gamma_pos_succ = pm.Normal("gamma_pos_succ", 0, 1.0, shape=n_pos)
        
        sigma_theta_succ = pm.Exponential("sigma_theta_succ", 1.0)
        theta_raw_succ = pm.Normal("theta_raw_succ", 0, 1, shape=n_players)
        theta_succ = pm.Deterministic("theta_succ", theta_raw_succ * sigma_theta_succ)
        
        sigma_opp_succ = pm.Exponential("sigma_opp_succ", 1.0)
        opp_raw_succ = pm.Normal("opp_raw_succ", 0, 1, shape=n_opp)
        delta_opp_succ = pm.Deterministic("delta_opp_succ", opp_raw_succ * sigma_opp_succ)
        
        sigma_comp_succ = pm.Exponential("sigma_comp_succ", 1.0)
        comp_raw_succ = pm.Normal("comp_raw_succ", 0, 1, shape=n_comp)
        zeta_comp_succ = pm.Deterministic("zeta_comp_succ", comp_raw_succ * sigma_comp_succ)
        
        logit_p = (alpha_succ + 
                   pm.math.dot(X_data_succ, beta_succ) + 
                   gamma_pos_succ[posid_succ] +
                   theta_succ[pid_succ] +
                   delta_opp_succ[oid_succ] +
                   zeta_comp_succ[cid_succ])
        
        p = pm.Deterministic("p", pm.math.invlogit(logit_p))
        pm.Bernoulli("y_succ_obs", p=p, observed=y_success)
        
        # --- VALUE RETENTION MODEL (Beta) ---
        X_data_val = pm.Data("X_val", X_val)
        pid_val = pm.Data("pid_val", player_idx_val)
        cid_val = pm.Data("cid_val", comp_idx_val)
        oid_val = pm.Data("oid_val", opp_idx_val)
        posid_val = pm.Data("posid_val", pos_idx_val)
        
        alpha_val = pm.Normal("alpha_val", 0, 1.5)
        beta_val = pm.Normal("beta_val", 0, 1.0, shape=n_features)
        gamma_pos_val = pm.Normal("gamma_pos_val", 0, 1.0, shape=n_pos)
        
        sigma_theta_val = pm.Exponential("sigma_theta_val", 1.0)
        theta_raw_val = pm.Normal("theta_raw_val", 0, 1, shape=n_players)
        theta_val = pm.Deterministic("theta_val", theta_raw_val * sigma_theta_val)
        
        sigma_opp_val = pm.Exponential("sigma_opp_val", 1.0)
        opp_raw_val = pm.Normal("opp_raw_val", 0, 1, shape=n_opp)
        delta_opp_val = pm.Deterministic("delta_opp_val", opp_raw_val * sigma_opp_val)
        
        sigma_comp_val = pm.Exponential("sigma_comp_val", 1.0)
        comp_raw_val = pm.Normal("comp_raw_val", 0, 1, shape=n_comp)
        zeta_comp_val = pm.Deterministic("zeta_comp_val", comp_raw_val * sigma_comp_val)
        
        logit_mu = (alpha_val + 
                    pm.math.dot(X_data_val, beta_val) + 
                    gamma_pos_val[posid_val] +
                    theta_val[pid_val] +
                    delta_opp_val[oid_val] +
                    zeta_comp_val[cid_val])
        
        mu = pm.Deterministic("mu", pm.math.invlogit(logit_mu))
        kappa = pm.Exponential("kappa", 0.1)  # Mean=10; stable concentration for Beta regression
        
        alpha_beta = mu * kappa
        beta_beta = (1 - mu) * kappa
        
        pm.Beta("y_val_obs", alpha=alpha_beta, beta=beta_beta, observed=y_val_scaled)
        
        logger.info("Starting MCMC sampling...")
        try:
            trace = pm.sample(
                draws=MODEL_SETTINGS['draws'],
                tune=MODEL_SETTINGS['tune'],
                chains=MODEL_SETTINGS['chains'],
                target_accept=MODEL_SETTINGS['target_accept'],
                random_seed=MODEL_SETTINGS['random_seed'],
                nuts_sampler=MODEL_SETTINGS['nuts_sampler'],
                return_inferencedata=True,
                progressbar=True
            )
        except Exception as e:
            logger.warning(f"numpyro failed: {e}. Falling back to default NUTS.")
            trace = pm.sample(
                draws=MODEL_SETTINGS['draws'],
                tune=MODEL_SETTINGS['tune'],
                chains=MODEL_SETTINGS['chains'],
                target_accept=MODEL_SETTINGS['target_accept'],
                random_seed=MODEL_SETTINGS['random_seed'],
                return_inferencedata=True,
                progressbar=True
            )
        
    trace_path = MODEL_TRACES_DIR / "pooled_trace.nc"
    trace.to_netcdf(trace_path)
    logger.info(f"Saved trace to {trace_path}")
    
    return trace

if __name__ == "__main__":
    fit_pooled_model()
