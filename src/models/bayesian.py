import logging
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pymc as pm
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

from config import PROCESSED_DATA_DIR, MODEL_TRACES_DIR, MODEL_SETTINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_pooled_model():
    """
    Fit the pooled hierarchical Beta regression model with value_preserved outcome.
    """
    dataset_path = PROCESSED_DATA_DIR / "all_pressure_dataset.parquet"
    if not dataset_path.exists():
        logger.error(f"Dataset {dataset_path} not found. Run build_dataset.py first.")
        return None
        
    df = pd.read_parquet(dataset_path)
    df = df.dropna()
    
    logger.info(f"Loaded dataset: {len(df)} observations")
    logger.info(f"Competitions: {df['competition'].unique()}")
    logger.info(f"Unique players: {df['player_id'].nunique()}")
    logger.info(f"Position groups: {df['position_group'].value_counts().to_dict()}")
    
    # Select features
    feature_cols = [
        'dist_nearest_opp', 'dist_2nd_nearest_opp', 'opps_within_1yd', 
        'opps_within_2yd', 'opps_within_4yd', 'angle_nearest_opp', 
        'coverage_arc', 'voronoi_area', 'n_free_teammates', 
        'max_free_triangle_area', 'dist_nearest_free_teammate', 
        'angle_nearest_free_teammate', 'pitch_control', 'opp_density_5yd',
        'has_progressive_option', 'xt_value', 'zone', 'game_state_diff',
        'minutes_elapsed', 'match_period'
    ]
    
    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_features)} features")
    
    X = df[available_features].values
    
    # Scale value_preserved to (0, 1) for Beta regression
    # Add small epsilon to avoid exact 0 and 1
    y_raw = df['value_preserved'].values
    max_value = y_raw.max()
    if max_value == 0:
        max_value = 0.15  # Fallback
    
    epsilon = 1e-6
    y_scaled = (y_raw / max_value) * (1 - 2*epsilon) + epsilon
    y_scaled = np.clip(y_scaled, epsilon, 1 - epsilon)
    
    logger.info(f"Value range: [{y_raw.min():.4f}, {y_raw.max():.4f}] -> [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler and max_value
    MODEL_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_TRACES_DIR / "pooled_scaler.pkl", "wb") as f:
        pickle.dump({
            'scaler': scaler, 
            'features': available_features,
            'max_value': max_value,
            'epsilon': epsilon
        }, f)
    
    # Create integer indices for random effects
    player_cats = df['player_id'].astype('category')
    comp_cats = df['competition'].astype('category')
    opp_team_cats = df['opponent_team_id'].astype('category')
    pos_cats = df['position_group'].astype('category')
    
    player_idx = player_cats.cat.codes.values
    comp_idx = comp_cats.cat.codes.values
    opp_idx = opp_team_cats.cat.codes.values
    pos_idx = pos_cats.cat.codes.values
    
    n_players = len(player_cats.cat.categories)
    n_comps = len(comp_cats.cat.categories)
    n_teams = len(opp_team_cats.cat.categories)
    n_pos = len(pos_cats.cat.categories)
    n_features = len(available_features)
    
    # Save mappings
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
        
    logger.info(f"Model dimensions: {len(df)} obs, {n_players} players, {n_comps} comps, {n_teams} teams, {n_pos} positions, {n_features} features")
    
    # Build Beta regression model
    with pm.Model() as model:
        # Data containers
        X_data = pm.Data("X", X_scaled)
        player_idx_data = pm.Data("player_idx", player_idx)
        comp_idx_data = pm.Data("comp_idx", comp_idx)
        opp_idx_data = pm.Data("opp_idx", opp_idx)
        pos_idx_data = pm.Data("pos_idx", pos_idx)
        
        # Fixed effects
        alpha = pm.Normal("alpha", 0, 1.5)
        beta = pm.Normal("beta", 0, 1.0, shape=n_features)
        
        # Position group fixed effect
        gamma_pos = pm.Normal("gamma_pos", 0, 1.0, shape=n_pos)
        
        # Player random effect (non-centered)
        sigma_theta = pm.Exponential("sigma_theta", 1.0)
        theta_raw = pm.Normal("theta_raw", 0, 1, shape=n_players)
        theta = pm.Deterministic("theta", theta_raw * sigma_theta)
        
        # Competition random effect (non-centered)
        sigma_comp = pm.Exponential("sigma_comp", 1.0)
        comp_raw = pm.Normal("comp_raw", 0, 1, shape=n_comps)
        gamma_comp = pm.Deterministic("gamma_comp", comp_raw * sigma_comp)
        
        # Opponent team random effect (centered)
        sigma_delta = pm.Exponential("sigma_delta", 1.0)
        delta_offset = pm.Normal("delta_offset", 0, sigma=sigma_delta, shape=n_teams)
        delta = pm.Deterministic("delta", delta_offset - delta_offset.mean())
        
        # Linear predictor
        logit_mu = (alpha + 
                    pm.math.dot(X_data, beta) + 
                    gamma_pos[pos_idx_data] +
                    theta[player_idx_data] + 
                    gamma_comp[comp_idx_data] + 
                    delta[opp_idx_data])
        
        mu = pm.Deterministic("mu", pm.math.invlogit(logit_mu))
        
        # Beta dispersion parameter
        kappa = pm.Exponential("kappa", 1.0)
        
        # Beta likelihood
        alpha_beta = mu * kappa
        beta_beta = (1 - mu) * kappa
        
        y_obs = pm.Beta("y", alpha=alpha_beta, beta=beta_beta, observed=y_scaled)
        
        # Sample
        logger.info("Starting MCMC sampling with numpyro...")
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
            logger.warning(f"numpyro sampling failed: {e}. Falling back to default NUTS.")
            trace = pm.sample(
                draws=MODEL_SETTINGS['draws'],
                tune=MODEL_SETTINGS['tune'],
                chains=MODEL_SETTINGS['chains'],
                target_accept=MODEL_SETTINGS['target_accept'],
                random_seed=MODEL_SETTINGS['random_seed'],
                return_inferencedata=True,
                progressbar=True
            )
        
        # Posterior predictive
        logger.info("Sampling posterior predictive...")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=MODEL_SETTINGS['random_seed'])
        
    trace_path = MODEL_TRACES_DIR / "pooled_trace.nc"
    trace.to_netcdf(trace_path)
    logger.info(f"Saved trace to {trace_path}")
    
    # Print diagnostics
    logger.info("\n=== MODEL DIAGNOSTICS ===")
    summary = az.summary(trace, var_names=['alpha', 'beta', 'gamma_pos', 'sigma_theta', 'sigma_comp', 'sigma_delta', 'kappa'])
    logger.info(f"\n{summary}")
    
    # Check for divergences
    divergences = trace.sample_stats.diverging.sum().values
    logger.info(f"\nDivergences: {divergences}")
    
    if divergences > 0:
        logger.warning(f"Model had {divergences} divergent transitions. Consider increasing target_accept.")
    
    return trace

if __name__ == "__main__":
    fit_pooled_model()
