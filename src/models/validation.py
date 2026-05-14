import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import logging
import pickle
import arviz as az
from config import TABLES_DIR, MODEL_TRACES_DIR, PROCESSED_DATA_DIR, SPATIAL_CONFIG, MIN_EVENTS_THRESHOLD
from src.data.validation import validate_model_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _require_paths(*paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required validation artifact(s) missing. Run data build, model fitting, "
            f"and leaderboard generation first: {', '.join(missing)}"
        )

def run_cross_validation():
    """
    Principled cross-validation: Correlate training PRS with 
    mean expected value residuals in the holdout dataset.
    Vectorized for performance.
    """
    logger.info("=== CROSS-VALIDATION: VALUE-RESIDUAL CORRELATION ===")
    
    holdout_path = PROCESSED_DATA_DIR / "holdout_pressure_dataset.parquet"
    trace_path = MODEL_TRACES_DIR / "pooled_trace.nc"
    mapping_path = MODEL_TRACES_DIR / "pooled_mappings.pkl"
    scaler_path = MODEL_TRACES_DIR / "pooled_scaler.pkl"
    leaderboard_path = TABLES_DIR / "prs_leaderboard.csv"
    _require_paths(holdout_path, trace_path, mapping_path, scaler_path, leaderboard_path)
    
    holdout_df = pd.read_parquet(holdout_path)
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
    feature_names = scaler_data['features']
    validate_model_dataset(holdout_df, feature_names, context="holdout pressure dataset")
    holdout_df = holdout_df[holdout_df['dist_nearest_opp'] <= SPATIAL_CONFIG['tight_pressure_radius']]
    
    trace = az.from_netcdf(trace_path)
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
    
    scaler = scaler_data['scaler']
    max_value = scaler_data['max_value']
    pos_mapping = mappings['position']
    
    post = trace.posterior
    # Success params
    alpha_succ = post['alpha_succ'].values.flatten()
    beta_succ = post['beta_succ'].values.reshape(-1, len(feature_names))
    gamma_pos_succ = post['gamma_pos_succ'].values.reshape(-1, len(pos_mapping))
    
    # Value params
    alpha_val = post['alpha_val'].values.flatten()
    beta_val = post['beta_val'].values.reshape(-1, len(feature_names))
    gamma_pos_val = post['gamma_pos_val'].values.reshape(-1, len(pos_mapping))

    # Marginalise opp/comp by posterior group mean — consistent with leaderboard
    delta_opp_succ = post['delta_opp_succ'].values.reshape(-1, post['delta_opp_succ'].shape[-1])
    zeta_comp_succ = post['zeta_comp_succ'].values.reshape(-1, post['zeta_comp_succ'].shape[-1])
    delta_opp_val  = post['delta_opp_val'].values.reshape(-1, post['delta_opp_val'].shape[-1])
    zeta_comp_val  = post['zeta_comp_val'].values.reshape(-1, post['zeta_comp_val'].shape[-1])
    mean_opp_succ  = delta_opp_succ.mean(axis=1)  # (n_samples,)
    mean_comp_succ = zeta_comp_succ.mean(axis=1)
    mean_opp_val   = delta_opp_val.mean(axis=1)
    mean_comp_val  = zeta_comp_val.mean(axis=1)
    
    X_holdout = holdout_df[feature_names].values
    X_holdout_scaled = scaler.transform(X_holdout)
    
    rev_pos_mapping = {v: k for k, v in pos_mapping.items()}
    holdout_pos_codes = holdout_df['position_group'].map(rev_pos_mapping).fillna(rev_pos_mapping.get('Midfielder', 0)).astype(int).values
    
    # Vectorised linear predictors
    logit_succ_base = alpha_succ[:, np.newaxis] + np.dot(beta_succ, X_holdout_scaled.T)
    logit_val_base = alpha_val[:, np.newaxis] + np.dot(beta_val, X_holdout_scaled.T)

    # Position effects
    logit_succ_base += gamma_pos_succ[:, holdout_pos_codes]
    logit_val_base += gamma_pos_val[:, holdout_pos_codes]

    # Marginalised opp/comp effects
    logit_succ_base += mean_opp_succ[:, np.newaxis] + mean_comp_succ[:, np.newaxis]
    logit_val_base  += mean_opp_val[:, np.newaxis]  + mean_comp_val[:, np.newaxis]
        
    p_succ = expit(logit_succ_base)
    mu_val = expit(logit_val_base) * max_value
    
    # Predicted Expected Value
    predicted_ev = (p_succ * mu_val).mean(axis=0)
    
    # Observed Expected Value (Success * Intended xT)
    observed_val = holdout_df['success'].values * holdout_df['value_preserved'].values
    residuals = observed_val - predicted_ev
    holdout_df['residual'] = residuals
    
    player_stats = holdout_df.groupby('player_id').agg({
        'residual': 'mean',
        'value_preserved': 'count',
        'player_name': 'first'
    }).reset_index()
    player_stats = player_stats[player_stats['value_preserved'] >= MIN_EVENTS_THRESHOLD]
    
    train_lb = pd.read_csv(leaderboard_path)
    merged = train_lb.merge(player_stats, on='player_id', suffixes=('_train', '_holdout'))
    
    if len(merged) < 5:
        logger.warning(f"Insufficient overlapping players for correlation (n={len(merged)})")
        return
        
    pearson_corr, p_p = pearsonr(merged['mean_PRS'], merged['residual'])
    spearman_corr, p_s = spearmanr(merged['mean_PRS'], merged['residual'])
    
    logger.info(f"Stability Analysis (n={len(merged)} overlapping players):")
    logger.info(f"  Pearson Correlation: {pearson_corr:.3f} (p={p_p:.4f})")
    logger.info(f"  Spearman Correlation: {spearman_corr:.3f} (p={p_s:.4f})")
    
    y_true_bin = holdout_df['success'].values.astype(int)
    y_pred_prob = p_succ.mean(axis=0)
    if len(np.unique(y_true_bin)) == 2:
        auc = roc_auc_score(y_true_bin, y_pred_prob)
        logger.info(f"  Holdout AUC (Binary Success): {auc:.3f}")
    else:
        auc = np.nan
        logger.warning("Holdout AUC skipped because only one success class is present.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES_DIR / "holdout_correlation_data.csv", index=False)
    
    metrics_df = pd.DataFrame([{
        'n_players': len(merged),
        'pearson': pearson_corr,
        'pearson_p': p_p,
        'auc': auc
    }])
    metrics_df.to_csv(TABLES_DIR / "holdout_metrics.csv", index=False)

if __name__ == "__main__":
    run_cross_validation()
