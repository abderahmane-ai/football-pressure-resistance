import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import logging
import pickle
import arviz as az
from config import TABLES_DIR, MODEL_TRACES_DIR, PROCESSED_DATA_DIR, SPATIAL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cross_validation():
    """
    Principled cross-validation: Correlate training PRS with 
    mean expected value residuals in the holdout dataset.
    Vectorized for performance.
    """
    logger.info("=== CROSS-VALIDATION: VALUE-RESIDUAL CORRELATION ===")
    
    holdout_path = PROCESSED_DATA_DIR / "holdout_pressure_dataset.parquet"
    if not holdout_path.exists():
        logger.error("Holdout dataset not found. Run build_dataset.py first.")
        return
    
    holdout_df = pd.read_parquet(holdout_path)
    holdout_df = holdout_df[holdout_df['dist_nearest_opp'] <= SPATIAL_CONFIG['tight_pressure_radius']]
    
    trace = az.from_netcdf(MODEL_TRACES_DIR / "pooled_trace.nc")
    with open(MODEL_TRACES_DIR / "pooled_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    with open(MODEL_TRACES_DIR / "pooled_scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)
    
    scaler = scaler_data['scaler']
    feature_names = scaler_data['features']
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
    
    X_holdout = holdout_df[feature_names].values
    X_holdout_scaled = scaler.transform(X_holdout)
    
    rev_pos_mapping = {v: k for k, v in pos_mapping.items()}
    holdout_pos_codes = holdout_df['position_group'].map(rev_pos_mapping).fillna(rev_pos_mapping.get('Midfielder', 0)).astype(int).values
    
    # Vectorized computation
    logit_succ_base = alpha_succ[:, np.newaxis] + np.dot(beta_succ, X_holdout_scaled.T)
    logit_val_base = alpha_val[:, np.newaxis] + np.dot(beta_val, X_holdout_scaled.T)
    
    # Add position effects fully vectorized
    logit_succ_base += gamma_pos_succ[:, holdout_pos_codes]
    logit_val_base += gamma_pos_val[:, holdout_pos_codes]
        
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
    player_stats = player_stats[player_stats['value_preserved'] >= 20]
    
    train_lb = pd.read_csv(TABLES_DIR / "prs_leaderboard.csv")
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
    auc = roc_auc_score(y_true_bin, y_pred_prob)
    logger.info(f"  Holdout AUC (Binary Success): {auc:.3f}")
    
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
