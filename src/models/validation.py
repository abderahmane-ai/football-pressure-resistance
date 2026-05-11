import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import logging
import pickle
import arviz as az
from config import TABLES_DIR, MODEL_TRACES_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cross_validation():
    """
    Principled cross-validation: Correlate training PRS (theta) with 
    mean value-preserved residuals in the holdout dataset.
    """
    logger.info("=== CROSS-VALIDATION: VALUE-RESIDUAL CORRELATION ===")
    
    # 1. Load holdout dataset
    holdout_path = PROCESSED_DATA_DIR / "holdout_pressure_dataset.parquet"
    if not holdout_path.exists():
        logger.error("Holdout dataset not found. Run build_dataset.py first.")
        return
    
    holdout_df = pd.read_parquet(holdout_path)
    # Tight-pressure filter is already applied during build_dataset, 
    # but we ensure it here for robustness.
    holdout_df = holdout_df[holdout_df['dist_nearest_opp'] <= 5.0]
    
    # 2. Load trained model artifacts
    trace = az.from_netcdf(MODEL_TRACES_DIR / "pooled_trace.nc")
    with open(MODEL_TRACES_DIR / "pooled_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    with open(MODEL_TRACES_DIR / "pooled_scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)
    
    scaler = scaler_data['scaler']
    feature_names = scaler_data['features']
    max_value = scaler_data['max_value']
    pos_mapping = mappings['position'] # e.g., {0: 'Defender', 1: 'Forward', 2: 'Midfielder'}
    
    # 3. Extract fixed effect samples
    post = trace.posterior
    alpha_samples = post['alpha'].values.flatten()
    beta_samples = post['beta'].values.reshape(-1, len(feature_names))
    gamma_pos_samples = post['gamma_pos'].values.reshape(-1, len(pos_mapping))
    
    # 4. Predict Expected Value for holdout events
    X_holdout = holdout_df[feature_names].values
    X_holdout_scaled = scaler.transform(X_holdout)
    
    # Get position codes for holdout events
    # We map 'position_group' string to the integer code used in the model
    rev_pos_mapping = {v: k for k, v in pos_mapping.items()}
    holdout_pos_codes = holdout_df['position_group'].map(rev_pos_mapping).fillna(rev_pos_mapping.get('Midfielder', 0)).astype(int).values
    
    # population_logit = alpha + X*beta + gamma_pos
    # We compute mean prediction over samples for efficiency
    logit_base = alpha_samples[:, np.newaxis] + np.dot(beta_samples, X_holdout_scaled.T)
    
    # Add position fixed effects
    for i in range(len(holdout_df)):
        logit_base[:, i] += gamma_pos_samples[:, holdout_pos_codes[i]]
        
    mu = 1 / (1 + np.exp(-logit_base))
    predicted_val_mean = (mu * max_value).mean(axis=0)
    
    # 5. Compute Residuals
    observed_val = holdout_df['value_preserved'].values
    residuals = observed_val - predicted_val_mean
    holdout_df['residual'] = residuals
    
    # 6. Aggregate by Player (Minimum 20 events in holdout)
    player_stats = holdout_df.groupby('player_id').agg({
        'residual': 'mean',
        'value_preserved': 'count',
        'player_name': 'first'
    }).reset_index()
    player_stats = player_stats[player_stats['value_preserved'] >= 20]
    
    # 7. Merge with training PRS (theta)
    train_lb = pd.read_csv(TABLES_DIR / "prs_leaderboard.csv")
    merged = train_lb.merge(player_stats, on='player_id', suffixes=('_train', '_holdout'))
    
    if len(merged) < 5:
        logger.warning(f"Insufficient overlapping players for correlation (n={len(merged)})")
        return
        
    # 8. Correlation Analysis
    pearson_corr, p_p = pearsonr(merged['mean_PRS'], merged['residual'])
    spearman_corr, p_s = spearmanr(merged['mean_PRS'], merged['residual'])
    
    logger.info(f"Stability Analysis (n={len(merged)} overlapping players):")
    logger.info(f"  Pearson Correlation: {pearson_corr:.3f} (p={p_p:.4f})")
    logger.info(f"  Spearman Correlation: {spearman_corr:.3f} (p={p_s:.4f})")
    
    # Supplementary AUC for binary success
    y_true_bin = holdout_df['success'].values.astype(int)
    y_pred_prob = mu.mean(axis=0)
    auc = roc_auc_score(y_true_bin, y_pred_prob)
    logger.info(f"  Holdout AUC (Binary Success): {auc:.3f}")
    
    # Save results
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
