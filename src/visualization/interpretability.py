import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import arviz as az
import pandas as pd
import numpy as np
import pickle
from config import MODEL_TRACES_DIR, TABLES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_interpretability_analysis():
    trace_path = MODEL_TRACES_DIR / "pooled_trace.nc"
    if not trace_path.exists():
        logger.warning(f"Trace not found for interpretability: {trace_path}")
        return
        
    trace = az.from_netcdf(trace_path)
    
    scaler_path = MODEL_TRACES_DIR / "pooled_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        feature_names = scaler_data['features']
        max_value = scaler_data.get('max_value', 1.0)
    
    mapping_path = MODEL_TRACES_DIR / "pooled_mappings.pkl"
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
        pos_mapping = mappings['position']
        player_mapping = mappings['player']
        pos_lookup = mappings.get('position_lookup', {})
        
    post = trace.posterior
    
    logger.info("=== INTERPRETABILITY ANALYSIS ===")
    
    # 1. Feature Importance (Beta)
    logger.info("\n1. Feature Importance")
    beta_summary = az.summary(trace, var_names=['beta'], hdi_prob=0.90)
    beta_summary.index = feature_names
    beta_summary['abs_mean'] = beta_summary['mean'].abs()
    beta_summary = beta_summary.sort_values(by='abs_mean', ascending=False)
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    beta_summary.to_csv(TABLES_DIR / "feature_importance.csv")
    logger.info(f"Top 5 features:")
    for feat in beta_summary.head(5).index:
        logger.info(f"  {feat}: {beta_summary.loc[feat, 'mean']:.3f}")
    
    # 2. Variance Decomposition
    logger.info("\n2. Variance Decomposition")
    sigma_theta = post['sigma_theta'].values.flatten()
    sigma_comp = post['sigma_comp'].values.flatten()
    sigma_delta = post['sigma_delta'].values.flatten()
    
    # Compute variances
    var_theta = np.mean(sigma_theta**2)
    var_comp = np.mean(sigma_comp**2)
    var_delta = np.mean(sigma_delta**2)
    
    # Feature variance (approximate from standardized betas)
    beta_samples = post['beta'].values.reshape(-1, len(feature_names))
    # Variance explained by features: var(X*beta) ≈ sum(beta^2) since X is standardized
    var_features = np.mean(np.sum(beta_samples**2, axis=1))
    
    total_var = var_theta + var_comp + var_delta + var_features
    
    var_df = pd.DataFrame({
        'Component': ['Player Skill (theta)', 'Competition (gamma)', 'Opponent Press (delta)', 'Spatial Features (beta)'],
        'Variance': [var_theta, var_comp, var_delta, var_features],
        'Proportion': [var_theta/total_var, var_comp/total_var, var_delta/total_var, var_features/total_var],
        'SD_Mean': [np.mean(sigma_theta), np.mean(sigma_comp), np.mean(sigma_delta), np.sqrt(var_features)]
    })
    var_df.to_csv(TABLES_DIR / "variance_decomposition.csv", index=False)
    
    logger.info("Variance decomposition:")
    for _, row in var_df.iterrows():
        logger.info(f"  {row['Component']}: {row['Proportion']:.1%} (var={row['Variance']:.3f})")
    
    # 3. Population Marginal Effects
    logger.info("\n3. Population Marginal Effects")
    alpha_samples = post['alpha'].values.flatten()
    gamma_pos_samples = post['gamma_pos'].values.reshape(-1, len(pos_mapping))
    
    # Find index for Midfielder baseline
    mid_pos_code = None
    for code, name in pos_mapping.items():
        if name == 'Midfielder':
            mid_pos_code = code
            break
    if mid_pos_code is None: mid_pos_code = 0
    
    def compute_marginal(feat_idx, values_range):
        results = []
        for val in values_range:
            std_val = (val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
            logits = alpha_samples + beta_samples[:, feat_idx] * std_val + gamma_pos_samples[:, mid_pos_code]
            probs = (1 / (1 + np.exp(-logits))) * max_value
            results.append({
                'value': val,
                'mean_p': np.mean(probs),
                'hdi_5%': np.percentile(probs, 5),
                'hdi_95%': np.percentile(probs, 95)
            })
        return pd.DataFrame(results)

    if 'dist_nearest_opp' in feature_names:
        idx = feature_names.index('dist_nearest_opp')
        grid = np.linspace(0.5, 15.0, 50)
        df_marginal = compute_marginal(idx, grid)
        df_marginal.to_csv(TABLES_DIR / "marginal_dist.csv", index=False)
        logger.info("Saved marginal effect: dist_nearest_opp")

    if 'coverage_arc' in feature_names:
        idx = feature_names.index('coverage_arc')
        grid = np.linspace(0.0, np.pi, 50)
        df_marginal = compute_marginal(idx, grid)
        df_marginal.to_csv(TABLES_DIR / "marginal_arc.csv", index=False)
        logger.info("Saved marginal effect: coverage_arc")
    
    # 4. ICE Curves (Individual Conditional Expectation)
    logger.info("\n4. ICE Curves for Top/Bottom Players")
    
    # Load leaderboard to get top/bottom players
    lb_path = TABLES_DIR / "prs_leaderboard.csv"
    if lb_path.exists():
        lb_df = pd.read_csv(lb_path)
        top_3 = lb_df.head(3)
        bottom_3 = lb_df.tail(3)
        
        theta_samples = post['theta'].values.reshape(-1, len(mappings['player']))
        
        # Find indices for these players
        selected_players = []
        for _, row in pd.concat([top_3, bottom_3]).iterrows():
            player_id = row['player_id']
            player_pos = row['position_group']
            pos_code = next((code for code, name in pos_mapping.items() if name == player_pos), mid_pos_code)
            
            for idx, pid in player_mapping.items():
                if pid == player_id:
                    selected_players.append({
                        'name': row['player_name'],
                        'idx': idx,
                        'pos_code': pos_code,
                        'rank': 'Top' if row['player_name'] in top_3['player_name'].values else 'Bottom'
                    })
                    break
        
        # Compute ICE for distance to nearest opponent
        if 'dist_nearest_opp' in feature_names:
            feat_idx = feature_names.index('dist_nearest_opp')
            dist_grid = np.linspace(0.5, 15.0, 50)
            
            ice_results = []
            for player in selected_players:
                player_theta = theta_samples[:, player['idx']]
                player_pos_effect = gamma_pos_samples[:, player['pos_code']]
                for dist_val in dist_grid:
                    std_val = (dist_val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
                    logits = alpha_samples + beta_samples[:, feat_idx] * std_val + player_pos_effect + player_theta
                    probs = (1 / (1 + np.exp(-logits))) * max_value
                    ice_results.append({
                        'player_name': player['name'],
                        'rank_group': player['rank'],
                        'distance': dist_val,
                        'mean_p': np.mean(probs)
                    })
            
            ice_df = pd.DataFrame(ice_results)
            ice_df.to_csv(TABLES_DIR / "ice_curves.csv", index=False)
            logger.info("Saved ICE curves for 6 players")
    
    # 5. Counterfactual Analysis
    logger.info("\n5. Counterfactual Analysis")
    
    if lb_path.exists():
        lb_df = pd.read_csv(lb_path)
        if len(lb_df) > 0:
            top_player = lb_df.iloc[0]
            player_id = top_player['player_id']
            player_pos = top_player['position_group']
            pos_code = next((code for code, name in pos_mapping.items() if name == player_pos), mid_pos_code)
            
            # Find player index
            player_idx = None
            for idx, pid in player_mapping.items():
                if pid == player_id:
                    player_idx = idx
                    break
            
            if player_idx is not None:
                player_theta = theta_samples[:, player_idx]
                player_pos_effect = gamma_pos_samples[:, pos_code]
                
                # Define tight pressure scenario
                tight_scenario = {
                    'dist_nearest_opp': 1.5,
                    'coverage_arc': 1.5,
                    'opps_within_2yd': 2
                }
                
                # Standardize
                scenario_vec = np.zeros(len(feature_names))
                for feat, val in tight_scenario.items():
                    if feat in feature_names:
                        f_idx = feature_names.index(feat)
                        scenario_vec[f_idx] = (val - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
                
                # With player effect
                logits_with = alpha_samples + np.dot(beta_samples, scenario_vec) + player_pos_effect + player_theta
                probs_with = (1 / (1 + np.exp(-logits_with))) * max_value
                
                # Without player effect (population mean for that position)
                logits_without = alpha_samples + np.dot(beta_samples, scenario_vec) + player_pos_effect
                probs_without = (1 / (1 + np.exp(-logits_without))) * max_value
                
                counterfactual_df = pd.DataFrame([{
                    'player_name': top_player['player_name'],
                    'scenario': 'Tight Pressure (1.5yd, arc=1.5)',
                    'expected_value_with_skill': np.mean(probs_with),
                    'expected_value_population': np.mean(probs_without),
                    'advantage': np.mean(probs_with) - np.mean(probs_without)
                }])
                counterfactual_df.to_csv(TABLES_DIR / "counterfactual_analysis.csv", index=False)
                
                logger.info(f"Counterfactual for {top_player['player_name']}:")
                logger.info(f"  With skill: {np.mean(probs_with):.3f}")
                logger.info(f"  Population: {np.mean(probs_without):.3f}")
                logger.info(f"  Advantage: {np.mean(probs_with) - np.mean(probs_without):.3f}")

if __name__ == "__main__":
    run_interpretability_analysis()
