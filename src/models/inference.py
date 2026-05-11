import pandas as pd
import numpy as np
import arviz as az
import pickle
import logging
from config import MODEL_TRACES_DIR, TABLES_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_posterior_analysis():
    """
    Generate player leaderboards with PRS and 'Best Under' scenario classification.
    Fixes the scenario bug by ensuring correct feature indexing and standardization.
    """
    trace_path = MODEL_TRACES_DIR / "pooled_trace.nc"
    if not trace_path.exists():
        logger.error(f"Trace not found at {trace_path}")
        return
        
    trace = az.from_netcdf(trace_path)
    
    # Load mappings
    mapping_path = MODEL_TRACES_DIR / "pooled_mappings.pkl"
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
    player_mapping = mappings['player']
    pos_mapping = mappings['position']
    name_lookup = mappings.get('name_lookup', {})
    pos_lookup = mappings.get('position_lookup', {})
    
    # Load scaler and feature metadata
    scaler_path = MODEL_TRACES_DIR / "pooled_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        feature_names = scaler_data['features']
        max_value = scaler_data['max_value']
    
    df = pd.read_parquet(PROCESSED_DATA_DIR / "all_pressure_dataset.parquet")
    event_counts = df['player_id'].value_counts().to_dict()
    
    # Extract posterior samples
    post = trace.posterior
    alpha_samples = post['alpha'].values.flatten()
    beta_samples = post['beta'].values.reshape(-1, len(feature_names))
    theta_samples = post['theta'].values.reshape(-1, len(player_mapping))
    gamma_pos_samples = post['gamma_pos'].values.reshape(-1, len(pos_mapping))
    
    # Define 6 distinct scenarios
    # Geometric mapping relative to the opponent goal (0 radians):
    # Front: 0.0 (between player and goal), Lateral: π/2, Back: π (behind player)
    # Tight: dist=1.5, arc=1.5 | Loose: dist=3.0, arc=0.8
    scenarios = {
        'Front_Tight':   {'dist_nearest_opp': 1.5, 'angle_nearest_opp': 0.0, 'coverage_arc': 1.5},
        'Front_Loose':   {'dist_nearest_opp': 3.0, 'angle_nearest_opp': 0.0, 'coverage_arc': 0.8},
        'Lateral_Tight': {'dist_nearest_opp': 1.5, 'angle_nearest_opp': np.pi/2, 'coverage_arc': 1.5},
        'Lateral_Loose': {'dist_nearest_opp': 3.0, 'angle_nearest_opp': np.pi/2, 'coverage_arc': 0.8},
        'Back_Tight':    {'dist_nearest_opp': 1.5, 'angle_nearest_opp': np.pi, 'coverage_arc': 1.5},
        'Back_Loose':    {'dist_nearest_opp': 3.0, 'angle_nearest_opp': np.pi, 'coverage_arc': 0.8}
    }
    
    # Standardize scenario vectors using exact indices
    scenario_vectors = {}
    for name, s in scenarios.items():
        vec = np.zeros(len(feature_names))
        # Default all other features to 0 (mean)
        for f_name, val in s.items():
            if f_name in feature_names:
                f_idx = feature_names.index(f_name)
                vec[f_idx] = (val - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
        scenario_vectors[name] = vec
    
    # Reference position for population scenarios (Midfielder)
    mid_pos_code = None
    for code, name in pos_mapping.items():
        if name == 'Midfielder':
            mid_pos_code = code
            break
    if mid_pos_code is None: mid_pos_code = 0

    def get_expected_value(scenario_vec, player_idx=None, p_code=None):
        """Compute E[value] = invlogit(logit_mu) * max_value."""
        logit = alpha_samples + np.dot(beta_samples, scenario_vec)
        if p_code is not None:
            logit += gamma_pos_samples[:, p_code]
        else:
            logit += gamma_pos_samples[:, mid_pos_code]
            
        if player_idx is not None:
            logit += theta_samples[:, player_idx]
            
        mu = 1 / (1 + np.exp(-logit))
        return mu * max_value

    # Compute population baseline for each scenario
    pop_baselines = {name: get_expected_value(vec).mean() for name, vec in scenario_vectors.items()}
    
    # Summarize theta (PRS)
    theta_summary = az.summary(trace, var_names=['theta'], hdi_prob=0.90)
    
    leaderboard = []
    for idx, row in theta_summary.iterrows():
        try:
            # Parse index: 'theta[0]' -> 0
            idx_num = int(idx.split('[')[1].split(']')[0])
            player_id = player_mapping[idx_num]
            
            n_events = event_counts.get(player_id, 0)
            if n_events < 20: continue # Minimum sample size filter
            
            player_name = name_lookup.get(player_id, f"ID: {player_id}")
            position_group = pos_lookup.get(player_id, 'Midfielder')
            
            # Find position code for this player
            player_pos_code = mid_pos_code
            for code, name in pos_mapping.items():
                if name == position_group:
                    player_pos_code = code
                    break
            
            # Composure specialization
            best_scenario = "N/A"
            max_advantage = -np.inf
            
            for s_name, s_vec in scenario_vectors.items():
                player_expected = get_expected_value(s_vec, idx_num, player_pos_code).mean()
                advantage = player_expected - pop_baselines[s_name]
                if advantage > max_advantage:
                    max_advantage = advantage
                    best_scenario = s_name
            
            leaderboard.append({
                'player_id': player_id,
                'player_name': player_name,
                'position_group': position_group,
                'mean_PRS': row['mean'],
                'hdi_5%': row['hdi_5%'],
                'hdi_95%': row['hdi_95%'],
                'n_events': n_events,
                'best_under_scenario': best_scenario,
                'p_positive': (theta_samples[:, idx_num] > 0).mean()
            })
        except Exception as e:
            logger.debug(f"Error processing {idx}: {e}")
            
    lb_df = pd.DataFrame(leaderboard).sort_values(by='mean_PRS', ascending=False)
    
    out_path = TABLES_DIR / "prs_leaderboard.csv"
    lb_df.to_csv(out_path, index=False)
    logger.info(f"Saved leaderboard with {len(lb_df)} players (n>=20) to {out_path}")
    
    # Top specialized players log
    top_players = lb_df.head(10)
    logger.info("\n=== TOP 10 PRESSURE RESISTANCE SCORES ===")
    for _, p in top_players.iterrows():
        logger.info(f"{p['player_name']} ({p['position_group']}): {p['mean_PRS']:.3f} [Best: {p['best_under_scenario']}]")

if __name__ == "__main__":
    run_posterior_analysis()
