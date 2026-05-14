import pandas as pd
import numpy as np
import arviz as az
import pickle
import logging
from scipy.special import expit
from config import MODEL_TRACES_DIR, TABLES_DIR, PROCESSED_DATA_DIR, SPATIAL_CONFIG, MIN_EVENTS_THRESHOLD
from src.data.validation import validate_model_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _require_paths(*paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required model artifact(s) missing. Run the pipeline from data build "
            f"through model fitting first: {', '.join(missing)}"
        )

def run_posterior_analysis():
    """
    Generate player leaderboards with Ball Security and Value Retention scores.
    Uses expit for numeric stability and appropriately handles the Hurdle model outputs.
    """
    trace_path = MODEL_TRACES_DIR / "pooled_trace.nc"
    mapping_path = MODEL_TRACES_DIR / "pooled_mappings.pkl"
    scaler_path = MODEL_TRACES_DIR / "pooled_scaler.pkl"
    dataset_path = PROCESSED_DATA_DIR / "all_pressure_dataset.parquet"
    _require_paths(trace_path, mapping_path, scaler_path, dataset_path)
        
    trace = az.from_netcdf(trace_path)
    
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
    player_mapping = mappings['player']
    pos_mapping = mappings['position']
    name_lookup = mappings.get('name_lookup', {})
    pos_lookup = mappings.get('position_lookup', {})
    
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        feature_names = scaler_data['features']
        max_value = scaler_data['max_value']
    
    df = pd.read_parquet(dataset_path)
    validate_model_dataset(df, feature_names, context="posterior analysis dataset")
    event_counts = df['player_id'].value_counts().to_dict()
    
    post = trace.posterior
    
    # Success / Ball Security parameters
    alpha_succ = post['alpha_succ'].values.flatten()
    beta_succ = post['beta_succ'].values.reshape(-1, len(feature_names))
    theta_succ = post['theta_succ'].values.reshape(-1, len(player_mapping))
    gamma_pos_succ = post['gamma_pos_succ'].values.reshape(-1, len(pos_mapping))
    
    # Value Parameters
    alpha_val = post['alpha_val'].values.flatten()
    beta_val = post['beta_val'].values.reshape(-1, len(feature_names))
    theta_val = post['theta_val'].values.reshape(-1, len(player_mapping))
    gamma_pos_val = post['gamma_pos_val'].values.reshape(-1, len(pos_mapping))
    
    # Opponent and competition posterior effects
    delta_opp_succ = post['delta_opp_succ'].values.reshape(-1, post['delta_opp_succ'].shape[-1])
    zeta_comp_succ = post['zeta_comp_succ'].values.reshape(-1, post['zeta_comp_succ'].shape[-1])
    delta_opp_val  = post['delta_opp_val'].values.reshape(-1, post['delta_opp_val'].shape[-1])
    zeta_comp_val  = post['zeta_comp_val'].values.reshape(-1, post['zeta_comp_val'].shape[-1])
    
    # Marginalise by posterior group mean; shape: (n_samples,)
    mean_opp_succ  = delta_opp_succ.mean(axis=1)
    mean_comp_succ = zeta_comp_succ.mean(axis=1)
    mean_opp_val   = delta_opp_val.mean(axis=1)
    mean_comp_val  = zeta_comp_val.mean(axis=1)

    tight_dist = SPATIAL_CONFIG['tight_pressure_radius'] * 0.3
    loose_dist = SPATIAL_CONFIG['tight_pressure_radius'] * 0.6
    
    scenarios = {
        'Front_Tight':   {'dist_nearest_opp': tight_dist, 'angle_nearest_opp': 0.0, 'coverage_arc': 1.5},
        'Front_Loose':   {'dist_nearest_opp': loose_dist, 'angle_nearest_opp': 0.0, 'coverage_arc': 0.8},
        'Lateral_Tight': {'dist_nearest_opp': tight_dist, 'angle_nearest_opp': np.pi/2, 'coverage_arc': 1.5},
        'Lateral_Loose': {'dist_nearest_opp': loose_dist, 'angle_nearest_opp': np.pi/2, 'coverage_arc': 0.8},
        'Back_Tight':    {'dist_nearest_opp': tight_dist, 'angle_nearest_opp': np.pi, 'coverage_arc': 1.5},
        'Back_Loose':    {'dist_nearest_opp': loose_dist, 'angle_nearest_opp': np.pi, 'coverage_arc': 0.8}
    }
    
    scenario_vectors = {}
    for name, s in scenarios.items():
        vec = np.zeros(len(feature_names))
        for f_name, val in s.items():
            if f_name in feature_names:
                f_idx = feature_names.index(f_name)
                if f_name not in ['opps_within_1yd', 'opps_within_2yd', 'opps_within_4yd', 'has_progressive_option']:
                    vec[f_idx] = (val - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
                else:
                    # Binary/count features: use 0, not scaled mean
                    vec[f_idx] = (0 - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
        scenario_vectors[name] = vec
    
    mid_pos_code = next((code for code, name in pos_mapping.items() if name == 'Midfielder'), 0)

    def get_predictions(scenario_vec, player_idx=None, p_code=None):
        pos_effect_succ = gamma_pos_succ[:, p_code] if p_code is not None else gamma_pos_succ[:, mid_pos_code]
        pos_effect_val = gamma_pos_val[:, p_code] if p_code is not None else gamma_pos_val[:, mid_pos_code]
        
        logit_succ = (alpha_succ + np.dot(beta_succ, scenario_vec) + pos_effect_succ
                      + mean_opp_succ + mean_comp_succ)
        logit_val  = (alpha_val + np.dot(beta_val, scenario_vec) + pos_effect_val
                      + mean_opp_val + mean_comp_val)
        
        if player_idx is not None:
            logit_succ += theta_succ[:, player_idx]
            logit_val += theta_val[:, player_idx]
            
        prob_success = expit(logit_succ)
        expected_value_retention = expit(logit_val) * max_value
        
        # Overall EV under pressure
        total_ev = prob_success * expected_value_retention
        return prob_success, expected_value_retention, total_ev

    pop_baselines = {}
    for name, vec in scenario_vectors.items():
        p_succ, e_val, t_ev = get_predictions(vec)
        pop_baselines[name] = t_ev.mean()
    
    leaderboard = []
    for idx_num, player_id in player_mapping.items():
        try:
            n_events = event_counts.get(player_id, 0)
            if n_events < MIN_EVENTS_THRESHOLD:
                continue
            
            player_name = name_lookup.get(player_id, f"ID: {player_id}")
            position_group = pos_lookup.get(player_id, 'Midfielder')
            
            player_pos_code = next((code for code, name in pos_mapping.items() if name == position_group), mid_pos_code)
            
            best_scenario = "N/A"
            max_advantage = -np.inf
            
            for s_name, s_vec in scenario_vectors.items():
                _, _, t_ev = get_predictions(s_vec, idx_num, player_pos_code)
                advantage = t_ev.mean() - pop_baselines[s_name]
                if advantage > max_advantage:
                    max_advantage = advantage
                    best_scenario = s_name
            
            ball_security_samples = theta_succ[:, idx_num]
            value_retention_samples = theta_val[:, idx_num]
            prs_samples = ball_security_samples + value_retention_samples
            
            leaderboard.append({
                'player_id': player_id,
                'player_name': player_name,
                'position_group': position_group,
                'mean_Ball_Security_Score': ball_security_samples.mean(),
                'mean_Value_Retention_Score': value_retention_samples.mean(),
                'mean_PRS': prs_samples.mean(),
                'hdi_5%': np.percentile(prs_samples, 5),
                'hdi_95%': np.percentile(prs_samples, 95),
                'n_events': n_events,
                'best_under_scenario': best_scenario
            })
        except Exception as e:
            logger.debug(f"Error processing player {player_id}: {e}")
            
    lb_df = pd.DataFrame(leaderboard).sort_values(by='mean_PRS', ascending=False)
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "prs_leaderboard.csv"
    lb_df.to_csv(out_path, index=False)
    logger.info(f"Saved leaderboard with {len(lb_df)} players (n>=20) to {out_path}")
    
    top_players = lb_df.head(10)
    logger.info("\n=== TOP 10 PRESSURE RESISTANCE SCORES ===")
    for _, p in top_players.iterrows():
        logger.info(f"{p['player_name']} ({p['position_group']}): PRS={p['mean_PRS']:.3f} | BallSec={p['mean_Ball_Security_Score']:.3f} | ValueRet={p['mean_Value_Retention_Score']:.3f}")

if __name__ == "__main__":
    run_posterior_analysis()
