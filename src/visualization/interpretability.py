"""Posterior interpretability: feature importance, variance decomposition, marginal effects, ICE curves."""
from __future__ import annotations

import logging
import pickle
import warnings
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from config import (
    CROSS_VALIDATION_HOLDOUT,
    MODEL_TRACES_DIR,
    PROCESSED_DATA_DIR,
    SPATIAL_CONFIG,
    TABLES_DIR,
)
from src.features.spatial import expand_spline_features

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def run_interpretability_analysis() -> None:
    """Generate feature importance, variance decomposition, marginal effects, and ICE curves."""
    holdout = CROSS_VALIDATION_HOLDOUT
    trace_path = MODEL_TRACES_DIR / f"pooled_trace_{holdout}.nc"
    if not trace_path.exists():
        logger.warning("Trace not found for interpretability: %s", trace_path)
        return

    trace = az.from_netcdf(trace_path)

    scaler_path = MODEL_TRACES_DIR / f"pooled_scaler_{holdout}.pkl"
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
        scaler = scaler_data['scaler']
        feature_names = scaler_data['features']
        max_value = scaler_data.get('max_value', 1.0)
        min_value = scaler_data.get('min_value', 0.0)

    # Recompute masks for global vs. position-specific features
    scaler_psf = scaler_data.get("position_specific_features", [])
    def _is_pos_specific(feat: str) -> bool:
        if feat in scaler_psf:
            return True
        for root in scaler_psf:
            if feat.startswith(f"{root}_spline_"):
                return True
        return False

    pos_specific_mask = np.array([_is_pos_specific(f) for f in feature_names])
    global_mask = ~pos_specific_mask
    n_pos_specific = int(pos_specific_mask.sum())
    n_global = int(global_mask.sum())

    mapping_path = MODEL_TRACES_DIR / f"pooled_mappings_{holdout}.pkl"
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
        pos_mapping = mappings['position']
        player_mapping = mappings['player']

    post = trace.posterior

    logger.info("=== INTERPRETABILITY ANALYSIS ===")

    # ── 1. Feature importance ─────────────────────────────────────────────────
    # For global features: single coefficient from beta_global.
    # For position-specific features: mean-across-positions from beta_pos,
    # plus per-position breakdown saved separately.
    logger.info("1. Feature Importance")

    beta_global_succ = post["beta_global_succ"].values.reshape(-1, n_global)
    beta_pos_succ = post["beta_pos_succ"].values.reshape(-1, len(pos_mapping), n_pos_specific)
    beta_global_val = post["beta_global_val"].values.reshape(-1, n_global)
    beta_pos_val = post["beta_pos_val"].values.reshape(-1, len(pos_mapping), n_pos_specific)

    global_names = [feature_names[i] for i in range(len(feature_names)) if global_mask[i]]
    pos_names = [feature_names[i] for i in range(len(feature_names)) if pos_specific_mask[i]]

    # Aggregate position-specific β to mean across positions
    pos_mean_succ = beta_pos_succ.mean(axis=1)  # (n_samples, n_pos_specific)
    pos_mean_val = beta_pos_val.mean(axis=1)

    # Combine global + mean position-specific for a single feature importance table
    combined_beta_succ = np.column_stack([beta_global_succ, pos_mean_succ])
    combined_beta_val = np.column_stack([beta_global_val, pos_mean_val])
    combined_names = global_names + pos_names

    beta_succ_df = pd.DataFrame({"mean": combined_beta_succ.mean(axis=0)},
                                index=[f + "_ball_security" for f in combined_names])
    beta_val_df = pd.DataFrame({"mean": combined_beta_val.mean(axis=0)},
                               index=[f + "_value_retention" for f in combined_names])
    beta_summary = pd.concat([beta_succ_df, beta_val_df])
    beta_summary['abs_mean'] = beta_summary['mean'].abs()
    beta_summary = beta_summary.sort_values(by='abs_mean', ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    beta_summary.to_csv(TABLES_DIR / "feature_importance.csv")

    # Per-position feature importance for position-specific features
    pos_feat_rows = []
    for p_code, pos_name in pos_mapping.items():
        for j, f_name in enumerate(pos_names):
            for model_label, arr in [("ball_security", beta_pos_succ), ("value_retention", beta_pos_val)]:
                samples = arr[:, p_code, j]
                pos_feat_rows.append({
                    "position": pos_name, "feature": f_name, "model": model_label,
                    "mean": samples.mean(), "hdi_5%": np.percentile(samples, 5),
                    "hdi_95%": np.percentile(samples, 95),
                })
    pd.DataFrame(pos_feat_rows).to_csv(TABLES_DIR / "feature_importance_by_position.csv", index=False)

    # ── 2. Variance decomposition ─────────────────────────────────────────────
    # True sample variance of the linear predictor (Xβ) captures multi-collinearity
    # correctly — summing squared βs assumes zero correlation between features.
    # With position-specific β, the predictor is global_β·X_global + β_pos[p]·X_pos_specific.
    logger.info("2. Variance Decomposition")

    df = pd.read_parquet(PROCESSED_DATA_DIR / f"all_pressure_dataset_{holdout}.parquet").dropna()

    # Expand spline features using the fitted transformers saved with the scaler
    spline_transformers = scaler_data.get("spline_transformers")
    if spline_transformers:
        df = expand_spline_features(df, spline_transformers)

    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    rev_pos_mapping = {v: k for k, v in pos_mapping.items()}
    pos_idx = df["position_group"].map(rev_pos_mapping).fillna(rev_pos_mapping.get("Midfielder", 0)).astype(int).values


    # Ball security model — with correlated θ, extract sigma from packed Cholesky
    # theta_chol is stored as packed lower-triangular: [chol[0,0], chol[1,0], chol[1,1]]
    theta_chol_packed = post['theta_chol'].values
    theta_chol_flat = theta_chol_packed.reshape(-1, theta_chol_packed.shape[-1])
    sigma_theta_succ = theta_chol_flat[:, 0]
    var_theta_succ = float(np.mean(sigma_theta_succ**2))

    sigma_opp_succ = post['sigma_opp_succ'].values.flatten()
    var_opp_succ = float(np.mean(sigma_opp_succ**2))

    sigma_comp_succ = post['sigma_comp_succ'].values.flatten()
    var_comp_succ = float(np.mean(sigma_comp_succ**2))

    sigma_team_succ = post['sigma_team_succ'].values.flatten() if 'sigma_team_succ' in post else np.array([0.0])
    var_team_succ = float(np.mean(sigma_team_succ**2))

    X_global = X_scaled[:, global_mask]
    X_pos = X_scaled[:, pos_specific_mask]
    n_samples_succ = beta_global_succ.shape[0]
    global_contrib = np.dot(X_global, beta_global_succ.T)  # (n_obs, n_samples)
    pos_contrib = np.zeros_like(global_contrib)
    for s in range(n_samples_succ):
        pos_beta_at_pos = beta_pos_succ[s, pos_idx]  # (n_obs, n_pos_specific)
        pos_contrib[:, s] = np.sum(pos_beta_at_pos * X_pos, axis=1)
    linear_pred_succ = global_contrib + pos_contrib
    var_features_succ = float(np.mean(np.var(linear_pred_succ, axis=0)))

    total_var_succ = var_theta_succ + var_opp_succ + var_comp_succ + var_team_succ + var_features_succ

    # Value retention model
    sigma_theta_val = theta_chol_flat[:, 2]
    var_theta_val = float(np.mean(sigma_theta_val**2))

    sigma_opp_val = post['sigma_opp_val'].values.flatten()
    var_opp_val = float(np.mean(sigma_opp_val**2))

    sigma_comp_val = post['sigma_comp_val'].values.flatten()
    var_comp_val = float(np.mean(sigma_comp_val**2))

    sigma_team_val = post['sigma_team_val'].values.flatten() if 'sigma_team_val' in post else np.array([0.0])
    var_team_val = float(np.mean(sigma_team_val**2))

    n_samples_val = beta_global_val.shape[0]
    X_val_mask = df["success"].values.astype(bool)
    X_global_val = X_global[X_val_mask]
    X_pos_val = X_pos[X_val_mask]
    pos_idx_val = pos_idx[X_val_mask]
    global_contrib_val = np.dot(X_global_val, beta_global_val.T)
    pos_contrib_val = np.zeros_like(global_contrib_val)
    for s in range(n_samples_val):
        pos_beta_at_pos = beta_pos_val[s, pos_idx_val]
        pos_contrib_val[:, s] = np.sum(pos_beta_at_pos * X_pos_val, axis=1)
    linear_pred_val = global_contrib_val + pos_contrib_val
    var_features_val = float(np.mean(np.var(linear_pred_val, axis=0)))

    total_var_val = var_theta_val + var_opp_val + var_comp_val + var_team_val + var_features_val

    var_df = pd.DataFrame({
        'Component': [
            'Player Skill (Turnover)', 'Team Quality (Turnover)', 'Opp Quality (Turnover)',
            'Competition (Turnover)', 'Spatial Features (Turnover)',
            'Player Skill (Value)', 'Team Quality (Value)', 'Opp Quality (Value)',
            'Competition (Value)', 'Spatial Features (Value)',
        ],
        'Variance': [
            var_theta_succ, var_team_succ, var_opp_succ, var_comp_succ, var_features_succ,
            var_theta_val,  var_team_val,  var_opp_val,  var_comp_val,  var_features_val,
        ],
        'Proportion': [
            var_theta_succ/total_var_succ, var_team_succ/total_var_succ,
            var_opp_succ/total_var_succ, var_comp_succ/total_var_succ,
            var_features_succ/total_var_succ,
            var_theta_val/total_var_val, var_team_val/total_var_val,
            var_opp_val/total_var_val, var_comp_val/total_var_val,
            var_features_val/total_var_val,
        ]
    })
    var_df.to_csv(TABLES_DIR / "variance_decomposition.csv", index=False)

    # ── 3. Population marginal effects ────────────────────────────────────────
    # (Defined before SNR because both share alpha_succ/gamma_pos variables)
    logger.info("3. Population Marginal Effects")
    alpha_succ_samples = post['alpha_succ'].values.flatten()
    alpha_val_samples = post['alpha_val'].values.flatten()
    gamma_pos_succ = post['gamma_pos_succ'].values.reshape(-1, len(pos_mapping))
    gamma_pos_val = post['gamma_pos_val'].values.reshape(-1, len(pos_mapping))

    mid_pos_code = next((code for code, name in pos_mapping.items() if name == 'Midfielder'), 0)

    def _feat_contrib(vec: np.ndarray, beta_g: np.ndarray, beta_p: np.ndarray, p_code: int) -> np.ndarray:
        """Feature contribution: global + position-specific for a given position."""
        return np.dot(beta_g, vec[global_mask]) + np.dot(beta_p[:, p_code], vec[pos_specific_mask])

    # ── 2b. Signal-to-Noise Ratio ─────────────────────────────────────────────
    theta_succ_samples = post['theta_succ'].values.reshape(-1, len(mappings['player']))
    y_success_labels = df["success"].values.astype(int)
    logit_succ_mean = (
        alpha_succ_samples[:, np.newaxis]
        + np.column_stack([
            _feat_contrib(X_scaled[i], beta_global_succ, beta_pos_succ, pos_idx[i])
            for i in range(len(X_scaled))
        ])
        + gamma_pos_succ.mean(axis=0)[None, :]
    )
    p_post_mean = expit(logit_succ_mean).mean(axis=0)
    theta_succ_mean_post = theta_succ_samples.mean(axis=0)
    var_player_signal = float(np.var(theta_succ_mean_post))
    residual_var = float(np.var(y_success_labels - p_post_mean))
    snr_succ = var_player_signal / residual_var if residual_var > 0 else 0.0
    logger.info("  Ball Security SNR (signal/residual): %.4f", snr_succ)
    pd.DataFrame({"Metric": ["Ball Security SNR"], "Value": [snr_succ]}).to_csv(
        TABLES_DIR / "snr_decomposition.csv", index=False
    )

    def compute_marginal(feat_idx: int, values_range: np.ndarray) -> pd.DataFrame:
        """Marginalise over all posterior samples at each value of a single feature."""
        results = []
        scenario_vec = np.zeros(len(feature_names))
        for val in values_range:
            if global_mask[feat_idx]:
                std_val = (val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
            else:
                std_val = 0.0  # position-specific features handled below
            scenario_vec[feat_idx] = std_val

            feat_contrib = _feat_contrib(scenario_vec, beta_global_succ, beta_pos_succ, mid_pos_code)
            logit_succ = alpha_succ_samples + feat_contrib + gamma_pos_succ[:, mid_pos_code]
            feat_contrib_val = _feat_contrib(scenario_vec, beta_global_val, beta_pos_val, mid_pos_code)
            logit_val = alpha_val_samples + feat_contrib_val + gamma_pos_val[:, mid_pos_code]

            probs_succ = expit(logit_succ)
            ev_val = expit(logit_val) * max_value + min_value
            total_ev = probs_succ * ev_val

            results.append({
                'value': val,
                'mean_p': np.mean(total_ev),
                'hdi_5%': np.percentile(total_ev, 5),
                'hdi_95%': np.percentile(total_ev, 95)
            })
        return pd.DataFrame(results)

    # Grid capped at tight_pressure_radius; extrapolating beyond is
    # out-of-distribution because the builder filters to ≤ 5 yards.
    # Dist_nearest_opp is replaced by spline columns; marginal effects
    # for distance are computed via the spline basis if available,
    # otherwise fall back to the raw feature. coverage_arc is a base feature.
    dist_spline_cols = [c for c in feature_names if c.startswith('dist_nearest_opp_spline_')]
    if dist_spline_cols and spline_transformers:
        spline_tr = spline_transformers.get('dist_nearest_opp')
        if spline_tr is not None:
            grid = np.linspace(0.5, SPATIAL_CONFIG['tight_pressure_radius'], 50)
            dist_vals = grid.reshape(-1, 1)
            bases = spline_tr.transform(dist_vals)
            results = []
            for j, val in enumerate(grid):
                vec = np.zeros(len(feature_names))
                for k in range(bases.shape[1]):
                    vec[feature_names.index(dist_spline_cols[k])] = bases[j, k]
                # Spline columns are position-specific (root = dist_nearest_opp)
                # so _feat_contrib uses the position-specific beta
                feat_contrib = _feat_contrib(vec, beta_global_succ, beta_pos_succ, mid_pos_code)
                logit_succ = alpha_succ_samples + feat_contrib + gamma_pos_succ[:, mid_pos_code]
                feat_contrib_val = _feat_contrib(vec, beta_global_val, beta_pos_val, mid_pos_code)
                logit_val = alpha_val_samples + feat_contrib_val + gamma_pos_val[:, mid_pos_code]
                probs_succ = expit(logit_succ)
                ev_val = expit(logit_val) * max_value + min_value
                total_ev = probs_succ * ev_val
                results.append({
                    'value': val,
                    'mean_p': np.mean(total_ev),
                    'hdi_5%': np.percentile(total_ev, 5),
                    'hdi_95%': np.percentile(total_ev, 95)
                })
            pd.DataFrame(results).to_csv(TABLES_DIR / "marginal_dist.csv", index=False)
    elif 'dist_nearest_opp' in feature_names:
        # Fallback: raw feature (no spline expansion)
        idx = feature_names.index('dist_nearest_opp')
        grid = np.linspace(0.5, SPATIAL_CONFIG['tight_pressure_radius'], 50)
        df_marginal = compute_marginal(idx, grid)
        df_marginal.to_csv(TABLES_DIR / "marginal_dist.csv", index=False)

    if 'coverage_arc' in feature_names:
        idx = feature_names.index('coverage_arc')
        grid = np.linspace(0.0, np.pi, 50)
        df_marginal = compute_marginal(idx, grid)
        df_marginal.to_csv(TABLES_DIR / "marginal_arc.csv", index=False)

    # ── 4. ICE curves ─────────────────────────────────────────────────────────
    logger.info("4. ICE Curves for Top/Bottom Players")
    lb_path = TABLES_DIR / f"prs_leaderboard_{holdout}.csv"
    if not lb_path.exists():
        lb_path = TABLES_DIR / "prs_leaderboard.csv"
    if lb_path.exists():
        lb_df = pd.read_csv(lb_path)
        top_3 = lb_df.head(3)
        bottom_3 = lb_df.tail(3)

        theta_succ_samples = post['theta_succ'].values.reshape(-1, len(mappings['player']))
        theta_val_samples = post['theta_val'].values.reshape(-1, len(mappings['player']))

        selected_players: list[dict[str, Any]] = []
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

        dist_spline_cols = [c for c in feature_names if c.startswith('dist_nearest_opp_spline_')]
        if dist_spline_cols and spline_transformers:
            dist_tr = spline_transformers.get('dist_nearest_opp')
            if dist_tr is not None:
                max_dist = SPATIAL_CONFIG['tight_pressure_radius']
                dist_grid = np.linspace(0.5, max_dist, 50)

                ice_results: list[dict[str, Any]] = []
                for player in selected_players:
                    p_theta_succ = theta_succ_samples[:, player['idx']]
                    p_theta_val = theta_val_samples[:, player['idx']]
                    pc = player['pos_code']
                    p_pos_succ = gamma_pos_succ[:, pc]
                    p_pos_val = gamma_pos_val[:, pc]

                    dist_values = dist_grid.reshape(-1, 1)
                    spline_bases = dist_tr.transform(dist_values)

                    for j, dist_val in enumerate(dist_grid):
                        scenario_vec = np.zeros(len(feature_names))
                        for k in range(spline_bases.shape[1]):
                            col_name = dist_spline_cols[k]
                            col_idx = feature_names.index(col_name)
                            scenario_vec[col_idx] = (spline_bases[j, k] - scaler.mean_[col_idx]) / scaler.scale_[col_idx]

                        feat_contrib = _feat_contrib(scenario_vec, beta_global_succ, beta_pos_succ, pc)
                        logit_succ = alpha_succ_samples + feat_contrib + p_pos_succ + p_theta_succ
                        feat_contrib_val = _feat_contrib(scenario_vec, beta_global_val, beta_pos_val, pc)
                        logit_val = alpha_val_samples + feat_contrib_val + p_pos_val + p_theta_val

                        total_ev = expit(logit_succ) * (expit(logit_val) * max_value + min_value)

                        ice_results.append({
                            'player_name': player['name'],
                            'rank_group': player['rank'],
                            'distance': dist_val,
                            'mean_p': np.mean(total_ev)
                        })

                ice_df = pd.DataFrame(ice_results)
                ice_df.to_csv(TABLES_DIR / "ice_curves.csv", index=False)
        elif 'dist_nearest_opp' in feature_names:
            # Fallback: raw feature (no spline expansion)
            feat_idx = feature_names.index('dist_nearest_opp')
            max_dist = SPATIAL_CONFIG['tight_pressure_radius']
            dist_grid = np.linspace(0.5, max_dist, 50)
            ice_results = []
            for player in selected_players:
                p_theta_succ = theta_succ_samples[:, player['idx']]
                p_theta_val = theta_val_samples[:, player['idx']]
                pc = player['pos_code']
                p_pos_succ = gamma_pos_succ[:, pc]
                p_pos_val = gamma_pos_val[:, pc]
                for dist_val in dist_grid:
                    std_val = (dist_val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
                    scenario_vec = np.zeros(len(feature_names))
                    scenario_vec[feat_idx] = std_val
                    feat_contrib = _feat_contrib(scenario_vec, beta_global_succ, beta_pos_succ, pc)
                    logit_succ = alpha_succ_samples + feat_contrib + p_pos_succ + p_theta_succ
                    feat_contrib_val = _feat_contrib(scenario_vec, beta_global_val, beta_pos_val, pc)
                    logit_val = alpha_val_samples + feat_contrib_val + p_pos_val + p_theta_val
                    total_ev = expit(logit_succ) * (expit(logit_val) * max_value + min_value)
                    ice_results.append({
                        'player_name': player['name'],
                        'rank_group': player['rank'],
                        'distance': dist_val,
                        'mean_p': np.mean(total_ev)
                    })
            pd.DataFrame(ice_results).to_csv(TABLES_DIR / "ice_curves.csv", index=False)

if __name__ == "__main__":
    run_interpretability_analysis()
