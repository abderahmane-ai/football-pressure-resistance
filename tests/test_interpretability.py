"""Tests for interpretability analysis pipeline."""
import pickle

import arviz as az
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import CROSS_VALIDATION_HOLDOUT
from src.visualization import interpretability

FEATURES = ["dist_nearest_opp", "angle_nearest_opp", "coverage_arc", "xt_value"]


def _build_trace(path, n_global=1, n_pos_specific=3, n_pos=1):
    # Ensure all required posterior parameters are mock-sampled for interpretability.
    # All shapes must follow ArviZ convention: (n_chains, n_draws, *param_dims).
    # theta_chol: packed lower-triangular Cholesky factor [chol[0,0], chol[1,0], chol[1,1]]
    n_chains = 1
    n_draws = 1
    theta_chol_data = np.array([[0.5, 0.1, 0.4]])  # shape (1, 3): L00, L10, L11
    posterior = {
        "alpha_succ": np.zeros((n_chains, n_draws)),
        "beta_global_succ": np.zeros((n_chains, n_draws, n_global)),
        "beta_pos_succ": np.zeros((n_chains, n_draws, n_pos, n_pos_specific)),
        "theta_succ": np.array([[1.0, 0.2]]).reshape(n_chains, n_draws, 2),  # 2 players
        "gamma_pos_succ": np.zeros((n_chains, n_draws, n_pos)),
        "theta_chol": np.broadcast_to(theta_chol_data, (n_chains, n_draws, 3)).copy(),
        "sigma_opp_succ": np.zeros((n_chains, n_draws)) + 0.1,
        "sigma_comp_succ": np.zeros((n_chains, n_draws)) + 0.05,
        "sigma_team_succ": np.zeros((n_chains, n_draws)) + 0.03,
        "alpha_val": np.zeros((n_chains, n_draws)),
        "beta_global_val": np.zeros((n_chains, n_draws, n_global)),
        "beta_pos_val": np.zeros((n_chains, n_draws, n_pos, n_pos_specific)),
        "theta_val": np.array([[0.3, 0.1]]).reshape(n_chains, n_draws, 2),  # 2 players
        "gamma_pos_val": np.zeros((n_chains, n_draws, n_pos)),
        "sigma_opp_val": np.zeros((n_chains, n_draws)) + 0.15,
        "sigma_comp_val": np.zeros((n_chains, n_draws)) + 0.06,
        "sigma_team_val": np.zeros((n_chains, n_draws)) + 0.04,
        "delta_opp_succ": np.zeros((n_chains, n_draws, 1)),
        "zeta_comp_succ": np.zeros((n_chains, n_draws, 1)),
        "delta_opp_val": np.zeros((n_chains, n_draws, 1)),
        "zeta_comp_val": np.zeros((n_chains, n_draws, 1)),
    }
    az.from_dict(posterior=posterior).to_netcdf(path)


def _build_artifacts(tmp_path):
    traces_dir = tmp_path / "model_traces"
    tables_dir = tmp_path / "tables"
    processed_dir = tmp_path / "processed"
    traces_dir.mkdir()
    tables_dir.mkdir()
    processed_dir.mkdir()

    _build_trace(traces_dir / f"pooled_trace_{CROSS_VALIDATION_HOLDOUT}.nc")

    mappings = {
        "player": {0: "p1", 1: "p2"},
        "position": {0: "Midfielder"},
        "name_lookup": {"p1": "High Player", "p2": "Low Player"},
        "position_lookup": {"p1": "Midfielder", "p2": "Midfielder"},
    }
    with open(traces_dir / f"pooled_mappings_{CROSS_VALIDATION_HOLDOUT}.pkl", "wb") as f:
        pickle.dump(mappings, f)

    from config import POSITION_SPECIFIC_FEATURES

    scaler = StandardScaler().fit(np.zeros((2, len(FEATURES))))
    with open(traces_dir / f"pooled_scaler_{CROSS_VALIDATION_HOLDOUT}.pkl", "wb") as f:
        pickle.dump({
            "scaler": scaler, "features": FEATURES, "max_value": 0.15,
            "spline_transformers": None,
            "position_specific_features": list(POSITION_SPECIFIC_FEATURES),
        }, f)

    # Write synthetic prs_leaderboard_{holdout}.csv
    leaderboard_df = pd.DataFrame([
        {"player_id": "p1", "player_name": "High Player", "position_group": "Midfielder", "mean_PRS": 1.3},
        {"player_id": "p2", "player_name": "Low Player", "position_group": "Midfielder", "mean_PRS": 0.3},
    ])
    leaderboard_df.to_csv(tables_dir / f"prs_leaderboard_{CROSS_VALIDATION_HOLDOUT}.csv", index=False)

    rows = []
    for player_id, name in [("p1", "High Player"), ("p2", "Low Player")]:
        row = {
            "competition": "Euro_2024",
            "match_id": 1,
            "pressure_event_id": f"press_{player_id}",
            "ball_carrier_event_id": f"carry_{player_id}",
            "player_id": player_id,
            "player_name": name,
            "position_group": "Midfielder",
            "team_id": 1,
            "opponent_team_id": 2,
            "success": 1.0,
            "value_preserved": 0.05,
            "dist_nearest_opp": 1.0,
            "angle_nearest_opp": 0.0,
            "coverage_arc": 0.5,
            "xt_value": 0.01,
        }
        rows.append(row)
    pd.DataFrame(rows).to_parquet(
        processed_dir / f"all_pressure_dataset_{CROSS_VALIDATION_HOLDOUT}.parquet"
    )
    return traces_dir, tables_dir, processed_dir


def test_run_interpretability_analysis_outputs(tmp_path, monkeypatch):
    traces_dir, tables_dir, processed_dir = _build_artifacts(tmp_path)
    monkeypatch.setattr(interpretability, "MODEL_TRACES_DIR", traces_dir)
    monkeypatch.setattr(interpretability, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(interpretability, "PROCESSED_DATA_DIR", processed_dir)

    interpretability.run_interpretability_analysis()

    # Check that CSV tables were created
    assert (tables_dir / "feature_importance.csv").exists()
    assert (tables_dir / "variance_decomposition.csv").exists()
    assert (tables_dir / "marginal_dist.csv").exists()
    assert (tables_dir / "marginal_arc.csv").exists()
    assert (tables_dir / "ice_curves.csv").exists()
