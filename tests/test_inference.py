"""Synthetic posterior tests for leaderboard inference."""
import pickle

import arviz as az
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from config import CROSS_VALIDATION_HOLDOUT
from src.models import inference

FEATURES = ["dist_nearest_opp", "angle_nearest_opp", "coverage_arc"]


def _posterior_array(values):
    return np.asarray(values, dtype=float).reshape(1, 4)


def _build_trace(path):
    zeros_beta = np.zeros((1, 4, len(FEATURES)))
    posterior = {
        "alpha_succ": _posterior_array([0, 0, 0, 0]),
        "beta_succ": zeros_beta,
        "theta_succ": np.array([[[1.0, 0.2], [1.0, 0.2], [1.0, 0.2], [1.0, 0.2]]]),
        "gamma_pos_succ": np.zeros((1, 4, 1)),
        "alpha_val": _posterior_array([0, 0, 0, 0]),
        "beta_val": zeros_beta,
        "theta_val": np.array([[[0.3, 0.1], [0.3, 0.1], [0.3, 0.1], [0.3, 0.1]]]),
        "gamma_pos_val": np.zeros((1, 4, 1)),
        "delta_opp_succ": np.zeros((1, 4, 1)),
        "zeta_comp_succ": np.zeros((1, 4, 1)),
        "delta_opp_val": np.zeros((1, 4, 1)),
        "zeta_comp_val": np.zeros((1, 4, 1)),
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

    scaler = StandardScaler().fit(np.zeros((2, len(FEATURES))))
    with open(traces_dir / f"pooled_scaler_{CROSS_VALIDATION_HOLDOUT}.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "features": FEATURES, "max_value": 0.15}, f)

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
        }
        rows.append(row)
    pd.DataFrame(rows).to_parquet(
        processed_dir / f"all_pressure_dataset_{CROSS_VALIDATION_HOLDOUT}.parquet"
    )
    return traces_dir, tables_dir, processed_dir


def test_require_paths_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required model artifact"):
        inference._require_paths(tmp_path / "missing.nc")


def test_run_posterior_analysis_writes_sorted_leaderboard(tmp_path, monkeypatch):
    traces_dir, tables_dir, processed_dir = _build_artifacts(tmp_path)
    monkeypatch.setattr(inference, "MODEL_TRACES_DIR", traces_dir)
    monkeypatch.setattr(inference, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(inference, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(inference, "MIN_EVENTS_THRESHOLD", 1)

    inference.run_posterior_analysis()

    leaderboard = pd.read_csv(tables_dir / "prs_leaderboard.csv")
    assert leaderboard["player_id"].tolist() == ["p1", "p2"]
    assert leaderboard.loc[0, "mean_Ball_Security_Score"] == pytest.approx(1.0)
    assert leaderboard.loc[0, "mean_PRS"] == pytest.approx(1.3)
    assert "mean_Turnover_Risk" not in leaderboard.columns
