import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import SplineTransformer

# Paths
ROOT_DIR           = Path(__file__).parent
DATA_DIR           = ROOT_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR         = ROOT_DIR / "outputs"
MODEL_TRACES_DIR   = OUTPUT_DIR / "model_traces"
FIGURES_DIR        = OUTPUT_DIR / "figures"
TABLES_DIR         = OUTPUT_DIR / "tables"

# Competitions with verified 360 frame data available
COMPETITIONS = {
    "Bundesliga_2024":  {"comp_id": 9, "season_id": 281},
    "World_Cup_2022":   {"comp_id": 43, "season_id": 106},
    "Euro_2024":        {"comp_id": 55, "season_id": 282},
    "Euro_2020":        {"comp_id": 55, "season_id": 43},
    "MLS_2023":         {"comp_id": 44, "season_id": 107},
}

CROSS_VALIDATION_HOLDOUT = os.environ.get("PRS_HOLDOUT", "Euro_2020")

# Model Settings
MODEL_SETTINGS = {
    "draws": 2000,
    "tune": 1500,
    "chains": 4,
    "random_seed": 42,
    "target_accept": 0.9,
    "nuts_sampler": "numpyro",
}

MIN_EVENTS_THRESHOLD = 20  # Minimum pressure events for leaderboard inclusion

# Base features (before spline expansion)
MODEL_FEATURE_COLUMNS_BASE = [
    "dist_2nd_nearest_opp",
    "opps_within_1yd",
    "opps_within_2yd",
    "opps_within_4yd",
    "angle_nearest_opp",
    "coverage_arc",
    "voronoi_area",
    "n_free_teammates",
    "max_free_triangle_area",
    "dist_nearest_free_teammate",
    "angle_nearest_free_teammate",
    "pitch_control",
    "opp_density_5yd",
    "has_progressive_option",
    "xt_value",
    "game_state_diff",
    "minutes_elapsed",
    "match_period",
    "counter_press",
    "pass_height_ground",
    "pass_height_low",
    "pass_height_high",
    "recent_pressures",
]

# Features that get position-group-specific slopes (hierarchical β_pos[p] ~ N(β, σ)).
# These are spatial-geometry features whose effect on success/value differs by
# position — a CB facing coverage_arc=120° experiences it differently than a W.
POSITION_SPECIFIC_FEATURES = [
    "dist_nearest_opp",
    "dist_2nd_nearest_opp",
    "opps_within_2yd",
    "opps_within_4yd",
    "angle_nearest_opp",
    "coverage_arc",
    "voronoi_area",
    "n_free_teammates",
    "max_free_triangle_area",
    "dist_nearest_free_teammate",
    "angle_nearest_free_teammate",
    "pitch_control",
    "has_progressive_option",
]

# Features that get B-spline expansion (non-linear spatial effects)
SPLINE_FEATURES = ["bc_x", "bc_y", "dist_nearest_opp"]

# Final model feature columns: base + spline expansions
# Computed dynamically from the actual SplineTransformer output to avoid
# hardcoded desync when n_knots or degree change.
def _get_spline_n_basis() -> int:
    dummy = SplineTransformer(n_knots=5, degree=3, include_bias=False)
    dummy.fit(np.array([[0.0], [1.0]]))
    return int(dummy.n_features_out_)

_SPLINE_N_BASIS: int = _get_spline_n_basis()
MODEL_FEATURE_COLUMNS: list[str] = (
    MODEL_FEATURE_COLUMNS_BASE
    + [f"{feat}_spline_{i}" for feat in SPLINE_FEATURES for i in range(_SPLINE_N_BASIS)]
)

# Domain Spatial Constants
SPATIAL_CONFIG = {
    "pitch_length": 120.0,
    "pitch_width": 80.0,
    "goal_x": 120.0,
    "goal_y": 40.0,
    "tight_pressure_radius": 5.0,
    "coverage_arc_radius": 3.0,
    "xt_grid_cols": 12,
    "xt_grid_rows": 8,
    "pitch_control_sigma": 4.2,
    "pitch_control_max_radius": 15.0,
    "player_width": 0.5,
    "pass_clearance_radius": 1.5,
    "clear_pass_distance": 2.0,
    "progressive_distance": 5.0,
    "carry_lookahead_events": 5,
}

# VAEP Configuration
VAEP_CONFIG = {
    "lookahead": 10,
    "model_dir": str(DATA_DIR / "vaep_models"),
    "test_size": 0.2,
    "random_seed": 42,
    "lgb_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
}
