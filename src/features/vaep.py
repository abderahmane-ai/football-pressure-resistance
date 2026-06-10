"""VAEP (Valuing Actions by Estimating Probabilities) model.

Decroos et al. 2019 — models P(score) and P(concede) in next N actions.
Trained on all StatsBomb events, then applied to pressure events
to replace discrete Expected Threat (xT) with a value that accounts
for both offensive progression AND conceding risk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import SPATIAL_CONFIG, VAEP_CONFIG

logger = logging.getLogger(__name__)

# Conditional imports to allow basic testing and fallback without LightGBM
try:
    import joblib
    import lightgbm as lgb
    _HAS_ML_LIBS = True
except ImportError:
    _HAS_ML_LIBS = False
    joblib = None
    lgb = None

_VAEP_MODEL_DIR: Path = Path(cast(str, VAEP_CONFIG["model_dir"]))
_VAEP_MODELS: tuple[Any, Any] | None = None
_VAEP_SCORE_FILENAME: str = "vaep_score.pkl"
_VAEP_CONCEDE_FILENAME: str = "vaep_concede.pkl"


def _is_valid_loc(loc: Any) -> bool:
    if loc is not None and hasattr(loc, "__len__") and len(loc) >= 2:
        return True
    return False


def _extract_state_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from raw StatsBomb events.

    Returns a DataFrame with the same row count as *events* with columns:
    - loc_x, loc_y (normalised)
    - dist_to_goal, angle_to_goal
    - goal_diff, under_pressure
    - is_pass, is_dribble, is_shot, is_carry, is_duel (binary)
    - pass_ground, pass_low, pass_high (one-hot)
    """
    pitch_len = SPATIAL_CONFIG["pitch_length"]
    pitch_width = SPATIAL_CONFIG["pitch_width"]
    goal_x = SPATIAL_CONFIG["goal_x"]
    goal_y = SPATIAL_CONFIG["goal_y"]

    locs = events["location"].values
    ev_types = events["type"].values
    under_pressure_col = events.get("under_pressure", pd.Series([False] * len(events))).values
    goal_diff_col = events.get("goal_diff", pd.Series([0.0] * len(events))).values
    pass_infos = events.get("pass", [None] * len(events))
    pass_infos = pass_infos.values if isinstance(pass_infos, pd.Series) else pass_infos

    rows: list[dict[str, float]] = []
    n = len(events)

    for i in range(n):
        loc = locs[i]
        if not _is_valid_loc(loc):
            rows.append({})
            continue

        x, y = float(loc[0]), float(loc[1])
        dist_to_goal = float(np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2))
        angle_to_goal = float(np.arctan2(goal_y - y, goal_x - x))

        ev_type = ev_types[i] if not pd.isna(ev_types[i]) else ""
        ph_id = 0
        if ev_type == "Pass":
            pass_info = pass_infos[i]
            if isinstance(pass_info, dict):
                height_info = pass_info.get("height")
                if isinstance(height_info, dict):
                    ph_id = int(height_info.get("id", 0))

        rows.append({
            "loc_x": x / pitch_len,
            "loc_y": y / pitch_width,
            "dist_to_goal": dist_to_goal,
            "angle_to_goal": angle_to_goal,
            "goal_diff": float(goal_diff_col[i]),
            "under_pressure": 1.0 if under_pressure_col[i] else 0.0,
            "is_pass": 1.0 if ev_type == "Pass" else 0.0,
            "is_dribble": 1.0 if ev_type == "Dribble" else 0.0,
            "is_shot": 1.0 if ev_type == "Shot" else 0.0,
            "is_carry": 1.0 if ev_type == "Carry" else 0.0,
            "is_duel": 1.0 if ev_type == "Duel" else 0.0,
            "pass_ground": 1.0 if ph_id == 1 else 0.0,
            "pass_low": 1.0 if ph_id == 2 else 0.0,
            "pass_high": 1.0 if ph_id == 3 else 0.0,
            "_has_location": 1.0,
        })

    return pd.DataFrame(rows)


def _compute_labels(
    events: pd.DataFrame,
    lookahead: int = 10,
) -> pd.DataFrame:
    """For each event, label whether team scores or concedes within next *lookahead* actions.

    Returns DataFrame with ``scores_next`` and ``concedes_next`` binary columns,
    plus event id and match_id for merging.
    """
    if "index" in events.columns:
        events = events.sort_values(["match_id", "index"]).reset_index(drop=True)
    else:
        events = events.sort_values(["match_id", "timestamp"]).reset_index(drop=True)

    is_shot = events["type"] == "Shot"
    shot_outcome = events.get("shot_outcome")
    if shot_outcome is None:
        shot_outcome = events.get("shot_outcome_name")
    if shot_outcome is not None:
        is_goal_event = is_shot & (shot_outcome == "Goal")
    else:
        is_goal_event = pd.Series(False, index=events.index)
    is_own_goal = events["type"] == "Own Goal For"

    scores_next = np.zeros(len(events), dtype=np.int32)
    concedes_next = np.zeros(len(events), dtype=np.int32)

    for match_id in events["match_id"].unique():
        match_mask = events["match_id"] == match_id
        idx = np.where(match_mask)[0]
        teams = events.loc[match_mask, "team_id"].values
        goal_mask = is_goal_event.values[idx]
        own_mask = is_own_goal.values[idx]

        for j, gi in enumerate(idx):
            end = min(j + lookahead + 1, len(idx))
            for k in range(j + 1, end):
                if goal_mask[k]:
                    if teams[k] == teams[j]:
                        scores_next[gi] = 1
                    else:
                        concedes_next[gi] = 1
                    break
                if own_mask[k]:
                    if teams[k] != teams[j]:
                        scores_next[gi] = 1
                    else:
                        concedes_next[gi] = 1
                    break

    result = pd.DataFrame({
        "id": events["id"].values,
        "match_id": events["match_id"].values,
        "scores_next": scores_next,
        "concedes_next": concedes_next,
    })
    return result


def train_vaep_models(events: pd.DataFrame) -> tuple[Any, Any]:
    """Train LightGBM scoring and conceding classifiers on *events*.

    Two models are trained:
    - ``vaep_score`` : P(scores_next = 1 | state)
    - ``vaep_concede`` : P(concedes_next = 1 | state)

    Models are saved to ``VAEP_CONFIG['model_dir']``.
    Returns (score_model, concede_model).
    """
    if not _HAS_ML_LIBS or lgb is None or joblib is None:
        raise ImportError("lightgbm and joblib are required to train VAEP models.")
    _VAEP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    score_path = _VAEP_MODEL_DIR / _VAEP_SCORE_FILENAME
    concede_path = _VAEP_MODEL_DIR / _VAEP_CONCEDE_FILENAME

    logger.info("Extracting state features for VAEP training...")
    features = _extract_state_features(events)
    labels = _compute_labels(events, lookahead=cast(int, VAEP_CONFIG["lookahead"]))

    # Merge features and labels
    vaep_df = pd.concat([features, labels[["scores_next", "concedes_next"]]], axis=1)
    vaep_df = vaep_df[vaep_df["_has_location"] == 1.0].copy()
    feature_cols = [c for c in vaep_df.columns if c not in (
        "scores_next", "concedes_next", "id", "match_id", "_has_location"
    )]

    X = vaep_df[feature_cols].values
    y_score = vaep_df["scores_next"].values
    y_concede = vaep_df["concedes_next"].values

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_score, test_size=cast(float, VAEP_CONFIG["test_size"]),
        random_state=cast(int, VAEP_CONFIG["random_seed"]),
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_concede, test_size=cast(float, VAEP_CONFIG["test_size"]),
        random_state=cast(int, VAEP_CONFIG["random_seed"]),
    )

    lgb_params = cast(dict[str, Any], VAEP_CONFIG["lgb_params"])

    logger.info("Training VAEP score model (P(score))...")
    score_model = lgb.LGBMClassifier(**lgb_params, random_state=cast(int, VAEP_CONFIG["random_seed"]))
    score_model.fit(
        X_train_s, y_train_s,
        eval_set=[(X_test_s, y_test_s)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    logger.info("Training VAEP concede model (P(concede))...")
    concede_model = lgb.LGBMClassifier(**lgb_params, random_state=cast(int, VAEP_CONFIG["random_seed"]))
    concede_model.fit(
        X_train_c, y_train_c,
        eval_set=[(X_test_c, y_test_c)],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    joblib.dump(score_model, score_path)
    joblib.dump(concede_model, concede_path)
    logger.info("Saved VAEP models to %s", _VAEP_MODEL_DIR)

    return score_model, concede_model


def load_vaep_models() -> tuple[Any, Any] | None:
    """Load pre-trained VAEP models from disk. Returns None if not found.

    Models are cached at module level after first load to avoid repeated
    ``joblib.load`` deserialisation (called once per match during data building).
    """
    global _VAEP_MODELS
    if _VAEP_MODELS is not None:
        return _VAEP_MODELS
    if not _HAS_ML_LIBS or joblib is None:
        return None
    score_path = _VAEP_MODEL_DIR / _VAEP_SCORE_FILENAME
    concede_path = _VAEP_MODEL_DIR / _VAEP_CONCEDE_FILENAME
    if not score_path.exists() or not concede_path.exists():
        logger.warning(
            "VAEP models not found at %s. Run train_vaep_models() first.",
            _VAEP_MODEL_DIR,
        )
        return None
    score_model = joblib.load(score_path)
    concede_model = joblib.load(concede_path)
    _VAEP_MODELS = (score_model, concede_model)
    return _VAEP_MODELS


def compute_vaep(events: pd.DataFrame) -> np.ndarray | None:
    """Compute VAEP value for each event.

    VAEP = (P_score_after - P_score_before) - (P_concede_after - P_concede_before)

    This is the change in expected scoring probability minus the change
    in expected conceding probability, giving each action a net value
    that accounts for both offensive threat and defensive risk.

    Returns an array of VAEP values, same length as *events*.
    Returns None if VAEP models are not found.
    """
    models = load_vaep_models()
    if models is None:
        return None

    score_model, concede_model = models
    feature_cols = [
        "loc_x", "loc_y", "dist_to_goal", "angle_to_goal",
        "goal_diff", "under_pressure",
        "is_pass", "is_dribble", "is_shot", "is_carry", "is_duel",
        "pass_ground", "pass_low", "pass_high",
    ]

    features = _extract_state_features(events)
    has_loc = features["_has_location"].values == 1.0

    X = features[feature_cols].values

    # Baseline: uninformative state (centre spot, neutral)
    baseline_vec = np.zeros((1, len(feature_cols)))
    baseline_vec[0, 0] = 60.0 / SPATIAL_CONFIG["pitch_length"]  # loc_x
    baseline_vec[0, 1] = 40.0 / SPATIAL_CONFIG["pitch_width"]   # loc_y

    p_score_before = score_model.predict_proba(baseline_vec)[0, 1]
    p_concede_before = concede_model.predict_proba(baseline_vec)[0, 1]

    p_score_after = score_model.predict_proba(X)[:, 1]
    p_concede_after = concede_model.predict_proba(X)[:, 1]

    vaep = np.zeros(len(events))
    vaep[has_loc] = (p_score_after[has_loc] - p_score_before) - (p_concede_after[has_loc] - p_concede_before)
    vaep[~has_loc] = np.nan

    return vaep
