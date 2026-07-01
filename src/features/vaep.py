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
from src.common import (
    EVENT_TYPE_CARRY,
    EVENT_TYPE_DRIBBLE,
    EVENT_TYPE_DUEL,
    EVENT_TYPE_OWN_GOAL_FOR,
    EVENT_TYPE_PASS,
    EVENT_TYPE_SHOT,
    is_valid_loc,
)

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


def _vaep_model_paths() -> tuple[Path, Path]:
    """Return (score_path, concede_path) for the current holdout.

    VAEP models are holdout-specific so that each CV fold trains on all
    competitions EXCEPT its own holdout.  The holdout subdirectory mirrors
    the pattern used by ModelPaths for traces and scalers.
    """
    import os
    holdout = os.environ.get("PRS_HOLDOUT", "Euro_2020")
    base = _VAEP_MODEL_DIR / holdout
    return base / _VAEP_SCORE_FILENAME, base / _VAEP_CONCEDE_FILENAME


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
        if not is_valid_loc(loc):
            rows.append({})
            continue

        x, y = float(loc[0]), float(loc[1])
        dist_to_goal = float(np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2))
        angle_to_goal = float(np.arctan2(goal_y - y, goal_x - x))

        ev_type = ev_types[i] if not pd.isna(ev_types[i]) else ""
        ph_id = 0
        if ev_type == EVENT_TYPE_PASS:
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
            "is_pass": 1.0 if ev_type == EVENT_TYPE_PASS else 0.0,
            "is_dribble": 1.0 if ev_type == EVENT_TYPE_DRIBBLE else 0.0,
            "is_shot": 1.0 if ev_type == EVENT_TYPE_SHOT else 0.0,
            "is_carry": 1.0 if ev_type == EVENT_TYPE_CARRY else 0.0,
            "is_duel": 1.0 if ev_type == EVENT_TYPE_DUEL else 0.0,
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

    is_shot = events["type"] == EVENT_TYPE_SHOT
    shot_outcome = events.get("shot_outcome")
    if shot_outcome is None:
        shot_outcome = events.get("shot_outcome_name")
    if shot_outcome is not None:
        is_goal_event = is_shot & (shot_outcome == "Goal")
    else:
        is_goal_event = pd.Series(False, index=events.index)
    is_own_goal = events["type"] == EVENT_TYPE_OWN_GOAL_FOR

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
                    # Own Goal For team_id = beneficiary (same semantics as regular
                    # goal: teams[k] == teams[j] → our team scores)
                    if teams[k] != teams[j]:
                        concedes_next[gi] = 1
                    else:
                        scores_next[gi] = 1
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
    score_path, concede_path = _vaep_model_paths()
    score_path.parent.mkdir(parents=True, exist_ok=True)

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
    The cache is invalidated when the holdout changes (e.g. between CV folds).
    """
    global _VAEP_MODELS
    if _VAEP_MODELS is not None:
        return _VAEP_MODELS
    if not _HAS_ML_LIBS or joblib is None:
        return None
    score_path, concede_path = _vaep_model_paths()
    if not score_path.exists() or not concede_path.exists():
        logger.warning(
            "VAEP models not found at %s. Run train_vaep_models() first.",
            score_path.parent,
        )
        return None
    score_model = joblib.load(score_path)
    concede_model = joblib.load(concede_path)
    _VAEP_MODELS = (score_model, concede_model)
    return _VAEP_MODELS


def compute_vaep(events: pd.DataFrame) -> np.ndarray | None:
    """Compute VAEP value for each event.

    Canonical VAEP (Decroos et al. 2019):
    VAEP(action_i) = [P_score(S_i) − P_score(S_{i-1})] − [P_concede(S_i) − P_concede(S_{i-1})]

    where S_i is the state before action_i. The difference measures the
    change in scoring/conceding probability caused by the preceding action.

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

    # Sort chronologically per match so consecutive rows are consecutive actions.
    # Preserve original index so we can restore row order at the end.
    sort_col = "index" if "index" in events.columns else "timestamp"
    events_sorted = events.sort_values(["match_id", sort_col])

    features = _extract_state_features(events_sorted)
    has_loc = features["_has_location"].values == 1.0

    X = features[feature_cols]

    # Compute P(score|state) and P(concede|state) for every event state
    p_score = score_model.predict_proba(X)[:, 1]
    p_concede = concede_model.predict_proba(X)[:, 1]

    vaep_sorted = np.zeros(len(events_sorted))

    # Compute delta between consecutive events within each match
    for match_id in events_sorted["match_id"].unique():
        match_idx = np.where(events_sorted["match_id"].values == match_id)[0]

        for j, i in enumerate(match_idx):
            if not has_loc[i]:
                vaep_sorted[i] = np.nan
                continue

            if j == 0:
                vaep_sorted[i] = 0.0
            else:
                prev = match_idx[j - 1]
                vaep_sorted[i] = (p_score[i] - p_score[prev]) - (p_concede[i] - p_concede[prev])

    # Restore original row order using the preserved index
    return pd.Series(vaep_sorted, index=events_sorted.index).reindex(events.index).values
