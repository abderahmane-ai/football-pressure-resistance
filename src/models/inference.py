"""Posterior inference: generate PRS leaderboards from fitted Hurdle model."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from config import (
    CROSS_VALIDATION_HOLDOUT,
    MIN_EVENTS_THRESHOLD,
    MODEL_TRACES_DIR,
    PROCESSED_DATA_DIR,
    SPATIAL_CONFIG,
    TABLES_DIR,
)
from src.data.validation import validate_model_dataset

logger = logging.getLogger(__name__)


def _require_paths(*paths: Path) -> None:
    """Raise with an actionable message listing every missing artifact."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required model artifact(s) missing — run the pipeline from data "
            "build through model fitting first.\n  Missing: "
            + "\n  Missing: ".join(missing)
        )


def run_posterior_analysis() -> None:
    """
    Generate player leaderboards with Ball Security and Value Retention scores.
    Uses expit for numeric stability and appropriately handles the Hurdle model outputs.

    Design note — PRS aggregation:
    PRS = θ_succ + θ_val (additive on the logit scale). This gives equal
    weight to ball security and value retention. Both θ terms live on the
    logit scale with the same prior (Non-centered Normal), so addition is
    scale-consistent. An alternative would be multiplicative on the
    probability scale (p_succ × μ_val), but that conflates the two traits
    and makes the ranking sensitive to the base rate.
    """
    holdout = CROSS_VALIDATION_HOLDOUT
    trace_path: Path = MODEL_TRACES_DIR / f"pooled_trace_{holdout}.nc"
    mapping_path: Path = MODEL_TRACES_DIR / f"pooled_mappings_{holdout}.pkl"
    scaler_path: Path = MODEL_TRACES_DIR / f"pooled_scaler_{holdout}.pkl"
    dataset_path: Path = PROCESSED_DATA_DIR / f"all_pressure_dataset_{holdout}.parquet"
    _require_paths(trace_path, mapping_path, scaler_path, dataset_path)

    trace: az.InferenceData = az.from_netcdf(trace_path)

    with open(mapping_path, "rb") as f:
        mappings: dict[str, Any] = pickle.load(f)
    player_mapping: dict[int, Any] = mappings["player"]
    pos_mapping: dict[int, str] = mappings["position"]
    name_lookup: dict[Any, str] = mappings.get("name_lookup", {})
    pos_lookup: dict[Any, str] = mappings.get("position_lookup", {})

    with open(scaler_path, "rb") as f:
        scaler_data: dict[str, Any] = pickle.load(f)
        scaler = scaler_data["scaler"]
        feature_names: list[str] = scaler_data["features"]
        max_value: float = scaler_data["max_value"]

    df = pd.read_parquet(dataset_path)
    validate_model_dataset(df, feature_names, context="posterior analysis dataset")
    event_counts: dict[Any, int] = df["player_id"].value_counts().to_dict()

    # pyrefly: ignore [missing-attribute]
    post = trace.posterior

    # Success / Ball Security parameters
    alpha_succ: np.ndarray = post["alpha_succ"].values.flatten()
    beta_succ: np.ndarray = post["beta_succ"].values.reshape(-1, len(feature_names))
    theta_succ: np.ndarray = post["theta_succ"].values.reshape(-1, len(player_mapping))
    gamma_pos_succ: np.ndarray = post["gamma_pos_succ"].values.reshape(-1, len(pos_mapping))

    # Value Parameters
    alpha_val: np.ndarray = post["alpha_val"].values.flatten()
    beta_val: np.ndarray = post["beta_val"].values.reshape(-1, len(feature_names))
    theta_val: np.ndarray = post["theta_val"].values.reshape(-1, len(player_mapping))
    gamma_pos_val: np.ndarray = post["gamma_pos_val"].values.reshape(-1, len(pos_mapping))

    # Opponent and competition posterior effects
    delta_opp_succ: np.ndarray = post["delta_opp_succ"].values.reshape(-1, post["delta_opp_succ"].shape[-1])
    zeta_comp_succ: np.ndarray = post["zeta_comp_succ"].values.reshape(-1, post["zeta_comp_succ"].shape[-1])
    delta_opp_val: np.ndarray = post["delta_opp_val"].values.reshape(-1, post["delta_opp_val"].shape[-1])
    zeta_comp_val: np.ndarray = post["zeta_comp_val"].values.reshape(-1, post["zeta_comp_val"].shape[-1])

    # Marginalise by posterior group mean; shape: (n_samples,)
    mean_opp_succ: np.ndarray = delta_opp_succ.mean(axis=1)
    mean_comp_succ: np.ndarray = zeta_comp_succ.mean(axis=1)
    mean_opp_val: np.ndarray = delta_opp_val.mean(axis=1)
    mean_comp_val: np.ndarray = zeta_comp_val.mean(axis=1)

    tight_dist: float = SPATIAL_CONFIG["tight_pressure_radius"] * 0.3
    loose_dist: float = SPATIAL_CONFIG["tight_pressure_radius"] * 0.6

    scenarios: dict[str, dict[str, float]] = {
        "Front_Tight":   {"dist_nearest_opp": tight_dist, "angle_nearest_opp": 0.0, "coverage_arc": 1.5},
        "Front_Loose":   {"dist_nearest_opp": loose_dist, "angle_nearest_opp": 0.0, "coverage_arc": 0.8},
        "Lateral_Tight": {"dist_nearest_opp": tight_dist, "angle_nearest_opp": np.pi / 2, "coverage_arc": 1.5},
        "Lateral_Loose": {"dist_nearest_opp": loose_dist, "angle_nearest_opp": np.pi / 2, "coverage_arc": 0.8},
        "Back_Tight":    {"dist_nearest_opp": tight_dist, "angle_nearest_opp": np.pi, "coverage_arc": 1.5},
        "Back_Loose":    {"dist_nearest_opp": loose_dist, "angle_nearest_opp": np.pi, "coverage_arc": 0.8},
    }

    # Scenario feature vectors: unmentioned features are set to their
    # standardized mean (zero in scaled space), representing the "average
    # situation" baseline — an intentional design choice.
    scenario_vectors: dict[str, np.ndarray] = {}
    for name, s in scenarios.items():
        vec = np.zeros(len(feature_names))
        for f_name, val in s.items():
            if f_name in feature_names:
                f_idx = feature_names.index(f_name)
                if f_name not in ("opps_within_1yd", "opps_within_2yd", "opps_within_4yd", "has_progressive_option"):
                    vec[f_idx] = (val - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
                else:
                    # Binary/count features: use 0, not scaled mean
                    vec[f_idx] = (0 - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
        scenario_vectors[name] = vec

    mid_pos_code: int = next((code for code, name in pos_mapping.items() if name == "Midfielder"), 0)

    def get_predictions(
        scenario_vec: np.ndarray,
        player_idx: int | None = None,
        p_code: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_effect_succ = gamma_pos_succ[:, p_code] if p_code is not None else gamma_pos_succ[:, mid_pos_code]
        pos_effect_val = gamma_pos_val[:, p_code] if p_code is not None else gamma_pos_val[:, mid_pos_code]

        logit_succ = (alpha_succ + np.dot(beta_succ, scenario_vec) + pos_effect_succ
                      + mean_opp_succ + mean_comp_succ)
        logit_val = (alpha_val + np.dot(beta_val, scenario_vec) + pos_effect_val
                     + mean_opp_val + mean_comp_val)

        if player_idx is not None:
            logit_succ += theta_succ[:, player_idx]
            logit_val += theta_val[:, player_idx]

        prob_success: np.ndarray = expit(logit_succ)
        expected_value_retention: np.ndarray = expit(logit_val) * max_value

        # Overall EV under pressure
        total_ev: np.ndarray = prob_success * expected_value_retention
        return prob_success, expected_value_retention, total_ev

    pop_baselines: dict[str, float] = {}
    for name, vec in scenario_vectors.items():
        _, _, t_ev = get_predictions(vec)
        pop_baselines[name] = float(t_ev.mean())

    leaderboard: list[dict[str, Any]] = []
    for idx_num, player_id in player_mapping.items():
        try:
            n_events: int = event_counts.get(player_id, 0)
            if n_events < MIN_EVENTS_THRESHOLD:
                continue

            player_name: str = name_lookup.get(player_id, f"ID: {player_id}")
            position_group: str = pos_lookup.get(player_id, "Midfielder")

            if position_group not in pos_mapping.values():
                logger.warning(
                    "Player %s (%s) has position group '%s' which is not in model categories %s. "
                    "Defaulting to 'Midfielder'.",
                    player_id, player_name, position_group, list(pos_mapping.values())
                )

            player_pos_code: int = next(
                (code for code, name in pos_mapping.items() if name == position_group),
                mid_pos_code,
            )

            best_scenario = "N/A"
            max_advantage = -np.inf

            for s_name, s_vec in scenario_vectors.items():
                _, _, t_ev = get_predictions(s_vec, idx_num, player_pos_code)
                advantage = float(t_ev.mean()) - pop_baselines[s_name]
                if advantage > max_advantage:
                    max_advantage = advantage
                    best_scenario = s_name

            ball_security_samples: np.ndarray = theta_succ[:, idx_num]
            value_retention_samples: np.ndarray = theta_val[:, idx_num]
            prs_samples: np.ndarray = ball_security_samples + value_retention_samples

            # Use ArviZ HDI for proper Highest Density Interval
            # (not equal-tailed percentile interval)
            hdi_bounds = az.hdi(prs_samples, hdi_prob=0.90)

            leaderboard.append({
                "player_id": player_id,
                "player_name": player_name,
                "position_group": position_group,
                "mean_Ball_Security_Score": float(ball_security_samples.mean()),
                "mean_Value_Retention_Score": float(value_retention_samples.mean()),
                "mean_PRS": float(prs_samples.mean()),
                "hdi_5%": float(hdi_bounds[0]),
                "hdi_95%": float(hdi_bounds[1]),
                "n_events": n_events,
                "best_under_scenario": best_scenario,
            })
        except Exception as e:
            logger.debug("Error processing player %s: %s", player_id, e)

    lb_df = pd.DataFrame(leaderboard).sort_values(by="mean_PRS", ascending=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = TABLES_DIR / f"prs_leaderboard_{holdout}.csv"
    lb_df.to_csv(out_path, index=False)
    logger.info("Saved leaderboard with %d players (n>=%d) to %s", len(lb_df), MIN_EVENTS_THRESHOLD, out_path)

    top_players = lb_df.head(10)
    logger.info("\n=== TOP 10 PRESSURE RESISTANCE SCORES ===")
    for _, p in top_players.iterrows():
        logger.info(
            "%s (%s): PRS=%.3f | BallSec=%.3f | ValueRet=%.3f",
            p["player_name"], p["position_group"],
            p["mean_PRS"], p["mean_Ball_Security_Score"], p["mean_Value_Retention_Score"],
        )


if __name__ == "__main__":
    run_posterior_analysis()
