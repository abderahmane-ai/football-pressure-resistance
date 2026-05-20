"""Cross-validation: correlate training PRS with holdout value residuals."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from config import (MIN_EVENTS_THRESHOLD, MODEL_TRACES_DIR, PROCESSED_DATA_DIR,
                    SPATIAL_CONFIG, TABLES_DIR)
from src.data.validation import validate_model_dataset

logger = logging.getLogger(__name__)


def _require_paths(*paths: Path) -> None:
    """Raise with an actionable message listing every missing artifact."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required validation artifact(s) missing — run data build, model "
            "fitting, and leaderboard generation first.\n  Missing: "
            + "\n  Missing: ".join(missing)
        )


def run_cross_validation() -> None:
    """
    Principled cross-validation: Correlate training PRS with
    mean expected value residuals in the holdout dataset.
    Vectorized for performance.
    """
    logger.info("=== CROSS-VALIDATION: VALUE-RESIDUAL CORRELATION ===")

    holdout_path: Path = PROCESSED_DATA_DIR / "holdout_pressure_dataset.parquet"
    trace_path: Path = MODEL_TRACES_DIR / "pooled_trace.nc"
    mapping_path: Path = MODEL_TRACES_DIR / "pooled_mappings.pkl"
    scaler_path: Path = MODEL_TRACES_DIR / "pooled_scaler.pkl"
    leaderboard_path: Path = TABLES_DIR / "prs_leaderboard.csv"
    _require_paths(holdout_path, trace_path, mapping_path, scaler_path, leaderboard_path)

    holdout_df: pd.DataFrame = pd.read_parquet(holdout_path)
    with open(scaler_path, "rb") as f:
        scaler_data: dict[str, Any] = pickle.load(f)
    feature_names: list[str] = scaler_data["features"]
    validate_model_dataset(holdout_df, feature_names, context="holdout pressure dataset")
    holdout_df = holdout_df[holdout_df["dist_nearest_opp"] <= SPATIAL_CONFIG["tight_pressure_radius"]]

    trace: az.InferenceData = az.from_netcdf(trace_path)
    with open(mapping_path, "rb") as f:
        mappings: dict[str, Any] = pickle.load(f)

    scaler = scaler_data["scaler"]
    max_value: float = scaler_data["max_value"]
    pos_mapping: dict[int, str] = mappings["position"]

    post = trace.posterior  # type: ignore[attr-defined]
    # Success params
    alpha_succ: np.ndarray = post["alpha_succ"].values.flatten()
    beta_succ: np.ndarray = post["beta_succ"].values.reshape(-1, len(feature_names))
    gamma_pos_succ: np.ndarray = post["gamma_pos_succ"].values.reshape(-1, len(pos_mapping))

    # Value params
    alpha_val: np.ndarray = post["alpha_val"].values.flatten()
    beta_val: np.ndarray = post["beta_val"].values.reshape(-1, len(feature_names))
    gamma_pos_val: np.ndarray = post["gamma_pos_val"].values.reshape(-1, len(pos_mapping))

    # Marginalise opp/comp by posterior group mean — consistent with leaderboard
    delta_opp_succ: np.ndarray = post["delta_opp_succ"].values.reshape(-1, post["delta_opp_succ"].shape[-1])
    zeta_comp_succ: np.ndarray = post["zeta_comp_succ"].values.reshape(-1, post["zeta_comp_succ"].shape[-1])
    delta_opp_val: np.ndarray = post["delta_opp_val"].values.reshape(-1, post["delta_opp_val"].shape[-1])
    zeta_comp_val: np.ndarray = post["zeta_comp_val"].values.reshape(-1, post["zeta_comp_val"].shape[-1])
    mean_opp_succ: np.ndarray = delta_opp_succ.mean(axis=1)
    mean_comp_succ: np.ndarray = zeta_comp_succ.mean(axis=1)
    mean_opp_val: np.ndarray = delta_opp_val.mean(axis=1)
    mean_comp_val: np.ndarray = zeta_comp_val.mean(axis=1)

    X_holdout: np.ndarray = holdout_df[feature_names].values
    X_holdout_scaled: np.ndarray = scaler.transform(X_holdout)

    rev_pos_mapping: dict[str, int] = {v: k for k, v in pos_mapping.items()}
    holdout_pos_codes: np.ndarray = (
        holdout_df["position_group"]
        .map(rev_pos_mapping)
        .fillna(rev_pos_mapping.get("Midfielder", 0))
        .astype(int)
        .values
    )

    # Vectorised linear predictors
    logit_succ_base: np.ndarray = alpha_succ[:, np.newaxis] + np.dot(beta_succ, X_holdout_scaled.T)
    logit_val_base: np.ndarray = alpha_val[:, np.newaxis] + np.dot(beta_val, X_holdout_scaled.T)

    # Position effects
    logit_succ_base += gamma_pos_succ[:, holdout_pos_codes]
    logit_val_base += gamma_pos_val[:, holdout_pos_codes]

    # Marginalised opp/comp effects
    logit_succ_base += mean_opp_succ[:, np.newaxis] + mean_comp_succ[:, np.newaxis]
    logit_val_base += mean_opp_val[:, np.newaxis] + mean_comp_val[:, np.newaxis]

    p_succ: np.ndarray = expit(logit_succ_base)
    mu_val: np.ndarray = expit(logit_val_base) * max_value

    # Predicted Expected Value
    predicted_ev: np.ndarray = (p_succ * mu_val).mean(axis=0)

    # Observed Expected Value (Success × Intended xT)
    observed_val: np.ndarray = holdout_df["success"].values * holdout_df["value_preserved"].values
    residuals: np.ndarray = observed_val - predicted_ev
    holdout_df["residual"] = residuals

    player_stats = holdout_df.groupby("player_id").agg({
        "residual": "mean",
        "value_preserved": "count",
        "player_name": "first",
    }).reset_index()
    player_stats = player_stats[player_stats["value_preserved"] >= MIN_EVENTS_THRESHOLD]

    train_lb: pd.DataFrame = pd.read_csv(leaderboard_path)
    merged: pd.DataFrame = train_lb.merge(player_stats, on="player_id", suffixes=("_train", "_holdout"))

    if len(merged) < 5:
        logger.warning(
            "Only %d overlapping players between training leaderboard and holdout "
            "(minimum 5 required for meaningful correlation). "
            "Consider lowering MIN_EVENTS_THRESHOLD or adding more data.",
            len(merged),
        )
        return

    pearson_corr, p_p = pearsonr(merged["mean_PRS"], merged["residual"])
    # Spearman computed for diagnostic logging only; not persisted to CSV
    spearman_corr, p_s = spearmanr(merged["mean_PRS"], merged["residual"])

    logger.info("Stability Analysis (n=%d overlapping players):", len(merged))
    logger.info("  Pearson Correlation: %.3f (p=%.4f)", pearson_corr, p_p)
    logger.info("  Spearman Correlation: %.3f (p=%.4f)", spearman_corr, p_s)

    y_true_bin: np.ndarray = holdout_df["success"].values.astype(int)
    y_pred_prob: np.ndarray = p_succ.mean(axis=0)
    auc: float
    if len(np.unique(y_true_bin)) == 2:
        auc = float(roc_auc_score(y_true_bin, y_pred_prob))
        logger.info("  Holdout AUC (Binary Success): %.3f", auc)
    else:
        auc = np.nan
        logger.warning(
            "Holdout AUC skipped: only one success class present in holdout data. "
            "This typically means the holdout set is too small or homogeneous."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES_DIR / "holdout_correlation_data.csv", index=False)

    metrics_df = pd.DataFrame([{
        "n_players": len(merged),
        "pearson": pearson_corr,
        "pearson_p": p_p,
        "auc": auc,
    }])
    metrics_df.to_csv(TABLES_DIR / "holdout_metrics.csv", index=False)


if __name__ == "__main__":
    run_cross_validation()
