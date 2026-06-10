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

from config import (
    CROSS_VALIDATION_HOLDOUT,
    MIN_EVENTS_THRESHOLD,
    MODEL_TRACES_DIR,
    PROCESSED_DATA_DIR,
    SPATIAL_CONFIG,
    TABLES_DIR,
)
from src.data.validation import validate_model_dataset
from src.features.spatial import expand_spline_features

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

    holdout = CROSS_VALIDATION_HOLDOUT
    holdout_path: Path = PROCESSED_DATA_DIR / f"holdout_pressure_dataset_{holdout}.parquet"
    trace_path: Path = MODEL_TRACES_DIR / f"pooled_trace_{holdout}.nc"
    mapping_path: Path = MODEL_TRACES_DIR / f"pooled_mappings_{holdout}.pkl"
    scaler_path: Path = MODEL_TRACES_DIR / f"pooled_scaler_{holdout}.pkl"
    leaderboard_path: Path = TABLES_DIR / f"prs_leaderboard_{holdout}.csv"
    _require_paths(holdout_path, trace_path, mapping_path, scaler_path, leaderboard_path)

    holdout_df: pd.DataFrame = pd.read_parquet(holdout_path)
    with open(scaler_path, "rb") as f:
        scaler_data: dict[str, Any] = pickle.load(f)
    feature_names: list[str] = scaler_data["features"]

    # Filter to tight pressure BEFORE spline expansion (raw dist_nearest_opp is needed)
    if "dist_nearest_opp" in holdout_df.columns:
        holdout_df = holdout_df[holdout_df["dist_nearest_opp"] <= SPATIAL_CONFIG["tight_pressure_radius"]]

    # Expand spline features using the fitted transformers saved with the scaler
    spline_transformers = scaler_data.get("spline_transformers")
    if spline_transformers:
        holdout_df = expand_spline_features(holdout_df, spline_transformers)

    validate_model_dataset(holdout_df, feature_names, context="holdout pressure dataset")

    trace: az.InferenceData = az.from_netcdf(trace_path)
    with open(mapping_path, "rb") as f:
        mappings: dict[str, Any] = pickle.load(f)

    scaler = scaler_data["scaler"]
    max_value: float = scaler_data["max_value"]
    min_value: float = scaler_data.get("min_value", 0.0)
    pos_mapping: dict[int, str] = mappings["position"]

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
    n_global = int(global_mask.sum())
    n_pos_specific = int(pos_specific_mask.sum())

    post = trace.posterior  # type: ignore[attr-defined]
    # Success params
    alpha_succ: np.ndarray = post["alpha_succ"].values.flatten()
    beta_global_succ: np.ndarray = post["beta_global_succ"].values.reshape(-1, n_global)
    beta_pos_succ: np.ndarray = post["beta_pos_succ"].values.reshape(-1, len(pos_mapping), n_pos_specific)
    gamma_pos_succ: np.ndarray = post["gamma_pos_succ"].values.reshape(-1, len(pos_mapping))

    # Value params
    alpha_val: np.ndarray = post["alpha_val"].values.flatten()
    beta_global_val: np.ndarray = post["beta_global_val"].values.reshape(-1, n_global)
    beta_pos_val: np.ndarray = post["beta_pos_val"].values.reshape(-1, len(pos_mapping), n_pos_specific)
    gamma_pos_val: np.ndarray = post["gamma_pos_val"].values.reshape(-1, len(pos_mapping))

    # Marginalise opponent/competition effects using the prior mean (zero)
    # for holdout predictions.  The non-centered parameterisation means the
    # prior mean of each random effect is exactly zero, which is the
    # principled choice for out-of-sample data where the specific opponent
    # and competition are unseen during training.  (Previously, the
    # training-set posterior group mean was used, which would bias holdout
    # predictions if the training competitions were systematically
    # stronger/weaker than the holdout.)

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

    # Feature contributions: global + position-specific
    X_global = X_holdout_scaled[:, global_mask]
    X_pos = X_holdout_scaled[:, pos_specific_mask]
    global_contrib_succ = np.dot(beta_global_succ, X_global.T)
    global_contrib_val = np.dot(beta_global_val, X_global.T)

    # Position-specific: broadcast beta_pos over holdout position codes
    n_samples = beta_global_succ.shape[0]
    pos_contrib_succ = np.sum(
        beta_pos_succ[:, holdout_pos_codes, :] * X_pos[np.newaxis, :, :],
        axis=-1,
    )
    pos_contrib_val = np.sum(
        beta_pos_val[:, holdout_pos_codes, :] * X_pos[np.newaxis, :, :],
        axis=-1,
    )

    # Vectorised linear predictors (opponent/competition effects marginalised to zero)
    logit_succ_base: np.ndarray = alpha_succ[:, np.newaxis] + global_contrib_succ + pos_contrib_succ
    logit_val_base: np.ndarray = alpha_val[:, np.newaxis] + global_contrib_val + pos_contrib_val

    # Position effects (intercept)
    logit_succ_base += gamma_pos_succ[:, holdout_pos_codes]
    logit_val_base += gamma_pos_val[:, holdout_pos_codes]

    # No opponent/competition adjustments — prior mean is zero for unseen groups

    p_succ: np.ndarray = expit(logit_succ_base)
    mu_val: np.ndarray = expit(logit_val_base) * max_value + min_value

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

    # Calibration — Expected Calibration Error (ECE)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true_bin, y_pred_prob, n_bins=10)
    ece: float = float(np.mean(np.abs(prob_true - prob_pred)))
    logger.info("  Expected Calibration Error (ECE): %.4f", ece)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cal_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
    cal_df.to_csv(TABLES_DIR / f"calibration_curve_{holdout}.csv", index=False)
    merged.to_csv(TABLES_DIR / f"holdout_correlation_data_{holdout}.csv", index=False)

    metrics_df = pd.DataFrame([{
        "n_players": len(merged),
        "pearson": pearson_corr,
        "pearson_p": p_p,
        "auc": auc,
    }])
    metrics_df.to_csv(TABLES_DIR / f"holdout_metrics_{holdout}.csv", index=False)


if __name__ == "__main__":
    run_cross_validation()
