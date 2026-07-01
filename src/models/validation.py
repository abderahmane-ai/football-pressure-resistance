"""Cross-validation: correlate training PRS with holdout value residuals."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from config import (
    CROSS_VALIDATION_HOLDOUT,
    MIN_EVENTS_THRESHOLD,
    TABLES_DIR,
)
from src.models.posterior import load_posterior_context
from src.paths import ModelPaths

logger = logging.getLogger(__name__)


def run_cross_validation() -> None:
    """
    Principled cross-validation: Correlate training PRS with
    mean expected value residuals in the holdout dataset.
    Vectorized for performance.
    """
    logger.info("=== CROSS-VALIDATION: VALUE-RESIDUAL CORRELATION ===")

    ctx = load_posterior_context(dataset_kind="holdout")

    # Local aliases for the rest of the function body
    scaler = ctx.scaler
    feature_names = ctx.feature_names
    max_value = ctx.max_value
    min_value = ctx.min_value
    pos_mapping = ctx.pos_mapping

    # Marginalise opponent/competition effects using the prior mean (zero)
    # for holdout predictions.  The non-centered parameterisation means the
    # prior mean of each random effect is exactly zero, which is the
    # principled choice for out-of-sample data where the specific opponent
    # and competition are unseen during training.  (Previously, the
    # training-set posterior group mean was used, which would bias holdout
    # predictions if the training competitions were systematically
    # stronger/weaker than the holdout.)

    X_holdout: np.ndarray = ctx.df[feature_names].values
    X_holdout_scaled: np.ndarray = scaler.transform(X_holdout)

    rev_pos_mapping: dict[str, int] = {v: k for k, v in pos_mapping.items()}
    holdout_pos_codes: np.ndarray = (
        ctx.df["position_group"]
        .map(rev_pos_mapping)
        .fillna(rev_pos_mapping.get("CM", 0))
        .astype(int)
        .values
    )

    # Feature contributions: global + position-specific
    X_global = X_holdout_scaled[:, ctx.global_mask]
    X_pos = X_holdout_scaled[:, ctx.pos_specific_mask]
    global_contrib_succ = np.dot(ctx.beta_global_succ, X_global.T)
    global_contrib_val = np.dot(ctx.beta_global_val, X_global.T)

    # Position-specific: broadcast beta_pos over holdout position codes
    pos_contrib_succ = np.sum(
        ctx.beta_pos_succ[:, holdout_pos_codes, :] * X_pos[np.newaxis, :, :],
        axis=-1,
    )
    pos_contrib_val = np.sum(
        ctx.beta_pos_val[:, holdout_pos_codes, :] * X_pos[np.newaxis, :, :],
        axis=-1,
    )

    # Vectorised linear predictors (opponent/competition effects marginalised to zero)
    logit_succ_base: np.ndarray = ctx.alpha_succ[:, np.newaxis] + global_contrib_succ + pos_contrib_succ
    logit_val_base: np.ndarray = ctx.alpha_val[:, np.newaxis] + global_contrib_val + pos_contrib_val

    # Position effects (intercept)
    logit_succ_base += ctx.gamma_pos_succ[:, holdout_pos_codes]
    logit_val_base += ctx.gamma_pos_val[:, holdout_pos_codes]

    # No opponent/competition adjustments — prior mean is zero for unseen groups

    p_succ: np.ndarray = expit(logit_succ_base)
    mu_val: np.ndarray = expit(logit_val_base) * max_value + min_value

    # Predicted Expected Value
    predicted_ev: np.ndarray = (p_succ * mu_val).mean(axis=0)

    # Observed Expected Value (Success × Intended xT)
    observed_val: np.ndarray = ctx.df["success"].values * ctx.df["value_preserved"].values
    residuals: np.ndarray = observed_val - predicted_ev
    ctx.df["residual"] = residuals

    player_stats = ctx.df.groupby("player_id").agg({
        "residual": "mean",
        "value_preserved": "count",
        "player_name": "first",
    }).reset_index()
    player_stats = player_stats[player_stats["value_preserved"] >= MIN_EVENTS_THRESHOLD]

    out = ModelPaths(CROSS_VALIDATION_HOLDOUT)
    train_lb: pd.DataFrame = pd.read_csv(out.leaderboard)
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

    # Count player overlap: training vs holdout intersection
    n_train_players: int = len(train_lb)
    n_holdout_players: int = len(player_stats)
    n_overlap: int = len(merged)

    logger.info(
        "Stability Analysis: %d training players, %d holdout players, %d overlap",
        n_train_players, n_holdout_players, n_overlap,
    )
    logger.info("  Pearson Correlation: %.3f (p=%.4f)", pearson_corr, p_p)
    logger.info("  Spearman Correlation: %.3f (p=%.4f)", spearman_corr, p_s)

    y_true_bin: np.ndarray = ctx.df["success"].values.astype(int)
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
    cal_df.to_csv(out.calibration_curve, index=False)
    merged.to_csv(out.holdout_correlation, index=False)

    metrics_df = pd.DataFrame([{
        "n_overlap": n_overlap,
        "n_train_players": n_train_players,
        "n_holdout_players": n_holdout_players,
        "pearson": pearson_corr,
        "pearson_p": p_p,
        "auc": auc,
    }])
    metrics_df.to_csv(out.holdout_metrics, index=False)


if __name__ == "__main__":
    run_cross_validation()
