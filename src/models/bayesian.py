"""Bayesian Hierarchical Beta Hurdle Model for Pressure Resistance Score."""
from __future__ import annotations

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

from config import (
    CROSS_VALIDATION_HOLDOUT,
    MODEL_FEATURE_COLUMNS,
    MODEL_SETTINGS,
    MODEL_TRACES_DIR,
    POSITION_SPECIFIC_FEATURES,
)
from src.data.validation import DataValidationError, validate_model_dataset
from src.features.spatial import (
    expand_spline_features,
    fit_spline_transformers,
    is_position_specific,
)
from src.paths import ModelPaths

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Platforms accepted by the JAX/GPU verification.
# - "gpu" / "cuda": NVIDIA GPUs via CUDA
# - "metal": Apple Silicon via jax-metal
_ACCEPTED_GPU_PLATFORMS: frozenset[str] = frozenset({"gpu", "cuda", "metal"})


# ── JAX / device verification ────────────────────────────────────────────────

def _verify_jax_device() -> str:
    """Verify that JAX can see a hardware accelerator.

    Accepts NVIDIA CUDA GPUs and Apple Silicon Metal.  Falls back to CPU
    only when the ``PRS_ALLOW_CPU`` environment variable is explicitly set
    to ``1`` — this prevents silent multi-hour runs on cloud instances
    where the GPU driver failed to load.
    """
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError(
            "JAX is not installed. NumPyro NUTS requires JAX. "
            "Install with: pip install -r requirements-accelerated.txt"
        ) from exc

    devices = jax.devices()
    platform = devices[0].platform.lower() if devices else "unknown"

    if platform in _ACCEPTED_GPU_PLATFORMS:
        logger.info("JAX is using accelerator: %s (%s)", devices[0], platform)
        return platform

    # CPU fallback — only when explicitly opted-in
    if os.environ.get("PRS_ALLOW_CPU", "") == "1":
        logger.warning(
            "JAX is running on platform '%s' (CPU). GPU/Metal not detected. "
            "Continuing because PRS_ALLOW_CPU=1 is set. Expect SLOW sampling.",
            platform,
        )
        return platform

    raise RuntimeError(
        f"JAX is running on platform '{platform}', not a supported accelerator "
        f"({', '.join(sorted(_ACCEPTED_GPU_PLATFORMS))}). "
        f"Devices: {devices}. "
        "The CPU fallback is disabled to prevent multi-hour runs. "
        "Set PRS_ALLOW_CPU=1 to override, or ensure a GPU/Metal device is visible "
        "(e.g. JAX_PLATFORMS=cuda, pip install jax-metal)."
    )


# ── Intermediate caching ─────────────────────────────────────────────────────

def _load_cached_scaler(path: Path) -> dict[str, Any] | None:
    """Load a previously-saved scaler if it exists and matches the current feature set."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if list(data.get("features", [])) != list(MODEL_FEATURE_COLUMNS):
            logger.info(
                "Cached scaler feature set differs from MODEL_FEATURE_COLUMNS — "
                "refitting scaler."
            )
            return None
        return data  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning("Could not load cached scaler from %s: %s", path, exc)
        return None


def _save_scaler(
    path: Path,
    scaler: StandardScaler,
    features: list[str],
    max_value: float,
    min_value: float,
    epsilon: float,
    spline_transformers: dict[str, Any] | None = None,
    position_specific_features: list[str] | None = None,
) -> None:
    """Persist scaler, feature list, and scaling constants."""
    with open(path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "features": features,
            "max_value": max_value,
            "min_value": min_value,
            "epsilon": epsilon,
            "spline_transformers": spline_transformers,
            "position_specific_features": position_specific_features or POSITION_SPECIFIC_FEATURES,
        }, f)


def _is_position_specific(feature_name: str) -> bool:
    """Check whether *feature_name* should have position-group-specific slopes."""
    return is_position_specific(feature_name, list(POSITION_SPECIFIC_FEATURES))


def _save_mappings(
    path: Path,
    player_mapping: dict[int, Any],
    comp_mapping: dict[int, str],
    opp_team_mapping: dict[int, Any],
    team_mapping: dict[int, Any],
    pos_mapping: dict[int, str],
    name_lookup: dict[Any, str],
    pos_lookup: dict[Any, str],
) -> None:
    """Persist categorical index ↔ label mappings for inference/validation."""
    with open(path, "wb") as f:
        pickle.dump({
            "player": player_mapping,
            "competition": comp_mapping,
            "team": opp_team_mapping,
            "att_team": team_mapping,
            "position": pos_mapping,
            "name_lookup": name_lookup,
            "position_lookup": pos_lookup,
        }, f)


# ── Dataset preparation ──────────────────────────────────────────────────────

def prepare_model_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean the processed pressure dataset for model fitting."""
    validate_model_dataset(df, MODEL_FEATURE_COLUMNS)
    required_columns: list[str] = [
        *MODEL_FEATURE_COLUMNS,
        "success",
        "value_preserved",
        "player_id",
        "player_name",
        "competition",
        "opponent_team_id",
        "position_group",
    ]
    model_df = df.dropna(subset=required_columns).copy()
    if model_df.empty:
        raise DataValidationError(
            "Processed model dataset has no complete rows after dropping nulls. "
            f"Check that all required columns ({', '.join(required_columns)}) "
            "are populated in the processed parquet file."
        )
    if model_df["success"].sum() == 0:
        raise DataValidationError(
            "Processed model dataset has zero successful actions. "
            "The Beta value model requires at least some successes to fit."
        )
    return model_df, list(MODEL_FEATURE_COLUMNS)


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_pooled_model() -> az.InferenceData | None:
    """
    Fit a hurdle model separating Ball Security from Value Retention.
    """

    # ── Holdout-aware artifact paths ─────────────────────────────────────
    # The 4-fold CV retrains once per holdout. The trace, scaler, and
    # mappings must be keyed by holdout name, otherwise fold 1's model is
    # silently reused for folds 2-4 (data leakage in the CV metrics).
    holdout = CROSS_VALIDATION_HOLDOUT
    p = ModelPaths(holdout)

    if p.trace.exists() and not os.environ.get("PRS_FORCE_RETRAIN", "") == "1":
        logger.info(
            "Found existing trace at %s. Skipping MCMC (set PRS_FORCE_RETRAIN=1 to override).",
            p.trace,
        )
        return az.from_netcdf(str(p.trace))  # type: ignore[no-any-return]

    if not p.training_dataset.exists():
        logger.error(
            "Dataset not found at %s. Run src.data.builder first to generate "
            "the processed training data.",
            p.training_dataset,
        )
        return None

    df = pd.read_parquet(p.training_dataset)

    # Expand spatial features with B-spline basis (replaces bc_x, bc_y, dist_nearest_opp
    # with non-linear spline expansions that capture non-linear pitch geography)
    logger.info("Fitting B-spline transformers on spatial features...")
    spline_transformers = fit_spline_transformers(df, n_knots=5, degree=3)
    df = expand_spline_features(df, spline_transformers)
    logger.info(
        "Spline expansion complete: %d basis columns generated for %d raw features",
        sum(t.n_features_out_ for t in spline_transformers.values()),
        len(spline_transformers),
    )

    df, available_features = prepare_model_dataset(df)

    logger.info("Loaded dataset: %d observations, %d features", len(df), len(available_features))

    X: np.ndarray = np.asarray(df[available_features].values)

    y_success: np.ndarray = np.asarray(df["success"].values).astype(int)
    y_value: np.ndarray = np.asarray(df["value_preserved"].values)
    min_value: float = float(y_value.min())
    y_value_shifted: np.ndarray = y_value - min_value
    max_value: float = float(y_value_shifted.max()) if y_value_shifted.max() > 0 else 0.15

    epsilon: float = 1e-6
    y_value_scaled: np.ndarray = (y_value_shifted / max_value) * (1 - 2 * epsilon) + epsilon
    y_value_scaled = np.clip(y_value_scaled, epsilon, 1 - epsilon)

    # ── Scaler (with intermediate caching) ────────────────────────────────
    MODEL_TRACES_DIR.mkdir(parents=True, exist_ok=True)

    cached = _load_cached_scaler(p.scaler)
    if cached is not None and np.isclose(cached["max_value"], max_value) and np.isclose(cached.get("min_value", 0.0), min_value):
        scaler: StandardScaler = cached["scaler"]
        X_scaled: np.ndarray = scaler.transform(X)
        logger.info("Loaded cached scaler from %s", p.scaler)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        _save_scaler(p.scaler, scaler, available_features, max_value, min_value, epsilon,
                     spline_transformers, position_specific_features=list(POSITION_SPECIFIC_FEATURES))
        logger.info("Fitted and saved new scaler to %s", p.scaler)

    # ── Split into global vs. position-specific features ─────────────────
    pos_specific_mask: np.ndarray = np.array(
        [_is_position_specific(f) for f in available_features]
    )
    global_mask: np.ndarray = ~pos_specific_mask
    n_features_global: int = int(global_mask.sum())
    n_features_pos_specific: int = int(pos_specific_mask.sum())
    logger.info(
        "Features: %d global, %d position-specific (of %d total)",
        n_features_global, n_features_pos_specific, len(available_features),
    )

    X_scaled_global: np.ndarray = X_scaled[:, global_mask]
    X_scaled_pos_specific: np.ndarray = X_scaled[:, pos_specific_mask]

    # ── Categorical indices ───────────────────────────────────────────────
    player_cats = df["player_id"].astype("category")
    comp_cats = df["competition"].astype("category")
    opp_team_cats = df["opponent_team_id"].astype("category")
    team_cats = df["team_id"].astype("category")
    pos_cats = df["position_group"].astype("category")

    # type: ignore[assignment]
    player_idx: np.ndarray = player_cats.cat.codes.values
    # type: ignore[assignment]
    comp_idx: np.ndarray = comp_cats.cat.codes.values
    # type: ignore[assignment]
    opp_idx: np.ndarray = opp_team_cats.cat.codes.values
    # type: ignore[assignment]
    team_idx: np.ndarray = team_cats.cat.codes.values
    # type: ignore[assignment]
    pos_idx: np.ndarray = pos_cats.cat.codes.values

    n_players: int = len(player_cats.cat.categories)
    n_comp: int = len(comp_cats.cat.categories)
    n_opp: int = len(opp_team_cats.cat.categories)
    n_teams: int = len(team_cats.cat.categories)
    n_pos: int = len(pos_cats.cat.categories)
    player_mapping: dict[int, Any] = dict(enumerate(player_cats.cat.categories))
    comp_mapping: dict[int, str] = dict(enumerate(comp_cats.cat.categories))
    opp_team_mapping: dict[int, Any] = dict(enumerate(opp_team_cats.cat.categories))
    team_mapping: dict[int, Any] = dict(enumerate(team_cats.cat.categories))
    pos_mapping: dict[int, str] = dict(enumerate(pos_cats.cat.categories))

    name_lookup: dict[Any, str] = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    pos_lookup: dict[Any, str] = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()

    _save_mappings(
        p.mappings,
        player_mapping, comp_mapping, opp_team_mapping, team_mapping, pos_mapping,
        name_lookup, pos_lookup,
    )

    # Beta model subset: successes only
    mask: np.ndarray = y_success == 1
    X_val_global: np.ndarray = X_scaled_global[mask]
    X_val_pos_specific: np.ndarray = X_scaled_pos_specific[mask]
    y_val_scaled: np.ndarray = y_value_scaled[mask]
    player_idx_val: np.ndarray = player_idx[mask]
    comp_idx_val: np.ndarray = comp_idx[mask]
    opp_idx_val: np.ndarray = opp_idx[mask]
    team_idx_val: np.ndarray = team_idx[mask]
    pos_idx_val: np.ndarray = pos_idx[mask]

    with pm.Model():
        # --- BALL SECURITY MODEL (Logistic) ---
        # Data is passed directly as numpy arrays (becomes TensorConstant in the
        # graph) instead of via pm.Data (TensorSharedVariable). This is faster
        # under the NumPyro/JAX backend because TensorConstants are baked into
        # the JIT-compiled function, while SharedVariables require XLA to
        # read them at every step. The data is observation-only and never
        # mutated during sampling, so there is no functional difference.
        X_data_global = X_scaled_global
        X_data_pos_specific = X_scaled_pos_specific
        pid_succ = player_idx
        cid_succ = comp_idx
        oid_succ = opp_idx
        tid_succ = team_idx
        posid_succ = pos_idx

        alpha_succ = pm.Normal("alpha_succ", 0, 1.5)
        gamma_pos_succ = pm.Normal("gamma_pos_succ", 0, 1.0, shape=n_pos)

        # Global (non-position-specific) coefficients
        beta_global_succ = pm.Normal("beta_global_succ", 0, 1.0, shape=n_features_global)

        # Position-group-specific coefficients with hierarchical shrinkage
        beta_pos_global_succ = pm.Normal("beta_pos_global_succ", 0, 1.0, shape=n_features_pos_specific)
        sigma_pos_succ = pm.HalfNormal("sigma_pos_succ", 0.5, shape=n_features_pos_specific)
        beta_pos_raw_succ = pm.Normal("beta_pos_raw_succ", 0, 1, shape=(n_pos, n_features_pos_specific))
        beta_pos_succ = pm.Deterministic("beta_pos_succ",
            beta_pos_global_succ + sigma_pos_succ * beta_pos_raw_succ)

        sigma_opp_succ = pm.Exponential("sigma_opp_succ", 1.0)
        opp_raw_succ = pm.Normal("opp_raw_succ", 0, 1, shape=n_opp)
        delta_opp_succ = pm.Deterministic("delta_opp_succ", opp_raw_succ * sigma_opp_succ)

        sigma_comp_succ = pm.Exponential("sigma_comp_succ", 1.0)
        comp_raw_succ = pm.Normal("comp_raw_succ", 0, 1, shape=n_comp)
        zeta_comp_succ = pm.Deterministic("zeta_comp_succ", comp_raw_succ * sigma_comp_succ)

        sigma_team_succ = pm.Exponential("sigma_team_succ", 1.0)
        team_raw_succ = pm.Normal("team_raw_succ", 0, 1, shape=n_teams)
        eta_team_succ = pm.Deterministic("eta_team_succ", team_raw_succ * sigma_team_succ)

        # Multivariate Normal for correlated θ_succ / θ_val
        # LKJ(η=2) gives a weak prior favouring near-zero correlation
        # pm.LKJCholeskyCov returns (cholesky, corr_matrix, stds) tuple
        theta_chol_packed = pm.LKJCholeskyCov("theta_chol", n=2, eta=2.0,
                                               sd_dist=pm.Exponential.dist(1.0, shape=2))
        theta_chol, theta_corr, _ = theta_chol_packed
        theta_offset = pm.Normal("theta_offset", 0, 1, shape=(n_players, 2))
        theta = pm.Deterministic("theta", pm.math.dot(theta_offset, theta_chol.T))
        theta_succ = pm.Deterministic("theta_succ", theta[:, 0])
        theta_val = pm.Deterministic("theta_val", theta[:, 1])
        pm.Deterministic("rho_theta", theta_corr[0, 1])

        global_contrib = pm.math.dot(X_data_global, beta_global_succ)
        pos_feat_contrib = pm.math.sum(beta_pos_succ[posid_succ] * X_data_pos_specific, axis=-1)
        logit_p = (alpha_succ +
                   global_contrib + pos_feat_contrib +
                   gamma_pos_succ[posid_succ] +
                   theta_succ[pid_succ] +
                   delta_opp_succ[oid_succ] +
                   zeta_comp_succ[cid_succ] +
                   eta_team_succ[tid_succ])

        p = pm.Deterministic("p", pm.math.invlogit(logit_p))
        pm.Bernoulli("y_succ_obs", p=p, observed=y_success)

        # --- VALUE RETENTION MODEL (Beta) ---
        X_data_val_global = X_val_global
        X_data_val_pos_specific = X_val_pos_specific
        pid_val = player_idx_val
        cid_val = comp_idx_val
        oid_val = opp_idx_val
        tid_val = team_idx_val
        posid_val = pos_idx_val

        alpha_val = pm.Normal("alpha_val", 0, 1.5)
        gamma_pos_val = pm.Normal("gamma_pos_val", 0, 1.0, shape=n_pos)

        # Global (non-position-specific) coefficients
        beta_global_val = pm.Normal("beta_global_val", 0, 1.0, shape=n_features_global)

        # Position-group-specific coefficients with hierarchical shrinkage
        beta_pos_global_val = pm.Normal("beta_pos_global_val", 0, 1.0, shape=n_features_pos_specific)
        sigma_pos_val = pm.HalfNormal("sigma_pos_val", 0.5, shape=n_features_pos_specific)
        beta_pos_raw_val = pm.Normal("beta_pos_raw_val", 0, 1, shape=(n_pos, n_features_pos_specific))
        beta_pos_val = pm.Deterministic("beta_pos_val",
            beta_pos_global_val + sigma_pos_val * beta_pos_raw_val)

        sigma_opp_val = pm.Exponential("sigma_opp_val", 1.0)
        opp_raw_val = pm.Normal("opp_raw_val", 0, 1, shape=n_opp)
        delta_opp_val = pm.Deterministic("delta_opp_val", opp_raw_val * sigma_opp_val)

        sigma_comp_val = pm.Exponential("sigma_comp_val", 1.0)
        comp_raw_val = pm.Normal("comp_raw_val", 0, 1, shape=n_comp)
        zeta_comp_val = pm.Deterministic("zeta_comp_val", comp_raw_val * sigma_comp_val)

        sigma_team_val = pm.Exponential("sigma_team_val", 1.0)
        team_raw_val = pm.Normal("team_raw_val", 0, 1, shape=n_teams)
        eta_team_val = pm.Deterministic("eta_team_val", team_raw_val * sigma_team_val)

        global_contrib_val = pm.math.dot(X_data_val_global, beta_global_val)
        pos_feat_contrib_val = pm.math.sum(beta_pos_val[posid_val] * X_data_val_pos_specific, axis=-1)
        logit_mu = (alpha_val +
                    global_contrib_val + pos_feat_contrib_val +
                    gamma_pos_val[posid_val] +
                    theta_val[pid_val] +
                    delta_opp_val[oid_val] +
                    zeta_comp_val[cid_val] +
                    eta_team_val[tid_val])

        mu = pm.Deterministic("mu", pm.math.invlogit(logit_mu))
        kappa = pm.Exponential("kappa", 0.1)  # Mean=10; stable concentration for Beta regression

        alpha_beta = mu * kappa
        beta_beta = (1 - mu) * kappa

        pm.Beta("y_val_obs", alpha=alpha_beta, beta=beta_beta, observed=y_val_scaled)

        logger.info("Starting MCMC sampling on holdout=%s ...", holdout)
        logger.info(
            "Settings: draws=%d tune=%d chains=%d target_accept=%.2f sampler=%s",
            MODEL_SETTINGS["draws"], MODEL_SETTINGS["tune"],
            MODEL_SETTINGS["chains"], MODEL_SETTINGS["target_accept"],
            MODEL_SETTINGS["nuts_sampler"],
        )

        # Enable parallel chain execution on single GPU via vmap.
        # Must happen before any JAX operation (jax.devices() etc).
        import numpyro
        numpyro.set_host_device_count(MODEL_SETTINGS["chains"])

        # Verify JAX device (GPU/Metal/CPU-with-opt-in)
        _verify_jax_device()

        # Suppress PyMC's pmap warning — we use chain_method='vectorized',
        # so the pmap fallback is never used despite having only 1 device.
        warnings.filterwarnings("ignore", message="There are not enough devices to run parallel chains")

        # chain_method='vectorized' uses vmap to run all chains simultaneously
        # on a single device — the only correct choice for 1×GPU.
        # 'parallel' (pmap) requires one device per chain and silently falls
        # back to sequential with a single GPU, giving a 4× slowdown.
        # On CPU (PRS_ALLOW_CPU=1), vectorized is still the best option.
        trace = pm.sample(
            draws=MODEL_SETTINGS["draws"],
            tune=MODEL_SETTINGS["tune"],
            chains=MODEL_SETTINGS["chains"],
            target_accept=MODEL_SETTINGS["target_accept"],
            random_seed=MODEL_SETTINGS["random_seed"],
            nuts_sampler=MODEL_SETTINGS["nuts_sampler"],
            chain_method="vectorized",
            return_inferencedata=True,
            progressbar=True,
        )

    trace.to_netcdf(str(p.trace))
    logger.info("Saved trace to %s", p.trace)

    return trace  # type: ignore[no-any-return]


if __name__ == "__main__":
    fit_pooled_model()
