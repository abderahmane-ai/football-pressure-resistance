"""Bayesian Hierarchical Beta Hurdle Model for Pressure Resistance Score."""
from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler

from config import PROCESSED_DATA_DIR, MODEL_TRACES_DIR, MODEL_SETTINGS, MODEL_FEATURE_COLUMNS
from src.data.validation import DataValidationError, validate_model_dataset

warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        return data
    except Exception as exc:
        logger.warning("Could not load cached scaler from %s: %s", path, exc)
        return None


def _save_scaler(
    path: Path,
    scaler: StandardScaler,
    features: list[str],
    max_value: float,
    epsilon: float,
) -> None:
    """Persist scaler, feature list, and scaling constants."""
    with open(path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "features": features,
            "max_value": max_value,
            "epsilon": epsilon,
        }, f)


def _save_mappings(
    path: Path,
    player_mapping: dict[int, Any],
    comp_mapping: dict[int, str],
    opp_team_mapping: dict[int, Any],
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
    dataset_path: Path = PROCESSED_DATA_DIR / "all_pressure_dataset.parquet"
    if not dataset_path.exists():
        logger.error(
            "Dataset not found at %s. Run src.data.builder first to generate "
            "the processed training data.",
            dataset_path,
        )
        return None

    df = pd.read_parquet(dataset_path)
    df, available_features = prepare_model_dataset(df)

    logger.info("Loaded dataset: %d observations, %d features", len(df), len(available_features))

    X: np.ndarray = df[available_features].values

    y_success: np.ndarray = df["success"].values.astype(int)
    y_value: np.ndarray = df["value_preserved"].values
    max_value: float = float(y_value.max()) if y_value.max() > 0 else 0.15

    epsilon: float = 1e-6
    y_value_scaled: np.ndarray = (y_value / max_value) * (1 - 2 * epsilon) + epsilon
    y_value_scaled = np.clip(y_value_scaled, epsilon, 1 - epsilon)

    # ── Scaler (with intermediate caching) ────────────────────────────────
    scaler_path: Path = MODEL_TRACES_DIR / "pooled_scaler.pkl"
    MODEL_TRACES_DIR.mkdir(parents=True, exist_ok=True)

    cached = _load_cached_scaler(scaler_path)
    if cached is not None and cached["max_value"] == max_value:
        scaler: StandardScaler = cached["scaler"]
        X_scaled: np.ndarray = scaler.transform(X)
        logger.info("Loaded cached scaler from %s", scaler_path)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        _save_scaler(scaler_path, scaler, available_features, max_value, epsilon)
        logger.info("Fitted and saved new scaler to %s", scaler_path)

    # ── Categorical indices ───────────────────────────────────────────────
    player_cats = df["player_id"].astype("category")
    comp_cats = df["competition"].astype("category")
    opp_team_cats = df["opponent_team_id"].astype("category")
    pos_cats = df["position_group"].astype("category")

    player_idx: np.ndarray = player_cats.cat.codes.values
    comp_idx: np.ndarray = comp_cats.cat.codes.values
    opp_idx: np.ndarray = opp_team_cats.cat.codes.values
    pos_idx: np.ndarray = pos_cats.cat.codes.values

    n_players: int = len(player_cats.cat.categories)
    n_comp: int = len(comp_cats.cat.categories)
    n_opp: int = len(opp_team_cats.cat.categories)
    n_pos: int = len(pos_cats.cat.categories)
    n_features: int = len(available_features)

    player_mapping: dict[int, Any] = dict(enumerate(player_cats.cat.categories))
    comp_mapping: dict[int, str] = dict(enumerate(comp_cats.cat.categories))
    opp_team_mapping: dict[int, Any] = dict(enumerate(opp_team_cats.cat.categories))
    pos_mapping: dict[int, str] = dict(enumerate(pos_cats.cat.categories))

    name_lookup: dict[Any, str] = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    pos_lookup: dict[Any, str] = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()

    _save_mappings(
        MODEL_TRACES_DIR / "pooled_mappings.pkl",
        player_mapping, comp_mapping, opp_team_mapping, pos_mapping,
        name_lookup, pos_lookup,
    )

    # Beta model subset: successes only
    mask: np.ndarray = y_success == 1
    X_val: np.ndarray = X_scaled[mask]
    y_val_scaled: np.ndarray = y_value_scaled[mask]
    player_idx_val: np.ndarray = player_idx[mask]
    comp_idx_val: np.ndarray = comp_idx[mask]
    opp_idx_val: np.ndarray = opp_idx[mask]
    pos_idx_val: np.ndarray = pos_idx[mask]

    with pm.Model():
        # --- BALL SECURITY MODEL (Logistic) ---
        X_data_succ = pm.Data("X_succ", X_scaled)
        pid_succ = pm.Data("pid_succ", player_idx)
        cid_succ = pm.Data("cid_succ", comp_idx)
        oid_succ = pm.Data("oid_succ", opp_idx)
        posid_succ = pm.Data("posid_succ", pos_idx)

        alpha_succ = pm.Normal("alpha_succ", 0, 1.5)
        beta_succ = pm.Normal("beta_succ", 0, 1.0, shape=n_features)
        gamma_pos_succ = pm.Normal("gamma_pos_succ", 0, 1.0, shape=n_pos)

        sigma_theta_succ = pm.Exponential("sigma_theta_succ", 1.0)
        theta_raw_succ = pm.Normal("theta_raw_succ", 0, 1, shape=n_players)
        theta_succ = pm.Deterministic("theta_succ", theta_raw_succ * sigma_theta_succ)

        sigma_opp_succ = pm.Exponential("sigma_opp_succ", 1.0)
        opp_raw_succ = pm.Normal("opp_raw_succ", 0, 1, shape=n_opp)
        delta_opp_succ = pm.Deterministic("delta_opp_succ", opp_raw_succ * sigma_opp_succ)

        sigma_comp_succ = pm.Exponential("sigma_comp_succ", 1.0)
        comp_raw_succ = pm.Normal("comp_raw_succ", 0, 1, shape=n_comp)
        zeta_comp_succ = pm.Deterministic("zeta_comp_succ", comp_raw_succ * sigma_comp_succ)

        logit_p = (alpha_succ +
                   pm.math.dot(X_data_succ, beta_succ) +
                   gamma_pos_succ[posid_succ] +
                   theta_succ[pid_succ] +
                   delta_opp_succ[oid_succ] +
                   zeta_comp_succ[cid_succ])

        p = pm.Deterministic("p", pm.math.invlogit(logit_p))
        pm.Bernoulli("y_succ_obs", p=p, observed=y_success)

        # --- VALUE RETENTION MODEL (Beta) ---
        X_data_val = pm.Data("X_val", X_val)
        pid_val = pm.Data("pid_val", player_idx_val)
        cid_val = pm.Data("cid_val", comp_idx_val)
        oid_val = pm.Data("oid_val", opp_idx_val)
        posid_val = pm.Data("posid_val", pos_idx_val)

        alpha_val = pm.Normal("alpha_val", 0, 1.5)
        beta_val = pm.Normal("beta_val", 0, 1.0, shape=n_features)
        gamma_pos_val = pm.Normal("gamma_pos_val", 0, 1.0, shape=n_pos)

        sigma_theta_val = pm.Exponential("sigma_theta_val", 1.0)
        theta_raw_val = pm.Normal("theta_raw_val", 0, 1, shape=n_players)
        theta_val = pm.Deterministic("theta_val", theta_raw_val * sigma_theta_val)

        sigma_opp_val = pm.Exponential("sigma_opp_val", 1.0)
        opp_raw_val = pm.Normal("opp_raw_val", 0, 1, shape=n_opp)
        delta_opp_val = pm.Deterministic("delta_opp_val", opp_raw_val * sigma_opp_val)

        sigma_comp_val = pm.Exponential("sigma_comp_val", 1.0)
        comp_raw_val = pm.Normal("comp_raw_val", 0, 1, shape=n_comp)
        zeta_comp_val = pm.Deterministic("zeta_comp_val", comp_raw_val * sigma_comp_val)

        logit_mu = (alpha_val +
                    pm.math.dot(X_data_val, beta_val) +
                    gamma_pos_val[posid_val] +
                    theta_val[pid_val] +
                    delta_opp_val[oid_val] +
                    zeta_comp_val[cid_val])

        mu = pm.Deterministic("mu", pm.math.invlogit(logit_mu))
        kappa = pm.Exponential("kappa", 0.1)  # Mean=10; stable concentration for Beta regression

        alpha_beta = mu * kappa
        beta_beta = (1 - mu) * kappa

        pm.Beta("y_val_obs", alpha=alpha_beta, beta=beta_beta, observed=y_val_scaled)

        logger.info("Starting MCMC sampling...")
        try:
            # chain_method='vectorized' uses vmap to run all chains simultaneously
            # on a single GPU device — correct and fast for single-device setups.
            # 'parallel' (pmap) requires one device per chain and silently falls
            # back to sequential when only 1 GPU is present, giving a 4x slowdown.
            trace = pm.sample(
                draws=MODEL_SETTINGS["draws"],
                tune=MODEL_SETTINGS["tune"],
                chains=MODEL_SETTINGS["chains"],
                target_accept=MODEL_SETTINGS["target_accept"],
                random_seed=MODEL_SETTINGS["random_seed"],
                nuts_sampler=MODEL_SETTINGS["nuts_sampler"],
                nuts_sampler_kwargs={"chain_method": "vectorized"},
                return_inferencedata=True,
                progressbar=True,
            )
        except Exception as e:
            logger.warning(
                "numpyro vectorized sampling failed: %s. "
                "Falling back to default PyMC NUTS sampler.",
                e,
            )
            trace = pm.sample(
                draws=MODEL_SETTINGS["draws"],
                tune=MODEL_SETTINGS["tune"],
                chains=MODEL_SETTINGS["chains"],
                target_accept=MODEL_SETTINGS["target_accept"],
                random_seed=MODEL_SETTINGS["random_seed"],
                return_inferencedata=True,
                progressbar=True,
            )

    trace_path: Path = MODEL_TRACES_DIR / "pooled_trace.nc"
    trace.to_netcdf(trace_path)
    logger.info("Saved trace to %s", trace_path)

    return trace


if __name__ == "__main__":
    fit_pooled_model()
