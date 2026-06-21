"""Shared posterior access — eliminates duplicated array extraction across analysis modules.

``inference.py``, ``validation.py``, and ``interpretability.py`` previously
each had a ~25-line block that loaded the trace, mappings, and scaler,
recomputed position-specific masks, and reshaped every posterior array.
``PosteriorContext`` bundles everything into a single dataclass, and
``load_posterior_context()`` is the single factory for constructing it.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import CROSS_VALIDATION_HOLDOUT, SPATIAL_CONFIG
from src.data.validation import validate_model_dataset
from src.features.spatial import expand_spline_features, is_position_specific
from src.paths import ModelPaths, require_paths

logger = logging.getLogger(__name__)


@dataclass
class PosteriorContext:
    """All posterior arrays and metadata needed for inference / validation / interpretability.

    Attributes are populated by ``load_posterior_context()`` — do not construct directly.
    """

    holdout: str
    trace: az.InferenceData

    # ── Posterior arrays ────────────────────────────────────────────────────
    alpha_succ: np.ndarray
    alpha_val: np.ndarray
    beta_global_succ: np.ndarray
    beta_global_val: np.ndarray
    beta_pos_succ: np.ndarray
    beta_pos_val: np.ndarray
    theta_succ: np.ndarray
    theta_val: np.ndarray
    gamma_pos_succ: np.ndarray
    gamma_pos_val: np.ndarray

    # Group random effects
    delta_opp_succ: np.ndarray
    zeta_comp_succ: np.ndarray
    eta_team_succ: np.ndarray
    delta_opp_val: np.ndarray
    zeta_comp_val: np.ndarray
    eta_team_val: np.ndarray

    # Cholesky / variance parameters
    theta_chol_packed: np.ndarray | None = None
    sigma_opp_succ: np.ndarray | None = None
    sigma_comp_succ: np.ndarray | None = None
    sigma_team_succ: np.ndarray | None = None
    sigma_opp_val: np.ndarray | None = None
    sigma_comp_val: np.ndarray | None = None
    sigma_team_val: np.ndarray | None = None
    rho_theta: Any = None

    # ── Metadata ────────────────────────────────────────────────────────────
    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names: list[str] = field(default_factory=list)
    max_value: float = 1.0
    min_value: float = 0.0
    pos_mapping: dict[int, str] = field(default_factory=dict)
    player_mapping: dict[int, Any] = field(default_factory=dict)
    name_lookup: dict[Any, str] = field(default_factory=dict)
    pos_lookup: dict[Any, str] = field(default_factory=dict)
    spline_transformers: dict[str, Any] | None = None

    # Derived masks
    n_global: int = 0
    n_pos_specific: int = 0
    pos_specific_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    global_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Feature name → index lookup (O(1) instead of repeated list scans)
    name_to_idx: dict[str, int] = field(default_factory=dict)

    # The loaded dataset (training or holdout, with splines expanded)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_posterior_context(
    holdout: str | None = None,
    dataset_kind: str = "training",
) -> PosteriorContext:
    """Load trace, mappings, scaler, dataset and extract all posterior arrays.

    Parameters
    ----------
    holdout:
        Competition name used as CV holdout.  Defaults to ``PRS_HOLDOUT``
        environment variable / ``CROSS_VALIDATION_HOLDOUT``.
    dataset_kind:
        ``"training"`` loads the training dataset; ``"holdout"`` loads the
        holdout dataset.  Validation uses the holdout set because it evaluates
        out-of-sample residuals.

    Returns
    -------
    PosteriorContext
        Fully populated context ready for downstream analysis.

    Raises
    ------
    FileNotFoundError
        If any required artifact (trace, scaler, mappings, dataset) is missing.
    """
    holdout = holdout or CROSS_VALIDATION_HOLDOUT
    p = ModelPaths(holdout)

    dataset_path = p.training_dataset if dataset_kind == "training" else p.holdout_dataset
    require_paths(p.trace, p.mappings, p.scaler, dataset_path)

    trace: az.InferenceData = az.from_netcdf(p.trace)

    # ── Mappings ────────────────────────────────────────────────────────────
    with open(p.mappings, "rb") as f:
        mappings: dict[str, Any] = pickle.load(f)
    pos_mapping: dict[int, str] = mappings["position"]
    player_mapping: dict[int, Any] = mappings["player"]
    name_lookup: dict[Any, str] = mappings.get("name_lookup", {})
    pos_lookup: dict[Any, str] = mappings.get("position_lookup", {})

    # ── Scaler ──────────────────────────────────────────────────────────────
    with open(p.scaler, "rb") as f:
        scaler_data: dict[str, Any] = pickle.load(f)
    scaler: StandardScaler = scaler_data["scaler"]
    feature_names: list[str] = scaler_data["features"]
    max_value: float = scaler_data.get("max_value", 1.0)
    min_value: float = scaler_data.get("min_value", 0.0)
    spline_transformers = scaler_data.get("spline_transformers")

    # ── Dataset ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(dataset_path)

    if dataset_kind == "holdout" and "dist_nearest_opp" in df.columns:
        df = df[df["dist_nearest_opp"] <= SPATIAL_CONFIG["tight_pressure_radius"]]

    if spline_transformers:
        df = expand_spline_features(df, spline_transformers)

    validate_model_dataset(df, feature_names, context=f"{dataset_kind} pressure dataset")

    # ── Masks ───────────────────────────────────────────────────────────────
    scaler_psf = scaler_data.get("position_specific_features", [])
    pos_specific_mask = np.array([is_position_specific(f, scaler_psf) for f in feature_names])
    global_mask = ~pos_specific_mask
    n_global = int(global_mask.sum())
    n_pos_specific = int(pos_specific_mask.sum())

    # ── Posterior arrays ────────────────────────────────────────────────────
    post = trace.posterior  # type: ignore[attr-defined]

    alpha_succ = post["alpha_succ"].values.flatten()
    beta_global_succ = post["beta_global_succ"].values.reshape(-1, n_global)
    beta_pos_succ = post["beta_pos_succ"].values.reshape(-1, len(pos_mapping), n_pos_specific)
    theta_succ = post["theta_succ"].values.reshape(-1, len(player_mapping))
    gamma_pos_succ = post["gamma_pos_succ"].values.reshape(-1, len(pos_mapping))

    alpha_val = post["alpha_val"].values.flatten()
    beta_global_val = post["beta_global_val"].values.reshape(-1, n_global)
    beta_pos_val = post["beta_pos_val"].values.reshape(-1, len(pos_mapping), n_pos_specific)
    theta_val = post["theta_val"].values.reshape(-1, len(player_mapping))
    gamma_pos_val = post["gamma_pos_val"].values.reshape(-1, len(pos_mapping))

    # Group effects
    delta_opp_succ = post["delta_opp_succ"].values.reshape(-1, post["delta_opp_succ"].shape[-1])
    zeta_comp_succ = post["zeta_comp_succ"].values.reshape(-1, post["zeta_comp_succ"].shape[-1])
    eta_team_succ = (
        post["eta_team_succ"].values.reshape(-1, post["eta_team_succ"].shape[-1])
        if "eta_team_succ" in post
        else np.zeros_like(delta_opp_succ)
    )
    delta_opp_val = post["delta_opp_val"].values.reshape(-1, post["delta_opp_val"].shape[-1])
    zeta_comp_val = post["zeta_comp_val"].values.reshape(-1, post["zeta_comp_val"].shape[-1])
    eta_team_val = (
        post["eta_team_val"].values.reshape(-1, post["eta_team_val"].shape[-1])
        if "eta_team_val" in post
        else np.zeros_like(delta_opp_val)
    )

    # Optional variance parameters
    theta_chol_packed = None
    sigma_opp_succ = sigma_comp_succ = sigma_team_succ = None
    sigma_opp_succ = sigma_comp_succ = sigma_team_succ = None  # type: ignore[assignment]
    sigma_opp_val = sigma_comp_val = sigma_team_val = None  # type: ignore[assignment]
    if "theta_chol" in post:
        theta_chol_packed = post["theta_chol"].values
    if "sigma_opp_succ" in post:
        sigma_opp_succ = post["sigma_opp_succ"].values.flatten()  # type: ignore[assignment]
    if "sigma_comp_succ" in post:
        sigma_comp_succ = post["sigma_comp_succ"].values.flatten()  # type: ignore[assignment]
    if "sigma_team_succ" in post:
        sigma_team_succ = post["sigma_team_succ"].values.flatten()  # type: ignore[assignment]
    if "sigma_opp_val" in post:
        sigma_opp_val = post["sigma_opp_val"].values.flatten()  # type: ignore[assignment]
    if "sigma_comp_val" in post:
        sigma_comp_val = post["sigma_comp_val"].values.flatten()  # type: ignore[assignment]
    if "sigma_team_val" in post:
        sigma_team_val = post["sigma_team_val"].values.flatten()  # type: ignore[assignment]

    # Correlation rho(θ_succ, θ_val)
    rho_theta = None
    if "rho_theta" in post:
        rho_theta = post["rho_theta"]

    # ── Feature name → index ────────────────────────────────────────────────
    name_to_idx: dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    ctx = PosteriorContext(
        holdout=holdout,
        trace=trace,
        alpha_succ=alpha_succ,
        alpha_val=alpha_val,
        beta_global_succ=beta_global_succ,
        beta_global_val=beta_global_val,
        beta_pos_succ=beta_pos_succ,
        beta_pos_val=beta_pos_val,
        theta_succ=theta_succ,
        theta_val=theta_val,
        gamma_pos_succ=gamma_pos_succ,
        gamma_pos_val=gamma_pos_val,
        delta_opp_succ=delta_opp_succ,
        zeta_comp_succ=zeta_comp_succ,
        eta_team_succ=eta_team_succ,
        delta_opp_val=delta_opp_val,
        zeta_comp_val=zeta_comp_val,
        eta_team_val=eta_team_val,
        theta_chol_packed=theta_chol_packed,
        sigma_opp_succ=sigma_opp_succ,
        sigma_comp_succ=sigma_comp_succ,
        sigma_team_succ=sigma_team_succ,
        sigma_opp_val=sigma_opp_val,
        sigma_comp_val=sigma_comp_val,
        sigma_team_val=sigma_team_val,
        rho_theta=rho_theta,
        scaler=scaler,
        feature_names=feature_names,
        max_value=max_value,
        min_value=min_value,
        pos_mapping=pos_mapping,
        player_mapping=player_mapping,
        name_lookup=name_lookup,
        pos_lookup=pos_lookup,
        spline_transformers=spline_transformers,
        n_global=n_global,
        n_pos_specific=n_pos_specific,
        pos_specific_mask=pos_specific_mask,
        global_mask=global_mask,
        name_to_idx=name_to_idx,
        df=df,
    )
    logger.info("Loaded posterior context for holdout=%s, n_features=%d (%d global, %d pos-specific)",
                 holdout, len(feature_names), n_global, n_pos_specific)
    return ctx
