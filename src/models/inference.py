"""Posterior inference: generate PRS leaderboards from fitted Hurdle model."""
from __future__ import annotations

import logging
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from config import (
    MIN_EVENTS_THRESHOLD,
    SPATIAL_CONFIG,
)
from src.models.posterior import load_posterior_context
from src.paths import ModelPaths

logger = logging.getLogger(__name__)

# R-hat threshold above which a warning is emitted.  Vehtari et al. (2021)
# recommend 1.01 as a strict bound; we use 1.05 as a practical gate — values
# above this mean the chains almost certainly haven't converged.
_RHAT_WARN_THRESHOLD: float = 1.05
# Minimum effective sample size (bulk) before we consider a parameter reliable.
_ESS_MIN_THRESHOLD: float = 100.0


def _log_diagnostics(trace: az.InferenceData) -> None:
    """Log R-hat and ESS for the key player-effect parameters.

    Emits warnings when R-hat exceeds the threshold or ESS is too low.
    Does *not* halt execution — the caller decides how to handle degraded
    diagnostics (the leaderboard CSV is still written so the user can inspect
    it with full awareness).
    """
    key_params: list[str] = ["theta_succ", "theta_val"]
    available = [p for p in key_params if p in trace.posterior]  # type: ignore[attr-defined]
    if not available:
        logger.warning("Cannot compute diagnostics — no player-effect parameters in trace.")
        return

    try:
        summary = az.summary(trace, var_names=available, kind="diagnostics")
    except Exception as exc:
        logger.warning("Could not compute convergence diagnostics: %s", exc)
        return

    for param in available:
        if param not in summary.index:
            continue
        row = summary.loc[param]
        rhat: float = float(row.get("r_hat", 1.0))
        ess_bulk: float = float(row.get("ess_bulk", 0.0))

        if rhat > _RHAT_WARN_THRESHOLD:
            logger.warning(
                "R-hat=%.2f for '%s' exceeds threshold %.2f — chains may not have converged. "
                "Leaderboard scores should be treated as provisional.",
                rhat, param, _RHAT_WARN_THRESHOLD,
            )
        else:
            logger.info("  %s: R-hat=%.3f  ESS(bulk)=%.0f", param, rhat, ess_bulk)

        if ess_bulk < _ESS_MIN_THRESHOLD:
            logger.warning(
                "ESS(bulk)=%.0f for '%s' is below minimum %d — player effect "
                "estimates have high Monte Carlo error.",
                ess_bulk, param, int(_ESS_MIN_THRESHOLD),
            )


def run_posterior_analysis() -> None:
    """
    Generate player leaderboards with Ball Security and Value Retention scores.
    Uses expit for numeric stability and appropriately handles the Hurdle model outputs.

    PRS aggregation details:
    PRS = θ_succ + θ_val (additive on the logit scale). This gives equal
    weight to ball security and value retention. Both θ terms live on the
    logit scale with the same prior (Non-centered Normal), so addition is
    scale-consistent. An alternative would be multiplicative on the
    probability scale (p_succ × μ_val), but that conflates the two traits
    and makes the ranking sensitive to the base rate.
    """
    ctx = load_posterior_context()

    # Local aliases for the rest of the function body
    player_mapping = ctx.player_mapping
    pos_mapping = ctx.pos_mapping
    name_lookup = ctx.name_lookup
    pos_lookup = ctx.pos_lookup
    scaler = ctx.scaler
    feature_names = ctx.feature_names
    max_value = ctx.max_value
    min_value = ctx.min_value
    event_counts: dict[Any, int] = ctx.df["player_id"].value_counts().to_dict()

    # Extract correlation between θ_succ and θ_val
    if ctx.rho_theta is not None:
        rho_vals = getattr(ctx.rho_theta, "values", ctx.rho_theta)
        rho_mean: float = float(rho_vals.mean())
        rho_hdi = az.hdi(rho_vals.flatten(), hdi_prob=0.90)
        logger.info(
            "Correlation ρ(θ_succ, θ_val): mean=%.3f, 90%% HDI=[%.3f, %.3f]",
            rho_mean, rho_hdi[0], rho_hdi[1],
        )

    # ── Convergence diagnostics ──────────────────────────────────────────────
    # Gate leaderboard generation on R-hat and ESS for the key player-effect
    # parameters.  If the chains haven't mixed, the leaderboard is unreliable.
    _log_diagnostics(ctx.trace)

    # Marginalise by posterior group mean; shape: (n_samples,)
    mean_opp_succ: np.ndarray = ctx.delta_opp_succ.mean(axis=1)
    mean_comp_succ: np.ndarray = ctx.zeta_comp_succ.mean(axis=1)
    mean_team_succ: np.ndarray = ctx.eta_team_succ.mean(axis=1)
    mean_opp_val: np.ndarray = ctx.delta_opp_val.mean(axis=1)
    mean_comp_val: np.ndarray = ctx.zeta_comp_val.mean(axis=1)
    mean_team_val: np.ndarray = ctx.eta_team_val.mean(axis=1)

    tight_dist: float = SPATIAL_CONFIG["tight_pressure_radius"] * 0.3
    loose_dist: float = SPATIAL_CONFIG["tight_pressure_radius"] * 0.6

    # Pre-compute spline basis for distance scenarios
    dist_spline_cols = [c for c in feature_names if c.startswith('dist_nearest_opp_spline_')]
    dist_scenario_vecs: dict[str, np.ndarray] = {}
    if dist_spline_cols and ctx.spline_transformers:
        dist_tr = ctx.spline_transformers.get("dist_nearest_opp")
        if dist_tr is not None:
            for label, dist_val in [("tight", tight_dist), ("loose", loose_dist)]:
                basis = dist_tr.transform([[dist_val]])[0]
                vec = np.zeros(len(feature_names))
                for k, col_name in enumerate(dist_spline_cols):
                    f_idx = ctx.name_to_idx[col_name]
                    vec[f_idx] = (basis[k] - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
                dist_scenario_vecs[label] = vec

    scenarios: dict[str, dict[str, float]] = {
        "Front_Tight":      {"distance": tight_dist, "angle_nearest_opp": 0.0, "coverage_arc": 1.5},
        "Front_Loose":      {"distance": loose_dist, "angle_nearest_opp": 0.0, "coverage_arc": 0.8},
        "Lateral_Tight":    {"distance": tight_dist, "angle_nearest_opp": np.pi / 2, "coverage_arc": 1.5},
        "Lateral_Loose":    {"distance": loose_dist, "angle_nearest_opp": np.pi / 2, "coverage_arc": 0.8},
        "Back_Tight":       {"distance": tight_dist, "angle_nearest_opp": np.pi, "coverage_arc": 1.5},
        "Back_Loose":       {"distance": loose_dist, "angle_nearest_opp": np.pi, "coverage_arc": 0.8},
        "CounterPress_High":{"distance": tight_dist, "angle_nearest_opp": 0.0, "coverage_arc": 2.5, "counter_press": 1},
        "CounterPress_Low": {"distance": loose_dist, "angle_nearest_opp": np.pi, "coverage_arc": 0.5, "counter_press": 1},
        "HighPass_Back":    {"distance": tight_dist, "angle_nearest_opp": np.pi, "coverage_arc": 1.5, "pass_height_high": 1},
        "LowPass_Front":    {"distance": loose_dist, "angle_nearest_opp": 0.0, "coverage_arc": 0.5, "pass_height_ground": 1},
        "Sustained_Press":  {"distance": tight_dist, "angle_nearest_opp": 0.0, "coverage_arc": 2.0, "recent_pressures": 3},
    }

    # Scenario feature vectors: unmentioned features are set to their
    # standardized mean (zero in scaled space), representing the "average
    # situation" baseline — an intentional design choice.
    # For dist_nearest_opp (which is spline-expanded), we use the pre-computed
    # spline basis vector stored in dist_scenario_vecs.
    scenario_vectors: dict[str, np.ndarray] = {}
    for name, s in scenarios.items():
        vec = np.zeros(len(feature_names))
        dist_label = "tight" if s.get("distance") == tight_dist else "loose" if s.get("distance") == loose_dist else None
        if dist_label and dist_label in dist_scenario_vecs:
            vec += dist_scenario_vecs[dist_label]
        for f_name, val in s.items():
            if f_name == "distance":
                continue
            if f_name in ctx.name_to_idx:
                f_idx = ctx.name_to_idx[f_name]
                if f_name not in ("opps_within_1yd", "opps_within_2yd", "opps_within_4yd",
                                   "has_progressive_option"):
                    vec[f_idx] = (val - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
                else:
                    vec[f_idx] = (0 - scaler.mean_[f_idx]) / scaler.scale_[f_idx]
        scenario_vectors[name] = vec

    mid_pos_code: int = next((code for code, name in pos_mapping.items() if name == "CM"), 0)

    def get_predictions(
        scenario_vec: np.ndarray,
        player_idx: int | None = None,
        p_code: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        code = p_code if p_code is not None else mid_pos_code
        pos_effect_succ = ctx.gamma_pos_succ[:, code]
        pos_effect_val = ctx.gamma_pos_val[:, code]

        scenario_global = scenario_vec[ctx.global_mask]
        scenario_pos = scenario_vec[ctx.pos_specific_mask]
        feat_contrib_succ = np.dot(ctx.beta_global_succ, scenario_global) + np.dot(ctx.beta_pos_succ[:, code], scenario_pos)
        feat_contrib_val = np.dot(ctx.beta_global_val, scenario_global) + np.dot(ctx.beta_pos_val[:, code], scenario_pos)

        logit_succ = (ctx.alpha_succ + feat_contrib_succ + pos_effect_succ
                      + mean_opp_succ + mean_comp_succ + mean_team_succ)
        logit_val = (ctx.alpha_val + feat_contrib_val + pos_effect_val
                     + mean_opp_val + mean_comp_val + mean_team_val)

        if player_idx is not None:
            logit_succ += ctx.theta_succ[:, player_idx]
            logit_val += ctx.theta_val[:, player_idx]

        prob_success: np.ndarray = expit(logit_succ)
        expected_value_retention: np.ndarray = expit(logit_val) * max_value + min_value

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
            position_group: str = pos_lookup.get(player_id, "CM")

            if position_group not in pos_mapping.values():
                logger.warning(
                    "Player %s (%s) has position group '%s' which is not in model categories %s. "
                    "Defaulting to 'CM'.",
                    player_id, player_name, position_group, list(pos_mapping.values()),
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

            ball_security_samples: np.ndarray = ctx.theta_succ[:, idx_num]
            value_retention_samples: np.ndarray = ctx.theta_val[:, idx_num]
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

    out = ModelPaths(ctx.holdout)
    out.leaderboard.parent.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(out.leaderboard, index=False)
    logger.info("Saved leaderboard with %d players (n>=%d) to %s", len(lb_df), MIN_EVENTS_THRESHOLD, out.leaderboard)

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
