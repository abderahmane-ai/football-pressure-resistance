"""Extract spatial features from a single StatsBomb 360 freeze frame."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import SPATIAL_CONFIG

from .geometry import angular_span, lane_unblocked, pitch_control_value, voronoi_area, xt_value


def extract_spatial_features_from_frame(
    frame_data: Any,
    ball_carrier_player_id: int | str,
    team_id: int | str,
    opponent_team_id: int | str,
    match_context: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    """
    Compute spatial features from a single freeze frame.

    Returns a flat dict of feature-name → float, or *None* when the frame
    cannot be parsed (missing data, no ball-carrier location, etc.).
    """
    if pd.isna(frame_data) or not isinstance(frame_data, dict):
        return None

    freeze_frame = frame_data.get("freeze_frame", [])
    if isinstance(freeze_frame, np.ndarray):
        freeze_frame = freeze_frame.tolist()
    if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
        return None

    teammates: list[list[float]] = []
    opponents: list[list[float]] = []
    ball_carrier: list[float] | None = None
    all_players: list[list[float]] = []

    for p in freeze_frame:
        loc = p.get("location")
        if loc is None or len(loc) < 2:
            continue

        all_players.append(loc)

        if p.get("actor", False) or p.get("player_id") == ball_carrier_player_id:
            ball_carrier = loc

        if p.get("teammate", False):
            teammates.append(loc)
        else:
            opponents.append(loc)

    if ball_carrier is None:
        if teammates:
            ball_carrier = teammates[0]
        else:
            return None

    bc = np.array(ball_carrier)
    opps = np.array(opponents) if opponents else np.array([])
    # Exclude actor from teammate env; distance=0 self-inclusion biases pitch_control
    env_teammates = [loc for loc in teammates if not np.allclose(loc, ball_carrier)]
    tms = np.array(env_teammates) if env_teammates else np.array([])
    features: dict[str, float] = {}

    pitch_len: float = SPATIAL_CONFIG["pitch_length"]
    goal_x: float = SPATIAL_CONFIG["goal_x"]
    goal_y: float = SPATIAL_CONFIG["goal_y"]
    coverage_radius: float = SPATIAL_CONFIG["coverage_arc_radius"]

    # Pre-compute goal angle (used for both opponent and teammate orientation)
    goal_vec = np.array([goal_x, goal_y]) - bc
    goal_angle: float = float(np.arctan2(goal_vec[1], goal_vec[0]))

    if len(opps) > 0:
        opp_dists = np.linalg.norm(opps - bc, axis=1)
        sorted_opp_dists = np.sort(opp_dists)
        features["dist_nearest_opp"] = float(sorted_opp_dists[0])
        features["dist_2nd_nearest_opp"] = float(sorted_opp_dists[1]) if len(sorted_opp_dists) > 1 else pitch_len
        features["opps_within_1yd"] = int(np.sum(sorted_opp_dists <= 1))
        features["opps_within_2yd"] = int(np.sum(sorted_opp_dists <= 2))
        features["opps_within_4yd"] = int(np.sum(sorted_opp_dists <= 4))

        nearest_opp = opps[np.argmin(opp_dists)]
        opp_vec = nearest_opp - bc
        opp_angle = np.arctan2(opp_vec[1], opp_vec[0])

        rel_angle = opp_angle - goal_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        features["angle_nearest_opp"] = float(rel_angle)
    else:
        features["dist_nearest_opp"] = pitch_len
        features["dist_2nd_nearest_opp"] = pitch_len
        features["opps_within_1yd"] = 0
        features["opps_within_2yd"] = 0
        features["opps_within_4yd"] = 0
        features["angle_nearest_opp"] = 0.0

    # pyrefly: ignore [bad-argument-type]
    features["coverage_arc"] = angular_span(bc, opps, radius=coverage_radius)
    # pyrefly: ignore [bad-argument-type]
    features["voronoi_area"] = voronoi_area(bc, all_players)

    # pyrefly: ignore [bad-argument-type]
    features["pitch_control"] = pitch_control_value(bc, tms, opps)

    if len(opps) > 0:
        features["opp_density_5yd"] = int(np.sum(np.linalg.norm(opps - bc, axis=1) <= 5.0))
    else:
        features["opp_density_5yd"] = 0

    max_tri_area = 0.0
    free_tms: list[np.ndarray] = []
    if len(opps) > 0 and len(tms) > 0:
        for tm in tms:
            dist_to_opps = np.linalg.norm(opps - tm, axis=1)
            if np.min(dist_to_opps) > SPATIAL_CONFIG["clear_pass_distance"]:
                free_tms.append(tm)
    else:
        free_tms = list(tms)

    features["n_free_teammates"] = len(free_tms)

    if len(free_tms) >= 2:
        for i in range(len(free_tms)):
            for j in range(i + 1, len(free_tms)):
                p1, p2 = free_tms[i], free_tms[j]
                area = 0.5 * abs(bc[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - bc[1]) + p2[0] * (bc[1] - p1[1]))
                if area > max_tri_area:
                    max_tri_area = area

    features["max_free_triangle_area"] = max_tri_area

    if len(free_tms) > 0:
        free_tms_arr = np.array(free_tms)
        tm_dists = np.linalg.norm(free_tms_arr - bc, axis=1)
        min_tm_idx = int(np.argmin(tm_dists))
        features["dist_nearest_free_teammate"] = float(tm_dists[min_tm_idx])

        tm_vec = free_tms_arr[min_tm_idx] - bc
        tm_angle = np.arctan2(tm_vec[1], tm_vec[0])
        rel_tm_angle = tm_angle - goal_angle
        rel_tm_angle = (rel_tm_angle + np.pi) % (2 * np.pi) - np.pi
        features["angle_nearest_free_teammate"] = float(rel_tm_angle)
    else:
        features["dist_nearest_free_teammate"] = pitch_len
        features["angle_nearest_free_teammate"] = 0.0

    # Progressive option: teammate closer to goal in higher xT, with unblocked lane.
    # has_progressive_option is a binary indicator (0 or 1). After StandardScaler
    # centering/scaling it becomes a continuous value, which is valid but means the
    # model coefficient represents a per-SD effect rather than a per-unit effect.
    features["has_progressive_option"] = 0
    bc_xt = xt_value(bc[0], bc[1])

    if len(free_tms) > 0:
        for tm in free_tms:
            tm_xt = xt_value(tm[0], tm[1])
            # Check if progressive (higher xT or strictly closer to goal line)
            if tm_xt > bc_xt or tm[0] > bc[0] + 5.0:
                # pyrefly: ignore [bad-argument-type]
                if lane_unblocked(bc, tm, opps):
                    features["has_progressive_option"] = 1
                    break

    features["xt_value"] = bc_xt

    # Raw coordinates retain non-linear pitch geography; zone integers imposed a false linear ordering
    features["bc_x"] = float(bc[0])
    features["bc_y"] = float(bc[1])

    if match_context:
        features["game_state_diff"] = match_context.get("game_state_diff", 0)
        features["minutes_elapsed"] = match_context.get("minutes_elapsed", 0)
        features["match_period"] = match_context.get("match_period", 1)
    else:
        features["game_state_diff"] = 0
        features["minutes_elapsed"] = 0
        features["match_period"] = 1

    return features
