"""Event and match state calculation utilities."""
from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np
import pandas as pd

from config import SPATIAL_CONFIG
from src.data.labels import define_success
from src.data.pairing import pair_pressure_with_ball_carrier
from src.features.geometry import xt_value
from src.features.spatial import extract_spatial_features_from_frame

logger = logging.getLogger(__name__)


def _is_valid_loc(loc: Any) -> bool:
    """Accept list, tuple, or numpy array with at least 2 elements."""
    return loc is not None and hasattr(loc, "__len__") and len(loc) >= 2


def compute_game_state_for_match(match_events: pd.DataFrame) -> dict[str, int]:
    """
    Compute score differential (ball-carrier team − opponent) at the moment
    of every event in the match, using StatsBomb Shot/Own Goal events.
    Returns dict: event_id → int score diff.
    """
    if match_events.empty:
        return {}

    if "index" in match_events.columns:
        match_events = match_events.sort_values(by="index")
    elif "timestamp" in match_events.columns:
        match_events = match_events.sort_values(by="timestamp")

    teams = match_events["team_id"].dropna().unique()
    if len(teams) != 2:
        return {}
    team_a, team_b = teams[0], teams[1]

    # Vectorized check of goals scored
    is_shot = match_events["type"] == "Shot"
    shot_outcome_col = match_events.get("shot_outcome")
    shot_outcome_name_col = match_events.get("shot_outcome_name")
    if shot_outcome_col is not None and shot_outcome_name_col is not None:
        shot_outcome = shot_outcome_col.fillna(shot_outcome_name_col)
    elif shot_outcome_col is not None:
        shot_outcome = shot_outcome_col
    else:
        shot_outcome = shot_outcome_name_col

    is_goal = is_shot & (shot_outcome == "Goal") if shot_outcome is not None else pd.Series(False, index=match_events.index)
    is_own_goal = match_events["type"] == "Own Goal For"

    goal_a = (is_goal | is_own_goal) & (match_events["team_id"] == team_a)
    goal_b = (is_goal | is_own_goal) & (match_events["team_id"] == team_b)

    # Calculate cumulative goals before the current event (hence shift(1))
    goals_a = goal_a.cumsum().shift(fill_value=0)
    goals_b = goal_b.cumsum().shift(fill_value=0)

    score_diff_a = goals_a - goals_b
    score_diff = np.where(
        match_events["team_id"] == team_a,
        score_diff_a,
        np.where(match_events["team_id"] == team_b, -score_diff_a, 0)
    )

    return dict(zip(match_events["id"], score_diff.astype(int)))


def compute_intended_xt(
    item: dict[str, Any],
    match_events: pd.DataFrame,
    id_to_idx: dict[str, int] | None = None,
    match_events_list: list[dict[str, Any]] | None = None,
) -> float | None:
    """
    Compute the intended expected threat (xT) of the action, regardless of success.
    This separates the value of the action from its outcome for the Hurdle model.
    """
    bc_event_id: str = item["ball_carrier_event_id"]

    if id_to_idx is not None and (match_events_list is not None or not match_events.empty):
        bc_idx = id_to_idx.get(bc_event_id)
        if bc_idx is None:
            return None
        if match_events_list is not None:
            bc_event = match_events_list[bc_idx]
        else:
            bc_event = match_events.iloc[bc_idx]
    else:
        bc_event_rows = match_events[match_events["id"] == bc_event_id]
        if bc_event_rows.empty:
            return None
        bc_event = bc_event_rows.iloc[0]
        bc_idx = bc_event_rows.index[0]

    bc_loc = bc_event.get("location")
    if not _is_valid_loc(bc_loc):
        # Impute from previous event
        if bc_idx > 0:
            if match_events_list is not None:
                prev_event = match_events_list[bc_idx - 1]
            else:
                prev_event = match_events.iloc[bc_idx - 1]
            end_loc = prev_event.get("end_location")
            prev_loc = prev_event.get("location")
            if _is_valid_loc(end_loc):
                bc_loc = end_loc
            elif _is_valid_loc(prev_loc):
                bc_loc = prev_loc
            else:
                return None
        else:
            return None

    bc_loc_seq: Any = bc_loc
    next_xt: float = xt_value(bc_loc_seq[0], bc_loc_seq[1])

    bc_event_type = bc_event.get("type")

    if bc_event_type == "Pass":
        end_loc = bc_event.get("pass_end_location")
        if _is_valid_loc(end_loc):
            end_loc_seq_pass: Any = end_loc
            next_xt = xt_value(end_loc_seq_pass[0], end_loc_seq_pass[1])
    elif bc_event_type == "Carry":
        end_loc = bc_event.get("carry_end_location")
        if _is_valid_loc(end_loc):
            end_loc_seq_carry: Any = end_loc
            next_xt = xt_value(end_loc_seq_carry[0], end_loc_seq_carry[1])
    elif bc_event_type == "Dribble":
        length = len(match_events_list) if match_events_list is not None else len(match_events)
        if bc_idx + 1 < length:
            if match_events_list is not None:
                next_loc = match_events_list[bc_idx + 1].get("location")
            else:
                next_loc = match_events.iloc[bc_idx + 1].get("location")
            if _is_valid_loc(next_loc):
                next_loc_seq_dribble: Any = next_loc
                next_xt = xt_value(next_loc_seq_dribble[0], next_loc_seq_dribble[1])
    else:
        length = len(match_events_list) if match_events_list is not None else len(match_events)
        if bc_idx + 1 < length:
            if match_events_list is not None:
                next_loc = match_events_list[bc_idx + 1].get("location")
            else:
                next_loc = match_events.iloc[bc_idx + 1].get("location")
            if _is_valid_loc(next_loc):
                next_loc_seq_else: Any = next_loc
                next_xt = xt_value(next_loc_seq_else[0], next_loc_seq_else[1])

    return float(next_xt)


def _process_single_match(
    args: tuple[int, pd.DataFrame, pd.DataFrame, set[int], dict[Any, str], str],
) -> list[dict[str, Any]]:
    """
    Module-level worker for parallel match processing.
    Returns a list of processed row dicts for one match.

    Args tuple: (match_id, match_events, frames_df, gk_ids, position_groups, comp_name)
    """
    match_id, match_events, frames_df, gk_ids, position_groups, comp_name = args
    rows: list[dict[str, Any]] = []
    try:
        game_states = compute_game_state_for_match(match_events)
        n_pressure = len(match_events[match_events["type"] == "Pressure"])
        logger.debug("Match %d: events=%d, pressure_events=%d", match_id, len(match_events), n_pressure)
        paired_events = pair_pressure_with_ball_carrier(match_events, frames_df)
        logger.debug("Match %d: paired_events=%d", match_id, len(paired_events))
        labeled_events = define_success(match_events, paired_events)
        logger.debug("Match %d: labeled_events=%d", match_id, len(labeled_events))

        # Convert DataFrame to list of dict records once per match for fast C-level loop lookups
        match_events_list = match_events.to_dict(orient="records")

        # Build O(1) event lookup (dict of dicts)
        event_lookup: dict[str, dict[str, Any]] = {}
        if "id" in match_events.columns:
            event_lookup = match_events.set_index("id").to_dict(orient="index")

        # Build O(1) event index lookup
        id_to_idx: dict[str, int] = {eid: idx for idx, eid in enumerate(match_events["id"]) if isinstance(eid, str)}

        # Precompute player name lookup once per match
        player_name_lookup: dict[int, str] = {}
        if "player_id" in match_events.columns and "player" in match_events.columns:
            player_name_lookup = (
                match_events.dropna(subset=["player_id", "player"])
                .drop_duplicates("player_id")
                .set_index("player_id")["player"]
                .to_dict()
            )

        n_labeled = len(labeled_events)
        n_features_ok = 0
        n_dist_ok = 0
        n_xt_ok = 0

        for item in labeled_events:
            player_id = item.get("player_id")
            if player_id is None or player_id in gk_ids:
                continue

            ev = event_lookup.get(item["ball_carrier_event_id"])
            match_context: dict[str, Any] = {}
            if ev is not None:
                if "minute" in ev:
                    match_context["minutes_elapsed"] = ev["minute"]
                if "period" in ev:
                    match_context["match_period"] = ev["period"]
                match_context["game_state_diff"] = game_states.get(item["ball_carrier_event_id"], 0)

            features = extract_spatial_features_from_frame(
                frame_data=item["frame_data"],
                ball_carrier_player_id=player_id,
                team_id=item["team_id"],
                opponent_team_id=item["opponent_team_id"],
                match_context=match_context,
            )
            if features is None:
                continue
            n_features_ok += 1
            if features.get("dist_nearest_opp", 999) > SPATIAL_CONFIG["tight_pressure_radius"]:
                continue
            n_dist_ok += 1

            intended_xt = compute_intended_xt(item, match_events, id_to_idx=id_to_idx, match_events_list=match_events_list)
            if intended_xt is None:
                continue
            n_xt_ok += 1

            # Player name lookup via pre-caching
            player_name = player_name_lookup.get(player_id, f"Player_{player_id}")

            row: dict[str, Any] = {
                "competition": comp_name,
                "match_id": item["match_id"],
                "pressure_event_id": item["pressure_event_id"],
                "pressure_event_ids": item.get("pressure_event_ids", [item["pressure_event_id"]]),
                "n_pressure_events": item.get("n_pressure_events", 1),
                "ball_carrier_event_id": item["ball_carrier_event_id"],
                "player_id": player_id,
                "player_name": player_name,
                "position_group": position_groups.get(player_id, "Midfielder"),
                "team_id": item["team_id"],
                "opponent_team_id": item["opponent_team_id"],
                "success": item["success"],
                "value_preserved": intended_xt,
            }
            row.update(features)
            rows.append(row)
    except Exception as e:
        logger.warning(
            "Match %d (%s) worker failed: %s\n%s",
            match_id, comp_name, e, traceback.format_exc(),
        )
    if not rows:
        logger.debug("Match %d: n_labeled=%d, n_features_ok=%d, n_dist_ok=%d, n_xt_ok=%d", match_id, n_labeled, n_features_ok, n_dist_ok, n_xt_ok)
        logger.warning(
            "Match %d (%s): worker produced 0 rows — all events filtered or no pressure events found",
            match_id, comp_name,
        )
    return rows
