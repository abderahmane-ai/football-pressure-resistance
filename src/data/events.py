"""Event and match state calculation utilities."""
from __future__ import annotations

import logging
import threading
import traceback
from typing import Any

import numpy as np
import pandas as pd

from config import SPATIAL_CONFIG
from src.common import (
    BALL_CARRIER_EVENT_TYPES,
    EVENT_TYPE_CARRY,
    EVENT_TYPE_DRIBBLE,
    EVENT_TYPE_OWN_GOAL_FOR,
    EVENT_TYPE_PASS,
    EVENT_TYPE_PRESSURE,
    EVENT_TYPE_SHOT,
    is_valid_loc,
)
from src.data.labels import define_success
from src.data.pairing import pair_pressure_with_ball_carrier
from src.features.geometry import xt_value
from src.features.spatial import extract_spatial_features_from_frame

_vaep_cache = threading.local()
_has_vaep: bool = True
try:
    from src.features.vaep import compute_vaep
except ImportError:
    _has_vaep = False
    compute_vaep = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
    is_shot = match_events["type"] == EVENT_TYPE_SHOT
    shot_outcome_col = match_events.get("shot_outcome")
    shot_outcome_name_col = match_events.get("shot_outcome_name")
    if shot_outcome_col is not None and shot_outcome_name_col is not None:
        shot_outcome = shot_outcome_col.fillna(shot_outcome_name_col)
    elif shot_outcome_col is not None:
        shot_outcome = shot_outcome_col
    else:
        shot_outcome = shot_outcome_name_col

    is_goal = is_shot & (shot_outcome == "Goal") if shot_outcome is not None else pd.Series(False, index=match_events.index)
    is_own_goal = match_events["type"] == EVENT_TYPE_OWN_GOAL_FOR

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
    Compute the intended value of the action, regardless of success.

    Uses VAEP if pre-trained models are available (preferred), otherwise
    falls back to discrete Expected Threat (xT).  This separates the value
    of the action from its outcome for the Hurdle model.
    """
    bc_event_id: str = item["ball_carrier_event_id"]

    # Try VAEP first (per-thread cache, safe with ThreadPoolExecutor)
    match_id = item.get("match_id")
    cache: dict[int, dict[str, float]] | None = getattr(_vaep_cache, "data", None)
    if cache is not None and match_id is not None:
        match_vaep = cache.get(match_id)
        if match_vaep is not None:
            result = match_vaep.get(bc_event_id)
            if result is not None:
                return float(result)

    # Fall back to xT
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
    if not is_valid_loc(bc_loc):
        if bc_idx > 0:
            if match_events_list is not None:
                prev_event = match_events_list[bc_idx - 1]
            else:
                prev_event = match_events.iloc[bc_idx - 1]
            end_loc = prev_event.get("end_location")
            prev_loc = prev_event.get("location")
            if is_valid_loc(end_loc):
                bc_loc = end_loc
            elif is_valid_loc(prev_loc):
                bc_loc = prev_loc
            else:
                return None
        else:
            return None

    bc_loc_seq: Any = bc_loc
    next_xt: float = xt_value(bc_loc_seq[0], bc_loc_seq[1])

    bc_event_type = bc_event.get("type")

    if bc_event_type == EVENT_TYPE_PASS:
        end_loc = bc_event.get("pass_end_location")
        if is_valid_loc(end_loc):
            end_loc_seq_pass: Any = end_loc
            next_xt = xt_value(end_loc_seq_pass[0], end_loc_seq_pass[1])
    elif bc_event_type == EVENT_TYPE_CARRY:
        end_loc = bc_event.get("carry_end_location")
        if is_valid_loc(end_loc):
            end_loc_seq_carry: Any = end_loc
            next_xt = xt_value(end_loc_seq_carry[0], end_loc_seq_carry[1])
    elif bc_event_type == EVENT_TYPE_DRIBBLE:
        length = len(match_events_list) if match_events_list is not None else len(match_events)
        if bc_idx + 1 < length:
            if match_events_list is not None:
                next_loc = match_events_list[bc_idx + 1].get("location")
            else:
                next_loc = match_events.iloc[bc_idx + 1].get("location")
            if is_valid_loc(next_loc):
                next_loc_seq_dribble: Any = next_loc
                next_xt = xt_value(next_loc_seq_dribble[0], next_loc_seq_dribble[1])
    else:
        length = len(match_events_list) if match_events_list is not None else len(match_events)
        if bc_idx + 1 < length:
            if match_events_list is not None:
                next_loc = match_events_list[bc_idx + 1].get("location")
            else:
                next_loc = match_events.iloc[bc_idx + 1].get("location")
            if is_valid_loc(next_loc):
                next_loc_seq_else: Any = next_loc
                next_xt = xt_value(next_loc_seq_else[0], next_loc_seq_else[1])

    return float(next_xt)


# ── Private helpers for `process_single_match` ────────────────────────────────


def _precompute_match_lookups(
    match_events: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[int, str], list[dict[str, Any]]]:
    """Build O(1) lookup structures used by the per-match worker loop."""
    event_lookup: dict[str, dict[str, Any]] = {}
    if "id" in match_events.columns:
        event_lookup = match_events.set_index("id").to_dict(orient="index")

    id_to_idx: dict[str, int] = {
        eid: idx for idx, eid in enumerate(match_events["id"]) if isinstance(eid, str)
    }

    player_name_lookup: dict[int, str] = {}
    if "player_id" in match_events.columns and "player" in match_events.columns:
        player_name_lookup = (
            match_events.dropna(subset=["player_id", "player"])
            .drop_duplicates("player_id")
            .set_index("player_id")["player"]
            .to_dict()
        )

    match_events_list: list[dict[str, Any]] = match_events.to_dict(orient="records")
    return event_lookup, id_to_idx, player_name_lookup, match_events_list


def _precompute_vaep(match_events: pd.DataFrame, match_id: int) -> None:
    """Compute VAEP for all events in this match and store in the thread-local cache."""
    if not hasattr(_vaep_cache, "data"):
        _vaep_cache.data = {}
    _vaep_cache.data.clear()
    if _has_vaep and compute_vaep is not None:
        try:
            vaep_values = compute_vaep(match_events)
            if vaep_values is not None and "id" in match_events.columns:
                _vaep_cache.data[match_id] = dict(zip(match_events["id"], vaep_values))
        except Exception:
            logger.debug("VAEP computation failed for match %d, falling back to xT", match_id)


def _precompute_recent_pressures(
    match_events_list: list[dict[str, Any]],
    pressure_bc_ids: set[str],
) -> dict[str, int]:
    """Per-player rolling count of pressured actions among the last 5 touches."""
    player_carrier_events: dict[int, list[dict[str, Any]]] = {}
    for ev in match_events_list:
        if ev.get("type") in BALL_CARRIER_EVENT_TYPES:
            pid = ev.get("player_id")
            if pid is not None:
                player_carrier_events.setdefault(pid, []).append(ev)

    recent_pressures_lookup: dict[str, int] = {}
    for evs in player_carrier_events.values():
        for idx, ev in enumerate(evs):
            start_idx = max(0, idx - 5)
            prior_evs = evs[start_idx:idx]
            n_press = sum(1 for pev in prior_evs if pev.get("id") in pressure_bc_ids)
            recent_pressures_lookup[ev["id"]] = n_press
    return recent_pressures_lookup


def _compute_match_context(
    carrier_event_id: str,
    pressure_event_id: str,
    event_lookup: dict[str, dict[str, Any]],
    game_states: dict[str, int],
    recent_pressures_lookup: dict[str, int],
) -> dict[str, Any]:
    """Build the match_context dict for a single pressure-carrier pair."""
    ev = event_lookup.get(carrier_event_id)
    press_ev = event_lookup.get(pressure_event_id)
    ctx: dict[str, Any] = {}

    if ev is not None:
        ctx["minutes_elapsed"] = ev.get("minute", 0)
        ctx["match_period"] = ev.get("period", 1)
        ctx["game_state_diff"] = game_states.get(carrier_event_id, 0)

    ctx["counter_press"] = bool(press_ev.get("counterpress", False)) if press_ev else False
    ctx["recent_pressures"] = recent_pressures_lookup.get(carrier_event_id, 0)

    # Pass height (categorical: Ground/Low/High)
    ph_id = 0
    if ev is not None and ev.get("type") == EVENT_TYPE_PASS:
        pass_info = ev.get("pass")
        if isinstance(pass_info, dict):
            height_info = pass_info.get("height")
            if isinstance(height_info, dict):
                ph_id = int(height_info.get("id", 0))
    ctx["pass_height_id"] = ph_id

    return ctx


def process_single_match(
    args: tuple[int, pd.DataFrame, pd.DataFrame, set[int], dict[Any, str], str],
) -> list[dict[str, Any]]:
    """Module-level worker for parallel match processing.

    Args tuple: (match_id, match_events, frames_df, gk_ids, position_groups, comp_name)
    """
    match_id, match_events, frames_df, gk_ids, position_groups, comp_name = args
    rows: list[dict[str, Any]] = []
    n_labeled = 0
    n_features_ok = 0
    n_dist_ok = 0
    n_xt_ok = 0

    try:
        game_states = compute_game_state_for_match(match_events)
        n_pressure = len(match_events[match_events["type"] == EVENT_TYPE_PRESSURE])
        logger.debug("Match %d: events=%d, pressure_events=%d", match_id, len(match_events), n_pressure)

        paired_events = pair_pressure_with_ball_carrier(match_events, frames_df)
        logger.debug("Match %d: paired_events=%d", match_id, len(paired_events))

        labeled_events = define_success(match_events, paired_events)
        logger.debug("Match %d: labeled_events=%d", match_id, len(labeled_events))
        n_labeled = len(labeled_events)

        # ── Build lookups ────────────────────────────────────────────────────
        event_lookup, id_to_idx, player_name_lookup, match_events_list = (
            _precompute_match_lookups(match_events)
        )
        _precompute_vaep(match_events, match_id)

        pressure_bc_ids: set[str] = {item["ball_carrier_event_id"] for item in labeled_events}
        recent_pressures_lookup = _precompute_recent_pressures(
            match_events_list, pressure_bc_ids,
        )

        # ── Process each labelled event ──────────────────────────────────────
        for item in labeled_events:
            player_id = item.get("player_id")
            if player_id is None or player_id in gk_ids:
                continue

            match_context = _compute_match_context(
                item["ball_carrier_event_id"],
                item["pressure_event_id"],
                event_lookup,
                game_states,
                recent_pressures_lookup,
            )

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

            intended_xt = compute_intended_xt(
                item, match_events, id_to_idx=id_to_idx, match_events_list=match_events_list,
            )
            if intended_xt is None:
                continue
            n_xt_ok += 1

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
                "position_group": position_groups.get(player_id, "CM"),
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
        logger.debug(
            "Match %d: n_labeled=%d, n_features_ok=%d, n_dist_ok=%d, n_xt_ok=%d",
            match_id, n_labeled, n_features_ok, n_dist_ok, n_xt_ok,
        )
        logger.debug(
            "Match %d (%s): worker produced 0 rows — all events filtered or no pressure events found",
            match_id, comp_name,
        )
    return rows
