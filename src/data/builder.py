"""Build processed pressure datasets from raw StatsBomb data."""
from __future__ import annotations

import hashlib
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from statsbombpy import sb
from tqdm import tqdm

from config import (
    COMPETITIONS,
    CROSS_VALIDATION_HOLDOUT,
    MODEL_FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    SPATIAL_CONFIG,
)
from src.data.labels import define_success
from src.data.loader import load_all_competitions
from src.data.pairing import pair_pressure_with_ball_carrier
from src.data.validation import (
    validate_model_dataset,
    validate_statsbomb_events,
    validate_statsbomb_frames,
)
from src.features.geometry import xt_value
from src.features.spatial import extract_spatial_features_from_frame

logger = logging.getLogger(__name__)

N_WORKERS: int = min(os.cpu_count() or 4, 8)


# ── Data versioning ──────────────────────────────────────────────────────────

def _dataframe_hash(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 digest of a DataFrame's content for provenance tracking."""
    h = hashlib.sha256()
    # Hash column names (sorted for stability across column order)
    h.update(",".join(sorted(df.columns)).encode())
    # Hash shape
    h.update(f"{df.shape}".encode())
    # Hash a sample of the actual data (full hash is too slow for 200k+ rows)
    sample = df.head(500).to_csv(index=False).encode()
    h.update(sample)
    tail = df.tail(100).to_csv(index=False).encode()
    h.update(tail)
    return h.hexdigest()


def _save_parquet_with_metadata(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    *,
    source_hash: str,
    holdout: str,
    n_competitions: int,
) -> None:
    """Save a DataFrame as parquet with provenance metadata in the file footer."""
    table = pa.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    extra = {
        b"prs.source_hash": source_hash.encode(),
        b"prs.holdout": holdout.encode(),
        b"prs.n_competitions": str(n_competitions).encode(),
        b"prs.n_events": str(len(df)).encode(),
        b"prs.features": ",".join(MODEL_FEATURE_COLUMNS).encode(),
    }
    merged = {**existing_meta, **extra}
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, str(path))


# ── Match processing helpers ──────────────────────────────────────────────────

def compute_game_state_for_match(match_events: pd.DataFrame) -> dict[str, int]:
    """
    Compute score differential (ball-carrier team − opponent) at the moment
    of every event in the match, using StatsBomb Shot/Own Goal events.
    Returns dict: event_id → int score diff.
    """
    if "index" in match_events.columns:
        match_events = match_events.sort_values(by="index")
    elif "timestamp" in match_events.columns:
        match_events = match_events.sort_values(by="timestamp")

    teams = match_events["team_id"].dropna().unique()
    if len(teams) != 2:
        return {}
    team_a, team_b = teams[0], teams[1]
    score: dict[Any, int] = {team_a: 0, team_b: 0}
    event_state: dict[str, int] = {}

    for _, row in match_events.iterrows():
        event_id: str = row["id"]
        team = row.get("team_id")
        opp = team_b if team == team_a else team_a
        event_state[event_id] = score.get(team, 0) - score.get(opp, 0)

        ev_type = row.get("type", "")
        # Fallback handles both statsbombpy column name variants
        shot_outcome = row.get("shot_outcome") or row.get("shot_outcome_name", "")
        if ev_type == "Shot" and shot_outcome == "Goal":
            score[team] = score.get(team, 0) + 1
        elif ev_type == "Own Goal For":
            score[team] = score.get(team, 0) + 1

    return event_state


def _process_single_match(args: tuple[int, pd.DataFrame, pd.DataFrame, set[int], str]) -> list[dict[str, Any]]:
    """
    Module-level worker for parallel match processing.
    Returns a list of processed row dicts for one match.
    """
    match_id, match_events, frames_df, gk_ids, comp_name = args
    rows: list[dict[str, Any]] = []
    try:
        position_groups = get_player_position_groups(match_id, match_events)
        game_states = compute_game_state_for_match(match_events)
        paired_events = pair_pressure_with_ball_carrier(match_events, frames_df)
        labeled_events = define_success(match_events, paired_events)

        for item in labeled_events:
            player_id = item.get("player_id")
            if player_id in gk_ids:
                continue

            event_row = match_events[match_events["id"] == item["ball_carrier_event_id"]]
            match_context: dict[str, Any] = {}
            if not event_row.empty:
                ev = event_row.iloc[0]
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
            if features.get("dist_nearest_opp", 999) > SPATIAL_CONFIG["tight_pressure_radius"]:
                continue

            intended_xt = compute_intended_xt(item, match_events)
            if intended_xt is None:
                continue

            player_name = (
                match_events[match_events["player_id"] == player_id]["player"].iloc[0]
                if player_id in match_events["player_id"].values
                else f"Player_{player_id}"
            )
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
        logger.warning(
            "Match %d (%s): worker produced 0 rows — all events filtered or no pressure events found",
            match_id, comp_name,
        )
    return rows


def get_goalkeeper_ids(match_id: int) -> set[int]:
    """Get player IDs of goalkeepers from match lineups."""
    try:
        lineups = sb.lineups(match_id=match_id)
        gk_ids: set[int] = set()
        for team_name, lineup_df in lineups.items():
            if "positions" in lineup_df.columns:
                for _, player in lineup_df.iterrows():
                    positions = player["positions"]
                    if isinstance(positions, list):
                        for pos_dict in positions:
                            if isinstance(pos_dict, dict) and pos_dict.get("position") == "Goalkeeper":
                                gk_ids.add(player["player_id"])
                                break
            elif "player_position" in lineup_df.columns:
                gks = lineup_df[lineup_df["player_position"] == "Goalkeeper"]
                gk_ids.update(gks["player_id"].values)
        return gk_ids
    except Exception as e:
        logger.debug("Could not load lineups for match %d: %s", match_id, e)
        return set()


def get_player_position_groups(
    match_id: int,
    match_events: pd.DataFrame | None = None,
) -> dict[int, str]:
    """
    Get position group (Defender/Midfielder/Forward) for each player.
    Uses lineup data, and falls back to coordinate clustering if lineup is missing.
    """
    position_map: dict[int, str] = {}
    try:
        lineups = sb.lineups(match_id=match_id)
        for team_name, lineup_df in lineups.items():
            if "positions" in lineup_df.columns:
                for _, player in lineup_df.iterrows():
                    player_id: int = player["player_id"]
                    positions = player["positions"]

                    if isinstance(positions, list) and len(positions) > 0:
                        # Check all listed positions for the player
                        assigned = False
                        for pos_dict in positions:
                            if isinstance(pos_dict, dict):
                                pos_name: str = pos_dict.get("position", "").lower()
                                if any(x in pos_name for x in ["back", "defender", "wing back"]):
                                    position_map[player_id] = "Defender"
                                    assigned = True
                                    break
                                elif any(x in pos_name for x in ["forward", "striker", "wing", "winger"]):
                                    position_map[player_id] = "Forward"
                                    assigned = True
                                    break
                                elif "midfield" in pos_name:
                                    position_map[player_id] = "Midfielder"
                                    assigned = True
                                    break
                        if not assigned:
                            position_map[player_id] = "Midfielder"
    except Exception as e:
        logger.debug("Could not load lineups for match %d: %s", match_id, e)

    # Impute missing using event locations
    if match_events is not None:
        all_players = match_events["player_id"].dropna().unique()
        for pid in all_players:
            if pid not in position_map:
                player_events = match_events[
                    (match_events["player_id"] == pid) & (match_events["location"].notna())
                ]
                if not player_events.empty:
                    # Calculate average x coordinate (0 to 120)
                    locs = np.array(player_events["location"].tolist())
                    avg_x: float = float(np.mean(locs[:, 0]))
                    third: float = SPATIAL_CONFIG["pitch_length"] / 3.0
                    if avg_x < third:
                        position_map[pid] = "Defender"
                    elif avg_x > 2 * third:
                        position_map[pid] = "Forward"
                    else:
                        position_map[pid] = "Midfielder"
                else:
                    position_map[pid] = "Midfielder"

    return position_map


def _is_valid_loc(loc: Any) -> bool:
    """Accept list, tuple, or numpy array with at least 2 elements."""
    return loc is not None and hasattr(loc, "__len__") and len(loc) >= 2


def compute_intended_xt(item: dict[str, Any], match_events: pd.DataFrame) -> float | None:
    """
    Compute the intended expected threat (xT) of the action, regardless of success.
    This separates the value of the action from its outcome for the Hurdle model.
    """
    bc_event_id: str = item["ball_carrier_event_id"]
    bc_event_rows = match_events[match_events["id"] == bc_event_id]

    if bc_event_rows.empty:
        return None

    bc_event = bc_event_rows.iloc[0]
    bc_idx: int = bc_event_rows.index[0]

    bc_loc = bc_event.get("location")
    if not _is_valid_loc(bc_loc):
        # Impute from previous event
        if bc_idx > 0:
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

    # pyrefly: ignore [unsupported-operation]
    next_xt: float = xt_value(bc_loc[0], bc_loc[1])

    if bc_event["type"] == "Pass":
        end_loc = bc_event.get("pass_end_location")
        if _is_valid_loc(end_loc):
            # pyrefly: ignore [unsupported-operation]
            next_xt = xt_value(end_loc[0], end_loc[1])
    elif bc_event["type"] == "Carry":
        end_loc = bc_event.get("carry_end_location")
        if _is_valid_loc(end_loc):
            # pyrefly: ignore [unsupported-operation]
            next_xt = xt_value(end_loc[0], end_loc[1])
    elif bc_idx + 1 < len(match_events):
        next_loc = match_events.iloc[bc_idx + 1].get("location")
        if _is_valid_loc(next_loc):
            # pyrefly: ignore [unsupported-operation]
            next_xt = xt_value(next_loc[0], next_loc[1])

    return float(next_xt)


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_all_datasets(include_holdout: bool = False) -> pd.DataFrame | None:
    """Build the complete processed dataset for all competitions (parallelised per match)."""
    logger.info("Building dataset for all competitions...")

    # Skip rebuild if a per-holdout cached parquet already exists. Raw data
    # is content-addressed by the upstream StatsBomb API; if the raw cache
    # is unchanged, the processed dataset is byte-equivalent and rebuilding
    # it costs 15-30 min for no gain.
    out_file = PROCESSED_DATA_DIR / f"all_pressure_dataset_{CROSS_VALIDATION_HOLDOUT}.parquet"
    if out_file.exists() and not os.environ.get("FORCE_REBUILD_DATA", "") == "1":
        logger.info(
            "Found cached training dataset at %s. Skipping rebuild "
            "(set FORCE_REBUILD_DATA=1 to override).",
            out_file,
        )
        return pd.read_parquet(out_file)

    comp_names = list(COMPETITIONS.keys())
    if not include_holdout and CROSS_VALIDATION_HOLDOUT in comp_names:
        comp_names.remove(CROSS_VALIDATION_HOLDOUT)
        logger.info("Excluding holdout competition: %s", CROSS_VALIDATION_HOLDOUT)

    all_comp_data = load_all_competitions(comp_names)
    all_processed_data: list[dict[str, Any]] = []
    comp_event_counts: dict[str, int] = {}

    for comp_name, comp_data in all_comp_data.items():
        logger.info("Processing %s with %d workers...", comp_name, N_WORKERS)
        events_df: pd.DataFrame = comp_data["events"]
        frames_dict: dict[int, pd.DataFrame] = comp_data["frames"]
        if events_df.empty:
            continue
        validate_statsbomb_events(events_df, context=f"{comp_name} events")
        for match_id, frames_df in frames_dict.items():
            validate_statsbomb_frames(frames_df, context=f"{comp_name} match {match_id} frames")

        match_ids = events_df["match_id"].unique()

        # Pre-fetch goalkeeper IDs (I/O-bound API calls parallelised with threads)
        gk_ids_by_match: dict[int, set[int]] = {}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(get_goalkeeper_ids, mid): mid for mid in match_ids}
            for fut in tqdm(futures, desc=f"Loading lineups ({comp_name})"):
                gk_ids_by_match[futures[fut]] = fut.result()

        # Build args list, skipping matches with no 360 data
        worker_args = [
            (mid,
             events_df[events_df["match_id"] == mid].copy().reset_index(drop=True),
             frames_dict[mid],
             gk_ids_by_match.get(mid, set()),
             comp_name)
            for mid in match_ids
            if mid in frames_dict and not frames_dict[mid].empty
        ]

        comp_events = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for batch in tqdm(
                ex.map(_process_single_match, worker_args),
                total=len(worker_args),
                desc=f"Processing matches ({comp_name})",
            ):
                all_processed_data.extend(batch)
                comp_events += len(batch)

        comp_event_counts[comp_name] = comp_events
        logger.info("%s: %d events processed.", comp_name, comp_events)

    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        validate_model_dataset(dataset_df, MODEL_FEATURE_COLUMNS, context="training pressure dataset")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Key the training dataset by holdout so the 4-fold CV cache survives
        # across folds without overwriting an in-use file.
        source_hash = _dataframe_hash(dataset_df)
        _save_parquet_with_metadata(
            dataset_df,
            out_file,
            source_hash=source_hash,
            holdout=CROSS_VALIDATION_HOLDOUT,
            n_competitions=len(comp_names),
        )
        logger.info(
            "Saved dataset with %d events to %s (hash=%s)",
            len(dataset_df), out_file, source_hash[:12],
        )
        return dataset_df
    return None


def build_holdout_dataset() -> None:
    """Build dataset for holdout competition only (parallelised per match)."""
    logger.info("Building holdout dataset: %s", CROSS_VALIDATION_HOLDOUT)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DATA_DIR / f"holdout_pressure_dataset_{CROSS_VALIDATION_HOLDOUT}.parquet"
    if out_file.exists() and not os.environ.get("FORCE_REBUILD_DATA", "") == "1":
        logger.info(
            "Found cached holdout dataset at %s. Skipping rebuild "
            "(set FORCE_REBUILD_DATA=1 to override).",
            out_file,
        )
        return

    all_comp_data = load_all_competitions([CROSS_VALIDATION_HOLDOUT])
    all_processed_data: list[dict[str, Any]] = []

    for comp_name, comp_data in all_comp_data.items():
        events_df: pd.DataFrame = comp_data["events"]
        frames_dict: dict[int, pd.DataFrame] = comp_data["frames"]
        if events_df.empty:
            continue
        validate_statsbomb_events(events_df, context=f"{comp_name} events")
        for match_id, frames_df in frames_dict.items():
            validate_statsbomb_frames(frames_df, context=f"{comp_name} match {match_id} frames")

        match_ids = events_df["match_id"].unique()

        # Pre-fetch GK IDs (I/O-bound API calls parallelised with threads)
        gk_ids_by_match: dict[int, set[int]] = {}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(get_goalkeeper_ids, mid): mid for mid in match_ids}
            for fut in tqdm(futures, desc="Loading holdout lineups"):
                gk_ids_by_match[futures[fut]] = fut.result()

        worker_args = [
            (mid,
             events_df[events_df["match_id"] == mid].copy().reset_index(drop=True),
             frames_dict[mid],
             gk_ids_by_match.get(mid, set()),
             comp_name)
            for mid in match_ids
            if mid in frames_dict and not frames_dict[mid].empty
        ]

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for batch in tqdm(
                ex.map(_process_single_match, worker_args),
                total=len(worker_args),
                desc="Processing holdout matches",
            ):
                all_processed_data.extend(batch)

    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        validate_model_dataset(dataset_df, MODEL_FEATURE_COLUMNS, context="holdout pressure dataset")
        source_hash = _dataframe_hash(dataset_df)
        _save_parquet_with_metadata(
            dataset_df,
            out_file,
            source_hash=source_hash,
            holdout=CROSS_VALIDATION_HOLDOUT,
            n_competitions=1,
        )
        logger.info(
            "Saved holdout with %d events to %s (hash=%s)",
            len(dataset_df), out_file, source_hash[:12],
        )


if __name__ == "__main__":
    build_all_datasets(include_holdout=False)
    build_holdout_dataset()
