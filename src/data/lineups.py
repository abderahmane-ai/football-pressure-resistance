"""Lineup and position-related utilities for StatsBomb data."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsbombpy import sb

from config import SPATIAL_CONFIG

logger = logging.getLogger(__name__)


def _fetch_lineups(match_id: int) -> dict[str, pd.DataFrame]:
    """Fetch lineups for a match, returning an empty dict on failure."""
    try:
        return sb.lineups(match_id=match_id)  # type: ignore[no-any-return]
    except Exception as e:
        logger.debug("Could not load lineups for match %d: %s", match_id, e)
        return {}


def get_goalkeeper_ids_from_lineups(lineups: dict[str, pd.DataFrame]) -> set[int]:
    """Extract goalkeeper player IDs from pre-fetched lineup data."""
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


def get_player_position_groups_from_lineups(
    lineups: dict[str, pd.DataFrame],
    match_events: pd.DataFrame | None = None,
) -> dict[int, str]:
    """
    Get position group (Defender/Midfielder/Forward) for each player.
    Uses pre-fetched lineup data, and falls back to coordinate clustering
    if a player is missing from the lineups.
    """
    position_map: dict[int, str] = {}
    for team_name, lineup_df in lineups.items():
        if "positions" in lineup_df.columns:
            for _, player in lineup_df.iterrows():
                player_id: int = player["player_id"]
                positions = player["positions"]

                if isinstance(positions, list) and len(positions) > 0:
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

    # Impute missing using event locations (filter to open-play events
    # to avoid skewing from goal kicks, throw-ins, etc.)
    if match_events is not None:
        open_play_types = {"Pass", "Carry", "Dribble", "Ball Receipt*", "Shot"}
        all_players = match_events["player_id"].dropna().unique()
        for pid in all_players:
            if pid not in position_map:
                player_events = match_events[
                    (match_events["player_id"] == pid)
                    & (match_events["location"].notna())
                    & (match_events["type"].isin(open_play_types))
                ]
                if player_events.empty:
                    player_events = match_events[
                        (match_events["player_id"] == pid) & (match_events["location"].notna())
                    ]
                if not player_events.empty:
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
