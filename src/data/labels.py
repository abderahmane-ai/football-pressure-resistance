"""Define success labels for ball-carrier actions under pressure."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import SPATIAL_CONFIG

logger = logging.getLogger(__name__)


def define_success(
    events: pd.DataFrame,
    paired_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Define success for the ball-carrier based on the outcome of their event or subsequent events.
    Handles passes, carries, and dribbles safely, ensuring possession is truly retained.
    """
    if events.empty or not paired_results:
        return []

    if "index" in events.columns:
        events = events.sort_values(by=["match_id", "index"])
    else:
        events = events.sort_values(by=["match_id", "timestamp"])

    events = events.reset_index(drop=True)
    # pyrefly: ignore [bad-assignment]
    event_idx_map: dict[str, int] = {row["id"]: idx for idx, row in events.iterrows()}

    labeled_results: list[dict[str, Any]] = []

    for item in paired_results:
        bc_event_id: str = item["ball_carrier_event_id"]
        if bc_event_id not in event_idx_map:
            continue

        bc_idx: int = event_idx_map[bc_event_id]
        bc_event = events.iloc[bc_idx]

        success: float = np.nan

        if bc_event["type"] == "Pass":
            if pd.isna(bc_event.get("pass_outcome")):
                # NaN pass_outcome = complete in StatsBomb
                if pd.isna(bc_event.get("pass_recipient")) and bc_idx + 1 < len(events):
                    next_event = events.iloc[bc_idx + 1]
                    if next_event["team_id"] != bc_event["team_id"]:
                        success = 0.0
                    else:
                        success = 1.0
                else:
                    success = 1.0
            else:
                success = 0.0
        elif bc_event["type"] == "Dribble":
            if bc_event.get("dribble_outcome") == "Complete":
                success = 1.0
            else:
                success = 0.0
        elif bc_event["type"] == "Carry":
            # Default is NaN (unknown) — not 0.0 — so that carries whose
            # lookahead window contains only irrelevant event types (Ball
            # Receipt, Tactical Shift, etc.) are *excluded* rather than
            # silently labelled as failures.
            success = np.nan
            lookahead: int = SPATIAL_CONFIG["carry_lookahead_events"]
            for offset in range(1, min(lookahead + 1, len(events) - bc_idx)):
                next_event = events.iloc[bc_idx + offset]
                ev_type: str = next_event["type"]

                # Explicit possession-loss events — checked first because Dispossessed
                # is logged under the carrier's own team_id, which would otherwise
                # incorrectly satisfy the "same team acts next" condition below.
                if ev_type in ("Dispossessed", "Interception"):
                    success = 0.0
                    break
                # Opponent commits a foul on the carrier — free kick won
                elif ev_type == "Foul Committed" and next_event["team_id"] != bc_event["team_id"]:
                    success = 1.0
                    break
                # Opponent wins ball cleanly
                elif ev_type in ("Pass", "Carry", "Shot", "Clearance") and next_event["team_id"] != bc_event["team_id"]:
                    success = 0.0
                    break
                # Same team retains possession
                elif next_event["team_id"] == bc_event["team_id"]:
                    success = 1.0
                    break

        if not pd.isna(success):
            item["success"] = success
            labeled_results.append(item)

    return labeled_results
