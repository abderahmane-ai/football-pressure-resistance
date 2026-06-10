"""Pair StatsBomb Pressure events with the related ball-carrier action and 360 frame."""
from __future__ import annotations

import ast
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BALL_CARRIER_EVENT_TYPES: frozenset[str] = frozenset({"Pass", "Carry", "Dribble"})


def _normalise_related_events(value: Any) -> list[str]:
    """Return related event ids as a plain list across parquet/API variants."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        return []
    return [event_id for event_id in value if isinstance(event_id, str)]


def dedupe_pressure_events_by_carrier(
    paired_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Collapse multiple defender Pressure rows linked to the same carrier action.

    The modeling unit is the ball-carrier's action under pressure, not one row per
    pressing defender. Freeze-frame features already encode how many defenders
    are close, so repeated rows would overweight the same outcome.
    """
    by_carrier: dict[str, dict[str, Any]] = {}
    for item in paired_events:
        carrier_id: str | None = item.get("ball_carrier_event_id")
        if carrier_id is None:
            continue

        if carrier_id not in by_carrier:
            item = item.copy()
            item["pressure_event_ids"] = [item["pressure_event_id"]]
            item["n_pressure_events"] = 1
            by_carrier[carrier_id] = item
        else:
            existing = by_carrier[carrier_id]
            pressure_id = item.get("pressure_event_id")
            if pressure_id not in existing["pressure_event_ids"]:
                existing["pressure_event_ids"].append(pressure_id)
                existing["n_pressure_events"] += 1

    return list(by_carrier.values())


def pair_pressure_with_ball_carrier(
    events: pd.DataFrame,
    frames_dict: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Pair 'Pressure' events with the related ball-carrier event and its 360 freeze frame.

    Args:
        events: DataFrame of all events in a match.
        frames_dict: DataFrame of all 360 frames in a match.

    Returns:
        List of dictionaries with paired event data.
    """
    results: list[dict[str, Any]] = []

    if events.empty:
        return results

    logger.debug(
        "pairing (match %s): events=%d, frames=%s",
        events['match_id'].iloc[0] if 'match_id' in events.columns else 'unknown',
        len(events),
        len(frames_dict) if hasattr(frames_dict, 'len') or isinstance(frames_dict, pd.DataFrame) else 'no_len'
    )

    pressure_events = events[events["type"] == "Pressure"]

    # O(1) event lookup via dict — avoids iterrows over the full DataFrame
    events_lookup: dict[str, dict[str, Any]] = {}
    if "id" in events.columns:
        events_lookup = events.set_index("id").to_dict(orient="index")

    # O(1) frame lookup by event_uuid
    frames_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(frames_dict, pd.DataFrame) and not frames_dict.empty and "event_uuid" in frames_dict.columns:
        frames_lookup = {str(k): v for k, v in frames_dict.set_index("event_uuid").to_dict(orient="index").items()}

    for pressure_event in pressure_events.to_dict(orient="records"):
        try:
            related_event_ids = _normalise_related_events(pressure_event.get("related_events", []))
            if not related_event_ids:
                continue

            pressure_team_id = pressure_event.get("team_id", None)
            candidate_events: list[tuple[str, dict[str, Any]]] = []
            for related_id in related_event_ids:
                related_event = events_lookup.get(related_id)
                if related_event is None:
                    continue
                if related_event.get("type") not in BALL_CARRIER_EVENT_TYPES:
                    continue
                if pressure_team_id is not None and related_event.get("team_id") == pressure_team_id:
                    continue
                candidate_events.append((related_id, related_event))

            if not candidate_events:
                continue

            # Preserve StatsBomb's related_events order, but choose the carrier
            # action explicitly rather than assuming it is always the first id.
            related_id, related_event = candidate_events[0]
            frame_data = frames_lookup.get(related_id)
            if not frame_data:
                continue

            results.append({
                "match_id": pressure_event.get("match_id", None),
                "pressure_event_id": pressure_event["id"],
                "ball_carrier_event_id": related_id,
                "player_id": related_event.get("player_id", None),
                "team_id": related_event.get("team_id", None),
                "opponent_team_id": pressure_event.get("team_id", None),
                "frame_data": frame_data,
                "event_timestamp": related_event.get("timestamp", None),
            })

        except Exception as e:
            logger.debug(
                "Failed to pair pressure event %s: %s",
                pressure_event.get("id", "?"),
                e,
            )

    return dedupe_pressure_events_by_carrier(results)
