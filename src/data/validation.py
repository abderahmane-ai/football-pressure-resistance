"""Validation helpers for raw StatsBomb inputs and processed model data."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data violates the pipeline's expected contract."""


REQUIRED_EVENT_COLUMNS: set[str] = {
    "id",
    "index",
    "type",
    "team_id",
    "match_id",
    "timestamp",
    "related_events",
}

REQUIRED_FRAME_COLUMNS: set[str] = {
    "event_uuid",
    "freeze_frame",
}

REQUIRED_MODEL_COLUMNS: set[str] = {
    "competition",
    "match_id",
    "pressure_event_id",
    "ball_carrier_event_id",
    "player_id",
    "player_name",
    "position_group",
    "team_id",
    "opponent_team_id",
    "success",
    "value_preserved",
}


def _ensure_dataframe(df: object, context: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"{context}: expected a pandas DataFrame, got {type(df).__name__}."
        )


def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise DataValidationError(
            f"{context}: missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(sorted(df.columns))}"
        )


def _is_valid_location(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    if not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        return False
    val_seq: Any = value
    if len(val_seq) < 2:
        return False
    x, y = val_seq[0], val_seq[1]
    return bool(pd.notna(x) and pd.notna(y))


def validate_statsbomb_events(events_df: pd.DataFrame, context: str = "events") -> None:
    """Validate the raw StatsBomb event columns used by pairing/labels/building."""
    _ensure_dataframe(events_df, context)
    _ensure_columns(events_df, REQUIRED_EVENT_COLUMNS, context)

    if events_df.empty:
        raise DataValidationError(f"{context}: DataFrame is empty — no events to process.")
    for column in ["id", "type", "team_id", "match_id"]:
        n_null = int(events_df[column].isna().sum())
        if n_null > 0:
            raise DataValidationError(
                f"{context}: {n_null} null value(s) in required column '{column}'. "
                "Upstream data download may be incomplete."
            )
    n_dupes = int(events_df["id"].duplicated().sum())
    if n_dupes > 0:
        raise DataValidationError(
            f"{context}: {n_dupes} duplicate event id(s). "
            "Check for repeated event downloads or concatenation errors."
        )

    if "location" in events_df.columns:
        valid_mask = events_df["location"].dropna().map(_is_valid_location)
        if not valid_mask.all():
            n_bad = int((~valid_mask).sum())
            raise DataValidationError(
                f"{context}: {n_bad} event(s) with malformed locations "
                "(expected [x, y] arrays with numeric values)."
            )


def validate_statsbomb_frames(
    frames_df: pd.DataFrame,
    context: str = "360 frames",
    allow_empty: bool = True,
) -> None:
    """Validate the StatsBomb 360 frame columns required for event-frame lookup."""
    _ensure_dataframe(frames_df, context)
    if frames_df.empty:
        if allow_empty:
            return
        raise DataValidationError(f"{context}: DataFrame is empty — no frames available.")
    _ensure_columns(frames_df, REQUIRED_FRAME_COLUMNS, context)
    n_null = int(frames_df["event_uuid"].isna().sum())
    if n_null > 0:
        raise DataValidationError(
            f"{context}: {n_null} null event_uuid value(s). "
            "360 frame data may be corrupt or partially downloaded."
        )
    n_dupes = int(frames_df["event_uuid"].duplicated().sum())
    if n_dupes > 0:
        raise DataValidationError(
            f"{context}: {n_dupes} duplicate event_uuid(s). "
            "Each frame row must correspond to exactly one event."
        )


def validate_model_dataset(
    dataset_df: pd.DataFrame,
    feature_columns: Iterable[str],
    context: str = "processed model dataset",
) -> None:
    """Validate the processed pressure dataset before model fitting or inference."""
    _ensure_dataframe(dataset_df, context)
    _ensure_columns(dataset_df, REQUIRED_MODEL_COLUMNS, context)
    _ensure_columns(dataset_df, feature_columns, context)

    if dataset_df.empty:
        raise DataValidationError(f"{context}: dataset is empty after processing.")
    for column in ["player_id", "competition", "opponent_team_id", "position_group"]:
        n_null = int(dataset_df[column].isna().sum())
        if n_null > 0:
            raise DataValidationError(
                f"{context}: {n_null} null value(s) in grouping column '{column}'. "
                "Check upstream data pairing logic."
            )

    success_values = set(dataset_df["success"].dropna().unique())
    if not success_values <= {0, 0.0, 1, 1.0}:
        raise DataValidationError(
            f"{context}: success column contains non-binary values {success_values}. "
            "Expected only 0 and 1."
        )
    n_null_success = int(dataset_df["success"].isna().sum())
    if n_null_success > 0:
        raise DataValidationError(
            f"{context}: {n_null_success} null success label(s). "
            "define_success() may have failed to label some events."
        )
    n_null_vp = int(dataset_df["value_preserved"].isna().sum())
    if n_null_vp > 0:
        raise DataValidationError(
            f"{context}: {n_null_vp} null value_preserved value(s). "
            "compute_intended_xt() may have returned None for some events."
        )
    # VAEP can be negative (actions that increase conceding risk),
    # so negative value_preserved is valid when VAEP is active.

    numeric_columns = list(feature_columns) + ["value_preserved"]
    non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(dataset_df[col])]
    if non_numeric:
        raise DataValidationError(
            f"{context}: non-numeric model column(s): {', '.join(non_numeric)}. "
            "These must be numeric for StandardScaler and the MCMC sampler."
        )
