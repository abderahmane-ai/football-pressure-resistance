"""Validation helpers for raw StatsBomb inputs and processed model data."""
import math

import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data violates the pipeline's expected contract."""


REQUIRED_EVENT_COLUMNS = {
    "id",
    "index",
    "type",
    "team_id",
    "match_id",
    "timestamp",
    "related_events",
}

REQUIRED_FRAME_COLUMNS = {
    "event_uuid",
    "freeze_frame",
}

REQUIRED_MODEL_COLUMNS = {
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


def _ensure_dataframe(df, context):
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{context} must be a pandas DataFrame.")


def _ensure_columns(df, required_columns, context):
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise DataValidationError(f"{context} missing required column(s): {', '.join(missing)}")


def _is_valid_location(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    if not hasattr(value, "__len__") or len(value) < 2:
        return False
    x, y = value[0], value[1]
    return pd.notna(x) and pd.notna(y)


def validate_statsbomb_events(events_df, context="events"):
    """Validate the raw StatsBomb event columns used by pairing/labels/building."""
    _ensure_dataframe(events_df, context)
    _ensure_columns(events_df, REQUIRED_EVENT_COLUMNS, context)

    if events_df.empty:
        raise DataValidationError(f"{context} is empty.")
    for column in ["id", "type", "team_id", "match_id"]:
        if events_df[column].isna().any():
            raise DataValidationError(f"{context} contains null values in required column '{column}'.")
    if events_df["id"].duplicated().any():
        raise DataValidationError(f"{context} contains duplicate event ids.")

    if "location" in events_df.columns:
        invalid = events_df["location"].dropna().map(_is_valid_location)
        if not invalid.all():
            raise DataValidationError(f"{context} contains malformed event locations.")


def validate_statsbomb_frames(frames_df, context="360 frames", allow_empty=True):
    """Validate the StatsBomb 360 frame columns required for event-frame lookup."""
    _ensure_dataframe(frames_df, context)
    if frames_df.empty:
        if allow_empty:
            return
        raise DataValidationError(f"{context} is empty.")
    _ensure_columns(frames_df, REQUIRED_FRAME_COLUMNS, context)
    if frames_df["event_uuid"].isna().any():
        raise DataValidationError(f"{context} contains null event_uuid values.")
    if frames_df["event_uuid"].duplicated().any():
        raise DataValidationError(f"{context} contains duplicate event_uuid values.")


def validate_model_dataset(dataset_df, feature_columns, context="processed model dataset"):
    """Validate the processed pressure dataset before model fitting or inference."""
    _ensure_dataframe(dataset_df, context)
    _ensure_columns(dataset_df, REQUIRED_MODEL_COLUMNS, context)
    _ensure_columns(dataset_df, feature_columns, context)

    if dataset_df.empty:
        raise DataValidationError(f"{context} is empty.")
    for column in ["player_id", "competition", "opponent_team_id", "position_group"]:
        if dataset_df[column].isna().any():
            raise DataValidationError(f"{context} contains null values in '{column}'.")

    success_values = set(dataset_df["success"].dropna().unique())
    if not success_values <= {0, 0.0, 1, 1.0}:
        raise DataValidationError(f"{context} success must be binary 0/1.")
    if dataset_df["success"].isna().any():
        raise DataValidationError(f"{context} contains null success labels.")
    if dataset_df["value_preserved"].isna().any():
        raise DataValidationError(f"{context} contains null value_preserved values.")
    if (dataset_df["value_preserved"] < 0).any():
        raise DataValidationError(f"{context} contains negative value_preserved values.")

    numeric_columns = list(feature_columns) + ["value_preserved"]
    non_numeric = [column for column in numeric_columns if not pd.api.types.is_numeric_dtype(dataset_df[column])]
    if non_numeric:
        raise DataValidationError(f"{context} has non-numeric model column(s): {', '.join(non_numeric)}")
