"""Tests for explicit data validation contracts."""
import pandas as pd
import pytest

from config import MODEL_FEATURE_COLUMNS
from src.data.validation import (
    DataValidationError,
    validate_model_dataset,
    validate_statsbomb_events,
    validate_statsbomb_frames,
)


def valid_events_df():
    return pd.DataFrame([{
        "id": "e1",
        "index": 1,
        "type": "Pass",
        "team_id": 1,
        "match_id": 10,
        "timestamp": "00:01:00",
        "related_events": [],
        "location": [60.0, 40.0],
    }])


def valid_frames_df():
    return pd.DataFrame([{
        "event_uuid": "e1",
        "freeze_frame": [{"location": [60.0, 40.0], "actor": True, "teammate": True}],
    }])


def valid_model_df():
    row = {
        "competition": "Euro_2024",
        "match_id": 1,
        "pressure_event_id": "press_1",
        "ball_carrier_event_id": "carry_1",
        "player_id": 99,
        "player_name": "Player One",
        "position_group": "CM",
        "team_id": 1,
        "opponent_team_id": 2,
        "success": 1.0,
        "value_preserved": 0.05,
    }
    row.update({feature: 0.0 for feature in MODEL_FEATURE_COLUMNS})
    return pd.DataFrame([row])


def test_validate_statsbomb_events_accepts_valid_shape():
    validate_statsbomb_events(valid_events_df())


def test_validate_statsbomb_events_rejects_missing_required_column():
    df = valid_events_df().drop(columns=["related_events"])
    with pytest.raises(DataValidationError, match="related_events"):
        validate_statsbomb_events(df)


def test_validate_statsbomb_frames_rejects_duplicate_event_uuid():
    df = pd.concat([valid_frames_df(), valid_frames_df()], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_statsbomb_frames(df)


def test_validate_model_dataset_requires_binary_success():
    df = valid_model_df()
    df.loc[0, "success"] = 0.5
    with pytest.raises(DataValidationError, match="binary"):
        validate_model_dataset(df, MODEL_FEATURE_COLUMNS)


def test_validate_model_dataset_rejects_missing_feature():
    df = valid_model_df().drop(columns=[MODEL_FEATURE_COLUMNS[0]])
    with pytest.raises(DataValidationError, match=MODEL_FEATURE_COLUMNS[0]):
        validate_model_dataset(df, MODEL_FEATURE_COLUMNS)
