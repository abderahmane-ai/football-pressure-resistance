"""Data loading, pressure pairing, labeling, and validation helpers."""

from src.data.builder import build_all_datasets, build_holdout_dataset
from src.data.events import compute_game_state_for_match, compute_intended_xt
from src.data.labels import define_success
from src.data.pairing import pair_pressure_with_ball_carrier
from src.data.validation import (
    DataValidationError,
    validate_model_dataset,
    validate_statsbomb_events,
    validate_statsbomb_frames,
)

__all__ = [
    "DataValidationError",
    "build_all_datasets",
    "build_holdout_dataset",
    "compute_game_state_for_match",
    "compute_intended_xt",
    "define_success",
    "pair_pressure_with_ball_carrier",
    "validate_model_dataset",
    "validate_statsbomb_events",
    "validate_statsbomb_frames",
]
