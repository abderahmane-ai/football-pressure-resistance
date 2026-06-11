"""Data loading, pressure pairing, labeling, and validation helpers."""

from src.data.events import compute_game_state_for_match, compute_intended_xt
from src.data.labels import define_success
from src.data.pairing import pair_pressure_with_ball_carrier
from src.data.validation import (
    DataValidationError,
    validate_model_dataset,
    validate_statsbomb_events,
    validate_statsbomb_frames,
)


def __getattr__(name: str):
    """Lazy-import builder to avoid runpy warning (python -m src.data.builder)."""
    if name in ("build_all_datasets", "build_holdout_dataset"):
        from src.data.builder import build_all_datasets as _b1, build_holdout_dataset as _b2  # noqa: I001
        return {"build_all_datasets": _b1, "build_holdout_dataset": _b2}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataValidationError",
    "define_success",
    "pair_pressure_with_ball_carrier",
    "validate_model_dataset",
    "validate_statsbomb_events",
    "validate_statsbomb_frames",
    "build_all_datasets",
    "build_holdout_dataset",
    "compute_game_state_for_match",
    "compute_intended_xt",
]
