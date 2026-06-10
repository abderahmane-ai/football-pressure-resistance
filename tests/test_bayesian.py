"""Lightweight contract tests for Bayesian model input preparation."""
import pandas as pd
import pytest

from config import MODEL_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS_BASE, SPLINE_FEATURES
from src.data.validation import DataValidationError
from src.features.spatial import expand_spline_features, fit_spline_transformers
from src.models.bayesian import prepare_model_dataset


def _all_base_features() -> list[str]:
    """Return the complete list of base (pre-spline-expansion) feature columns."""
    return MODEL_FEATURE_COLUMNS_BASE + SPLINE_FEATURES


def make_model_df(success_values=(1.0, 0.0)):
    rows = []
    base_features = _all_base_features()
    for idx, success in enumerate(success_values):
        row = {
            "competition": "Euro_2024",
            "match_id": 1,
            "pressure_event_id": f"press_{idx}",
            "ball_carrier_event_id": f"carry_{idx}",
            "player_id": idx + 1,
            "player_name": f"Player {idx + 1}",
            "position_group": "Midfielder",
            "team_id": 10,
            "opponent_team_id": 20,
            "success": success,
            "value_preserved": 0.04 + idx * 0.01,
        }
        row.update({feature: float(idx) for feature in base_features})
        rows.append(row)
    return pd.DataFrame(rows)


def expand_and_prepare(df: pd.DataFrame):
    """Fit spline transformers, expand features, then call prepare_model_dataset."""
    transformers = fit_spline_transformers(df, n_knots=5, degree=3)
    df = expand_spline_features(df, transformers)
    return prepare_model_dataset(df)


def test_prepare_model_dataset_returns_clean_frame_and_feature_contract():
    df, features = expand_and_prepare(make_model_df())

    assert features == MODEL_FEATURE_COLUMNS
    assert len(df) == 2
    assert df["success"].tolist() == [1.0, 0.0]


def test_prepare_model_dataset_drops_only_rows_missing_required_model_values():
    raw = make_model_df(success_values=(1.0, 0.0, 1.0))
    raw.loc[1, "bc_x"] = None

    df, _ = expand_and_prepare(raw)

    assert df["ball_carrier_event_id"].tolist() == ["carry_0", "carry_2"]


def test_prepare_model_dataset_requires_success_rows_for_beta_component():
    raw = make_model_df(success_values=(0.0, 0.0))

    with pytest.raises(DataValidationError, match="zero successful actions"):
        expand_and_prepare(raw)


def test_mcmc_smoke(tmp_path, monkeypatch):
    from config import CROSS_VALIDATION_HOLDOUT
    from src.models import bayesian

    # 1. Setup temporary directories
    traces_dir = tmp_path / "model_traces"
    processed_dir = tmp_path / "processed"
    traces_dir.mkdir()
    processed_dir.mkdir()

    monkeypatch.setattr(bayesian, "MODEL_TRACES_DIR", traces_dir)
    monkeypatch.setattr(bayesian, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setenv("PRS_ALLOW_CPU", "1")

    # 2. Modify model settings to make it super fast
    monkeypatch.setitem(bayesian.MODEL_SETTINGS, "draws", 5)
    monkeypatch.setitem(bayesian.MODEL_SETTINGS, "tune", 5)
    monkeypatch.setitem(bayesian.MODEL_SETTINGS, "chains", 1)

    # 3. Create dummy dataset in processed directory (with base features)
    df = make_model_df(success_values=(1.0, 0.0, 1.0, 1.0, 0.0, 1.0))
    df["competition"] = CROSS_VALIDATION_HOLDOUT
    df.to_parquet(processed_dir / f"all_pressure_dataset_{CROSS_VALIDATION_HOLDOUT}.parquet")

    # 4. Fit the model (handles spline expansion internally)
    trace = bayesian.fit_pooled_model()

    assert trace is not None
    assert (traces_dir / f"pooled_trace_{CROSS_VALIDATION_HOLDOUT}.nc").exists()
    assert (traces_dir / f"pooled_scaler_{CROSS_VALIDATION_HOLDOUT}.pkl").exists()
    assert (traces_dir / f"pooled_mappings_{CROSS_VALIDATION_HOLDOUT}.pkl").exists()
