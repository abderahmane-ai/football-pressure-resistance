"""Model fitting, posterior inference, and validation entry points."""

from src.models.inference import run_posterior_analysis
from src.models.validation import run_cross_validation


def __getattr__(name: str):
    """Lazy-import bayesian to avoid runpy warning (python -m src.models.bayesian)."""
    if name in ("fit_pooled_model", "prepare_model_dataset"):
        from src.models.bayesian import fit_pooled_model as _f, prepare_model_dataset as _p  # noqa: I001
        return {"fit_pooled_model": _f, "prepare_model_dataset": _p}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "fit_pooled_model",
    "prepare_model_dataset",
    "run_cross_validation",
    "run_posterior_analysis",
]
