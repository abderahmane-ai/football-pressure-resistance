"""Entry point: train VAEP models on all available StatsBomb events.

Usage:
    python -m src.features.train_vaep

Trains two LightGBM classifiers (P(score | state) and P(concede | state))
on all events across all competitions in COMPETITIONS, then serialises them
to ``VAEP_CONFIG['model_dir']``.

Designed to be run once per data refresh, before the main PRS pipeline.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from config import COMPETITIONS, VAEP_CONFIG
from src.data.loader import load_all_competitions
from src.features.vaep import train_vaep_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== VAEP Model Training ===")

    # Skip if models already exist (VAEP is invariant to CV holdout)
    model_dir = Path(VAEP_CONFIG["model_dir"])  # type: ignore[arg-type]
    score_path = model_dir / "vaep_score.pkl"
    concede_path = model_dir / "vaep_concede.pkl"
    if score_path.exists() and concede_path.exists():
        if os.environ.get("PRS_FORCE_RETRAIN", "") != "1":
            logger.info(
                "VAEP models already exist at %s. Skipping (set PRS_FORCE_RETRAIN=1 to override).",
                model_dir,
            )
            return

    # Exclude the holdout competition from VAEP training to prevent data leakage.
    # The VAEP value signal is used as a target in the Hurdle model; training
    # VAEP on the holdout competition would contaminate out-of-sample evaluation.
    holdout = os.environ.get("PRS_HOLDOUT", "Euro_2020")
    comp_names = [c for c in COMPETITIONS if c != holdout]
    logger.info(
        "Loading competition events for VAEP training (excluding holdout: %s)...", holdout,
    )
    all_comp_data = load_all_competitions(comp_names)

    all_events: list[pd.DataFrame] = []
    for comp_name, comp_data in all_comp_data.items():
        events = comp_data["events"]
        logger.info("Loaded %s: %d events", comp_name, len(events))
        all_events.append(events)

    if not all_events:
        logger.error("No events loaded. Exiting.")
        sys.exit(1)

    combined = pd.concat(all_events, ignore_index=True)
    logger.info("Combined dataset: %d events", len(combined))

    train_vaep_models(combined)
    logger.info("VAEP training complete.")
    logger.info("Models saved to: %s", VAEP_CONFIG["model_dir"])


if __name__ == "__main__":
    main()
