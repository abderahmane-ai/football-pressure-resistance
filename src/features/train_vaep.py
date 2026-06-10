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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    logger.info("Loading all competition events...")
    all_comp_data = load_all_competitions(list(COMPETITIONS.keys()))

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

    score_model, concede_model = train_vaep_models(combined)
    logger.info("VAEP training complete.")
    logger.info("Models saved to: %s", VAEP_CONFIG["model_dir"])


if __name__ == "__main__":
    main()
