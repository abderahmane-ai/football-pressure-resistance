"""Build processed pressure datasets from raw StatsBomb data."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from config import (
    COMPETITIONS,
    MODEL_FEATURE_COLUMNS_BASE,
    PROCESSED_DATA_DIR,
)
from src.data.events import (
    process_single_match,
)
from src.data.lineups import (
    fetch_lineups,
    get_goalkeeper_ids_from_lineups,
    get_player_position_groups_from_lineups,
)
from src.data.loader import load_all_competitions
from src.data.validation import (
    validate_model_dataset,
    validate_statsbomb_events,
    validate_statsbomb_frames,
)
from src.data.writer import (
    dataframe_hash,
    save_parquet_with_metadata,
)

logger = logging.getLogger(__name__)

N_WORKERS: int = min(os.cpu_count() or 4, 8)


def _build_dataset_for_competitions(
    comp_names: list[str],
    output_file: Path,
    *,
    label: str = "training",
    lineups_label: str = "lineups",
    matches_label: str = "matches",
) -> pd.DataFrame | None:
    """Shared worker: download, validate, and process matches for *comp_names*.

    All the parallelised logic that was duplicated between
    ``build_all_datasets`` and ``build_holdout_dataset`` lives here.
    """
    all_comp_data = load_all_competitions(comp_names)
    all_processed_data: list[dict[str, Any]] = []

    for comp_name, comp_data in all_comp_data.items():
        logger.info("Processing %s with %d workers...", comp_name, N_WORKERS)
        events_df: pd.DataFrame = comp_data["events"]
        frames_dict: dict[int, pd.DataFrame] = comp_data["frames"]
        if events_df.empty:
            continue
        validate_statsbomb_events(events_df, context=f"{comp_name} events")
        for match_id, frames_df in frames_dict.items():
            validate_statsbomb_frames(frames_df, context=f"{comp_name} match {match_id} frames")

        match_ids = events_df["match_id"].unique()

        # Pre-fetch lineup data ONCE per match
        lineups_by_match: dict[int, dict[str, pd.DataFrame]] = {}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(fetch_lineups, mid): mid for mid in match_ids}
            for fut in tqdm(futures, desc=f"Loading {lineups_label} ({comp_name})"):
                lineups_by_match[futures[fut]] = fut.result()

        # Pre-group events by match_id to avoid re-filtering the full DataFrame
        match_groups: dict[int, pd.DataFrame] = {k: v for k, v in events_df.groupby("match_id")}

        gk_ids_by_match: dict[int, set[int]] = {}
        pos_groups_by_match: dict[int, dict[int, str]] = {}
        for mid in match_ids:
            lineups = lineups_by_match.get(mid, {})
            gk_ids_by_match[mid] = get_goalkeeper_ids_from_lineups(lineups)
            match_ev = match_groups.get(mid, events_df.iloc[:0])
            pos_groups_by_match[mid] = get_player_position_groups_from_lineups(lineups, match_ev)

        worker_args = [
            (mid,
             match_groups[mid].copy().reset_index(drop=True),
             frames_dict[mid],
             gk_ids_by_match.get(mid, set()),
             pos_groups_by_match.get(mid, {}),
             comp_name)
            for mid in match_ids
            if mid in frames_dict and not frames_dict[mid].empty
        ]

        comp_events = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for batch in tqdm(
                ex.map(process_single_match, worker_args),
                total=len(worker_args),
                desc=f"Processing {matches_label} ({comp_name})",
            ):
                all_processed_data.extend(batch)
                comp_events += len(batch)

        logger.info("%s: %d events processed.", comp_name, comp_events)

    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        validate_model_dataset(dataset_df, MODEL_FEATURE_COLUMNS_BASE, context=f"{label} pressure dataset")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        source_hash = dataframe_hash(dataset_df)
        save_parquet_with_metadata(
            dataset_df,
            output_file,
            source_hash=source_hash,
            holdout=os.environ.get("PRS_HOLDOUT", "Euro_2020"),
            n_competitions=len(comp_names),
        )
        logger.info(
            "Saved dataset with %d events to %s (hash=%s)",
            len(dataset_df), output_file, source_hash[:12],
        )
        return dataset_df
    return None


def build_all_datasets(include_holdout: bool = False) -> pd.DataFrame | None:
    """Build the complete processed dataset for all competitions (parallelised per match)."""
    logger.info("Building dataset for all competitions...")
    holdout = os.environ.get("PRS_HOLDOUT", "Euro_2020")
    out_file = PROCESSED_DATA_DIR / f"all_pressure_dataset_{holdout}.parquet"

    if out_file.exists() and not os.environ.get("PRS_FORCE_REBUILD_DATA", "") == "1":
        logger.info(
            "Found cached training dataset at %s. Skipping rebuild "
            "(set PRS_FORCE_REBUILD_DATA=1 to override).",
            out_file,
        )
        return pd.read_parquet(out_file)

    comp_names = list(COMPETITIONS.keys())
    if not include_holdout and holdout in comp_names:
        comp_names.remove(holdout)
        logger.info("Excluding holdout competition: %s", holdout)

    return _build_dataset_for_competitions(comp_names, out_file, label="training")


def build_holdout_dataset() -> None:
    """Build dataset for holdout competition only (parallelised per match)."""
    holdout = os.environ.get("PRS_HOLDOUT", "Euro_2020")
    logger.info("Building holdout dataset: %s", holdout)
    out_file = PROCESSED_DATA_DIR / f"holdout_pressure_dataset_{holdout}.parquet"

    if out_file.exists() and not os.environ.get("PRS_FORCE_REBUILD_DATA", "") == "1":
        logger.info(
            "Found cached holdout dataset at %s. Skipping rebuild "
            "(set PRS_FORCE_REBUILD_DATA=1 to override).",
            out_file,
        )
        return

    _build_dataset_for_competitions(
        [holdout], out_file,
        label="holdout",
        lineups_label="holdout lineups",
        matches_label="holdout matches",
    )


if __name__ == "__main__":
    build_all_datasets(include_holdout=False)
    build_holdout_dataset()
