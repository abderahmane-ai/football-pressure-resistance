"""Download and cache raw StatsBomb event + 360-frame data."""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from tqdm import tqdm

from config import COMPETITIONS, RAW_DATA_DIR
from src.data.validation import validate_statsbomb_events, validate_statsbomb_frames

warnings.filterwarnings("ignore", message="credentials were not supplied.*")

logger = logging.getLogger(__name__)

# Maximum number of retries for StatsBomb API calls
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: float = 2.0  # seconds; doubled on each retry


def _retry_api_call(fn, *args, description: str = "API call", **kwargs):  # type: ignore[no-untyped-def]
    """Retry a StatsBomb API call with exponential backoff."""
    delay = _RETRY_BACKOFF
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == _MAX_RETRIES:
                raise
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.0fs",
                description, attempt, _MAX_RETRIES, e, delay,
            )
            time.sleep(delay)
            delay *= 2
    return None  # unreachable, but keeps mypy happy


def load_competition_events(comp_id: int, season_id: int) -> pd.DataFrame:
    """Load all events for a given competition and season from the StatsBomb API."""
    matches = _retry_api_call(
        sb.matches, competition_id=comp_id, season_id=season_id,
        description=f"matches({comp_id}/{season_id})",
    )
    all_events: list[pd.DataFrame] = []

    for match_id in matches["match_id"]:
        try:
            events = _retry_api_call(
                sb.events, match_id=match_id,
                description=f"events(match {match_id})",
            )
            events["match_id"] = match_id
            # pyrefly: ignore [bad-argument-type]
            all_events.append(events)
        except Exception as e:
            logger.warning(
                "Could not load events for match %d (comp=%d, season=%d): %s",
                match_id, comp_id, season_id, e,
            )

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
        validate_statsbomb_events(events_df, context=f"StatsBomb events {comp_id}/{season_id}")
        return events_df
    return pd.DataFrame()


def load_match_frames(match_id: int) -> pd.DataFrame:
    """Load 360 frames for a specific match from the StatsBomb API."""
    try:
        frames_data = _retry_api_call(
            sb.frames, match_id=match_id, fmt="dict",
            description=f"frames(match {match_id})",
        )
        if isinstance(frames_data, dict):
            frames_data = list(frames_data.values())
        frames_df = pd.DataFrame(frames_data)
        validate_statsbomb_frames(frames_df, context=f"StatsBomb 360 frames {match_id}")
        return frames_df
    except Exception as e:
        logger.warning(
            "Could not load 360 frames for match %d: %s. "
            "This match will have no spatial features.",
            match_id, e,
        )
        return pd.DataFrame()


def download_all(competition_name: str) -> None:
    """Download and cache all data for a competition to disk (idempotent)."""
    if competition_name not in COMPETITIONS:
        available = ", ".join(sorted(COMPETITIONS))
        raise ValueError(
            f"Unknown competition '{competition_name}'. Available: {available}"
        )

    comp_info = COMPETITIONS[competition_name]
    comp_id: int = comp_info["comp_id"]
    season_id: int = comp_info["season_id"]

    out_dir: Path = RAW_DATA_DIR / competition_name
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / "events.parquet"
    if not events_path.exists():
        logger.info("Downloading events for %s...", competition_name)
        events_df = load_competition_events(comp_id, season_id)
        if not events_df.empty:
            events_df.to_parquet(events_path)
    else:
        logger.info("Events for %s already cached.", competition_name)

    # Fetch match list once for frames download (avoid duplicate API call)
    matches = _retry_api_call(
        sb.matches, competition_id=comp_id, season_id=season_id,
        description=f"matches({comp_id}/{season_id})",
    )
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    for match_id in tqdm(matches["match_id"], desc=f"Downloading frames ({competition_name})"):
        # Support both parquet (preferred) and legacy pickle
        frame_path_parquet = frames_dir / f"{match_id}.parquet"
        frame_path_legacy = frames_dir / f"{match_id}.pkl"
        if not frame_path_parquet.exists() and not frame_path_legacy.exists():
            frames_df = load_match_frames(match_id)
            if isinstance(frames_df, pd.DataFrame) and not frames_df.empty:
                frames_df.to_parquet(frame_path_parquet)
        else:
            logger.debug("Frames for match %d already cached.", match_id)


def _read_frame_file(path: Path) -> pd.DataFrame:
    """Read a frame file, supporting both parquet and legacy pickle."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def load_all_competitions(
    competition_names: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame | dict[int, pd.DataFrame]]]:
    """
    Load events and frames for all specified competitions.

    Returns dict: competition_name → {"events": DataFrame, "frames": {match_id: DataFrame}}
    """
    if competition_names is None:
        competition_names = list(COMPETITIONS.keys())

    all_data: dict[str, dict] = {}

    for comp_name in tqdm(competition_names, desc="Loading competitions"):

        events_path = RAW_DATA_DIR / comp_name / "events.parquet"
        frames_dir = RAW_DATA_DIR / comp_name / "frames"

        # Check if frames folder exists and has at least one .parquet or .pkl file
        has_frames = False
        if frames_dir.exists():
            for f in frames_dir.glob("*"):
                if f.suffix in (".parquet", ".pkl"):
                    has_frames = True
                    break

        if not events_path.exists() or not has_frames:
            logger.info("Data not fully cached for %s, ensuring download...", comp_name)
            download_all(comp_name)

        events_df = pd.read_parquet(events_path)
        validate_statsbomb_events(events_df, context=f"{comp_name} events")

        # Load frames (supports both parquet and legacy pickle)
        frames_dir = RAW_DATA_DIR / comp_name / "frames"
        frames_dict: dict[int, pd.DataFrame] = {}

        if frames_dir.exists():
            for frame_file in sorted(frames_dir.glob("*")):
                if frame_file.suffix not in (".parquet", ".pkl"):
                    continue
                match_id = int(frame_file.stem)
                # Skip legacy pickle if parquet exists for same match
                if frame_file.suffix == ".pkl" and (frames_dir / f"{match_id}.parquet").exists():
                    continue
                try:
                    frames_df = _read_frame_file(frame_file)
                    if not frames_df.empty:
                        validate_statsbomb_frames(frames_df, context=f"{comp_name} match {match_id} frames")
                        frames_dict[match_id] = frames_df
                except Exception as e:
                    logger.warning(
                        "Could not load cached frames for %s match %d: %s",
                        comp_name, match_id, e,
                    )

        all_data[comp_name] = {
            "events": events_df,
            "frames": frames_dict,
        }

        logger.info(
            "Loaded %s: %d events, %d matches with 360 data",
            comp_name, len(events_df), len(frames_dict),
        )

    return all_data
