import logging
import warnings
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
from config import RAW_DATA_DIR, COMPETITIONS
from src.data.validation import validate_statsbomb_events, validate_statsbomb_frames

warnings.filterwarnings("ignore", message="credentials were not supplied.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_competition_events(comp_id, season_id):
    """Load all events for a given competition and season."""
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    all_events = []
    
    for match_id in matches['match_id']:
        try:
            events = sb.events(match_id=match_id)
            events['match_id'] = match_id
            all_events.append(events)
        except Exception as e:
            logger.warning(f"Could not load events for match {match_id}: {e}")
            
    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
        validate_statsbomb_events(events_df, context=f"StatsBomb events {comp_id}/{season_id}")
        return events_df
    return pd.DataFrame()

def load_match_frames(match_id):
    """Load 360 frames for a specific match."""
    try:
        frames_data = sb.frames(match_id=match_id, fmt='dict')
        if isinstance(frames_data, dict):
            frames_data = list(frames_data.values())
        frames_df = pd.DataFrame(frames_data)
        validate_statsbomb_frames(frames_df, context=f"StatsBomb 360 frames {match_id}")
        return frames_df
    except Exception as e:
        logger.warning(f"Could not load 360 frames for match {match_id}: {e}")
        return pd.DataFrame()

def download_all(competition_name):
    """Download and cache all data for a competition."""
    if competition_name not in COMPETITIONS:
        raise ValueError(f"Unknown competition: {competition_name}")
        
    comp_info = COMPETITIONS[competition_name]
    comp_id = comp_info['comp_id']
    season_id = comp_info['season_id']
    
    out_dir = RAW_DATA_DIR / competition_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    events_path = out_dir / "events.parquet"
    if not events_path.exists():
        logger.info(f"Downloading events for {competition_name}...")
        events_df = load_competition_events(comp_id, season_id)
        if not events_df.empty:
            events_df.to_parquet(events_path)
    else:
        logger.info(f"Events for {competition_name} already cached.")
        
    # Download frames
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    for match_id in tqdm(matches['match_id'], desc=f"Downloading frames ({competition_name})"):
        frame_path = frames_dir / f"{match_id}.pkl"
        if not frame_path.exists():
            frames_df = load_match_frames(match_id)
            if isinstance(frames_df, pd.DataFrame) and not frames_df.empty:
                frames_df.to_pickle(frame_path)
        else:
            logger.debug(f"Frames for match {match_id} already cached.")

def load_all_competitions(competition_names=None):
    """
    Load events and frames for all specified competitions.
    Returns dict with competition_name -> {'events': df, 'frames': dict}
    """
    if competition_names is None:
        competition_names = list(COMPETITIONS.keys())
    
    all_data = {}
    
    for comp_name in tqdm(competition_names, desc="Loading competitions"):
        comp_info = COMPETITIONS[comp_name]
        
        # Load or download events
        events_path = RAW_DATA_DIR / comp_name / "events.parquet"
        if events_path.exists():
            events_df = pd.read_parquet(events_path)
        else:
            logger.info(f"Events not cached for {comp_name}, downloading...")
            download_all(comp_name)
            events_df = pd.read_parquet(events_path)
        validate_statsbomb_events(events_df, context=f"{comp_name} events")
        
        # Load frames
        frames_dir = RAW_DATA_DIR / comp_name / "frames"
        frames_dict = {}
        
        if frames_dir.exists():
            for frame_file in frames_dir.glob("*.pkl"):
                match_id = int(frame_file.stem)
                try:
                    frames_df = pd.read_pickle(frame_file)
                    if not frames_df.empty:
                        validate_statsbomb_frames(frames_df, context=f"{comp_name} match {match_id} frames")
                        frames_dict[match_id] = frames_df
                except Exception as e:
                    logger.warning(f"Could not load frames for match {match_id}: {e}")
        
        all_data[comp_name] = {
            'events': events_df,
            'frames': frames_dict
        }
        
        logger.info(f"Loaded {comp_name}: {len(events_df)} events, {len(frames_dict)} matches with 360 data")
    
    return all_data
