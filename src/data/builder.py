import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from statsbombpy import sb

from config import PROCESSED_DATA_DIR, COMPETITIONS, CROSS_VALIDATION_HOLDOUT
from src.data.loader import load_all_competitions
from src.data.pairing import pair_pressure_with_ball_carrier
from src.data.labels import define_success
from src.features.spatial import extract_spatial_features_from_frame
from src.features.geometry import xt_value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_goalkeeper_ids(match_id):
    """Get player IDs of goalkeepers from match lineups."""
    try:
        lineups = sb.lineups(match_id=match_id)
        gk_ids = set()
        for team_name, lineup_df in lineups.items():
            if 'positions' in lineup_df.columns:
                for _, player in lineup_df.iterrows():
                    positions = player['positions']
                    if isinstance(positions, list):
                        for pos_dict in positions:
                            if isinstance(pos_dict, dict) and pos_dict.get('position') == 'Goalkeeper':
                                gk_ids.add(player['player_id'])
                                break
            elif 'player_position' in lineup_df.columns:
                gks = lineup_df[lineup_df['player_position'] == 'Goalkeeper']
                gk_ids.update(gks['player_id'].values)
        return gk_ids
    except Exception as e:
        logger.debug(f"Could not load lineups for match {match_id}: {e}")
        return set()

def get_player_position_groups(match_id):
    """Get position group (Defender/Midfielder/Forward) for each player."""
    try:
        lineups = sb.lineups(match_id=match_id)
        position_map = {}
        
        for team_name, lineup_df in lineups.items():
            if 'positions' in lineup_df.columns:
                for _, player in lineup_df.iterrows():
                    player_id = player['player_id']
                    positions = player['positions']
                    
                    if isinstance(positions, list) and len(positions) > 0:
                        # Use first position
                        pos_dict = positions[0]
                        if isinstance(pos_dict, dict):
                            pos_name = pos_dict.get('position', '').lower()
                            
                            # Map to position group
                            if any(x in pos_name for x in ['back', 'defender', 'wing back']):
                                position_map[player_id] = 'Defender'
                            elif any(x in pos_name for x in ['forward', 'striker', 'wing', 'winger']):
                                position_map[player_id] = 'Forward'
                            else:
                                position_map[player_id] = 'Midfielder'
        
        return position_map
    except Exception as e:
        logger.debug(f"Could not load position groups for match {match_id}: {e}")
        return {}

def compute_value_preserved(item, match_events):
    """
    Compute value preserved = success * next_xT_value.
    Simplified: find next event; if it has location, use its xT; 
    if not, use the ball-carrier's own location xT.
    """
    success = item.get('success', 0)
    
    if success == 0:
        return 0.0
    
    # Find ball-carrier event and its location
    bc_event_id = item['ball_carrier_event_id']
    bc_event_rows = match_events[match_events['id'] == bc_event_id]
    
    if bc_event_rows.empty:
        return 0.0
        
    bc_event = bc_event_rows.iloc[0]
    bc_idx = bc_event_rows.index[0]
    
    bc_loc = bc_event.get('location')
    if not isinstance(bc_loc, (list, tuple)) or len(bc_loc) < 2:
        bc_loc = [0, 0] # Fallback
        
    # Get next event
    next_xt = xt_value(bc_loc[0], bc_loc[1]) # Default to current location xT
    
    # Special case for Pass: destination is end_location
    if bc_event['type'] == 'Pass' and 'pass_end_location' in bc_event:
        end_loc = bc_event['pass_end_location']
        if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2:
            next_xt = xt_value(end_loc[0], end_loc[1])
    elif bc_idx + 1 < len(match_events):
        next_event = match_events.iloc[bc_idx + 1]
        next_loc = next_event.get('location')
        if isinstance(next_loc, (list, tuple)) and len(next_loc) >= 2:
            next_xt = xt_value(next_loc[0], next_loc[1])
            
    return float(success * next_xt)

def build_all_datasets(include_holdout=False):
    """Build the complete processed dataset for all competitions."""
    logger.info("Building dataset for all competitions...")
    
    comp_names = list(COMPETITIONS.keys())
    if not include_holdout and CROSS_VALIDATION_HOLDOUT in comp_names:
        comp_names.remove(CROSS_VALIDATION_HOLDOUT)
        logger.info(f"Excluding holdout competition: {CROSS_VALIDATION_HOLDOUT}")
    
    all_comp_data = load_all_competitions(comp_names)
    
    all_processed_data = []
    comp_event_counts = {}
    
    for comp_name, comp_data in all_comp_data.items():
        logger.info(f"Processing {comp_name}...")
        events_df = comp_data['events']
        frames_dict = comp_data['frames']
        
        if events_df.empty:
            continue
        
        match_ids = events_df['match_id'].unique()
        comp_events = 0
        
        gk_ids_by_match = {}
        position_groups_by_match = {}
        
        for match_id in tqdm(match_ids, desc=f"Loading lineups ({comp_name})"):
            gk_ids_by_match[match_id] = get_goalkeeper_ids(match_id)
            position_groups_by_match[match_id] = get_player_position_groups(match_id)
        
        for match_id in tqdm(match_ids, desc=f"Processing matches ({comp_name})"):
            match_events = events_df[events_df['match_id'] == match_id].copy().reset_index(drop=True)
            
            if match_id not in frames_dict:
                continue
            
            frames_df = frames_dict[match_id]
            if frames_df.empty:
                continue
            
            paired_events = pair_pressure_with_ball_carrier(match_events, frames_df)
            labeled_events = define_success(match_events, paired_events)
            
            gk_ids = gk_ids_by_match.get(match_id, set())
            position_groups = position_groups_by_match.get(match_id, {})
            
            for item in labeled_events:
                player_id = item.get('player_id')
                
                # Exclude Goalkeepers
                if player_id in gk_ids:
                    continue
                
                event_row = match_events[match_events['id'] == item['ball_carrier_event_id']]
                match_context = {}
                if not event_row.empty:
                    event_row = event_row.iloc[0]
                    if 'minute' in event_row: match_context['minutes_elapsed'] = event_row['minute']
                    if 'period' in event_row: match_context['match_period'] = event_row['period']
                
                features = extract_spatial_features_from_frame(
                    frame_data=item['frame_data'],
                    ball_carrier_player_id=player_id,
                    team_id=item['team_id'],
                    opponent_team_id=item['opponent_team_id'],
                    match_context=match_context
                )
                
                if features is None:
                    continue
                
                # TIGHT-PRESSURE FILTER: only include if nearest opponent ≤ 5 yards
                if features.get('dist_nearest_opp', 999) > 5.0:
                    continue
                
                # Compute value preserved
                value_preserved = compute_value_preserved(item, match_events)
                
                player_name = match_events[match_events['player_id'] == player_id]['player'].iloc[0] if player_id in match_events['player_id'].values else "Unknown"
                position_group = position_groups.get(player_id, 'Midfielder')
                
                row = {
                    'competition': comp_name,
                    'match_id': item['match_id'],
                    'pressure_event_id': item['pressure_event_id'],
                    'ball_carrier_event_id': item['ball_carrier_event_id'],
                    'player_id': player_id,
                    'player_name': player_name,
                    'position_group': position_group,
                    'team_id': item['team_id'],
                    'opponent_team_id': item['opponent_team_id'],
                    'success': item['success'],
                    'value_preserved': value_preserved,
                }
                row.update(features)
                all_processed_data.append(row)
                comp_events += 1
        
        comp_event_counts[comp_name] = comp_events
        logger.info(f"{comp_name}: {comp_events} events processed (after tight filter).")
    
    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PROCESSED_DATA_DIR / "all_pressure_dataset.parquet"
        dataset_df.to_parquet(out_file)
        logger.info(f"Saved dataset with {len(dataset_df)} events to {out_file}")
        return dataset_df
    return None

def build_holdout_dataset():
    """Build dataset for holdout competition only."""
    logger.info(f"Building holdout dataset: {CROSS_VALIDATION_HOLDOUT}")
    all_comp_data = load_all_competitions([CROSS_VALIDATION_HOLDOUT])
    all_processed_data = []
    
    for comp_name, comp_data in all_comp_data.items():
        events_df = comp_data['events']
        frames_dict = comp_data['frames']
        if events_df.empty: continue
        
        match_ids = events_df['match_id'].unique()
        for match_id in tqdm(match_ids, desc="Processing holdout"):
            match_events = events_df[events_df['match_id'] == match_id].copy().reset_index(drop=True)
            if match_id not in frames_dict: continue
            
            gk_ids = get_goalkeeper_ids(match_id)
            position_groups = get_player_position_groups(match_id)
            
            paired = pair_pressure_with_ball_carrier(match_events, frames_dict[match_id])
            labeled = define_success(match_events, paired)
            
            for item in labeled:
                player_id = item.get('player_id')
                if player_id in gk_ids: continue
                
                features = extract_spatial_features_from_frame(
                    frame_data=item['frame_data'],
                    ball_carrier_player_id=player_id,
                    team_id=item['team_id'],
                    opponent_team_id=item['opponent_team_id']
                )
                if features is None or features.get('dist_nearest_opp', 999) > 5.0: continue
                
                row = {
                    'competition': comp_name,
                    'match_id': item['match_id'],
                    'player_id': player_id,
                    'player_name': match_events[match_events['player_id'] == player_id]['player'].iloc[0] if player_id in match_events['player_id'].values else "Unknown",
                    'position_group': position_groups.get(player_id, 'Midfielder'),
                    'success': item['success'],
                    'value_preserved': compute_value_preserved(item, match_events),
                }
                row.update(features)
                all_processed_data.append(row)
                
    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        out_file = PROCESSED_DATA_DIR / "holdout_pressure_dataset.parquet"
        dataset_df.to_parquet(out_file)
        logger.info(f"Saved holdout to {out_file}")

if __name__ == "__main__":
    build_all_datasets(include_holdout=False)
    build_holdout_dataset()
