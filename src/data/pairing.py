import pandas as pd
import logging
import ast

logger = logging.getLogger(__name__)

def pair_pressure_with_ball_carrier(events, frames_dict):
    """
    Pair 'Pressure' events with the related ball-carrier event and its 360 freeze frame.
    
    Args:
        events: DataFrame of all events in a match.
        frames_dict: DataFrame of all 360 frames in a match.
        
    Returns:
        List of dictionaries with paired event data.
    """
    results = []
    
    if events.empty:
        return results
        
    pressure_events = events[events['type'] == 'Pressure']
    
    # Pre-process frames into a dict by event_uuid for O(1) lookup
    frames_lookup = {}
    if isinstance(frames_dict, pd.DataFrame) and not frames_dict.empty and 'event_uuid' in frames_dict.columns:
        for _, row in frames_dict.iterrows():
            frames_lookup[row['event_uuid']] = row.to_dict()
        
    for _, pressure_event in pressure_events.iterrows():
        try:
            related_event_ids = pressure_event.get('related_events', [])
            # Handle string representation of lists if loaded from certain formats
            if isinstance(related_event_ids, str):
                try:
                    related_event_ids = ast.literal_eval(related_event_ids)
                except Exception:
                    continue
            # Handle numpy arrays from parquet
            elif hasattr(related_event_ids, 'tolist'):
                related_event_ids = related_event_ids.tolist()
                    
            if not isinstance(related_event_ids, list) or len(related_event_ids) == 0:
                continue
                
            related_id = related_event_ids[0]
            
            related_event = events[events['id'] == related_id]
            if related_event.empty:
                continue
                
            related_event = related_event.iloc[0]
            
            if related_event['type'] not in ['Pass', 'Carry', 'Dribble']:
                continue
                
            frame_data = frames_lookup.get(related_id)
            if not frame_data:
                continue
                
            results.append({
                'match_id': pressure_event.get('match_id', None),
                'pressure_event_id': pressure_event['id'],
                'ball_carrier_event_id': related_id,
                'player_id': related_event.get('player_id', None),
                'team_id': related_event.get('team_id', None),
                'opponent_team_id': pressure_event.get('team_id', None),
                'frame_data': frame_data,
                'event_timestamp': related_event.get('timestamp', None)
            })
            
        except Exception as e:
            logger.debug(f"Error pairing event {pressure_event.get('id')}: {e}")
            
    return results
