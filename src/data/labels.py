import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def define_success(events, paired_results):
    """
    Define success for the ball-carrier based on the outcome of their event or the next event.
    """
    if events.empty or not paired_results:
        return []
        
    if 'index' in events.columns:
        events = events.sort_values(by=['match_id', 'index'])
    else:
        events = events.sort_values(by=['match_id', 'timestamp'])
        
    events = events.reset_index(drop=True)
    event_idx_map = {row['id']: idx for idx, row in events.iterrows()}
    
    labeled_results = []
    
    for item in paired_results:
        bc_event_id = item['ball_carrier_event_id']
        if bc_event_id not in event_idx_map:
            continue
            
        bc_idx = event_idx_map[bc_event_id]
        bc_event = events.iloc[bc_idx]
        
        success = np.nan
        
        if bc_event['type'] == 'Pass':
            if pd.isna(bc_event.get('pass_outcome')):
                success = 1.0
            else:
                success = 0.0
        elif bc_event['type'] == 'Dribble':
            if bc_event.get('dribble_outcome') == 'Complete':
                success = 1.0
            else:
                success = 0.0
        elif bc_event['type'] == 'Carry':
            if bc_idx + 1 < len(events):
                next_event = events.iloc[bc_idx + 1]
                if next_event['team_id'] == bc_event['team_id']:
                    success = 1.0
                else:
                    success = 0.0
                    
        if not pd.isna(success):
            item['success'] = success
            labeled_results.append(item)
            
    return labeled_results
