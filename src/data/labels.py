import pandas as pd
import numpy as np
import logging
from config import SPATIAL_CONFIG

logger = logging.getLogger(__name__)

def define_success(events, paired_results):
    """
    Define success for the ball-carrier based on the outcome of their event or subsequent events.
    Handles passes, carries, and dribbles safely, ensuring possession is truly retained.
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
                # In StatsBomb, a NaN pass_outcome typically implies a completed pass.
                # To be rigorous, we check if the recipient is explicit or the next event is by the same team
                # but for standard definition we will rely on NaN = Complete, but add defensive checks.
                if pd.isna(bc_event.get('pass_recipient')) and bc_idx + 1 < len(events):
                     next_event = events.iloc[bc_idx + 1]
                     if next_event['team_id'] != bc_event['team_id']:
                         success = 0.0
                     else:
                         success = 1.0
                else:
                    success = 1.0
            else:
                success = 0.0
        elif bc_event['type'] == 'Dribble':
            if bc_event.get('dribble_outcome') == 'Complete':
                success = 1.0
            else:
                success = 0.0
        elif bc_event['type'] == 'Carry':
            # Trace subsequent events to see if possession is retained
            success = 0.0
            lookahead = SPATIAL_CONFIG['carry_lookahead_events']
            # range end is lookahead+1 so we check exactly `lookahead` events (was off-by-one)
            for offset in range(1, min(lookahead + 1, len(events) - bc_idx)):
                next_event = events.iloc[bc_idx + offset]
                # If same team makes next action, carry was successful
                if next_event['team_id'] == bc_event['team_id']:
                    success = 1.0
                    break
                # Opponent commits a foul on the carrier — carrier wins a free kick (success)
                elif (next_event['type'] == 'Foul Committed'
                      and next_event['team_id'] != bc_event['team_id']):
                    success = 1.0
                    break
                # Opponent wins ball cleanly — possession lost
                elif next_event['type'] in ['Pass', 'Carry', 'Shot', 'Clearance', 'Interception', 'Dispossessed']:
                    success = 0.0
                    break
                    
        if not pd.isna(success):
            item['success'] = success
            labeled_results.append(item)
            
    return labeled_results
