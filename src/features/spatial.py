import numpy as np
import pandas as pd
from .geometry import voronoi_area, angular_span, point_in_pitch, pitch_control_value, xt_value

def extract_spatial_features_from_frame(frame_data, ball_carrier_player_id, team_id, opponent_team_id, match_context=None):
    """
    Compute spatial features from a single freeze frame.
    """
    if pd.isna(frame_data) or not isinstance(frame_data, dict):
        return None
        
    freeze_frame = frame_data.get('freeze_frame', [])
    if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
        return None
        
    teammates = []
    opponents = []
    ball_carrier = None
    all_players = []
    
    for p in freeze_frame:
        loc = p.get('location', [0, 0])
        all_players.append(loc)
        
        is_teammate = p.get('teammate', False)
        
        if p.get('actor', False) or p.get('player_id') == ball_carrier_player_id:
            ball_carrier = loc
            
        if is_teammate:
            teammates.append(loc)
        else:
            opponents.append(loc)
            
    if ball_carrier is None:
        if teammates:
            ball_carrier = teammates[0]
        else:
            return None
            
    bc = np.array(ball_carrier)
    opps = np.array(opponents) if opponents else np.array([])
    tms = np.array(teammates) if teammates else np.array([])
    
    features = {}
    
    if len(opps) > 0:
        opp_dists = np.linalg.norm(opps - bc, axis=1)
        sorted_opp_dists = np.sort(opp_dists)
        features['dist_nearest_opp'] = sorted_opp_dists[0]
        features['dist_2nd_nearest_opp'] = sorted_opp_dists[1] if len(sorted_opp_dists) > 1 else 50.0
        features['opps_within_1yd'] = np.sum(sorted_opp_dists <= 1)
        features['opps_within_2yd'] = np.sum(sorted_opp_dists <= 2)
        features['opps_within_4yd'] = np.sum(sorted_opp_dists <= 4)
        
        nearest_opp = opps[np.argmin(opp_dists)]
        goal_vec = np.array([120.0, 40.0]) - bc
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        opp_vec = nearest_opp - bc
        opp_angle = np.arctan2(opp_vec[1], opp_vec[0])
        
        rel_angle = opp_angle - goal_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        features['angle_nearest_opp'] = rel_angle
    else:
        features['dist_nearest_opp'] = 50.0
        features['dist_2nd_nearest_opp'] = 50.0
        features['opps_within_1yd'] = 0
        features['opps_within_2yd'] = 0
        features['opps_within_4yd'] = 0
        features['angle_nearest_opp'] = 0.0
        
    features['coverage_arc'] = angular_span(bc, opps, radius=3.0)
    features['voronoi_area'] = voronoi_area(bc, all_players)
    
    # Pitch control at ball-carrier location
    features['pitch_control'] = pitch_control_value(bc, tms, opps)
    
    # Opponent density (within 5 yards)
    if len(opps) > 0:
        features['opp_density_5yd'] = np.sum(np.linalg.norm(opps - bc, axis=1) <= 5.0)
    else:
        features['opp_density_5yd'] = 0
    
    max_tri_area = 0.0
    free_tms = []
    if len(opps) > 0 and len(tms) > 0:
        for tm in tms:
            dist_to_opps = np.linalg.norm(opps - tm, axis=1)
            if np.min(dist_to_opps) > 2.0:
                free_tms.append(tm)
    else:
        free_tms = list(tms)
        
    features['n_free_teammates'] = len(free_tms)
    
    if len(free_tms) >= 2:
        for i in range(len(free_tms)):
            for j in range(i+1, len(free_tms)):
                p1, p2 = free_tms[i], free_tms[j]
                area = 0.5 * abs(bc[0]*(p1[1] - p2[1]) + p1[0]*(p2[1] - bc[1]) + p2[0]*(bc[1] - p1[1]))
                if area > max_tri_area:
                    max_tri_area = area
                    
    features['max_free_triangle_area'] = max_tri_area
    
    if len(free_tms) > 0:
        free_tms_arr = np.array(free_tms)
        tm_dists = np.linalg.norm(free_tms_arr - bc, axis=1)
        min_tm_idx = np.argmin(tm_dists)
        features['dist_nearest_free_teammate'] = tm_dists[min_tm_idx]
        
        tm_vec = free_tms_arr[min_tm_idx] - bc
        tm_angle = np.arctan2(tm_vec[1], tm_vec[0])
        rel_tm_angle = tm_angle - np.arctan2(120.0 - bc[0], 40.0 - bc[1])
        rel_tm_angle = (rel_tm_angle + np.pi) % (2 * np.pi) - np.pi
        features['angle_nearest_free_teammate'] = rel_tm_angle
    else:
        features['dist_nearest_free_teammate'] = 50.0
        features['angle_nearest_free_teammate'] = 0.0
    
    # Progressive option: any free teammate closer to opponent goal
    features['has_progressive_option'] = 0
    if len(free_tms) > 0:
        free_tms_arr = np.array(free_tms)
        # Opponent goal is at x=120
        if np.any(free_tms_arr[:, 0] > bc[0]):
            features['has_progressive_option'] = 1
    
    # Expected threat at ball-carrier location
    features['xt_value'] = xt_value(bc[0], bc[1])
    
    # Pitch zone (6x4 grid)
    zone_x = int(np.clip(bc[0] / 20.0, 0, 5))  # 6 zones along x
    zone_y = int(np.clip(bc[1] / 20.0, 0, 3))  # 4 zones along y
    features['zone'] = zone_x * 4 + zone_y  # Single integer 0-23
        
    if match_context:
        features['game_state_diff'] = match_context.get('game_state_diff', 0)
        features['minutes_elapsed'] = match_context.get('minutes_elapsed', 0)
        features['match_period'] = match_context.get('match_period', 1)  # 1=first half, 2=second half
    else:
        features['game_state_diff'] = 0
        features['minutes_elapsed'] = 0
        features['match_period'] = 1
        
    return features
