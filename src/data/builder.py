import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsbombpy import sb
from concurrent.futures import ThreadPoolExecutor

from config import PROCESSED_DATA_DIR, COMPETITIONS, CROSS_VALIDATION_HOLDOUT, SPATIAL_CONFIG
from src.data.loader import load_all_competitions
from src.data.pairing import pair_pressure_with_ball_carrier
from src.data.labels import define_success
from src.features.spatial import extract_spatial_features_from_frame
from src.features.geometry import xt_value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_WORKERS = min(os.cpu_count() or 4, 8)

def compute_game_state_for_match(match_events):
    """
    Compute score differential (ball-carrier team - opponent) at the moment
    of every event in the match, using StatsBomb Shot/Own Goal events.
    Returns dict: event_id -> int score diff.
    """
    teams = match_events['team_id'].dropna().unique()
    if len(teams) != 2:
        return {}
    team_a, team_b = teams[0], teams[1]
    score = {team_a: 0, team_b: 0}
    event_state = {}

    for _, row in match_events.iterrows():
        event_id = row['id']
        team = row.get('team_id')
        opp = team_b if team == team_a else team_a
        event_state[event_id] = score.get(team, 0) - score.get(opp, 0)

        ev_type = row.get('type', '')
        if ev_type == 'Shot' and row.get('shot_outcome') == 'Goal':
            score[team] = score.get(team, 0) + 1
        elif ev_type == 'Own Goal For':
            score[team] = score.get(team, 0) + 1

    return event_state


def _process_single_match(args):
    """
    Module-level worker for parallel match processing.
    Returns a list of processed row dicts for one match.
    """
    match_id, match_events, frames_df, gk_ids, comp_name = args
    rows = []
    try:
        position_groups = get_player_position_groups(match_id, match_events)
        game_states = compute_game_state_for_match(match_events)
        paired_events = pair_pressure_with_ball_carrier(match_events, frames_df)
        labeled_events = define_success(match_events, paired_events)

        for item in labeled_events:
            player_id = item.get('player_id')
            if player_id in gk_ids:
                continue

            event_row = match_events[match_events['id'] == item['ball_carrier_event_id']]
            match_context = {}
            if not event_row.empty:
                ev = event_row.iloc[0]
                if 'minute' in ev: match_context['minutes_elapsed'] = ev['minute']
                if 'period' in ev: match_context['match_period'] = ev['period']
                match_context['game_state_diff'] = game_states.get(item['ball_carrier_event_id'], 0)

            features = extract_spatial_features_from_frame(
                frame_data=item['frame_data'],
                ball_carrier_player_id=player_id,
                team_id=item['team_id'],
                opponent_team_id=item['opponent_team_id'],
                match_context=match_context
            )
            if features is None:
                continue
            if features.get('dist_nearest_opp', 999) > SPATIAL_CONFIG['tight_pressure_radius']:
                continue

            intended_xt = compute_intended_xt(item, match_events)
            if intended_xt is None:
                continue

            player_name = (
                match_events[match_events['player_id'] == player_id]['player'].iloc[0]
                if player_id in match_events['player_id'].values
                else f"Player_{player_id}"
            )
            row = {
                'competition': comp_name,
                'match_id': item['match_id'],
                'pressure_event_id': item['pressure_event_id'],
                'ball_carrier_event_id': item['ball_carrier_event_id'],
                'player_id': player_id,
                'player_name': player_name,
                'position_group': position_groups.get(player_id, 'Midfielder'),
                'team_id': item['team_id'],
                'opponent_team_id': item['opponent_team_id'],
                'success': item['success'],
                'value_preserved': intended_xt,
            }
            row.update(features)
            rows.append(row)
    except Exception as e:
        logger.warning(f"Match {match_id} worker failed: {e}")
    return rows


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

def get_player_position_groups(match_id, match_events=None):
    """
    Get position group (Defender/Midfielder/Forward) for each player.
    Uses lineup data, and falls back to coordinate clustering if lineup is missing.
    """
    position_map = {}
    try:
        lineups = sb.lineups(match_id=match_id)
        for team_name, lineup_df in lineups.items():
            if 'positions' in lineup_df.columns:
                for _, player in lineup_df.iterrows():
                    player_id = player['player_id']
                    positions = player['positions']
                    
                    if isinstance(positions, list) and len(positions) > 0:
                        # Check all listed positions for the player
                        assigned = False
                        for pos_dict in positions:
                            if isinstance(pos_dict, dict):
                                pos_name = pos_dict.get('position', '').lower()
                                if any(x in pos_name for x in ['back', 'defender', 'wing back']):
                                    position_map[player_id] = 'Defender'
                                    assigned = True
                                    break
                                elif any(x in pos_name for x in ['forward', 'striker', 'wing', 'winger']):
                                    position_map[player_id] = 'Forward'
                                    assigned = True
                                    break
                                elif 'midfield' in pos_name:
                                    position_map[player_id] = 'Midfielder'
                                    assigned = True
                                    break
                        if not assigned:
                            position_map[player_id] = 'Midfielder'
    except Exception as e:
        logger.debug(f"Could not load lineups for match {match_id}: {e}")
        
    # Impute missing using event locations
    if match_events is not None:
        all_players = match_events['player_id'].dropna().unique()
        for pid in all_players:
            if pid not in position_map:
                player_events = match_events[(match_events['player_id'] == pid) & (match_events['location'].notna())]
                if not player_events.empty:
                    # Calculate average x coordinate (0 to 120)
                    locs = np.array(player_events['location'].tolist())
                    avg_x = np.mean(locs[:, 0])
                    third = SPATIAL_CONFIG['pitch_length'] / 3.0
                    if avg_x < third:
                        position_map[pid] = 'Defender'
                    elif avg_x > 2 * third:
                        position_map[pid] = 'Forward'
                    else:
                        position_map[pid] = 'Midfielder'
                else:
                    position_map[pid] = 'Midfielder'
                    
    return position_map

def compute_intended_xt(item, match_events):
    """
    Compute the intended expected threat (xT) of the action, regardless of success.
    This separates the value of the action from its outcome for the ZIB model.
    """
    bc_event_id = item['ball_carrier_event_id']
    bc_event_rows = match_events[match_events['id'] == bc_event_id]
    
    if bc_event_rows.empty:
        return None
        
    bc_event = bc_event_rows.iloc[0]
    bc_idx = bc_event_rows.index[0]
    
    bc_loc = bc_event.get('location')
    if not isinstance(bc_loc, (list, tuple)) or len(bc_loc) < 2:
        # Check previous event for location imputation
        if bc_idx > 0:
            prev_event = match_events.iloc[bc_idx - 1]
            if 'end_location' in prev_event and isinstance(prev_event['end_location'], (list, tuple)):
                bc_loc = prev_event['end_location']
            elif 'location' in prev_event and isinstance(prev_event['location'], (list, tuple)):
                bc_loc = prev_event['location']
            else:
                return None
        else:
            return None
            
    next_xt = xt_value(bc_loc[0], bc_loc[1])
    
    if bc_event['type'] == 'Pass' and 'pass_end_location' in bc_event:
        end_loc = bc_event['pass_end_location']
        if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2:
            next_xt = xt_value(end_loc[0], end_loc[1])
    elif bc_idx + 1 < len(match_events):
        next_event = match_events.iloc[bc_idx + 1]
        next_loc = next_event.get('location')
        if isinstance(next_loc, (list, tuple)) and len(next_loc) >= 2:
            next_xt = xt_value(next_loc[0], next_loc[1])
            
    return float(next_xt)

def build_all_datasets(include_holdout=False):
    """Build the complete processed dataset for all competitions (parallelised per match)."""
    logger.info("Building dataset for all competitions...")

    comp_names = list(COMPETITIONS.keys())
    if not include_holdout and CROSS_VALIDATION_HOLDOUT in comp_names:
        comp_names.remove(CROSS_VALIDATION_HOLDOUT)
        logger.info(f"Excluding holdout competition: {CROSS_VALIDATION_HOLDOUT}")

    all_comp_data = load_all_competitions(comp_names)
    all_processed_data = []
    comp_event_counts = {}

    for comp_name, comp_data in all_comp_data.items():
        logger.info(f"Processing {comp_name} with {N_WORKERS} workers...")
        events_df = comp_data['events']
        frames_dict = comp_data['frames']
        if events_df.empty:
            continue

        match_ids = events_df['match_id'].unique()

        # Pre-fetch goalkeeper IDs in parallel (I/O-bound API calls)
        gk_ids_by_match = {}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(get_goalkeeper_ids, mid): mid for mid in match_ids}
            for fut in tqdm(futures, desc=f"Loading lineups ({comp_name})"):
                gk_ids_by_match[futures[fut]] = fut.result()

        # Build args list, skipping matches with no 360 data
        worker_args = [
            (mid,
             events_df[events_df['match_id'] == mid].copy().reset_index(drop=True),
             frames_dict[mid],
             gk_ids_by_match.get(mid, set()),
             comp_name)
            for mid in match_ids
            if mid in frames_dict and not frames_dict[mid].empty
        ]

        comp_events = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for batch in tqdm(
                ex.map(_process_single_match, worker_args),
                total=len(worker_args),
                desc=f"Processing matches ({comp_name})"
            ):
                all_processed_data.extend(batch)
                comp_events += len(batch)

        comp_event_counts[comp_name] = comp_events
        logger.info(f"{comp_name}: {comp_events} events processed.")

    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PROCESSED_DATA_DIR / "all_pressure_dataset.parquet"
        dataset_df.to_parquet(out_file)
        logger.info(f"Saved dataset with {len(dataset_df)} events to {out_file}")
        return dataset_df
    return None

def build_holdout_dataset():
    """Build dataset for holdout competition only (parallelised per match)."""
    logger.info(f"Building holdout dataset: {CROSS_VALIDATION_HOLDOUT}")
    all_comp_data = load_all_competitions([CROSS_VALIDATION_HOLDOUT])
    all_processed_data = []

    for comp_name, comp_data in all_comp_data.items():
        events_df = comp_data['events']
        frames_dict = comp_data['frames']
        if events_df.empty:
            continue

        match_ids = events_df['match_id'].unique()

        # Pre-fetch GK IDs in parallel
        gk_ids_by_match = {}
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(get_goalkeeper_ids, mid): mid for mid in match_ids}
            for fut in tqdm(futures, desc="Loading holdout lineups"):
                gk_ids_by_match[futures[fut]] = fut.result()

        worker_args = [
            (mid,
             events_df[events_df['match_id'] == mid].copy().reset_index(drop=True),
             frames_dict[mid],
             gk_ids_by_match.get(mid, set()),
             comp_name)
            for mid in match_ids
            if mid in frames_dict and not frames_dict[mid].empty
        ]

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            for batch in tqdm(
                ex.map(_process_single_match, worker_args),
                total=len(worker_args),
                desc="Processing holdout matches"
            ):
                all_processed_data.extend(batch)

    if all_processed_data:
        dataset_df = pd.DataFrame(all_processed_data)
        out_file = PROCESSED_DATA_DIR / "holdout_pressure_dataset.parquet"
        dataset_df.to_parquet(out_file)
        logger.info(f"Saved holdout with {len(dataset_df)} events to {out_file}")

if __name__ == "__main__":
    build_all_datasets(include_holdout=False)
    build_holdout_dataset()
