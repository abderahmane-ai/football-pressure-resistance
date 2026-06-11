"""Tests for src.data.events.compute_game_state_for_match."""
import pandas as pd

from src.data.events import compute_game_state_for_match


def make_events(*rows):
    return pd.DataFrame(list(rows))


def ev(id_, type_, team, shot_outcome=None):
    return {'id': id_, 'type': type_, 'team_id': team, 'shot_outcome': shot_outcome}


class TestComputeGameState:
    def test_no_goals_all_zero(self):
        events = make_events(
            ev('e1', 'Pass', 'A'),
            ev('e2', 'Pass', 'B'),
        )
        state = compute_game_state_for_match(events)
        assert state['e1'] == 0
        assert state['e2'] == 0

    def test_state_recorded_before_goal(self):
        """Event at the shot moment should see pre-shot score."""
        events = make_events(
            ev('e1', 'Pass', 'A'),
            ev('e2', 'Shot', 'A', shot_outcome='Goal'),
            ev('e3', 'Pass', 'B'),
        )
        state = compute_game_state_for_match(events)
        assert state['e1'] == 0   # Before goal
        assert state['e2'] == 0   # At shot — state recorded before update
        assert state['e3'] == -1  # B is 0-1 down

    def test_multiple_goals_correct_diff(self):
        events = make_events(
            ev('e1', 'Shot', 'A', shot_outcome='Goal'),
            ev('e2', 'Shot', 'A', shot_outcome='Goal'),
            ev('e3', 'Pass', 'A'),
            ev('e4', 'Pass', 'B'),
        )
        state = compute_game_state_for_match(events)
        assert state['e3'] == 2   # A scored 2
        assert state['e4'] == -2  # B is 2 down

    def test_own_goal_for_credits_correct_team(self):
        events = make_events(
            ev('e1', 'Own Goal For', 'B'),
            ev('e2', 'Pass', 'A'),
            ev('e3', 'Pass', 'B'),
        )
        state = compute_game_state_for_match(events)
        assert state['e1'] == 0    # Before own goal
        assert state['e2'] == -1   # A is 0-1 down (B got the own goal)
        assert state['e3'] == 1    # B is 1-0 up

    def test_shot_outcome_name_fallback(self):
        """statsbombpy sometimes uses shot_outcome_name instead of shot_outcome."""
        events = make_events(
            {'id': 'e1', 'type': 'Shot', 'team_id': 'A',
             'shot_outcome': None, 'shot_outcome_name': 'Goal'},
            {'id': 'e2', 'type': 'Pass', 'team_id': 'B',
             'shot_outcome': None, 'shot_outcome_name': None},
        )
        state = compute_game_state_for_match(events)
        assert state['e2'] == -1  # Fallback correctly detected goal

    def test_single_team_returns_empty(self):
        events = make_events(ev('e1', 'Pass', 'A'))
        assert compute_game_state_for_match(events) == {}

    def test_non_goal_shot_not_counted(self):
        events = make_events(
            ev('e1', 'Shot', 'A', shot_outcome='Saved'),
            ev('e2', 'Pass', 'B'),
        )
        state = compute_game_state_for_match(events)
        assert state['e2'] == 0

    def test_uses_event_index_order_when_available(self):
        events = make_events(
            {'id': 'e2', 'type': 'Pass', 'team_id': 'B', 'shot_outcome': None, 'index': 2},
            {'id': 'e1', 'type': 'Shot', 'team_id': 'A', 'shot_outcome': 'Goal', 'index': 1},
        )
        state = compute_game_state_for_match(events)
        assert state['e1'] == 0
        assert state['e2'] == -1
