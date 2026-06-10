"""Tests for builder helper functions: normalise_related_events, compute_intended_xt, compute_game_state."""
import numpy as np
import pandas as pd

from src.data.builder import compute_intended_xt
from src.data.pairing import _normalise_related_events


class TestNormaliseRelatedEvents:
    def test_list_passthrough(self):
        assert _normalise_related_events(["a", "b", "c"]) == ["a", "b", "c"]

    def test_string_json_parsed(self):
        assert _normalise_related_events("['a', 'b']") == ["a", "b"]

    def test_empty_list(self):
        assert _normalise_related_events([]) == []

    def test_none_returns_empty(self):
        assert _normalise_related_events(None) == []

    def test_numpy_array_converted(self):
        arr = np.array(["x", "y"])
        assert _normalise_related_events(arr) == ["x", "y"]

    def test_malformed_string_returns_empty(self):
        assert _normalise_related_events("not a list") == []

    def test_filters_non_string_items(self):
        result = _normalise_related_events(["a", 123, "b", None])
        assert result == ["a", "b"]


class TestComputeIntendedXt:
    def _make_events(self, *rows):
        return pd.DataFrame(list(rows)).reset_index(drop=True)

    def test_pass_uses_pass_end_location(self):
        events = self._make_events(
            {"id": "e1", "type": "Pass", "location": [60, 40],
             "pass_end_location": [90, 40], "team_id": "A"},
        )
        item = {"ball_carrier_event_id": "e1"}
        result = compute_intended_xt(item, events)
        assert result is not None
        assert result > 0

    def test_carry_uses_carry_end_location(self):
        events = self._make_events(
            {"id": "e1", "type": "Carry", "location": [60, 40],
             "carry_end_location": [70, 40], "team_id": "A"},
        )
        item = {"ball_carrier_event_id": "e1"}
        result = compute_intended_xt(item, events)
        assert result is not None
        assert result > 0

    def test_dribble_uses_next_event_location(self):
        """Dribbles don't have end_location; should fall through to next event."""
        events = self._make_events(
            {"id": "e1", "type": "Dribble", "location": [60, 40], "team_id": "A"},
            {"id": "e2", "type": "Carry", "location": [65, 40], "team_id": "A"},
        )
        item = {"ball_carrier_event_id": "e1"}
        result = compute_intended_xt(item, events)
        assert result is not None

    def test_missing_event_returns_none(self):
        events = self._make_events(
            {"id": "e1", "type": "Pass", "location": [60, 40], "team_id": "A"},
        )
        item = {"ball_carrier_event_id": "MISSING"}
        assert compute_intended_xt(item, events) is None

    def test_missing_location_uses_previous_event(self):
        events = self._make_events(
            {"id": "e0", "type": "Pass", "location": [50, 40],
             "end_location": [55, 40], "team_id": "A"},
            {"id": "e1", "type": "Carry", "location": None,
             "carry_end_location": None, "team_id": "A"},
        )
        item = {"ball_carrier_event_id": "e1"}
        # Should use previous event's end_location
        result = compute_intended_xt(item, events)
        assert result is not None


class TestRecentPressures:
    def test_recent_pressures_rolling_count(self):
        from src.data.events import process_single_match

        # Carrier events (Pass/Carry/Dribble) in chronological order
        # for player_id="p1", with pressures paired on e1, e3, e5.
        events_list = [
            {'id': 'e1', 'type': 'Pass', 'team_id': 'A', 'player_id': 'p1', 'location': [60.0, 40.0], 'pass': {'height': {'id': 1}}, 'minute': 5, 'period': 1, 'index': 1, 'match_id': 100},
            {'id': 'p_e1', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['e1'], 'index': 2, 'match_id': 100},
            {'id': 'e2', 'type': 'Carry', 'team_id': 'A', 'player_id': 'p1', 'location': [61.0, 40.0], 'carry_end_location': [62.0, 40.0], 'minute': 6, 'period': 1, 'index': 3, 'match_id': 100},
            {'id': 'e3', 'type': 'Pass', 'team_id': 'A', 'player_id': 'p1', 'location': [62.0, 40.0], 'pass': {'height': {'id': 1}}, 'minute': 7, 'period': 1, 'index': 4, 'match_id': 100},
            {'id': 'p_e3', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['e3'], 'index': 5, 'match_id': 100},
            {'id': 'e4', 'type': 'Carry', 'team_id': 'A', 'player_id': 'p1', 'location': [63.0, 40.0], 'carry_end_location': [64.0, 40.0], 'minute': 8, 'period': 1, 'index': 6, 'match_id': 100},
            {'id': 'e5', 'type': 'Pass', 'team_id': 'A', 'player_id': 'p1', 'location': [64.0, 40.0], 'pass': {'height': {'id': 1}}, 'minute': 9, 'period': 1, 'index': 7, 'match_id': 100},
            {'id': 'p_e5', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['e5'], 'index': 8, 'match_id': 100},
            {'id': 'e6', 'type': 'Carry', 'team_id': 'A', 'player_id': 'p1', 'location': [65.0, 40.0], 'carry_end_location': [66.0, 40.0], 'minute': 10, 'period': 1, 'index': 9, 'match_id': 100},
            {'id': 'e7', 'type': 'Pass', 'team_id': 'A', 'player_id': 'p1', 'location': [66.0, 40.0], 'pass': {'height': {'id': 1}}, 'minute': 11, 'period': 1, 'index': 10, 'match_id': 100},
        ]
        events = pd.DataFrame(events_list)

        frames = pd.DataFrame([
            {'event_uuid': 'e1', 'freeze_frame': [
                {'location': [60.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [61.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
            {'event_uuid': 'e2', 'freeze_frame': [
                {'location': [61.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [62.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
            {'event_uuid': 'e3', 'freeze_frame': [
                {'location': [62.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [63.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
            {'event_uuid': 'e4', 'freeze_frame': [
                {'location': [63.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [64.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
            {'event_uuid': 'e5', 'freeze_frame': [
                {'location': [64.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [65.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
            {'event_uuid': 'e6', 'freeze_frame': [
                {'location': [65.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [66.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
            {'event_uuid': 'e7', 'freeze_frame': [
                {'location': [66.0, 40.0], 'actor': True, 'teammate': True, 'player_id': 'p1'},
                {'location': [67.0, 40.0], 'actor': False, 'teammate': False, 'player_id': 'opp1'}
            ]},
        ])

        gk_ids = set()
        position_groups = {'p1': 'Midfielder'}

        args = (100, events, frames, gk_ids, position_groups, 'Test_Comp')
        rows = process_single_match(args)

        rp_by_id = {r['ball_carrier_event_id']: r['recent_pressures'] for r in rows}

        assert rp_by_id['e1'] == 0
        assert rp_by_id['e3'] == 1
        assert rp_by_id['e5'] == 2

