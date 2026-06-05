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
