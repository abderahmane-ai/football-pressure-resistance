"""Tests for src.features.spatial.extract_spatial_features_from_frame."""
import numpy as np

from config import MODEL_FEATURE_COLUMNS_BASE, SPLINE_FEATURES
from src.features.spatial import extract_spatial_features_from_frame


def _frame(actor_loc, teammates, opponents, actor_id="p1", team_id="A"):
    """Build a minimal freeze frame dict."""
    ff = [{"location": actor_loc, "actor": True, "teammate": True, "player_id": actor_id}]
    for loc in teammates:
        ff.append({"location": loc, "actor": False, "teammate": True, "player_id": "tm"})
    for loc in opponents:
        ff.append({"location": loc, "actor": False, "teammate": False, "player_id": "op"})
    return {"freeze_frame": ff, "event_uuid": "test-uuid"}


class TestBasicExtraction:
    def test_returns_none_for_empty_frame(self):
        result = extract_spatial_features_from_frame(
            frame_data={"freeze_frame": []},
            ball_carrier_player_id="p1",
            team_id="A",
            opponent_team_id="B",
        )
        assert result is None

    def test_returns_none_for_nan_frame(self):
        result = extract_spatial_features_from_frame(
            frame_data=float("nan"),
            ball_carrier_player_id="p1",
            team_id="A",
            opponent_team_id="B",
        )
        assert result is None

    def test_returns_all_base_features(self):
        frame = _frame([60, 40], [[55, 38]], [[62, 40]])
        result = extract_spatial_features_from_frame(
            frame_data=frame,
            ball_carrier_player_id="p1",
            team_id="A",
            opponent_team_id="B",
        )
        assert result is not None
        base_cols = MODEL_FEATURE_COLUMNS_BASE + SPLINE_FEATURES
        for col in base_cols:
            assert col in result, f"Missing feature: {col}"


class TestOpponentFeatures:
    def test_nearest_opponent_distance(self):
        frame = _frame([60, 40], [], [[62, 40], [65, 40]])
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert abs(result["dist_nearest_opp"] - 2.0) < 1e-6
        assert abs(result["dist_2nd_nearest_opp"] - 5.0) < 1e-6

    def test_nearest_opponent_angle(self):
        # Carrier at 60,40. Goal at 120,40. Goal vector is [60, 0] (angle 0).
        # Nearest opponent is at [61, 41]. Opponent vector is [1, 1] (angle pi/4).
        # Relative angle should be pi/4.
        frame = _frame([60, 40], [], [[61, 41]])
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert abs(result["angle_nearest_opp"] - np.pi/4) < 1e-2

    def test_opponents_in_radii(self):
        frame = _frame([60, 40], [], [[60.5, 40], [61.5, 40], [63.5, 40], [66, 40]])
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert result["opps_within_1yd"] == 1
        assert result["opps_within_2yd"] == 2
        assert result["opps_within_4yd"] == 3
        assert result["opp_density_5yd"] == 3


class TestTeammateFeatures:
    def test_free_teammates_and_triangles(self):
        # 2 teammates. Opponent at [60, 10] (far from all).
        # Both teammates should be free.
        frame = _frame([60, 40], [[65, 45], [65, 35]], [[60, 10]])
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert result["n_free_teammates"] == 2
        # Triangle area of [60,40], [65,45], [65,35] is 0.5 * base * height = 0.5 * 10 * 5 = 25.0
        assert abs(result["max_free_triangle_area"] - 25.0) < 1e-6

    def test_progressive_option(self):
        # Teammate is progressive: tm[0] > bc[0] + 5.0 (66 > 60 + 5.0).
        # Opponent at [60, 10] (lane is clear).
        frame = _frame([60, 40], [[66, 40]], [[60, 10]])
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert result["has_progressive_option"] == 1

        # Teammate blocked by opponent at [63, 40] (lane clearance radius is 1.5)
        frame_blocked = _frame([60, 40], [[66, 40]], [[63, 40]])
        result_blocked = extract_spatial_features_from_frame(
            frame_blocked, "p1", "A", "B"
        )
        assert result_blocked is not None
        assert result_blocked["has_progressive_option"] == 0


class TestContextPropagation:
    def test_propagates_match_context(self):
        frame = _frame([60, 40], [], [])
        context = {"game_state_diff": -1, "minutes_elapsed": 75, "match_period": 2}
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B", match_context=context
        )
        assert result is not None
        assert result["game_state_diff"] == -1
        assert result["minutes_elapsed"] == 75
        assert result["match_period"] == 2

    def test_fallback_player_location(self):
        # No player with player_id="p1" or actor=True.
        # It should fall back to using the first teammate's location.
        ff = [
            {"location": [60, 40], "actor": False, "teammate": True, "player_id": "tm1"},
            {"location": [62, 40], "actor": False, "teammate": False, "player_id": "op1"},
        ]
        frame = {"freeze_frame": ff}
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert result["bc_x"] == 60.0
        assert result["bc_y"] == 40.0

    def test_supports_numpy_array_freeze_frame(self):
        ff = np.array([
            {"location": np.array([60, 40]), "actor": True, "teammate": True, "player_id": "p1"},
            {"location": np.array([62, 40]), "actor": False, "teammate": False, "player_id": "op1"},
        ])
        frame = {"freeze_frame": ff}
        result = extract_spatial_features_from_frame(
            frame, "p1", "A", "B"
        )
        assert result is not None
        assert result["bc_x"] == 60.0

