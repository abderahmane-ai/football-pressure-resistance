"""Tests for geometry primitives: voronoi, pitch control, lane clearance, xT lookup."""
import numpy as np

from src.features.geometry import (
    angular_span,
    lane_unblocked,
    pitch_control_value,
    voronoi_area,
    xt_value,
)


def _point_in_pitch(x: float, y: float) -> bool:
    """Check if a point is within the pitch dimensions (120x80)."""
    return 0 <= x <= 120 and 0 <= y <= 80


class TestPointInPitch:
    def test_center_is_in_pitch(self):
        assert _point_in_pitch(60, 40) is True

    def test_origin_is_in_pitch(self):
        assert _point_in_pitch(0, 0) is True

    def test_outside_x_is_not_in_pitch(self):
        assert _point_in_pitch(121, 40) is False

    def test_outside_y_is_not_in_pitch(self):
        assert _point_in_pitch(60, 81) is False

    def test_negative_is_not_in_pitch(self):
        assert _point_in_pitch(-1, 40) is False


class TestXtValue:
    def test_origin_returns_lowest_value(self):
        corner = xt_value(0, 0)
        center = xt_value(60, 40)
        assert corner < center

    def test_near_goal_returns_higher_value(self):
        far = xt_value(10, 40)
        close = xt_value(110, 40)
        assert close > far

    def test_returns_nonnegative(self):
        for x in [0, 30, 60, 90, 120]:
            for y in [0, 20, 40, 60, 80]:
                assert xt_value(x, y) >= 0


class TestPitchControl:
    def test_no_players_returns_zero(self):
        assert pitch_control_value([60, 40], [], []) == 0.0

    def test_only_teammates_returns_positive(self):
        val = pitch_control_value([60, 40], [[58, 40], [62, 40]], [])
        assert val > 0

    def test_only_opponents_returns_negative(self):
        val = pitch_control_value([60, 40], [], [[58, 40], [62, 40]])
        assert val < 0

    def test_symmetric_players_returns_near_zero(self):
        val = pitch_control_value([60, 40], [[58, 40]], [[62, 40]])
        assert abs(val) < 0.3  # Approximately balanced

    def test_bounded_between_minus_one_and_one(self):
        val = pitch_control_value([60, 40], [[55, 35], [55, 45]], [[65, 35], [65, 45]])
        assert -1.0 <= val <= 1.0


class TestVoronoiArea:
    def test_single_player_gets_full_pitch(self):
        """Single player should get the entire pitch area."""
        area = voronoi_area([60, 40], [[60, 40]])
        assert area > 0

    def test_more_players_gives_smaller_cell(self):
        area_alone = voronoi_area([60, 40], [[60, 40]])
        area_crowd = voronoi_area(
            [60, 40],
            [[60, 40], [30, 20], [90, 60], [30, 60], [90, 20]],
        )
        assert area_crowd < area_alone

    def test_non_negative(self):
        area = voronoi_area([60, 40], [[60, 40], [30, 20]])
        assert area >= 0


class TestAngularSpanExtended:
    """Extended angular span tests covering body-width accumulation."""

    def test_two_close_opponents_accumulate_body_width(self):
        """Two opponents very close together should have a span ≈ sum of body widths, not a huge gap coverage."""
        bc = [60, 40]
        # Two opponents 0.5 yards apart, both 2 yards away
        opps = [[62, 39.75], [62, 40.25]]
        span = angular_span(bc, opps, radius=5.0)
        # Each individual arc is ≈ 2*arctan(0.3/2) ≈ 0.297 rad
        # Total should be close to 2 × 0.297 = 0.594, not 2π - tiny_gap
        assert span < np.pi  # Should not cover more than half the circle

    def test_evenly_spread_opponents_give_large_span(self):
        """4 opponents on cardinal directions = large coverage."""
        bc = [60, 40]
        opps = [[60, 41], [60, 39], [61, 40], [59, 40]]
        span = angular_span(bc, opps, radius=5.0)
        assert span > np.pi / 2  # At least a quarter circle

    def test_capped_at_two_pi(self):
        """Even with many opponents, span cannot exceed 2π."""
        bc = [60, 40]
        opps = [[60 + np.cos(a), 40 + np.sin(a)] for a in np.linspace(0, 2*np.pi, 20, endpoint=False)]
        span = angular_span(bc, opps, radius=5.0)
        assert span <= 2 * np.pi + 1e-9


class TestLaneUnblocked:
    def test_empty_opponents_always_clear(self):
        assert lane_unblocked([60, 40], [80, 40], []) is True

    def test_opponent_on_lane_blocks(self):
        # Opponent at [70, 40] is directly on the lane from [60,40] to [80,40]
        assert lane_unblocked([60, 40], [80, 40], [[70, 40]]) is False

    def test_opponent_far_from_lane_does_not_block(self):
        # Opponent at [70, 50] is 10 yards off the lane
        assert lane_unblocked([60, 40], [80, 40], [[70, 50]]) is True

    def test_custom_clearance_radius(self):
        # Opponent at [70, 41] is 1 yard from the lane
        assert lane_unblocked([60, 40], [80, 40], [[70, 41]], clearance_radius=0.5) is True
        assert lane_unblocked([60, 40], [80, 40], [[70, 41]], clearance_radius=1.5) is False
