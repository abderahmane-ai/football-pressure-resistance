"""Tests for src.features.geometry.angular_span."""
import numpy as np

from config import SPATIAL_CONFIG
from src.features.geometry import angular_span

RADIUS = 3.0
PLAYER_WIDTH = SPATIAL_CONFIG['player_width']


class TestAngularSpan:
    def test_no_opponents_returns_zero(self):
        assert angular_span([60.0, 40.0], [], radius=RADIUS) == 0.0

    def test_opponent_outside_radius_returns_zero(self):
        opp = [[60.0, 50.0]]  # 10 yards away
        assert angular_span([60.0, 40.0], opp, radius=RADIUS) == 0.0

    def test_single_opponent_uses_trig_formula(self):
        """Single opp: span = 2·arctan(player_width/2 / dist)."""
        bc = [60.0, 40.0]
        opp = [[62.0, 40.0]]  # 2 yards directly ahead
        result = angular_span(bc, opp, radius=RADIUS)
        expected = 2.0 * np.arctan((PLAYER_WIDTH / 2.0) / 2.0)
        assert abs(result - expected) < 1e-6

    def test_single_opponent_closer_gives_larger_span(self):
        bc = [60.0, 40.0]
        close = [[60.5, 40.0]]
        far   = [[62.0, 40.0]]
        assert angular_span(bc, close, radius=RADIUS) > angular_span(bc, far, radius=RADIUS)

    def test_opponents_spread_apart_give_larger_span(self):
        """Opponents on opposite sides of carrier block more total angle."""
        bc = [60.0, 40.0]
        together = [[62.0, 40.25], [62.0, 40.5]]    # Clustered
        spread   = [[62.0, 40.0], [60.0, 42.0]]     # Spread
        assert angular_span(bc, spread, radius=5.0) > angular_span(bc, together, radius=5.0)

    def test_full_surround_exceeds_pi(self):
        """4 opponents on cardinal points = heavily surrounded."""
        bc = [60.0, 40.0]
        opps = [
            [60.0, 40.3],  # N
            [60.0, 39.7],  # S
            [60.3, 40.0],  # E
            [59.7, 40.0],  # W
        ]
        assert angular_span(bc, opps, radius=5.0) > np.pi

    def test_result_is_non_negative(self):
        bc = [60.0, 40.0]
        opps = [[60.5, 40.5], [60.2, 39.8]]
        assert angular_span(bc, opps, radius=RADIUS) >= 0.0

    def test_result_bounded_below_two_pi(self):
        bc = [60.0, 40.0]
        opps = [[60.0 + 0.1 * i, 40.0 + 0.1 * j] for i in range(-2, 3) for j in range(-2, 3) if i or j]
        span = angular_span(bc, opps, radius=5.0)
        assert span <= 2 * np.pi + 1e-9
