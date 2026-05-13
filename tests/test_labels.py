"""Tests for src.data.labels.define_success."""
import numpy as np
import pandas as pd
import pytest

from src.data.labels import define_success


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_events(*rows):
    """Build a minimal events DataFrame sorted by timestamp."""
    df = pd.DataFrame(list(rows))
    return df.reset_index(drop=True)


def base(id_, type_, team, ts, **kw):
    """Minimal event row with required columns."""
    return {
        'id': id_, 'type': type_, 'team_id': team,
        'match_id': 'm1', 'timestamp': ts,
        'pass_outcome': np.nan, 'pass_recipient': np.nan,
        'dribble_outcome': np.nan,
        **kw
    }


def paired(bc_event_id, team='team_a', opp='team_b'):
    return [{
        'match_id': 'm1',
        'pressure_event_id': 'press1',
        'ball_carrier_event_id': bc_event_id,
        'player_id': 'p1',
        'team_id': team,
        'opponent_team_id': opp,
        'frame_data': {},
        'event_timestamp': None,
    }]


# ── Pass ─────────────────────────────────────────────────────────────────────

class TestPass:
    def test_nan_outcome_with_recipient_is_success(self):
        events = make_events(
            base('e1', 'Pass', 'team_a', '00:01:00',
                 pass_outcome=np.nan, pass_recipient='p2'),
        )
        result = define_success(events, paired('e1'))
        assert result[0]['success'] == 1.0

    def test_non_nan_outcome_is_failure(self):
        events = make_events(
            base('e1', 'Pass', 'team_a', '00:01:00',
                 pass_outcome='Incomplete', pass_recipient=np.nan),
        )
        result = define_success(events, paired('e1'))
        assert result[0]['success'] == 0.0

    def test_nan_outcome_nan_recipient_next_same_team_is_success(self):
        events = make_events(
            base('e1', 'Pass', 'team_a', '00:01:00'),
            base('e2', 'Pass', 'team_a', '00:01:01'),
        )
        result = define_success(events, paired('e1'))
        assert result[0]['success'] == 1.0

    def test_nan_outcome_nan_recipient_next_opp_team_is_failure(self):
        events = make_events(
            base('e1', 'Pass', 'team_a', '00:01:00'),
            base('e2', 'Pass', 'team_b', '00:01:01'),
        )
        result = define_success(events, paired('e1'))
        assert result[0]['success'] == 0.0


# ── Dribble ───────────────────────────────────────────────────────────────────

class TestDribble:
    def test_complete_is_success(self):
        events = make_events(
            base('e1', 'Dribble', 'team_a', '00:01:00', dribble_outcome='Complete'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 1.0

    def test_incomplete_is_failure(self):
        events = make_events(
            base('e1', 'Dribble', 'team_a', '00:01:00', dribble_outcome='Incomplete'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 0.0


# ── Carry ─────────────────────────────────────────────────────────────────────

class TestCarry:
    def test_same_team_next_is_success(self):
        events = make_events(
            base('e1', 'Carry', 'team_a', '00:01:00'),
            base('e2', 'Pass',  'team_a', '00:01:01'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 1.0

    def test_opponent_possession_is_failure(self):
        events = make_events(
            base('e1', 'Carry', 'team_a', '00:01:00'),
            base('e2', 'Pass',  'team_b', '00:01:01'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 0.0

    def test_opponent_foul_is_success(self):
        """Opponent fouling the carrier = free kick won."""
        events = make_events(
            base('e1', 'Carry',          'team_a', '00:01:00'),
            base('e2', 'Foul Committed', 'team_b', '00:01:01'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 1.0

    def test_dispossessed_is_failure(self):
        events = make_events(
            base('e1', 'Carry',        'team_a', '00:01:00'),
            base('e2', 'Dispossessed', 'team_a', '00:01:01'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 0.0

    def test_interception_is_failure(self):
        events = make_events(
            base('e1', 'Carry',         'team_a', '00:01:00'),
            base('e2', 'Interception',  'team_b', '00:01:01'),
        )
        assert define_success(events, paired('e1'))[0]['success'] == 0.0

    def test_empty_paired_returns_empty(self):
        events = make_events(base('e1', 'Carry', 'team_a', '00:01:00'))
        assert define_success(events, []) == []

    def test_unknown_event_id_skipped(self):
        events = make_events(base('e1', 'Carry', 'team_a', '00:01:00'))
        assert define_success(events, paired('MISSING')) == []
