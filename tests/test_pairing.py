"""Tests for pressure-to-ball-carrier event pairing."""
import pandas as pd

from src.data.pairing import pair_pressure_with_ball_carrier


def make_events(*rows):
    return pd.DataFrame(list(rows))


def make_frames(*event_ids):
    return pd.DataFrame([
        {'event_uuid': event_id, 'freeze_frame': [{'location': [60.0, 40.0], 'actor': True, 'teammate': True}]}
        for event_id in event_ids
    ])


def test_uses_ball_carrier_related_event_not_first_related_id():
    events = make_events(
        {'id': 'pressure_1', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['duel_1', 'pass_1']},
        {'id': 'duel_1', 'type': 'Duel', 'team_id': 'B'},
        {'id': 'pass_1', 'type': 'Pass', 'team_id': 'A', 'player_id': 'p1'},
    )

    result = pair_pressure_with_ball_carrier(events, make_frames('pass_1'))

    assert len(result) == 1
    assert result[0]['ball_carrier_event_id'] == 'pass_1'
    assert result[0]['player_id'] == 'p1'


def test_ignores_same_team_related_pressure_action():
    events = make_events(
        {'id': 'pressure_1', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['carry_b', 'carry_a']},
        {'id': 'carry_b', 'type': 'Carry', 'team_id': 'B', 'player_id': 'defender'},
        {'id': 'carry_a', 'type': 'Carry', 'team_id': 'A', 'player_id': 'carrier'},
    )

    result = pair_pressure_with_ball_carrier(events, make_frames('carry_a', 'carry_b'))

    assert len(result) == 1
    assert result[0]['ball_carrier_event_id'] == 'carry_a'
    assert result[0]['player_id'] == 'carrier'


def test_collapses_multiple_pressures_on_same_carrier_action():
    events = make_events(
        {'id': 'pressure_1', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['pass_1']},
        {'id': 'pressure_2', 'type': 'Pressure', 'team_id': 'B', 'related_events': ['pass_1']},
        {'id': 'pass_1', 'type': 'Pass', 'team_id': 'A', 'player_id': 'p1'},
    )

    result = pair_pressure_with_ball_carrier(events, make_frames('pass_1'))

    assert len(result) == 1
    assert result[0]['pressure_event_id'] == 'pressure_1'
    assert result[0]['pressure_event_ids'] == ['pressure_1', 'pressure_2']
    assert result[0]['n_pressure_events'] == 2
