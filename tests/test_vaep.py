
import pandas as pd

from src.features.vaep import (
    _compute_labels,
    _extract_state_features,
)


def test_extract_state_features() -> None:
    events = pd.DataFrame([
        {
            "id": "e1",
            "type": "Pass",
            "location": [60.0, 40.0],
            "pass": {"height": {"id": 1, "name": "Ground"}},
            "under_pressure": True,
            "goal_diff": 1,
            "team_id": 1,
            "match_id": 101,
        },
        {
            "id": "e2",
            "type": "Shot",
            "location": [110.0, 38.0],
            "under_pressure": False,
            "goal_diff": 1,
            "team_id": 1,
            "match_id": 101,
        },
        {
            "id": "e3",
            "type": "Carry",
            "location": None,
            "team_id": 1,
            "match_id": 101,
        }
    ])

    features = _extract_state_features(events)
    assert len(features) == 3
    assert features.loc[0, "loc_x"] == 60.0 / 120.0
    assert features.loc[0, "loc_y"] == 40.0 / 80.0
    assert features.loc[0, "is_pass"] == 1.0
    assert features.loc[0, "pass_ground"] == 1.0
    assert features.loc[0, "pass_high"] == 0.0
    assert features.loc[0, "under_pressure"] == 1.0
    assert pd.isna(features.loc[2, "loc_x"]) or "_has_location" not in features.loc[2] or features.loc[2].isna().all()

def test_compute_labels() -> None:
    events = pd.DataFrame([
        {"id": "e1", "type": "Pass", "team_id": "TeamA", "match_id": 101, "index": 1},
        {"id": "e2", "type": "Carry", "team_id": "TeamA", "match_id": 101, "index": 2},
        {"id": "e3", "type": "Shot", "shot_outcome": "Goal", "team_id": "TeamA", "match_id": 101, "index": 3},
        {"id": "e4", "type": "Pass", "team_id": "TeamB", "match_id": 101, "index": 4},
        {"id": "e5", "type": "Shot", "shot_outcome": "Goal", "team_id": "TeamB", "match_id": 101, "index": 5},
    ])

    labels = _compute_labels(events, lookahead=3)
    assert len(labels) == 5
    # e1 (TeamA) is followed by e3 (TeamA goal) within 3 actions -> scores_next = 1
    assert labels.loc[labels["id"] == "e1", "scores_next"].values[0] == 1
    assert labels.loc[labels["id"] == "e1", "concedes_next"].values[0] == 0

    # e2 (TeamA) is followed by e3 (TeamA goal) within 2 actions -> scores_next = 1
    assert labels.loc[labels["id"] == "e2", "scores_next"].values[0] == 1

    # e4 (TeamB) is followed by e5 (TeamB goal) -> scores_next = 1
    assert labels.loc[labels["id"] == "e4", "scores_next"].values[0] == 1
    assert labels.loc[labels["id"] == "e4", "concedes_next"].values[0] == 0

    # Add a case where TeamB does an action, but TeamA scores next (concede)
    events2 = pd.DataFrame([
        {"id": "e1", "type": "Pass", "team_id": "TeamB", "match_id": 101, "index": 1},
        {"id": "e2", "type": "Shot", "shot_outcome": "Goal", "team_id": "TeamA", "match_id": 101, "index": 2},
    ])
    labels2 = _compute_labels(events2, lookahead=2)
    # e1 (TeamB) is followed by e2 (TeamA Goal) -> concedes_next = 1, scores_next = 0
    assert labels2.loc[labels2["id"] == "e1", "concedes_next"].values[0] == 1
    assert labels2.loc[labels2["id"] == "e1", "scores_next"].values[0] == 0

    # Add a case where TeamB does an action, and TeamB scores next (score)
    # First event of the match (index 0) is by TeamA, so teams[0] = TeamA
    events3 = pd.DataFrame([
        {"id": "e1", "type": "Pass", "team_id": "TeamA", "match_id": 101, "index": 1},
        {"id": "e2", "type": "Pass", "team_id": "TeamA", "match_id": 101, "index": 2},
        {"id": "e3", "type": "Pass", "team_id": "TeamB", "match_id": 101, "index": 3},
        {"id": "e4", "type": "Pass", "team_id": "TeamB", "match_id": 101, "index": 4},
        {"id": "e5", "type": "Shot", "shot_outcome": "Goal", "team_id": "TeamB", "match_id": 101, "index": 5},
    ])
    labels3 = _compute_labels(events3, lookahead=3)
    # e3 (TeamB) is followed by e5 (TeamB Goal) -> scores_next = 1, concedes_next = 0
    assert labels3.loc[labels3["id"] == "e3", "scores_next"].values[0] == 1
    assert labels3.loc[labels3["id"] == "e3", "concedes_next"].values[0] == 0
