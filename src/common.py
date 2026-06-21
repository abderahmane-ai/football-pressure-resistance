"""Shared constants and validation primitives used across the PRS pipeline."""

import math

# ── Event type constants ────────────────────────────────────────────────────────
# Single source of truth for StatsBomb event type strings.  Using constants
# rather than raw strings means typos are caught at import time and IDE
# autocomplete / find-references works correctly.

EVENT_TYPE_PASS: str = "Pass"
EVENT_TYPE_CARRY: str = "Carry"
EVENT_TYPE_DRIBBLE: str = "Dribble"
EVENT_TYPE_SHOT: str = "Shot"
EVENT_TYPE_PRESSURE: str = "Pressure"
EVENT_TYPE_DUEL: str = "Duel"
EVENT_TYPE_CLEARANCE: str = "Clearance"
EVENT_TYPE_INTERCEPTION: str = "Interception"
EVENT_TYPE_DISPOSSESSED: str = "Dispossessed"
EVENT_TYPE_FOUL_COMMITTED: str = "Foul Committed"
EVENT_TYPE_OWN_GOAL_FOR: str = "Own Goal For"
EVENT_TYPE_BALL_RECEIPT: str = "Ball Receipt"

BALL_CARRIER_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_TYPE_PASS,
    EVENT_TYPE_CARRY,
    EVENT_TYPE_DRIBBLE,
})


# ── Validation helpers ──────────────────────────────────────────────────────────

def is_valid_loc(value: object) -> bool:
    """Return True when *value* is a non-null 2D coordinate [x, y].

    Handles ``None``, ``float('nan')``, missing ``__len__`` / ``__getitem__``,
    and elements that are NaN or null — the union of all edge cases encountered
    across the StatsBomb event schemas (numpy arrays, Python lists, pandas
    nulls, and malformed JSON exports).
    """
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        return False
    # type narrowing — we've verified __len__ and __getitem__
    val_seq = value  # type: ignore[assignment]
    if len(val_seq) < 2:  # type: ignore[arg-type]
        return False
    x, y = val_seq[0], val_seq[1]  # type: ignore[index]
    # pd.notna is the most reliable NaN/null check across numpy/float/None
    import pandas as pd
    return bool(pd.notna(x) and pd.notna(y))
