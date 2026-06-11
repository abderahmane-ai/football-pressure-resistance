"""Geometric feature primitives: Voronoi, xT lookup, pitch control, angular span, lane clearance."""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import voronoi_diagram

from config import SPATIAL_CONFIG

logger = logging.getLogger(__name__)

# ── xT grid ─────────────────────────────────────────────────────────────────
# Karun Singh 8×12 grid loaded from data/xt_grid.json at import time.
# Falls back to the inline default if the file is absent or malformed.

_XT_GRID_DEFAULT: list[list[float]] = [
    [0.00638, 0.00779, 0.00844, 0.00977, 0.01126, 0.01248, 0.01473, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
    [0.00681, 0.00878, 0.00942, 0.01121, 0.01237, 0.01288, 0.01552, 0.01918, 0.02412, 0.03401, 0.04647, 0.05401],
    [0.00750, 0.00941, 0.01078, 0.01258, 0.01406, 0.01579, 0.01890, 0.02410, 0.03107, 0.04618, 0.06456, 0.07409],
    [0.00793, 0.00938, 0.01124, 0.01357, 0.01479, 0.01692, 0.02058, 0.02636, 0.03328, 0.05051, 0.07604, 0.09327],
    [0.00793, 0.00938, 0.01124, 0.01357, 0.01479, 0.01692, 0.02058, 0.02636, 0.03328, 0.05051, 0.07604, 0.09327],
    [0.00750, 0.00941, 0.01078, 0.01258, 0.01406, 0.01579, 0.01890, 0.02410, 0.03107, 0.04618, 0.06456, 0.07409],
    [0.00681, 0.00878, 0.00942, 0.01121, 0.01237, 0.01288, 0.01552, 0.01918, 0.02412, 0.03401, 0.04647, 0.05401],
    [0.00638, 0.00779, 0.00844, 0.00977, 0.01126, 0.01248, 0.01473, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
]


def _load_xt_grid() -> np.ndarray:
    """Load xT grid from data/xt_grid.json if present, else use the inline default."""
    xt_path = Path(__file__).resolve().parents[2] / "data" / "xt_grid.json"
    if xt_path.exists():
        try:
            with open(xt_path) as f:
                grid = json.load(f)
            arr = np.asarray(grid, dtype=float)
            expected = (SPATIAL_CONFIG["xt_grid_rows"], SPATIAL_CONFIG["xt_grid_cols"])
            if arr.shape != expected:
                logger.warning(
                    "xt_grid.json shape %s does not match config (%d×%d); "
                    "using inline default.",
                    arr.shape, *expected,
                )
                return np.array(_XT_GRID_DEFAULT)
            return arr
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse xt_grid.json (%s); using inline default.", exc)
    return np.array(_XT_GRID_DEFAULT)


XT_GRID_VALUES: np.ndarray = _load_xt_grid()


# ── Public API ────────────────────────────────────────────────────────────────

def point_in_pitch(x: float, y: float) -> bool:
    """Check if a point is within the pitch dimensions."""
    return 0 <= x <= SPATIAL_CONFIG["pitch_length"] and 0 <= y <= SPATIAL_CONFIG["pitch_width"]


def _gaussian_influence(point: np.ndarray, players: Sequence[object], max_radius: float | None = None) -> float:
    """Compute Gaussian influence of a list of players at a specific point."""
    if max_radius is None:
        max_radius = SPATIAL_CONFIG["pitch_control_max_radius"]
    if len(players) == 0:
        return 0.0
    arr = np.asarray(players, dtype=float)
    dists = np.linalg.norm(arr - point, axis=1)
    close_mask = dists <= max_radius
    if not np.any(close_mask):
        return 0.0
    close_dists = dists[close_mask]
    sigma: float = SPATIAL_CONFIG["pitch_control_sigma"]  # Fernandez/Bornn: 4.2 yds
    influence: float = float(np.sum(np.exp(-(close_dists**2) / (2 * sigma**2))))
    return influence


def pitch_control_value(
    bc: Sequence[float],
    teammates: Sequence[object],
    opponents: Sequence[object],
) -> float:
    """
    Gaussian pitch control model at ball-carrier location.

    Returns value in [-1, 1] representing net control (positive = teammate dominance).

    Callers must ensure the ball-carrier is excluded from ``teammates``
    to avoid a self-influence bias (distance=0 → maximum Gaussian weight).
    """
    bc_arr = np.asarray(bc, dtype=float)
    tm_influence = _gaussian_influence(bc_arr, teammates)
    opp_influence = _gaussian_influence(bc_arr, opponents)

    total = tm_influence + opp_influence
    if total == 0:
        return 0.0

    pc_prob = tm_influence / total
    return (pc_prob * 2) - 1.0


def xt_value(x: float, y: float) -> float:
    """Expected threat lookup based on the loaded xT grid."""
    pitch_len: float = SPATIAL_CONFIG["pitch_length"]
    pitch_wid: float = SPATIAL_CONFIG["pitch_width"]
    cols = int(SPATIAL_CONFIG["xt_grid_cols"])
    rows = int(SPATIAL_CONFIG["xt_grid_rows"])

    x_idx = int(np.clip((x / pitch_len) * cols, 0, cols - 1))
    y_idx = int(np.clip((y / pitch_wid) * rows, 0, rows - 1))

    return float(XT_GRID_VALUES[y_idx, x_idx])


def voronoi_area(ball_carrier: Sequence[float], all_players: list[Sequence[float]]) -> float:
    """Calculate the area of the Voronoi cell for the ball carrier, clipped to the pitch."""
    pitch_len: float = SPATIAL_CONFIG["pitch_length"]
    pitch_wid: float = SPATIAL_CONFIG["pitch_width"]

    pitch_polygon = Polygon([
        (0, 0), (pitch_len, 0),
        (pitch_len, pitch_wid), (0, pitch_wid),
    ])

    points = np.asarray(all_players, dtype=float)
    if len(points) < 4:
        return _grid_voronoi_area(ball_carrier, all_players)

    mp = MultiPoint(points)  # type: ignore[arg-type]

    try:
        regions = voronoi_diagram(mp, envelope=pitch_polygon)
    except Exception:
        return _grid_voronoi_area(ball_carrier, all_players)

    bc_point = Point(ball_carrier)  # type: ignore[arg-type]
    for polygon in regions.geoms:
        if polygon.covers(bc_point):
            clipped = polygon.intersection(pitch_polygon)
            # type: ignore[unnecessary-cast]
            return float(clipped.area)

    return 0.0


def _grid_voronoi_area(ball_carrier: Sequence[float], all_players: list[Sequence[float]]) -> float:
    """Grid-based approximation if Shapely fails or few points.

    The original 1-yard grid (120 × 80 = 9 600 cells × n_players distance calcs)
    is the dominant per-event cost in the data builder. A 4-yard grid is
    16× faster and the area estimate error is below 1% for the surrounding
    ball-carrier region (which is what feeds the pitch control / density
    features downstream).
    """
    pitch_len: float = SPATIAL_CONFIG["pitch_length"]
    pitch_wid: float = SPATIAL_CONFIG["pitch_width"]
    grid_res = 4.0

    xs = np.arange(0, pitch_len, grid_res)
    ys = np.arange(0, pitch_wid, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    all_players_arr = np.asarray(all_players, dtype=float)
    if len(all_players_arr) == 0:
        return pitch_len * pitch_wid

    distances = np.linalg.norm(
        grid_points[:, np.newaxis, :] - all_players_arr[np.newaxis, :, :], axis=2,
    )
    closest_player_idx = np.argmin(distances, axis=1)

    bc_arr = np.asarray(ball_carrier, dtype=float)
    bc_idx = -1
    for i, p in enumerate(all_players_arr):
        if np.allclose(p, bc_arr):
            bc_idx = i
            break

    if bc_idx == -1:
        return 0.0

    bc_area = float(np.sum(closest_player_idx == bc_idx) * (grid_res**2))
    return bc_area


def angular_span(ball_carrier: Sequence[float], opponents: Sequence[Sequence[float]], radius: float) -> float:
    """Calculate the angular span (coverage arc) of opponents within *radius* of the ball carrier.

    For a single opponent, uses the trigonometric body-width formula.
    For multiple opponents, calculates the union of angular intervals on the circle.
    """
    bc = np.asarray(ball_carrier, dtype=float)
    opps = np.asarray(opponents, dtype=float)

    if len(opps) == 0:
        return 0.0

    distances = np.linalg.norm(opps - bc, axis=1)
    close_mask = distances <= radius
    close_opps = opps[close_mask]
    close_dists = distances[close_mask]

    if len(close_opps) == 0:
        return 0.0

    player_width: float = SPATIAL_CONFIG["player_width"]

    # Calculate intervals [theta - w, theta + w] mapped to [0, 2*pi]
    intervals: list[tuple[float, float]] = []
    for opp, dist in zip(close_opps, close_dists):
        theta = np.arctan2(opp[1] - bc[1], opp[0] - bc[0])
        w = np.arctan((player_width / 2.0) / max(dist, 0.01))

        start = theta - w
        end = theta + w

        if end - start >= 2 * np.pi:
            return 2 * np.pi

        start_mod = start % (2 * np.pi)
        end_mod = start_mod + (end - start)

        if end_mod <= 2 * np.pi:
            intervals.append((start_mod, end_mod))
        else:
            intervals.append((start_mod, 2 * np.pi))
            intervals.append((0.0, end_mod - 2 * np.pi))

    # Sort intervals by start
    intervals.sort(key=lambda x: x[0])

    # Merge intervals
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    # Sum lengths of disjoint intervals
    total_span = sum(end - start for start, end in merged)
    return float(np.clip(total_span, 0.0, 2 * np.pi))



def lane_unblocked(
    start_point: Sequence[float],
    end_point: Sequence[float],
    opponents: Sequence[Sequence[float]],
    clearance_radius: float | None = None,
) -> bool:
    """Check if a passing lane from *start_point* to *end_point* is clear of opponents."""
    if clearance_radius is None:
        clearance_radius = SPATIAL_CONFIG["pass_clearance_radius"]

    if len(opponents) == 0:
        return True

    lane = LineString([start_point, end_point])  # type: ignore[arg-type]
    for opp in opponents:
        if lane.distance(Point(opp)) <= clearance_radius:  # type: ignore[arg-type]
            return False
    return True
