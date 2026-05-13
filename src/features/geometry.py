import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import voronoi_diagram

from config import SPATIAL_CONFIG

def point_in_pitch(x, y):
    """Check if a point is within the pitch dimensions."""
    return 0 <= x <= SPATIAL_CONFIG['pitch_length'] and 0 <= y <= SPATIAL_CONFIG['pitch_width']

def _gaussian_influence(point, players, max_radius=15.0):
    """
    Compute Gaussian influence of a list of players at a specific point.
    """
    if len(players) == 0:
        return 0.0
    players = np.array(players)
    dists = np.linalg.norm(players - point, axis=1)
    # Only consider players within max_radius
    close_mask = dists <= max_radius
    if not np.any(close_mask):
        return 0.0
    close_dists = dists[close_mask]
    sigma = SPATIAL_CONFIG['pitch_control_sigma']  # Fernandez/Bornn: 4.2 yards
    influence = np.sum(np.exp(-(close_dists**2) / (2 * sigma**2)))
    return influence

def pitch_control_value(bc, teammates, opponents):
    """
    Gaussian pitch control model at ball-carrier location.
    Returns value in [-1, 1] range representing net control.
    """
    bc = np.array(bc)
    tm_influence = _gaussian_influence(bc, teammates)
    opp_influence = _gaussian_influence(bc, opponents)
    
    total = tm_influence + opp_influence
    if total == 0:
        return 0.0
        
    pc_prob = tm_influence / total
    # Scale to [-1, 1]
    return (pc_prob * 2) - 1.0

# Authentic Karun Singh xT grid values
XT_GRID_VALUES = np.array([
    [0.00638, 0.00779, 0.00844, 0.00977, 0.01126, 0.01248, 0.01473, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792],
    [0.00681, 0.00878, 0.00942, 0.01121, 0.01237, 0.01288, 0.01552, 0.01918, 0.02412, 0.03401, 0.04647, 0.05401],
    [0.00750, 0.00941, 0.01078, 0.01258, 0.01406, 0.01579, 0.01890, 0.02410, 0.03107, 0.04618, 0.06456, 0.07409],
    [0.00793, 0.00938, 0.01124, 0.01357, 0.01479, 0.01692, 0.02058, 0.02636, 0.03328, 0.05051, 0.07604, 0.09327],
    [0.00793, 0.00938, 0.01124, 0.01357, 0.01479, 0.01692, 0.02058, 0.02636, 0.03328, 0.05051, 0.07604, 0.09327],
    [0.00750, 0.00941, 0.01078, 0.01258, 0.01406, 0.01579, 0.01890, 0.02410, 0.03107, 0.04618, 0.06456, 0.07409],
    [0.00681, 0.00878, 0.00942, 0.01121, 0.01237, 0.01288, 0.01552, 0.01918, 0.02412, 0.03401, 0.04647, 0.05401],
    [0.00638, 0.00779, 0.00844, 0.00977, 0.01126, 0.01248, 0.01473, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792]
])

def xt_value(x, y):
    """
    Expected threat grid based on Karun Singh's model.
    """
    pitch_len = SPATIAL_CONFIG['pitch_length']
    pitch_wid = SPATIAL_CONFIG['pitch_width']
    cols = SPATIAL_CONFIG['xt_grid_cols']
    rows = SPATIAL_CONFIG['xt_grid_rows']
    
    x_idx = int(np.clip((x / pitch_len) * cols, 0, cols - 1))
    y_idx = int(np.clip((y / pitch_wid) * rows, 0, rows - 1))
    
    return XT_GRID_VALUES[y_idx, x_idx]

def voronoi_area(ball_carrier, all_players):
    """
    Calculate the area of the Voronoi cell for the ball carrier, clipped to the pitch.
    """
    pitch_len = SPATIAL_CONFIG['pitch_length']
    pitch_wid = SPATIAL_CONFIG['pitch_width']
    
    pitch_polygon = Polygon([
        (0, 0), (pitch_len, 0), 
        (pitch_len, pitch_wid), (0, pitch_wid)
    ])
    
    points = np.array(all_players)
    if len(points) < 4:
        return _grid_voronoi_area(ball_carrier, all_players)
        
    from shapely.geometry import MultiPoint
    mp = MultiPoint(points)
    
    try:
        regions = voronoi_diagram(mp, envelope=pitch_polygon)
    except Exception:
        return _grid_voronoi_area(ball_carrier, all_players)
        
    bc_point = Point(ball_carrier)
    for polygon in regions.geoms:
        if polygon.contains(bc_point):
            clipped = polygon.intersection(pitch_polygon)
            return clipped.area
            
    return 0.0

def _grid_voronoi_area(ball_carrier, all_players):
    """Grid-based approximation if Shapely fails or few points."""
    pitch_len = SPATIAL_CONFIG['pitch_length']
    pitch_wid = SPATIAL_CONFIG['pitch_width']
    grid_res = 1.0
    
    xs = np.arange(0, pitch_len, grid_res)
    ys = np.arange(0, pitch_wid, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    all_players = np.array(all_players)
    if len(all_players) == 0:
        return pitch_len * pitch_wid
        
    distances = np.linalg.norm(grid_points[:, np.newaxis, :] - all_players[np.newaxis, :, :], axis=2)
    closest_player_idx = np.argmin(distances, axis=1)
    
    bc_idx = -1
    for i, p in enumerate(all_players):
        if np.allclose(p, ball_carrier):
            bc_idx = i
            break
            
    if bc_idx == -1:
        return 0.0
        
    bc_area = np.sum(closest_player_idx == bc_idx) * (grid_res ** 2)
    return bc_area

def angular_span(ball_carrier, opponents, radius):
    """
    Calculate the angular span (coverage arc) of opponents within a certain radius.
    """
    bc = np.array(ball_carrier)
    opps = np.array(opponents)
    
    if len(opps) == 0:
        return 0.0
        
    distances = np.linalg.norm(opps - bc, axis=1)
    close_opps = opps[distances <= radius]
    
    if len(close_opps) == 0:
        return 0.0
        
    # Note: arctan2 signature is (y, x). The previous code used (y, x).
    angles = np.arctan2(close_opps[:, 1] - bc[1], close_opps[:, 0] - bc[0])
    
    angles = np.sort(angles)
    if len(angles) == 1:
        # Principled trigonometric formula from methodology: 2 * arctan(player_width/2 / distance)
        dist = distances[distances <= radius][0]
        dist = max(dist, 0.01)  # Guard against division by zero
        player_width = SPATIAL_CONFIG['player_width']
        return 2.0 * np.arctan((player_width / 2.0) / dist)
        
    angles_wrapped = np.append(angles, angles[0] + 2 * np.pi)
    gaps = np.diff(angles_wrapped)
    max_gap = np.max(gaps)
    
    span = 2 * np.pi - max_gap
    return span

def lane_unblocked(start_point, end_point, opponents, clearance_radius=None):
    """
    Check if a passing lane from start to end is unblocked by opponents.
    """
    if clearance_radius is None:
        clearance_radius = SPATIAL_CONFIG['pass_clearance_radius']
        
    if len(opponents) == 0:
        return True
    
    lane = LineString([start_point, end_point])
    for opp in opponents:
        opp_point = Point(opp)
        if lane.distance(opp_point) <= clearance_radius:
            return False
    return True
