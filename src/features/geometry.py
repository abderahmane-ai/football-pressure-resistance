import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.ops import voronoi_diagram

def point_in_pitch(x, y, pitch_dims=(120.0, 80.0)):
    """Check if a point is within the pitch dimensions."""
    return 0 <= x <= pitch_dims[0] and 0 <= y <= pitch_dims[1]

def pitch_control_value(bc, teammates, opponents):
    """
    Simple additive pitch control model at ball-carrier location.
    Returns value in [-1, 1] range.
    """
    bc = np.array(bc)
    
    tm_influence = 0.0
    if len(teammates) > 0:
        tms = np.array(teammates)
        tm_dists_sq = np.sum((tms - bc)**2, axis=1)
        tm_influence = np.sum(1.0 / (1.0 + tm_dists_sq))
    
    opp_influence = 0.0
    if len(opponents) > 0:
        opps = np.array(opponents)
        opp_dists_sq = np.sum((opps - bc)**2, axis=1)
        opp_influence = np.sum(1.0 / (1.0 + opp_dists_sq))
    
    pc_value = tm_influence - opp_influence
    return np.clip(pc_value, -1.0, 1.0)

def xt_value(x, y):
    """
    Expected threat grid (12x8) based on Karun Singh's model.
    Simplified version with higher values closer to goal.
    """
    # 12 zones along x (0-120), 8 zones along y (0-80)
    x_idx = int(np.clip(x / 10.0, 0, 11))
    y_idx = int(np.clip(y / 10.0, 0, 7))
    
    # Simplified xT grid (higher values near opponent goal)
    xt_grid = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.05],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.04, 0.08],
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.06, 0.12],
        [0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.08, 0.15],
        [0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.08, 0.15],
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.06, 0.12],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.02, 0.04, 0.08],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.05],
    ])
    
    return xt_grid[y_idx, x_idx]

def voronoi_area(ball_carrier, all_players, pitch_dims=(120.0, 80.0)):
    """
    Calculate the area of the Voronoi cell for the ball carrier, clipped to the pitch.
    """
    pitch_polygon = Polygon([
        (0, 0), (pitch_dims[0], 0), 
        (pitch_dims[0], pitch_dims[1]), (0, pitch_dims[1])
    ])
    
    points = np.array(all_players)
    if len(points) < 4: # Voronoi needs at least 4 points to be well defined usually, or at least 3
        # Approximate with a grid if too few players
        return _grid_voronoi_area(ball_carrier, all_players, pitch_dims)
        
    # We use Shapely to create the Voronoi diagram
    from shapely.geometry import MultiPoint
    mp = MultiPoint(points)
    
    # It might fail if points are collinear, fallback to grid
    try:
        regions = voronoi_diagram(mp, envelope=pitch_polygon)
    except Exception:
        return _grid_voronoi_area(ball_carrier, all_players, pitch_dims)
        
    bc_point = Point(ball_carrier)
    for polygon in regions.geoms:
        if polygon.contains(bc_point):
            clipped = polygon.intersection(pitch_polygon)
            return clipped.area
            
    return 0.0

def _grid_voronoi_area(ball_carrier, all_players, pitch_dims=(120.0, 80.0)):
    """Grid-based approximation if Shapely fails or few points."""
    grid_res = 1.0
    xs = np.arange(0, pitch_dims[0], grid_res)
    ys = np.arange(0, pitch_dims[1], grid_res)
    xx, yy = np.meshgrid(xs, ys)
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    all_players = np.array(all_players)
    if len(all_players) == 0:
        return pitch_dims[0] * pitch_dims[1]
        
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

def angular_span(ball_carrier, opponents, radius=3.0):
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
        
    angles = np.arctan2(close_opps[:, 1] - bc[1], close_opps[:, 0] - bc[0])
    
    angles = np.sort(angles)
    if len(angles) == 1:
        return 0.5 # Default width for a single player
        
    angles_wrapped = np.append(angles, angles[0] + 2 * np.pi)
    gaps = np.diff(angles_wrapped)
    max_gap = np.max(gaps)
    
    span = 2 * np.pi - max_gap
    return span
