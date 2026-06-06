import numpy as np

def generate_ball(radius: float, n: int, dim: int, ord) -> np.ndarray:
    """Generate :n: random points on the boundary of a ball of radius :radius: w.r.t. the :ord:-norm in R^:dim:."""
    rng = np.random.default_rng()
    points = rng.normal(size=(n,dim))
    norms = np.linalg.norm(points, axis=1, keepdims=True, ord=ord)
    norms[norms == 0] = 1.0
    return radius * points / norms

def generate_torus(inner_radius: float, outer_radius: float, n: int,) -> np.ndarray:
    '''Generate :n: random points on the boundary of a torus with r = :inner_radius: and R = :outer_radius:'''
    rng = np.random.default_rng()
    theta = rng.random(n) * 2 * np.pi
    phi = rng.random(n) * 2 * np.pi
    
    x = (outer_radius + inner_radius*np.sin(theta))*np.cos(phi)
    y = (outer_radius + inner_radius*np.sin(theta))*np.sin(phi)
    z = inner_radius*np.cos(theta)
    
    points = np.stack([x, y, z], axis=0).T
    print(points.shape)
    return points

def rotate(points: np.ndarray, yaw: float, pitch: float = 0.0, roll: float = 0.0):
    '''Rotate :points: among z-axis with :yaw:, among y-axis with :pitch: and among x-axis with :roll:'''
    n = points.shape[1]
    if n == 2:
        A = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
    elif n == 3:
        A = np.array([
            [np.cos(yaw)*np.sin(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
            [np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) + np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) - np.sin(yaw)*np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
        ])
    else:
        raise ValueError("Points should be either 2 or 3 dimensional")

    return points @ A.T

__all__ = [
    "generate_ball",
    "generate_torus",
    "rotate",
]
