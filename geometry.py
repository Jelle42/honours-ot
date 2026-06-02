import numpy as np


def generate_circle(radius: float, n: int) -> np.ndarray:
    """Generate n random points on the boundary of a circle in R^2."""
    rng = np.random.default_rng()
    angles = rng.random(n) * 2 * np.pi
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def generate_square(radius: float, n: int) -> np.ndarray:
    """Generate n random points on the boundary of a square in R^2."""
    rng = np.random.default_rng()
    points = rng.random((n, 2)) - 0.5
    scales = np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1]))
    scales[scales == 0] = 1.0
    points = points / scales[:, None] * radius
    return points


def generate_sphere(radius: float, n: int) -> np.ndarray:
    """Generate n random points on the boundary of a sphere in R^3."""
    rng = np.random.default_rng()
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return radius * points / norms


def generate_cube(radius: float, n: int) -> np.ndarray:
    """Generate n random points on the boundary of a cube in R^3."""
    rng = np.random.default_rng()
    points = rng.random((n, 3)) - 0.5
    scales = np.maximum.reduce([np.abs(points[:, 0]), np.abs(points[:, 1]), np.abs(points[:, 2])])
    scales[scales == 0] = 1.0
    return points / scales[:, None] * radius


__all__ = [
    "generate_circle",
    "generate_square",
    "generate_sphere",
    "generate_cube",
]
