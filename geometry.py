import numpy as np

def generate_ball(radius: float, n: int, dim: int, ord) -> np.ndarray:
    """Generate :n: random points on the boundary of a ball of radius :radius: w.r.t. the :ord:-norm in R^:dim:."""
    rng = np.random.default_rng()
    points = rng.normal(size=(n,dim))
    norms = np.linalg.norm(points, axis=1, keepdims=True, ord=ord)
    norms[norms == 0] = 1.0
    return radius * points / norms

__all__ = [
    "generate_ball",
]
