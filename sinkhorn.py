import numpy as np

def sinkhorn(K: np.ndarray, a: np.ndarray, b: np.ndarray, precision: float = 1e-3) -> tuple[np.ndarray, int]:
    '''
    Computes OT matrix P by finding u and v st P = diag(u) K diag(v)
    '''
    n, m = np.shape(K)

    assert np.shape(a)[0] == m, "a or K is not of desired shape"
    assert np.shape(b)[0] == n, "b or K is not of desired shap"

    v = np.ones(m)
    u = np.zeros(n)

    iter = 0
    while True:
        v_prev = v
        u_prev = u
        u = a / (K @ v)
        v = b / (K.T @ u)

        iter += 1

        if np.linalg.norm(v_prev - v) + np.linalg.norm(u_prev - u) < precision:
            break

    return np.diag(u) @ K @ np.diag(v), iter


if __name__ == "__main__":
    a = np.array([0.5, 0.2, 0.3])
    b = np.array([0.1, 0.3, 0.6])

    C = np.array([
        np.array([1, 2, 0.3]),
        np.array([3, np.pi, 3]),
        np.array([1, 0, 4])
    ])

    gamma = 0.3

    K = np.exp(C / gamma)

    res = sinkhorn(K, a, b)
    print(res[0])
    print(res[1])