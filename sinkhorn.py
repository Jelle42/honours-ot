import numpy as np

def sinkhorn(K: np.ndarray, a: np.ndarray, b: np.ndarray, precision: float = 1e-3) -> tuple[np.ndarray, int]:
    '''
    Computes OT matrix P by finding u and v st P = diag(u) K diag(v)
    '''
    n, m = np.shape(K)

    assert np.shape(a)[0] == n, "a or K is not of desired shape"
    assert np.shape(b)[0] == m, "b or K is not of desired shape"

    assert np.abs(np.sum(a) - 1) < 1e-6, "a should sum to 1"
    assert np.abs(np.sum(b) - 1) < 1e-6, "b should sum to 1"
    
    v = np.ones(m)
    u = np.ones(n)

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
    a = np.array([0.5, 0.2, 0.2, 0.1])
    b = np.array([0.1, 0.3, 0.6])

    C = np.array([
        [0.1, 0.2, 0.3],
        [0.3, np.pi/4, 0.3],
        [1, 0.2, 0.4],
        [0.2, 0.4, 0.5]
    ])

    gamma = 2
    K = np.exp(C / gamma)

    res = sinkhorn(K, a, b)
    print(res[0])
    print(res[1])