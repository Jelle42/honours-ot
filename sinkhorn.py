import numpy as np

def sinkhorn(K: np.ndarray, a: np.ndarray, b: np.ndarray, precision: float = 1e-3):
    '''
    Computes OT matrix P by finding u and v st P = diag(u) K diag(v)
    '''
    n, m = np.shape(K)

    assert np.shape(a)[0] == m, "a or K is not of desired shape"
    assert np.shape(b)[0] == n, "b or K is not of desired shap"

    v = np.ones(n)

    while True:
        v_prev = v
        u_prev = u
        u = a / (K @ v)
        v = b / (K.T @ u)

        if np.linalg.norm(v_prev - v) + np.linalg.norm(u_prev - u) < precision:
            break

    return np.diag(u) @ K @ np.diag(v) 


if __name__ == "__main__":
    pass