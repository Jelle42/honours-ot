import numpy as np

def sinkhorn(K: np.ndarray, C: np.ndarray, a: np.ndarray, b: np.ndarray, precision: float = 1e-3):
    '''
    Computes OT matrix P by finding u and v st P_ij^y = u_i e^(-C_ij/y) v_j
    '''
    n, m = np.shape(K)

    v = np.ones(n)

    while True:
        v_prev = v
        u_prev = u
        u = a / (K @ v)
        v = b / (K.T @ u)

        if np.linalg.norm(v_prev - v) + np.linalg.norm(u_prev - u) < precision:
            break