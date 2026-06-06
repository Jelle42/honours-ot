import numpy as np

def sinkhorn(kernel: np.ndarray, source_mass: np.ndarray, target_mass: np.ndarray, tolerance: float = 1e-9, eps: float = 1e-12) -> tuple[np.ndarray, int]:
    '''
    Computes OT matrix P by finding u and v so that P = diag(u) K diag(v)
    '''
    n, m = np.shape(kernel)

    assert np.shape(source_mass)[0] == n, "source_mass or kernel is not of desired shape"
    assert np.shape(target_mass)[0] == m, "target_mass or kernel is not of desired shape"

    assert np.abs(np.sum(source_mass) - 1) < eps, "source_mass should sum to 1"
    assert np.abs(np.sum(target_mass) - 1) < eps, "target_mass should sum to 1"
    
    col_scaling = np.ones(m)
    row_scaling = np.ones(n)

    iteration = 0
    while True:
        row_scaling = source_mass / (kernel @ col_scaling)
        col_scaling = target_mass / (kernel.T @ row_scaling)

        iteration += 1
        
        transport = np.diag(row_scaling) @ kernel @ np.diag(col_scaling)
        
        row_err = np.max(np.abs(np.sum(transport, 1) - source_mass))
        col_err = np.max(np.abs(np.sum(transport, 0) - target_mass))

        if row_err < tolerance and col_err < tolerance:
            break

    # print(f"sinkhorn iter: {iteration}")
    
    return transport, iteration


if __name__ == "__main__":
    np.set_printoptions(precision=6)
    
    a = np.array([0.4, 0.2, 0.2, 0.1, 0.05, 0.05])
    b = np.array([0.1, 0.3, 0.4, 0.2])

    C = np.array([
        [0.1, 0.2, 0.3, 1],
        [0.3, np.pi/4, 0.3, 1],
        [1, 0.2, 0.4, 1],
        [0.2, 0.4, 0.5, 1],
        [1, 1, 1, 1],
        [0, 2, 4, 10]
    ])

    gamma = 2
    K = np.exp(C / gamma)

    P, i = sinkhorn(K, a, b)
    print(P)
    print(f"Took {i} iterations")
    
    print(np.sum(P, 1))
    print(a)
    print(np.sum(P, 0))
    print(b)