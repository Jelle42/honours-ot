from sinkhorn import sinkhorn
import numpy as np
import plotly.graph_objects as go

def generateCircle(radius: float, n: int) -> np.ndarray:
    '''
    Randomly generate :n: points on the boundary of a circle in R^2
    '''
    
    result: list[np.ndarray] = []
    for _ in range(n):
        x = np.random.random() - 0.5
        y = np.random.random() - 0.5
        norm = np.sqrt((x**2 + y**2)) / radius
        result.append(np.array([x/norm, y/norm]))
    return np.array(result)

def generateSquare(radius: float, n: int) -> np.ndarray:
    '''
    Randomly generate n points on the boundary of a square in R^2
    '''
    result: list[np.ndarray] = []
    for _ in range(n):
        x = np.random.random() - 0.5
        y = np.random.random() - 0.5
        norm = max(abs(x), abs(y)) / radius
        result.append(np.array([x/norm, y/norm]))
    return np.array(result)

def generateSphere(radius: float, n: int) -> np.ndarray:
    '''
    Randomly generate :n: points on the boundary of a sphere in R^3
    '''
    result: list[np.ndarray] = []
    for _ in range(n):
        x = np.random.random() - 0.5
        y = np.random.random() - 0.5
        z = np.random.random() - 0.5
        norm = np.sqrt(x**2 + y**2 + z**2) / radius
        result.append(np.array([x/norm, y/norm, z/norm]))
    return np.array(result)
    
def generateCube(radius: float, n: int) -> np.ndarray:
    '''
    Randomly generate n points on the boundary of a cube in R^3
    '''
    result: list[np.ndarray] = []
    for _ in range(n):
        x = np.random.random() - 0.5
        y = np.random.random() - 0.5
        z = np.random.random() - 0.5
        norm = max(abs(x), abs(y), abs(z)) / radius
        result.append(np.array([x/norm, y/norm, z/norm]))
    return np.array(result)
    

def visualize2d(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float,
    num_particles: int = 1_000,
    num_frames: int = 120
    ) -> None:
    '''
    Given coefficients a_i, b_j and points x_i, y_j, visualize transport from 
    '''
    n = np.shape(source_points)[0]
    m = np.shape(target_points)[0]
    
    assert np.shape(source_mass)[0] == n, "you need to provide a mass for ALL points in :source_points:"
    assert np.shape(target_mass)[0] == m, "you need to provide a mass for ALL points in :target_points:"
    assert np.shape(source_points)[1] == np.shape(target_points)[1] == 2, "There is literally 2d in the name, why are you not inputting 2d points"
    
    # use euclidean distance to compute cost
    cost_matrix: np.ndarray = np.array([
        [np.linalg.norm(source_points[i] - target_points[j]) for j in range(m)]  for i in range(n)
    ])
    
    kernel = np.exp(-cost_matrix / gamma)
    
    P, sinkhorn_iterations = sinkhorn(kernel, source_mass, target_mass)
    P_norm: np.ndarray = P / np.sum(P)
    
    flat = P_norm.ravel()
    
    # randomly sample :num_particles: points wrt the rescaled P matrix as prob. distr.
    edge_indices = np.random.choice(
        len(flat),
        size=num_particles,
        p=flat
    )
    
    ii, jj = np.unravel_index(
        edge_indices,
        P.shape
    )
    
    starts: np.ndarray = source_points[ii]
    ends: np.ndarray = target_points[jj]
    
    # animate using plotly
    frames = []
    for frame in range(num_frames):
        t = frame / (num_frames - 1)
        pos: np.ndarray = (1-t)*starts + t*ends 
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=pos[:,0],
                        y=pos[:,1],
                        mode="markers",
                        marker=dict(size=2)
                    )
                ],
                name=str(frame)
            )
        )
    
    pos0 = starts
    
    scatter = go.Scatter(
        x=pos0[:,0],
        y=pos0[:,1],
        mode="markers",
        marker=dict(size=2)
    )
    
    source_cloud = go.Scatter(
        x=source_points[:,0],
        y=source_points[:,1],
        mode="markers",
        marker=dict(size=3),
        opacity=0.2,
        name="Source"
    )
    
    target_cloud = go.Scatter(
        x=target_points[:,0],
        y=target_points[:,1],
        mode="markers",
        marker=dict(size=3),
        opacity=0.2,
        name="Target"
    )
    
    fig = go.Figure(
        data=[
            source_cloud,
            target_cloud,
            scatter
        ],
        frames=frames
    )
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=10, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True
                            )
                        ]
                    )
                ]
            )
        ]
    )
    
    all_pts = np.vstack([source_points, target_points])

    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[xmin, xmax]),
            yaxis=dict(range=[ymin, ymax]),
            aspectmode="data"
        )
    )

    import os.path
    file_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            'ot_animation.html'
        )
    )
    
    fig.write_html(file_path)
    fig.show()
    
    

    

if __name__ == "__main__":
    n = 1000
    circle_points = generateCircle(1, n)
    circle_weights = np.array([1/n]*n)
    m = 800
    square_points = generateSquare(1, m)
    square_weights = np.array([1/m]*m)
    
    visualize2d(square_weights, circle_weights, square_points, circle_points, 0.2)