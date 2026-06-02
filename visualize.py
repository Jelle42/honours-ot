import io
import os

import numpy as np
import plotly.graph_objects as go
import imageio.v3 as iio
from PIL import Image

from sinkhorn import sinkhorn
from geometry import generate_circle, generate_cube, generate_square, generate_sphere


def compute_cost_matrix(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance cost matrix."""
    return np.array([
        [np.linalg.norm(source_points[i] - target_points[j]) for j in range(target_points.shape[0])]
        for i in range(source_points.shape[0])
    ])


def compute_transport_plan(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, int]:
    """Compute the normalized transport plan using Sinkhorn iterations."""
    cost_matrix = compute_cost_matrix(source_points, target_points)
    kernel = np.exp(-cost_matrix / gamma)
    transport, iterations = sinkhorn(kernel, source_mass, target_mass)
    return transport / np.sum(transport), iterations


def sample_transport_edges(
    transport_plan: np.ndarray,
    num_particles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample particle transports from the transport plan."""
    flat = transport_plan.ravel()
    indices = np.random.choice(len(flat), size=num_particles, p=flat)
    return np.unravel_index(indices, transport_plan.shape)

def _write_html_output(fig: go.Figure, output_file: str | None, iterations: int) -> str | None:
    if output_file is None:
        return None
    fig.write_html(output_file)
    print(f"Saved animation to {output_file} (Sinkhorn iterations: {iterations})")
    return output_file


def build_transport_frames(
    starts: np.ndarray,
    ends: np.ndarray,
    num_frames: int,
) -> list[go.Frame]:
    """Build Plotly animation frames for the particle transport in 2D."""
    return [
        go.Frame(
            data=[
                go.Scatter(
                    x=((1 - t) * starts + t * ends)[:, 0],
                    y=((1 - t) * starts + t * ends)[:, 1],
                    mode="markers",
                    marker=dict(size=2),
                )
            ],
            name=str(frame_index),
            traces=[2],
        )
        for frame_index, t in enumerate(np.linspace(0.0, 1.0, num_frames))
    ]


def build_transport_frames3d(
    starts: np.ndarray,
    ends: np.ndarray,
    num_frames: int,
) -> list[go.Frame]:
    """Build Plotly animation frames for the particle transport in 3D."""
    return [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=((1 - t) * starts + t * ends)[:, 0],
                    y=((1 - t) * starts + t * ends)[:, 1],
                    z=((1 - t) * starts + t * ends)[:, 2],
                    mode="markers",
                    marker=dict(size=2),
                )
            ],
            name=str(frame_index),
            traces=[2],
        )
        for frame_index, t in enumerate(np.linspace(0.0, 1.0, num_frames))
    ]


def build_figure2d(
    source_points: np.ndarray,
    target_points: np.ndarray,
    frames: list[go.Frame],
    particle_positions: np.ndarray,
) -> go.Figure:
    """Construct the Plotly figure with source/target clouds and particle frames."""
    source_cloud = go.Scatter(
        x=source_points[:, 0],
        y=source_points[:, 1],
        mode="markers",
        marker=dict(size=3),
        opacity=0.67,
        name="Source",
    )
    target_cloud = go.Scatter(
        x=target_points[:, 0],
        y=target_points[:, 1],
        mode="markers",
        marker=dict(size=3),
        opacity=0.67,
        name="Target",
    )
    particle_scatter = go.Scatter(
        x=particle_positions[:, 0],
        y=particle_positions[:, 1],
        mode="markers",
        marker=dict(size=2),
        name="Particles",
    )

    stacked = np.vstack([source_points, target_points])
    fig = go.Figure(data=[source_cloud, target_cloud, particle_scatter], frames=frames)
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
                                frame=dict(duration=50, redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=True,
                            ),
                        ],
                    )
                ],
            )
        ],
        xaxis=dict(range=[stacked[:, 0].min(), stacked[:, 0].max()]),
        yaxis=dict(range=[stacked[:, 1].min(), stacked[:, 1].max()], scaleanchor="x", scaleratio=1),
    )
    return fig


def build_figure3d(
    source_points: np.ndarray,
    target_points: np.ndarray,
    frames: list[go.Frame],
    particle_positions: np.ndarray,
) -> go.Figure:
    """Construct the Plotly figure with source/target clouds and particle frames."""
    source_cloud = go.Scatter3d(
        x=source_points[:, 0],
        y=source_points[:, 1],
        z=source_points[:, 2],
        mode="markers",
        marker=dict(size=3),
        opacity=0.67,
        name="Source",
    )
    target_cloud = go.Scatter3d(
        x=target_points[:, 0],
        y=target_points[:, 1],
        z=target_points[:, 2],
        mode="markers",
        marker=dict(size=3),
        opacity=0.67,
        name="Target",
    )
    particle_scatter = go.Scatter3d(
        x=particle_positions[:, 0],
        y=particle_positions[:, 1],
        z=particle_positions[:, 2],
        mode="markers",
        marker=dict(size=2),
        name="Particles",
    )

    stacked = np.vstack([source_points, target_points])
    fig = go.Figure(data=[source_cloud, target_cloud, particle_scatter], frames=frames)
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
                                frame=dict(duration=50, redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=True,
                            ),
                        ],
                    )
                ],
            )
        ],
        scene=dict(
            xaxis=dict(range=[stacked[:, 0].min(), stacked[:, 0].max()]),
            yaxis=dict(range=[stacked[:, 1].min(), stacked[:, 1].max()]),
            zaxis=dict(range=[stacked[:, 2].min(), stacked[:, 2].max()]),
            aspectmode="data",
        ),
    )
    return fig


def visualize2d(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float,
    num_particles: int = 1_000,
    num_frames: int = 120,
    output_file: str | None = None,
    show_plot: bool = True,
) -> None:
    """Visualize optimal transport between two 2D discrete measures."""
    assert source_points.shape[1] == target_points.shape[1] == 2, (
        "visualize2d requires 2D points for both source and target."
    )
    assert source_mass.shape[0] == source_points.shape[0], (
        "source_mass must match the number of source points."
    )
    assert target_mass.shape[0] == target_points.shape[0], (
        "target_mass must match the number of target points."
    )

    transport_plan, iterations = compute_transport_plan(
        source_mass,
        target_mass,
        source_points,
        target_points,
        gamma,
    )

    row_indices, col_indices = sample_transport_edges(transport_plan, num_particles)
    starts = source_points[row_indices]
    ends = target_points[col_indices]
    frames = build_transport_frames(starts, ends, num_frames)
    fig = build_figure2d(source_points, target_points, frames, starts)

    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "ot_animation_2d.html")
    _write_html_output(fig, output_file, iterations)


    if show_plot:
        fig.show()


def visualize3d(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float,
    num_particles: int = 1_000,
    num_frames: int = 120,
    output_file: str | None = None,
    show_plot: bool = True,
) -> None:
    """Visualize optimal transport between two 3D discrete measures."""
    assert source_points.shape[1] == target_points.shape[1] == 3, (
        "visualize3d requires 3D points for both source and target."
    )
    assert source_mass.shape[0] == source_points.shape[0], (
        "source_mass must match the number of source points."
    )
    assert target_mass.shape[0] == target_points.shape[0], (
        "target_mass must match the number of target points."
    )

    transport_plan, iterations = compute_transport_plan(
        source_mass,
        target_mass,
        source_points,
        target_points,
        gamma,
    )

    row_indices, col_indices = sample_transport_edges(transport_plan, num_particles)
    starts = source_points[row_indices]
    ends = target_points[col_indices]
    frames = build_transport_frames3d(starts, ends, num_frames)
    fig = build_figure3d(source_points, target_points, frames, starts)

    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "ot_animation_3d.html")
    _write_html_output(fig, output_file, iterations)

    if show_plot:
        fig.show()


if __name__ == "__main__":
    file_path = os.path.dirname(__file__)
    
    n = 1000
    sphere_points = generate_sphere(1.0, n)
    sphere_weights = np.full(n, 1.0 / n)

    m = 1100
    cube_points = generate_cube(1.0, m)
    cube_weights = np.full(m, 1.0 / m)

    visualize3d(
        source_mass=cube_weights,
        target_mass=sphere_weights,
        source_points=cube_points,
        target_points=sphere_points,
        gamma=0.2,
        show_plot=False,
        output_file=file_path + "/ot_animation_3d.html"
    )
