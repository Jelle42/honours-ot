import os
import numpy as np
import plotly.graph_objects as go
import imageio.v2 as imageio
import kaleido
from pathlib import Path

from sinkhorn import sinkhorn
from geometry import *

def figure_to_video(
    fig: go.Figure,
    output_file: str,
    fps: int = 60,
    width: int = 1000,
    height: int = 800,
) -> str:
    """
    Convert a Plotly figure with frames to a video file (MP4 or GIF).
    
    Args:
        fig: Plotly figure with animation frames
        output_file: Output path (must end with .mp4 or .gif)
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
    
    Returns:
        Path to output file
    """
    if kaleido is None:
        raise ImportError(
            "kaleido is required for video export. Install with: pip install kaleido"
        )
    
    if not fig.frames:
        raise ValueError("Figure has no animation frames")
    
    file_ext = Path(output_file).suffix.lower()
    if file_ext not in [".mp4", ".gif"]:
        raise ValueError("output_file must end with .mp4 or .gif")
    
    # Render each frame as an image
    frames_images = []
    
    # Initial frame
    img = fig.to_image(format="png", width=width, height=height)
    frames_images.append(imageio.imread(img))
    
    # Animation frames
    for frame_data in fig.frames:
        fig_copy = go.Figure(fig)
        if frame_data.data:
            trace_indices = list(frame_data.traces) if frame_data.traces is not None else list(range(len(frame_data.data)))
            for trace_idx, trace_data in zip(trace_indices, frame_data.data):
                if trace_idx < len(fig_copy.data):
                    fig_copy.data[trace_idx].update(trace_data)

        if frame_data.layout is not None:
            fig_copy.layout.update(frame_data.layout)

        img = fig_copy.to_image(format="png", width=width, height=height)
        frames_images.append(imageio.imread(img))
    
    # Write to file
    if file_ext == ".gif":
        imageio.mimwrite(output_file, frames_images, duration=20 / fps, loop=0)
    else:  # .mp4
        imageio.mimwrite(output_file, frames_images, fps=fps)
    
    print(f"Video saved to {output_file}")
    return output_file


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
    dot_size: float,
) -> list[go.Frame]:
    """Build Plotly animation frames for the particle transport in 2D."""
    return [
        go.Frame(
            data=[
                go.Scatter(
                    x=((1 - t) * starts + t * ends)[:, 0],
                    y=((1 - t) * starts + t * ends)[:, 1],
                    mode="markers",
                    marker=dict(size=dot_size),
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
    dot_size: float,
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
                    marker=dict(size=dot_size),
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
    dot_size: float = 3
) -> go.Figure:
    """Construct the Plotly figure with source/target clouds and particle frames."""
    source_cloud = go.Scatter(
        x=source_points[:, 0],
        y=source_points[:, 1],
        mode="markers",
        marker=dict(size=3),
        opacity=0.067,
        name="Source",
    )
    target_cloud = go.Scatter(
        x=target_points[:, 0],
        y=target_points[:, 1],
        mode="markers",
        marker=dict(size=dot_size),
        opacity=0.067,
        name="Target",
    )
    particle_scatter = go.Scatter(
        x=particle_positions[:, 0],
        y=particle_positions[:, 1],
        mode="markers",
        marker=dict(size=dot_size),
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
    dot_size: float = 3
) -> go.Figure:
    """Construct the Plotly figure with source/target clouds and particle frames."""
    source_cloud = go.Scatter3d(
        x=source_points[:, 0],
        y=source_points[:, 1],
        z=source_points[:, 2],
        mode="markers",
        marker=dict(size=dot_size),
        opacity=0.067,
        name="Source",
    )
    target_cloud = go.Scatter3d(
        x=target_points[:, 0],
        y=target_points[:, 1],
        z=target_points[:, 2],
        mode="markers",
        marker=dict(size=dot_size),
        opacity=0.067,
        name="Target",
    )
    particle_scatter = go.Scatter3d(
        x=particle_positions[:, 0],
        y=particle_positions[:, 1],
        z=particle_positions[:, 2],
        mode="markers",
        marker=dict(size=dot_size),
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
            aspectmode="cube",
        ),
    )
    return fig


def visualize2d(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float,
    dot_size: float = 3,
    num_particles: int = 1_000,
    num_frames: int = 120,
    output_file: str | None = None,
    show_plot: bool = True,
    video_output: str | None = None,
    video_fps: int = 24,
) -> None:
    """Visualize optimal transport between two 2D discrete measures.
    
    Args:
        video_output: If provided, export to video file (.mp4 or .gif)
        video_fps: Frames per second for video export
    """
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
    frames = build_transport_frames(starts, ends, num_frames, dot_size)
    fig = build_figure2d(source_points, target_points, frames, starts, dot_size)

    if output_file is not None:
        _write_html_output(fig, output_file, iterations)

    if video_output is not None:
        figure_to_video(fig, video_output, fps=video_fps)

    if show_plot:
        fig.show()


def visualize3d(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    gamma: float,
    dot_size: float,
    num_particles: int = 1_000,
    num_frames: int = 120,
    output_file: str | None = None,
    show_plot: bool = True,
    video_output: str | None = None,
    video_fps: int = 24,
) -> None:
    """Visualize optimal transport between two 3D discrete measures.
    
    Args:
        video_output: If provided, export to video file (.mp4 or .gif)
        video_fps: Frames per second for video export
    """
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
    frames = build_transport_frames3d(starts, ends, num_frames, dot_size)
    fig = build_figure3d(source_points, target_points, frames, starts, dot_size)

    if output_file is not None:
        _write_html_output(fig, output_file, iterations)

    if video_output is not None:
        figure_to_video(fig, video_output, fps=video_fps)

    if show_plot:
        fig.show()


if __name__ == "__main__":
    file_path = os.path.dirname(__file__)
    import time
    start = time.time()
    
    n = 1500
    sphere_points = generate_ball(1.0, n, 3, 2)
    sphere_weights = np.full(n, 1.0 / n)
    
    circle_points = generate_ball(1.0, n, 2, 2)
    circle_weights = np.full(n, 1.0 / n)

    m = 1700
    cube_points = generate_ball(1.0, m, 3, float("inf"))
    cube_weights = np.full(m, 1.0 / m)
    
    square_points = generate_ball(1.0, m, 2, float("inf"))
    square_weights = np.full(m, 1.0 / m)
    
    square2_points = rotate(generate_ball(1.0, m, 2, float("inf")), np.pi/4)
    square2_weights = np.full(m, 1.0 / m)
    
    torus_points = generate_torus(0.5, 1, m)
    torus_weights = np.full(m, 1.0 / m)
    
    torus2_points = rotate(generate_torus(0.5, 1, m), 0, np.pi/2, np.pi/2)
    torus2_weights = np.full(m, 1.0 / m)
    
    triangle_points = generate_polygon(
        np.array([
            [0,-1],
            [2,1],
            [-2,1],
        ]),
        np.array([0.5,2,2]),
        m
        )
    triangle_weights = np.full(m, 1.0 / m)
    
    print(f"Generating Shapes took {(time.time() - start):.4f} seconds")
    start = time.time()
    

    visualize2d(
        source_mass=square_weights,
        target_mass=triangle_weights,
        source_points=square_points,
        target_points=triangle_points,
        gamma=0.2,
        dot_size=3,
        show_plot=False,
        num_frames=60,
        num_particles=2000,
        video_output=file_path + f"/animations/ot_animation_2d_{10_000*np.random.random():.0f}.gif"
    )
    
    print(f"Rendering took {(time.time() - start):.4f} seconds")