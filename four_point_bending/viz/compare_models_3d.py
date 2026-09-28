#!/usr/bin/env python3
"""Render ground truth and three RC-beam rollouts as synchronized 3D views."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm

from compare_models import (
    MODEL_SPECS,
    field_arrays,
    load_rollout,
    loading_centers_from_case_id,
    resolve_case_file,
    validate_shared_reference,
)


ROW_LABELS = {
    "Dual-Graph GConvGRU": "Dual-Graph",
    "Single-Graph E": "CGN-E",
    "Single-Graph N": "CGN-N",
}


def ordered_face(coords: np.ndarray, nodes: np.ndarray, plane: str) -> np.ndarray:
    if plane == "x":
        uv = coords[nodes][:, [2, 1]]
    elif plane == "y":
        uv = coords[nodes][:, [2, 0]]
    else:
        uv = coords[nodes][:, [0, 1]]
    center = uv.mean(axis=0)
    angle = np.arctan2(uv[:, 1] - center[1], uv[:, 0] - center[0])
    return nodes[np.argsort(angle)]


def visible_surface_faces(
    coords: np.ndarray, connectivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return front, top and end faces visible from the fixed camera."""
    element_coords = coords[connectivity]
    limits = {
        "x_min": float(coords[:, 0].min()),
        "y_max": float(coords[:, 1].max()),
        "z_min": float(coords[:, 2].min()),
        "z_max": float(coords[:, 2].max()),
    }
    definitions = (
        ("x", "min", limits["x_min"]),
        ("y", "max", limits["y_max"]),
        ("z", "min", limits["z_min"]),
        ("z", "max", limits["z_max"]),
    )
    axis_index = {"x": 0, "y": 1, "z": 2}
    face_elements: list[int] = []
    face_nodes: list[np.ndarray] = []
    for plane, _, value in definitions:
        axis = axis_index[plane]
        masks = np.isclose(
            element_coords[:, :, axis], value, rtol=0.0, atol=1.0e-8
        )
        element_ids = np.flatnonzero(masks.sum(axis=1) == 4)
        for element_id in element_ids:
            nodes = connectivity[element_id][masks[element_id]]
            face_elements.append(int(element_id))
            face_nodes.append(ordered_face(coords, nodes, plane))
    return np.asarray(face_elements, dtype=np.int64), np.stack(face_nodes)


def plot_vertices(
    coords: np.ndarray,
    displacement: np.ndarray,
    face_nodes: np.ndarray,
    frame: int,
    scale: float,
) -> np.ndarray:
    deformed = coords + scale * displacement[frame]
    # Plot coordinates: beam axis z, section depth x, vertical y.
    return deformed[face_nodes][:, :, [2, 0, 1]]


def oblique_projection(vertices: np.ndarray) -> np.ndarray:
    """Project (beam-axis z, section-depth x, vertical y) to a fixed 3D view."""
    projected = np.empty(vertices.shape[:-1] + (2,), dtype=np.float64)
    projected[..., 0] = vertices[..., 0] + 0.38 * vertices[..., 1]
    projected[..., 1] = vertices[..., 2] + 0.24 * vertices[..., 1]
    return projected


def cuboid_faces(
    z0: float, z1: float, x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    vertices = np.asarray(
        [
            [z0, x0, y0],
            [z1, x0, y0],
            [z1, x1, y0],
            [z0, x1, y0],
            [z0, x0, y1],
            [z1, x0, y1],
            [z1, x1, y1],
            [z0, x1, y1],
        ],
        dtype=np.float64,
    )
    return vertices[
        np.asarray(
            [
                [0, 1, 2, 3],
                [4, 7, 6, 5],
                [0, 4, 5, 1],
                [1, 5, 6, 2],
                [2, 6, 7, 3],
                [3, 7, 4, 0],
            ]
        )
    ]


def loading_block_y(
    coords: np.ndarray,
    displacement: np.ndarray,
    frame: int,
    center_z: float,
    scale: float,
) -> float:
    top_y = float(coords[:, 1].max())
    selected = (
        (np.abs(coords[:, 2] - center_z) <= 50.0)
        & np.isclose(coords[:, 1], top_y, rtol=0.0, atol=1.0e-5)
    )
    if not np.any(selected):
        return top_y

    deformed_top = coords[selected, 1] + scale * displacement[frame, selected, 1]
    # Keep the rigid block just above the highest node in its contact patch.  The
    # small clearance avoids projection/z-fighting without changing the scale.
    return float(deformed_top.max()) + 0.75


def render(
    *,
    case_id: str,
    gif_path: Path,
    png_path: Path,
    coords: np.ndarray,
    connectivity: np.ndarray,
    progress: np.ndarray,
    row_labels: list[str],
    row_displacements: list[np.ndarray],
    row_values: list[np.ndarray],
    target_values: np.ndarray,
    field_label: str,
    field_unit: str,
    load_centers: tuple[float, float],
    deformation_scale: float,
    fps: int,
    dpi: int,
    bands: int,
) -> None:
    face_elements, face_nodes = visible_surface_faces(coords, connectivity)
    vmin = 0.0
    vmax = max(float(target_values.max()), 1.0)
    levels = np.linspace(vmin, vmax, bands + 1)
    cmap = plt.get_cmap("jet", bands)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    figure, axes = plt.subplots(
        len(row_labels),
        1,
        figsize=(13.2, 8.6),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.075, right=0.90, top=0.94, bottom=0.035, hspace=0.12
    )
    axes = list(axes[:, 0])
    surface_artists: list[PolyCollection] = []
    loading_artists: list[list[PolyCollection]] = []

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    support_x0, support_x1 = x_min - 12.0, x_max + 12.0
    loading_x0, loading_x1 = x_min - 8.0, x_max + 8.0

    for row, (axis, label, displacement, values) in enumerate(
        zip(axes, row_labels, row_displacements, row_values, strict=True)
    ):
        surface = PolyCollection(
            oblique_projection(
                plot_vertices(
                    coords, displacement, face_nodes, 0, deformation_scale
                )
            ),
            linewidths=0.08,
            edgecolors=(0.05, 0.05, 0.05, 0.16),
            antialiased=True,
            zorder=2,
        )
        surface.set_facecolor(cmap(norm(np.clip(values[0, face_elements], vmin, vmax))))
        axis.add_collection(surface)
        surface_artists.append(surface)

        supports = []
        for center in (150.0, 2550.0):
            support = PolyCollection(
                oblique_projection(
                    cuboid_faces(
                        center - 50.0,
                        center + 50.0,
                        support_x0,
                        support_x1,
                        y_min - 34.0,
                        y_min,
                    )
                ),
                facecolors="#4d5965",
                edgecolors="#20262c",
                linewidths=0.35,
                zorder=1,
            )
            axis.add_collection(support)
            supports.append(support)

        loads = []
        for center in load_centers:
            base_y = loading_block_y(
                coords, displacement, 0, center, deformation_scale
            )
            block = PolyCollection(
                oblique_projection(
                    cuboid_faces(
                        center - 50.0,
                        center + 50.0,
                        loading_x0,
                        loading_x1,
                        base_y,
                        base_y + 32.0,
                    )
                ),
                facecolors="#5a6570",
                edgecolors="#20262c",
                linewidths=0.35,
                zorder=3,
            )
            axis.add_collection(block)
            loads.append(block)
        loading_artists.append(loads)

        axis.text(
            -0.018,
            0.50,
            label,
            transform=axis.transAxes,
            fontsize=7.7,
            rotation=90,
            ha="center",
            va="center",
            clip_on=False,
        )
        axis.set_xlim(z_min - 75.0, z_max + 0.38 * x_max + 75.0)
        axis.set_ylim(y_min - 55.0, y_max + 0.24 * x_max + 65.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_axis_off()

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colorbar_axis = figure.add_axes([0.92, 0.12, 0.014, 0.75])
    colorbar = figure.colorbar(mappable, cax=colorbar_axis)
    colorbar.set_label(f"{field_label} ({field_unit})", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    title = figure.suptitle("", fontsize=11)

    def update(frame: int):
        artists: list[object] = [title]
        for surface, loads, displacement, values in zip(
            surface_artists,
            loading_artists,
            row_displacements,
            row_values,
            strict=True,
        ):
            surface.set_verts(
                oblique_projection(
                    plot_vertices(
                        coords,
                        displacement,
                        face_nodes,
                        frame,
                        deformation_scale,
                    )
                )
            )
            surface.set_facecolor(
                cmap(norm(np.clip(values[frame, face_elements], vmin, vmax)))
            )
            artists.append(surface)
            for center, block in zip(load_centers, loads, strict=True):
                base_y = loading_block_y(
                    coords, displacement, frame, center, deformation_scale
                )
                block.set_verts(
                    oblique_projection(
                        cuboid_faces(
                            center - 50.0,
                            center + 50.0,
                            loading_x0,
                            loading_x1,
                            base_y,
                            base_y + 32.0,
                        )
                    )
                )
                artists.append(block)
        title.set_text(
            f"{case_id} — 3D ground truth and three-model rollout | "
            f"p={progress[frame]:.2f} | frame {frame}/{progress.size - 1} | "
            f"shared {field_label} scale {vmin:g}–{vmax:.4g} {field_unit}"
        )
        return tuple(artists)

    update(0)
    animation = FuncAnimation(
        figure,
        update,
        frames=progress.size,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    update(progress.size - 1)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi)
    plt.close(figure)

    print(f"visible surface faces: {face_nodes.shape[0]}")
    print(f"frames: {progress.size}")
    print(f"shared field range: {vmin:.6g} to {vmax:.6g} {field_unit}")
    print(f"wrote: {gif_path}")
    print(f"wrote: {png_path}")


def main() -> int:
    benchmark_root = Path(__file__).resolve().parents[1]
    output_root = benchmark_root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--case", default="Job-new0a0b0")
    parser.add_argument("--field", choices=("stress", "peeq"), default="stress")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=105)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 50 or args.bands < 2:
        parser.error("fps >= 1, dpi >= 50 and bands >= 2 are required")
    if args.gif is None:
        args.gif = output_root / f"{args.case}-three-model-{args.field}-3d.gif"
    if args.png is None:
        args.png = output_root / f"{args.case}-three-model-{args.field}-3d.png"

    rollouts = [
        load_rollout(
            spec.label,
            resolve_case_file(args.train_root / spec.directory, args.case),
        )
        for spec in MODEL_SPECS
    ]
    validate_shared_reference(rollouts)
    reference = rollouts[0]
    target_values, prediction_values, field_label, field_unit, _ = field_arrays(
        args.field, reference, rollouts
    )
    row_labels = [
        "Ground truth",
        *(ROW_LABELS[rollout.label] for rollout in rollouts),
    ]
    row_displacements = [
        reference.true_displacement_mm,
        *(rollout.displacement_mm for rollout in rollouts),
    ]
    row_values = [target_values, *prediction_values]
    render(
        case_id=args.case,
        gif_path=args.gif,
        png_path=args.png,
        coords=reference.coords_mm,
        connectivity=reference.connectivity,
        progress=reference.progress,
        row_labels=row_labels,
        row_displacements=row_displacements,
        row_values=row_values,
        target_values=target_values,
        field_label=field_label,
        field_unit=field_unit,
        load_centers=loading_centers_from_case_id(args.case),
        deformation_scale=args.deformation_scale,
        fps=args.fps,
        dpi=args.dpi,
        bands=args.bands,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
