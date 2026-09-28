#!/usr/bin/env python3
"""Render one RC-beam HDF5 case as an Abaqus-style ground-truth GIF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import FancyArrowPatch, Rectangle


def exposed_side_faces(
    coords_mm: np.ndarray, connectivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return elements and ordered node ids on the minimum-x exterior face."""
    element_coords = coords_mm[connectivity]
    centroid_x = element_coords[:, :, 0].mean(axis=1)
    side_elements = np.flatnonzero(
        np.isclose(centroid_x, centroid_x.min(), rtol=0.0, atol=1.0e-8)
    )

    ordered_faces: list[np.ndarray] = []
    for element_index in side_elements:
        node_ids = connectivity[element_index]
        local = coords_mm[node_ids]
        face_mask = np.isclose(local[:, 0], local[:, 0].min(), atol=1.0e-8)
        face_nodes = node_ids[face_mask]
        if face_nodes.size != 4:
            raise ValueError(
                f"element {element_index} has {face_nodes.size} nodes on its side face"
            )
        yz = coords_mm[face_nodes][:, [2, 1]]
        center = yz.mean(axis=0)
        angles = np.arctan2(yz[:, 1] - center[1], yz[:, 0] - center[0])
        ordered_faces.append(face_nodes[np.argsort(angles)])

    return side_elements, np.stack(ordered_faces)


def loading_centers_mm(coords_mm: np.ndarray, node_type: np.ndarray) -> tuple[float, float]:
    """Recover left/right loading-patch centers from canonical node flags."""
    z_values = np.unique(np.round(coords_mm[node_type == 1, 2], decimals=6))
    if z_values.size != 10:
        raise ValueError(f"expected 10 flagged z planes, found {z_values.size}")
    split = int(np.argmax(np.diff(z_values))) + 1
    if split != 5:
        raise ValueError("loading-surface flags do not form two five-plane patches")
    return float(z_values[:split].mean()), float(z_values[split:].mean())


def field_settings(
    handle: h5py.File, field: str, side_elements: np.ndarray
) -> tuple[np.ndarray, str, str, str]:
    if field == "stress":
        values = handle["response/element/solid/stress_scalar"][...] / 1.0e6
        return values[:, side_elements], "von Mises stress", "MPa", "{:.1f}"
    if field == "peeq":
        values = handle["response/element/solid/peeq"][...]
        return values[:, side_elements], "Equivalent plastic strain (PEEQ)", "–", "{:.4f}"
    raise ValueError(f"unsupported field: {field}")


def render(
    h5_path: Path,
    gif_path: Path,
    png_path: Path | None,
    *,
    field: str,
    fps: int,
    dpi: int,
    bands: int,
    deformation_scale: float,
) -> None:
    with h5py.File(h5_path, "r") as handle:
        case_id = str(handle["metadata"].attrs["case_id"])
        coords_mm = handle["nodes/coords"][...] * 1000.0
        displacement_mm = handle["response/node/displacement"][...] * 1000.0
        connectivity = handle["elements/solid/connectivity"][...]
        node_type = handle["nodes/node_type"][...]
        progress = handle["response/time/t"][...]
        head_down_mm = -handle["response/global/head_u2"][...] * 1000.0
        total_load_kn = -handle["response/global/rf2"][...] / 1000.0

        side_elements, face_nodes = exposed_side_faces(coords_mm, connectivity)
        values, field_label, field_unit, tick_format = field_settings(
            handle, field, side_elements
        )

    left_load, right_load = loading_centers_mm(coords_mm, node_type)
    n_frames = displacement_mm.shape[0]
    vmin = 0.0 if float(values.min()) >= 0.0 else float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    levels = np.linspace(vmin, vmax, bands + 1)
    cmap = plt.get_cmap("jet", bands)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    figure = plt.figure(figsize=(12.0, 5.9), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(3.25, 1.35))
    fringe_ax = figure.add_subplot(grid[0, 0])
    curve_ax = figure.add_subplot(grid[1, 0])

    def frame_positions(frame: int) -> np.ndarray:
        return coords_mm + deformation_scale * displacement_mm[frame]

    def face_vertices(frame: int) -> np.ndarray:
        deformed = frame_positions(frame)
        return deformed[face_nodes][:, :, [2, 1]]

    collection = PolyCollection(
        face_vertices(0),
        array=values[0],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        antialiased=False,
    )
    fringe_ax.add_collection(collection)
    colorbar = figure.colorbar(collection, ax=fringe_ax, pad=0.015, fraction=0.025)
    colorbar.set_label(f"{field_label} ({field_unit})")
    colorbar.set_ticks(np.linspace(vmin, vmax, 7))
    colorbar.set_ticklabels(
        [tick_format.format(value) for value in np.linspace(vmin, vmax, 7)]
    )

    fringe_ax.scatter(
        [150.0, 2550.0],
        [-15.0, -15.0],
        marker="^",
        s=70,
        facecolor="#4d4d4d",
        edgecolor="none",
        zorder=5,
    )
    block_width = 100.0
    block_height = 28.0
    blocks = [
        Rectangle(
            (center - block_width / 2.0, 250.0),
            block_width,
            block_height,
            facecolor="#666666",
            edgecolor="#222222",
            linewidth=0.7,
            zorder=6,
        )
        for center in (left_load, right_load)
    ]
    for block in blocks:
        fringe_ax.add_patch(block)
    arrows = [
        FancyArrowPatch(
            (center, 335.0),
            (center, 285.0),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#222222",
            zorder=7,
        )
        for center in (left_load, right_load)
    ]
    for arrow in arrows:
        fringe_ax.add_patch(arrow)

    fringe_ax.set_xlim(-40.0, 2740.0)
    fringe_ax.set_ylim(-55.0, 355.0)
    fringe_ax.set_aspect("equal", adjustable="box")
    fringe_ax.set_xlabel("Beam-axis coordinate z (mm)")
    fringe_ax.set_ylabel("Vertical coordinate y (mm)")
    fringe_ax.grid(False)

    curve_ax.plot(
        head_down_mm,
        total_load_kn,
        color="#b8b8b8",
        linewidth=1.4,
        label="Full ground-truth path",
    )
    history_line, = curve_ax.plot(
        head_down_mm[:1],
        total_load_kn[:1],
        color="#155fa0",
        linewidth=2.2,
        label="Current history",
    )
    current_point, = curve_ax.plot(
        head_down_mm[:1],
        total_load_kn[:1],
        marker="o",
        markersize=5.5,
        color="#b2182b",
        linestyle="none",
    )
    curve_ax.set_xlim(0.0, max(1.0, float(head_down_mm.max()) * 1.04))
    curve_ax.set_ylim(0.0, max(1.0, float(total_load_kn.max()) * 1.12))
    curve_ax.set_xlabel("Prescribed downward displacement (mm)")
    curve_ax.set_ylabel("Total load −RF2 (kN)")
    curve_ax.grid(True, color="#dedede", linewidth=0.6)
    curve_ax.legend(loc="best", frameon=False, fontsize=8)

    scale_note = "" if deformation_scale == 1.0 else f" | deformation ×{deformation_scale:g}"
    title = figure.suptitle("", fontsize=11)

    def update(frame: int):
        collection.set_verts(face_vertices(frame))
        collection.set_array(values[frame])
        block_y = 250.0 - head_down_mm[frame] * deformation_scale
        for block in blocks:
            block.set_y(block_y)
        for center, arrow in zip((left_load, right_load), arrows, strict=True):
            arrow.set_positions((center, block_y + 78.0), (center, block_y + block_height + 3.0))
        history_line.set_data(head_down_mm[: frame + 1], total_load_kn[: frame + 1])
        current_point.set_data([head_down_mm[frame]], [total_load_kn[frame]])
        title.set_text(
            f"{case_id} — ground truth | p={progress[frame]:.2f} | "
            f"head={head_down_mm[frame]:.2f} mm | load={total_load_kn[frame]:.2f} kN"
            f"{scale_note}"
        )
        return collection, history_line, current_point, title, *blocks, *arrows

    update(0)
    animation = FuncAnimation(
        figure,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)

    if png_path is not None:
        update(n_frames - 1)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png_path, dpi=dpi)
    plt.close(figure)

    print(f"case: {case_id}")
    print(f"frames: {n_frames}")
    print(f"visible side elements: {side_elements.size}")
    print(f"field range: {vmin:.6g} to {vmax:.6g} {field_unit}")
    print(f"wrote: {gif_path}")
    if png_path is not None:
        print(f"wrote: {png_path}")


def main() -> int:
    benchmark_root = Path(__file__).resolve().parents[1]
    h5_root = Path(
        os.environ.get(
            "H5_ROOT",
            Path("cases"),
        )
    )
    default_case = "Job-new0a0b0"
    default_output = benchmark_root / "outputs"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=h5_root / f"{default_case}.h5",
        help="canonical case HDF5 (default: Job-new0a0b0 under H5_ROOT)",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        help="output GIF (default: outputs/<case>-<field>.gif)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        help="final-frame PNG (default: outputs/<case>-<field>-final.png)",
    )
    parser.add_argument("--field", choices=("stress", "peeq"), default="stress")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.gif is None:
        args.gif = default_output / f"{args.h5.stem}-{args.field}.gif"
    if args.png is None:
        args.png = default_output / f"{args.h5.stem}-{args.field}-final.png"
    if args.fps < 1 or args.dpi < 50 or args.bands < 2:
        parser.error("fps >= 1, dpi >= 50, and bands >= 2 are required")
    if not args.h5.is_file():
        parser.error(
            f"HDF5 file not found: {args.h5}. Set --h5 explicitly or set H5_ROOT."
        )
    render(
        args.h5,
        args.gif,
        args.png,
        field=args.field,
        fps=args.fps,
        dpi=args.dpi,
        bands=args.bands,
        deformation_scale=args.deformation_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

