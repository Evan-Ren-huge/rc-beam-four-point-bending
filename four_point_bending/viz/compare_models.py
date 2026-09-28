#!/usr/bin/env python3
"""Compare the Job-new rollouts saved by the three local surrogate pipelines.

This reader normalizes them in memory; it never renames or rewrites predictions.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import FancyArrowPatch, Rectangle


@dataclass(frozen=True)
class ModelSpec:
    label: str
    directory: str


@dataclass
class Rollout:
    label: str
    path: Path
    coords_mm: np.ndarray
    connectivity: np.ndarray
    progress: np.ndarray
    displacement_mm: np.ndarray
    stress_mpa: np.ndarray
    peeq: np.ndarray
    true_displacement_mm: np.ndarray
    true_stress_mpa: np.ndarray
    true_peeq: np.ndarray


MODEL_SPECS = (
    ModelSpec("Dual-Graph GConvGRU", "double_pre"),
    ModelSpec("Single-Graph E", "single_element_pre"),
    ModelSpec("Single-Graph N", "single_node_pre"),
)

FIGURE_LABELS = {
    "Dual-Graph GConvGRU": "Dual-Graph",
    "Single-Graph E": "CGN-E",
    "Single-Graph N": "CGN-N",
}

ALIASES = {
    "coords": ("node_coords", "coords"),
    "time": ("frame_times", "time", "frame_times_used"),
    "pred_displacement": ("pred_disp", "displacement_pred", "disp_pred"),
    "true_displacement": ("true_disp", "displacement_true", "disp_true"),
    "pred_stress": ("pred_s_elem", "stress_pred", "s_elem_pred"),
    "true_stress": ("true_s_elem", "stress_true", "s_elem_true"),
    "pred_peeq": ("pred_peeq_elem", "peeq_pred", "peeq_elem_pred"),
    "true_peeq": ("true_peeq_elem", "peeq_true", "peeq_elem_true"),
}


def normalize_case_id(name: str) -> str:
    """Normalize Job-new0 filename variants used by the prediction folders."""
    stem = Path(name).stem
    stem = re.sub(r"^test[-_]?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[-_]?pred$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-z]", "", stem.lower())


def loading_centers_from_case_id(case_id: str) -> tuple[float, float]:
    """Return (left, right) loading centers from Job-new<i>a<A>b<B>."""
    match = re.search(r"a(?P<right>-?\d+)b(?P<left>-?\d+)$", case_id)
    if match is None:
        raise ValueError(
            f"cannot recover loading positions from case id {case_id!r}; "
            "expected a trailing 'a<A>b<B>'"
        )
    right_shift = float(match.group("right"))
    left_shift = float(match.group("left"))
    return 950.0 + left_shift, 1750.0 + right_shift


def resolve_case_file(directory: Path, case_id: str) -> Path:
    target = normalize_case_id(case_id)
    matches = [
        path
        for path in sorted(directory.glob("*.npz"))
        if normalize_case_id(path.name) == target
    ]
    if not matches:
        raise FileNotFoundError(
            f"no rollout matching {case_id!r} in {directory}; "
            "both '<case>_pred.npz' and 'test-<case>.npz' are recognized"
        )
    if len(matches) != 1:
        raise ValueError(f"multiple files in {directory} map to {case_id}: {matches}")
    return matches[0]


def array_from(payload: np.lib.npyio.NpzFile, canonical_key: str) -> np.ndarray:
    for key in ALIASES[canonical_key]:
        if key in payload:
            return np.asarray(payload[key])
    tried = ", ".join(ALIASES[canonical_key])
    raise KeyError(f"missing {canonical_key}; tried keys: {tried}")


def load_rollout(label: str, path: Path) -> Rollout:
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(array_from(payload, "coords"), dtype=np.float64)
        connectivity = np.asarray(payload["connectivity"], dtype=np.int64)
        if connectivity.min() == 1:
            connectivity = connectivity - 1
        elif connectivity.min() != 0:
            raise ValueError(f"unsupported connectivity base in {path}")

        progress = np.asarray(array_from(payload, "time"), dtype=np.float64)
        pred_disp = np.asarray(
            array_from(payload, "pred_displacement"), dtype=np.float64
        )
        true_disp = np.asarray(
            array_from(payload, "true_displacement"), dtype=np.float64
        )
        pred_stress = np.asarray(array_from(payload, "pred_stress"), dtype=np.float64)
        true_stress = np.asarray(array_from(payload, "true_stress"), dtype=np.float64)
        pred_peeq = np.asarray(array_from(payload, "pred_peeq"), dtype=np.float64)
        true_peeq = np.asarray(array_from(payload, "true_peeq"), dtype=np.float64)

    n_frames = progress.size
    expected = {
        "pred displacement": (n_frames, coords.shape[0], 3),
        "true displacement": (n_frames, coords.shape[0], 3),
        "pred stress": (n_frames, connectivity.shape[0]),
        "true stress": (n_frames, connectivity.shape[0]),
        "pred PEEQ": (n_frames, connectivity.shape[0]),
        "true PEEQ": (n_frames, connectivity.shape[0]),
    }
    arrays = {
        "pred displacement": pred_disp,
        "true displacement": true_disp,
        "pred stress": pred_stress,
        "true stress": true_stress,
        "pred PEEQ": pred_peeq,
        "true PEEQ": true_peeq,
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{path}: {name} shape {arrays[name].shape}, expected {shape}")

    return Rollout(
        label=label,
        path=path,
        coords_mm=coords,
        connectivity=connectivity,
        progress=progress,
        displacement_mm=pred_disp,
        stress_mpa=pred_stress,
        peeq=pred_peeq,
        true_displacement_mm=true_disp,
        true_stress_mpa=true_stress,
        true_peeq=true_peeq,
    )


def validate_shared_reference(rollouts: list[Rollout]) -> None:
    reference = rollouts[0]
    for rollout in rollouts[1:]:
        checks = (
            ("coordinates", rollout.coords_mm, reference.coords_mm, 1.0e-5),
            ("progress", rollout.progress, reference.progress, 1.0e-8),
            (
                "ground-truth displacement",
                rollout.true_displacement_mm,
                reference.true_displacement_mm,
                1.0e-5,
            ),
            (
                "ground-truth stress",
                rollout.true_stress_mpa,
                reference.true_stress_mpa,
                1.0e-5,
            ),
            ("ground-truth PEEQ", rollout.true_peeq, reference.true_peeq, 1.0e-8),
        )
        if not np.array_equal(rollout.connectivity, reference.connectivity):
            raise ValueError(f"connectivity differs between {reference.path} and {rollout.path}")
        for name, candidate, truth, atol in checks:
            if not np.allclose(candidate, truth, rtol=1.0e-6, atol=atol):
                delta = float(np.max(np.abs(candidate - truth)))
                raise ValueError(
                    f"{name} differs between {reference.path.name} and "
                    f"{rollout.path.name}; max |delta|={delta:.6g}"
                )


def exposed_side_faces(
    coords_mm: np.ndarray, connectivity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return element ids and ordered nodes on the minimum-transverse-x side."""
    element_coords = coords_mm[connectivity]
    centroid_x = element_coords[:, :, 0].mean(axis=1)
    side_elements = np.flatnonzero(
        np.isclose(centroid_x, centroid_x.min(), rtol=0.0, atol=1.0e-8)
    )
    faces: list[np.ndarray] = []
    for element_index in side_elements:
        node_ids = connectivity[element_index]
        local = coords_mm[node_ids]
        face_nodes = node_ids[
            np.isclose(local[:, 0], local[:, 0].min(), rtol=0.0, atol=1.0e-8)
        ]
        if face_nodes.size != 4:
            raise ValueError(
                f"element {element_index} has {face_nodes.size} side-face nodes"
            )
        zy = coords_mm[face_nodes][:, [2, 1]]
        center = zy.mean(axis=0)
        angles = np.arctan2(zy[:, 1] - center[1], zy[:, 0] - center[0])
        faces.append(face_nodes[np.argsort(angles)])
    return side_elements, np.stack(faces)


def field_arrays(
    field: str, reference: Rollout, rollouts: list[Rollout]
) -> tuple[np.ndarray, list[np.ndarray], str, str, str]:
    if field == "stress":
        return (
            reference.true_stress_mpa,
            [rollout.stress_mpa for rollout in rollouts],
            "von Mises stress",
            "MPa",
            ".3f",
        )
    if field == "peeq":
        return (
            reference.true_peeq,
            [rollout.peeq for rollout in rollouts],
            "equivalent plastic strain (PEEQ)",
            "–",
            ".6f",
        )
    raise ValueError(f"unsupported field: {field}")


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def rel_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), 1e-12))


def face_vertices(
    coords_mm: np.ndarray,
    displacement_mm: np.ndarray,
    face_nodes: np.ndarray,
    frame: int,
    deformation_scale: float,
) -> np.ndarray:
    deformed = coords_mm + deformation_scale * displacement_mm[frame]
    return deformed[face_nodes][:, :, [2, 1]]


def style_axis(axis: plt.Axes, *, bottom: bool, left: bool) -> None:
    axis.set_xlim(-35.0, 2735.0)
    axis.set_ylim(-55.0, 330.0)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    axis.tick_params(labelsize=7, length=2.5, width=0.6)
    if bottom:
        axis.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    else:
        axis.set_xticklabels([])
    if left:
        axis.set_yticks([0, 125, 250])
    else:
        axis.set_yticklabels([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#555555")


def add_fixture_artists(axis: plt.Axes, load_centers: tuple[float, float]):
    supports = axis.scatter(
        [150.0, 2550.0],
        [-17.0, -17.0],
        marker="^",
        s=18,
        facecolor="#4b4b4b",
        edgecolor="none",
        zorder=5,
    )
    blocks = [
        Rectangle(
            (center - 50.0, 252.0),
            100.0,
            23.0,
            facecolor="#616161",
            edgecolor="#202020",
            linewidth=0.45,
            zorder=6,
        )
        for center in load_centers
    ]
    arrows = [
        FancyArrowPatch(
            (center, 323.0),
            (center, 280.0),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.75,
            color="#202020",
            zorder=7,
        )
        for center in load_centers
    ]
    for artist in (*blocks, *arrows):
        axis.add_artist(artist)
    return supports, blocks, arrows


def update_fixture_artists(
    blocks: list[Rectangle],
    arrows: list[FancyArrowPatch],
    load_centers: tuple[float, float],
    coords_mm: np.ndarray,
    displacement_mm: np.ndarray,
    frame: int,
    deformation_scale: float,
) -> None:
    top_y = float(coords_mm[:, 1].max())
    for center, block, arrow in zip(load_centers, blocks, arrows, strict=True):
        patch_nodes = (
            (np.abs(coords_mm[:, 2] - center) <= 62.5)
            & np.isclose(coords_mm[:, 1], top_y, rtol=0.0, atol=1.0e-5)
        )
        dy = (
            float(displacement_mm[frame, patch_nodes, 1].mean())
            if np.any(patch_nodes)
            else 0.0
        )
        block_y = 252.0 + deformation_scale * dy
        block.set_y(block_y)
        arrow.set_positions((center, block_y + 71.0), (center, block_y + 28.0))


def render_static(
    case_id: str,
    output_png: Path,
    output_pdf: Path | None,
    coords_mm: np.ndarray,
    face_nodes: np.ndarray,
    side_elements: np.ndarray,
    progress: np.ndarray,
    row_labels: list[str],
    row_displacements: list[np.ndarray],
    row_values: list[np.ndarray],
    target_values: np.ndarray,
    frame_indices: list[int],
    field_label: str,
    field_unit: str,
    metric_format: str,
    cmap,
    norm: BoundaryNorm,
    vmin: float,
    vmax: float,
    load_centers: tuple[float, float],
    deformation_scale: float,
    dpi: int,
) -> None:
    n_rows = len(row_labels)
    n_columns = len(frame_indices)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.15 * n_columns + 1.15, 1.22 * n_rows + 1.35),
        constrained_layout=True,
        squeeze=False,
    )
    collections: list[PolyCollection] = []
    for row, (label, displacement, values) in enumerate(
        zip(row_labels, row_displacements, row_values, strict=True)
    ):
        metric_note = ""
        if row:
            metric_note = (
                f"\nRMSE {format(rmse(values, target_values), metric_format)} {field_unit}"
                f"\nrel. L2 {rel_l2(values, target_values):.3f}"
            )
        for column, frame in enumerate(frame_indices):
            axis = axes[row, column]
            collection = PolyCollection(
                face_vertices(
                    coords_mm, displacement, face_nodes, frame, deformation_scale
                ),
                array=np.clip(values[frame, side_elements], vmin, vmax),
                cmap=cmap,
                norm=norm,
                edgecolors="none",
                antialiased=False,
                rasterized=True,
            )
            axis.add_collection(collection)
            collections.append(collection)
            _, blocks, arrows = add_fixture_artists(axis, load_centers)
            update_fixture_artists(
                blocks,
                arrows,
                load_centers,
                coords_mm,
                displacement,
                frame,
                deformation_scale,
            )
            style_axis(axis, bottom=row == n_rows - 1, left=column == 0)
            if row == 0:
                axis.set_title(
                    f"p = {progress[frame]:.2f}  (frame {frame})", fontsize=9, pad=3
                )
            if column == 0:
                axis.set_ylabel(label + metric_note, fontsize=8.2, labelpad=7)
            if row == n_rows - 1:
                axis.set_xlabel("beam-axis coordinate z (mm)", fontsize=8)

    colorbar = figure.colorbar(
        collections[0], ax=axes.ravel().tolist(), fraction=0.014, pad=0.012
    )
    colorbar.set_label(f"{field_label} ({field_unit})", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"{case_id} — ground truth and three-model rollout comparison\n"
        f"shared {field_label} scale: {vmin:g}–{vmax:.4g} {field_unit}",
        fontsize=11,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=dpi)
    if output_pdf is not None:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf, dpi=dpi)
    plt.close(figure)


def render_animation(
    case_id: str,
    output_gif: Path,
    coords_mm: np.ndarray,
    face_nodes: np.ndarray,
    side_elements: np.ndarray,
    progress: np.ndarray,
    row_labels: list[str],
    row_displacements: list[np.ndarray],
    row_values: list[np.ndarray],
    target_values: np.ndarray,
    field_label: str,
    field_unit: str,
    metric_format: str,
    cmap,
    norm: BoundaryNorm,
    vmin: float,
    vmax: float,
    load_centers: tuple[float, float],
    deformation_scale: float,
    fps: int,
    dpi: int,
) -> None:
    n_rows = len(row_labels)
    figure, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12.2, 1.58 * n_rows + 0.85),
        constrained_layout=True,
        squeeze=False,
    )
    axes_1d = axes[:, 0]
    collections: list[PolyCollection] = []
    block_rows: list[list[Rectangle]] = []
    arrow_rows: list[list[FancyArrowPatch]] = []
    for row, (axis, label, displacement, values) in enumerate(
        zip(axes_1d, row_labels, row_displacements, row_values, strict=True)
    ):
        collection = PolyCollection(
            face_vertices(coords_mm, displacement, face_nodes, 0, deformation_scale),
            array=np.clip(values[0, side_elements], vmin, vmax),
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            antialiased=False,
        )
        axis.add_collection(collection)
        collections.append(collection)
        _, blocks, arrows = add_fixture_artists(axis, load_centers)
        block_rows.append(blocks)
        arrow_rows.append(arrows)
        metric_note = ""
        if row:
            metric_note = (
                f"\nRMSE {format(rmse(values, target_values), metric_format)} {field_unit}"
                f"\nrel. L2 {rel_l2(values, target_values):.3f}"
            )
        axis.set_ylabel(label + metric_note, fontsize=8.4, labelpad=8)
        style_axis(axis, bottom=row == n_rows - 1, left=True)
        if row == n_rows - 1:
            axis.set_xlabel("beam-axis coordinate z (mm)", fontsize=8.5)

    colorbar = figure.colorbar(
        collections[0], ax=axes_1d.tolist(), fraction=0.022, pad=0.012
    )
    colorbar.set_label(f"{field_label} ({field_unit})", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    title = figure.suptitle("", fontsize=11)

    def update(frame: int):
        artists: list[object] = [title]
        for collection, blocks, arrows, displacement, values in zip(
            collections,
            block_rows,
            arrow_rows,
            row_displacements,
            row_values,
            strict=True,
        ):
            collection.set_verts(
                face_vertices(
                    coords_mm, displacement, face_nodes, frame, deformation_scale
                )
            )
            collection.set_array(np.clip(values[frame, side_elements], vmin, vmax))
            update_fixture_artists(
                blocks,
                arrows,
                load_centers,
                coords_mm,
                displacement,
                frame,
                deformation_scale,
            )
            artists.extend((collection, *blocks, *arrows))
        title.set_text(
            f"{case_id} — ground truth and three models | "
            f"p={progress[frame]:.2f} | frame {frame}/{progress.size - 1} | "
            f"shared {field_label} scale {vmin:g}–{vmax:.4g} {field_unit}"
        )
        return artists

    update(0)
    animation = FuncAnimation(
        figure,
        update,
        frames=progress.size,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_gif, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)


def parse_frame_indices(raw: str, n_frames: int) -> list[int]:
    frames = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not frames:
        raise ValueError("at least one static frame is required")
    if min(frames) < 0 or max(frames) >= n_frames:
        raise ValueError(f"static frames must lie in [0, {n_frames - 1}]")
    return frames


def main() -> int:
    benchmark_root = Path(__file__).resolve().parents[1]
    output_root = benchmark_root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True, help="Directory containing the model rollout folders")
    parser.add_argument("--case", default="Job-new0a0b0")
    parser.add_argument("--field", choices=("stress", "peeq"), default="stress")
    parser.add_argument("--frames", default="1,5,10,15,20")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--bands", type=int, default=12)
    parser.add_argument("--deformation-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.fps < 1 or args.dpi < 50 or args.bands < 2:
        parser.error("fps >= 1, dpi >= 50, and bands >= 2 are required")

    if args.gif is None:
        args.gif = output_root / f"{args.case}-three-model-{args.field}.gif"
    if args.png is None:
        args.png = output_root / f"{args.case}-three-model-{args.field}.png"
    if args.pdf is None:
        args.pdf = output_root / f"{args.case}-three-model-{args.field}.pdf"

    rollouts = [
        load_rollout(
            spec.label,
            resolve_case_file(args.train_root / spec.directory, args.case),
        )
        for spec in MODEL_SPECS
    ]
    validate_shared_reference(rollouts)
    reference = rollouts[0]
    side_elements, face_nodes = exposed_side_faces(
        reference.coords_mm, reference.connectivity
    )
    target_values, prediction_values, field_label, field_unit, metric_format = (
        field_arrays(args.field, reference, rollouts)
    )
    frame_indices = parse_frame_indices(args.frames, reference.progress.size)

    row_labels = [
        "Ground truth",
        *(FIGURE_LABELS[rollout.label] for rollout in rollouts),
    ]
    row_displacements = [
        reference.true_displacement_mm,
        *(rollout.displacement_mm for rollout in rollouts),
    ]
    row_values = [target_values, *prediction_values]

    vmin = 0.0
    vmax = float(np.max(target_values))
    if vmax <= vmin:
        vmax = 1.0
    levels = np.linspace(vmin, vmax, args.bands + 1)
    cmap = plt.get_cmap("jet", args.bands)
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    load_centers = loading_centers_from_case_id(args.case)

    render_static(
        args.case,
        args.png,
        args.pdf,
        reference.coords_mm,
        face_nodes,
        side_elements,
        reference.progress,
        row_labels,
        row_displacements,
        row_values,
        target_values,
        frame_indices,
        field_label,
        field_unit,
        metric_format,
        cmap,
        norm,
        vmin,
        vmax,
        load_centers,
        args.deformation_scale,
        args.dpi,
    )
    render_animation(
        args.case,
        args.gif,
        reference.coords_mm,
        face_nodes,
        side_elements,
        reference.progress,
        row_labels,
        row_displacements,
        row_values,
        target_values,
        field_label,
        field_unit,
        metric_format,
        cmap,
        norm,
        vmin,
        vmax,
        load_centers,
        args.deformation_scale,
        args.fps,
        args.dpi,
    )

    print(f"case: {args.case}")
    print(f"frames: {reference.progress.size}; static frames: {frame_indices}")
    print(f"visible side elements: {side_elements.size}")
    print(f"shared field range: {vmin:.6g} to {vmax:.6g} {field_unit}")
    for rollout, values in zip(rollouts, prediction_values, strict=True):
        print(
            f"{rollout.label}: {rollout.path.name}; "
            f"RMSE={rmse(values, target_values):{metric_format}} {field_unit}; "
            f"relL2={rel_l2(values, target_values):.6f}"
        )
    print(f"wrote: {args.gif}")
    print(f"wrote: {args.png}")
    print(f"wrote: {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

