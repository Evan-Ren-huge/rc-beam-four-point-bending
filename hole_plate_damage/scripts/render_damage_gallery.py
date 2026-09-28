"""Reproducible finite-element gallery; reads NPZs without modifying source data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#2166ac', '#d95f02', '#1b9e77', '#8055ad', '#c63666', '#947126']
DAMAGE_CMAP = LinearSegmentedColormap.from_list('damage', ['#e3eaf0', '#fee08b', '#f89d3c', '#df4937', '#980f34'])
DELETED = '#192c45'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#38485a', 'text.color': '#192c45',
                     'axes.titlecolor': '#192c45', 'savefig.facecolor': 'white'})


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_cases(source):
    results = {r['case_id']: r for r in read_csv(source / 'results.csv')}
    pool = [r for r in read_csv(source / 'cases_500.csv')
            if r['split'] == 'ood_test' and results[r['case_id']]['analysis_status'] == 'COMPLETED']
    choices = [('Smallest hole', lambda r: float(r['R_mm']), min),
               ('Largest hole', lambda r: float(r['R_mm']), max),
               ('Lowest L/W', lambda r: float(r['L_mm']) / float(r['W_mm']), min),
               ('Highest L/W', lambda r: float(r['L_mm']) / float(r['W_mm']), max),
               ('Thinnest plate', lambda r: float(r['t_mm']), min),
               ('Thickest plate', lambda r: float(r['t_mm']), max)]
    cases = []
    for title, key, extremum in choices:
        row = extremum(pool, key=key)
        cid = row['case_id']
        npz = source / cid / (cid + '_damage_gnn.npz')
        summary = json.loads((source / cid / (cid + '_damage_summary.json')).read_text())
        with np.load(npz, allow_pickle=False) as data:
            d = {k: data[k] for k in data.files}
        labels = {int(v): i for i, v in enumerate(d['node_labels'])}
        conn = np.array([[labels[int(v)] for v in elem] for elem in d['connectivity']])
        p = d['normalized_progress']
        assert len(p) == 101 and np.allclose(p, np.linspace(0, 1, 101), atol=1e-5)
        assert d['integration_point_damage_history'].shape == (101, 1580)
        assert d['nodal_displacement_history'].shape == (101, 1684, 2)
        assert np.isfinite(d['integration_point_damage_history']).all()
        assert summary['load_application'] == 'FULL_RIGHT_EDGE_UNIFORM_U1'
        d.update(row=row, summary=summary, conn=conn, title=title, short=cid.replace('_DMG_FULL', ''),
                 sha256=hashlib.sha256(npz.read_bytes()).hexdigest(), npz_path=str(npz))
        cases.append(d)
    assert len({d['short'] for d in cases}) == 6, 'Geometry extrema overlap; revise selection explicitly.'
    return cases


def vertices(d, frame, scale=3):
    return (d['node_coordinates'] + scale * d['nodal_displacement_history'][frame])[d['conn']]


def field_colors(d, frame, field):
    if field == 'damage':
        rgba = DAMAGE_CMAP(np.clip(d['integration_point_damage_history'][frame], 0, 1))
    else:
        rgba = plt.get_cmap('turbo')(np.clip(d['integration_point_mises_history_mpa'][frame] / 400, 0, 1))
    rgba[d['element_status_history'][frame] <= .5] = matplotlib.colors.to_rgba(DELETED)
    return rgba


def title_block(fig, title, subtitle):
    fig.text(.05, .957, title, fontsize=23, weight='bold')
    fig.text(.05, .926, subtitle, fontsize=10, color='#536477')


def footer(fig):
    fig.text(.05, .022, 'Abaqus finite-element results | Synthetic ductile-damage baseline; not calibrated to a specific alloy.',
             fontsize=9, color='#68788a')


def make_dashboard(cases, out, field='damage', animate=False):
    titles = {'damage': 'Damage evolution', 'stress': 'von Mises stress', 'displacement': 'Displacement magnitude'}
    vmax = {'damage': 1, 'stress': 400}.get(field)
    if field == 'displacement':
        for d in cases:
            d['umag'] = np.linalg.norm(d['nodal_displacement_history'], axis=2)
            assert np.isfinite(d['umag']).all()
        vmax = float(np.ceil(max(d['umag'].max() for d in cases) * 2) / 2)
        print(f'Displacement shared scale: 0 to {vmax} mm', flush=True)
    cmap = DAMAGE_CMAP if field == 'damage' else ('viridis' if field == 'displacement' else 'turbo')
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(14, 10), dpi=100)
    title_block(fig, 'HolePlate2D / ' + titles[field],
                '6 OOD geometry extrema from the latest 500-case batch | Full-edge tension | 101 saved loading frames')
    gs = fig.add_gridspec(3, 3, left=.06, right=.90, bottom=.16, top=.815,
                          height_ratios=[1, 1, .95], hspace=.62, wspace=.22)
    meshes, labels = [], []
    for i, d in enumerate(cases):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        if field == 'displacement':
            xy = d['node_coordinates'] + 3 * d['nodal_displacement_history'][100]
            triangles = np.concatenate([d['conn'][:, [0, 1, 2]], d['conn'][:, [0, 2, 3]]])
            tri = Triangulation(xy[:, 0].copy(), xy[:, 1].copy(), triangles)
            tri.set_mask(np.tile(d['element_status_history'][100] <= .5, 2))
            artist = TriMesh(tri, cmap=cmap, norm=norm)
            artist.set_array(d['umag'][100])
            ax.add_collection(artist)
            deleted_artist = PolyCollection(vertices(d, 100)[d['element_status_history'][100] <= .5], facecolors=DELETED, edgecolors='none')
            ax.add_collection(deleted_artist)
            mesh = (tri, artist, deleted_artist)
        else:
            mesh = PolyCollection(vertices(d, 100), facecolors=field_colors(d, 100, field), edgecolors='none')
            ax.add_collection(mesh)
        ax.set(xlim=(-130, 140), ylim=(-70, 70), aspect='equal')
        ax.set_xticks([-100, 0, 100]); ax.set_yticks([-50, 0, 50])
        ax.tick_params(labelsize=8, colors='#758394')
        r = d['row']
        ax.set_title(f"{d['short']}  /  {d['title']}\nL={float(r['L_mm']):.1f}   W={float(r['W_mm']):.1f}   t={float(r['t_mm']):.2f}   D={2*float(r['R_mm']):.1f} mm", loc='left', fontsize=9.4, pad=9, color=COLORS[i])
        label = ax.text(.02, .02, '', transform=ax.transAxes, fontsize=8, color='#536477')
        meshes.append(mesh); labels.append(label)
    cax = fig.add_axes([.928, .47, .012, .32])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label({'damage': 'Damage SDEG', 'stress': 'von Mises stress [MPa]', 'displacement': 'Displacement magnitude |U| [mm]'}[field], fontsize=10)
    fig.legend(handles=[Patch(facecolor=DELETED, label='Deleted (STATUS = 0)')], loc='upper left',
               bbox_to_anchor=(.735, .907), frameon=False, fontsize=9)
    progress = fig.text(.05, .877, '', fontsize=12, weight='bold')
    ax = fig.add_subplot(gs[2, :])
    ax.set_title('Force response / synchronized with the fields above', loc='left', fontsize=12, weight='bold', pad=10)
    lines, dots = [], []
    for i, d in enumerate(cases):
        strain = d['prescribed_displacement_history_mm'] / float(d['row']['L_mm']) * 100
        force = d['tensile_force_magnitude_history_n'] / 1000
        d['strain'] = strain; d['force'] = force
        ax.plot(strain, force, color=COLORS[i], lw=1, alpha=.14)
        line, = ax.plot([], [], color=COLORS[i], lw=2, label=d['short'])
        dot, = ax.plot([], [], 'o', color=COLORS[i], ms=5)
        lines.append(line); dots.append(dot)
    ax.set(xlim=(0, 1.52), ylim=(0, max(d['force'].max() for d in cases)*1.10),
           xlabel='Nominal axial strain [%]', ylabel='Reaction force [kN]')
    ax.grid(alpha=.16); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.37), ncol=6, fontsize=8, frameon=False)
    note = ('Shared x/y limits in mm; deformation 3x. Colors: actual nodal |U| (mm), interpolated within each element. Deleted elements shown in navy.'
            if field == 'displacement' else
            'Shared x/y limits in mm; deformation magnified 3x. Original element values; no nodal smoothing. Deleted elements retain their last saved position.')
    fig.text(.06, .055, note, fontsize=8.5, color='#536477')
    footer(fig)

    def update(frame):
        progress.set_text(f'Loading progress  {frame:3d}%    |    nominal strain  {cases[0]["strain"][frame]:.3f}%')
        for d, mesh, label, line, dot in zip(cases, meshes, labels, lines, dots):
            if field == 'displacement':
                tri, artist, deleted_artist = mesh
                xy = d['node_coordinates'] + 3 * d['nodal_displacement_history'][frame]
                tri.x[:] = xy[:, 0]; tri.y[:] = xy[:, 1]
                deleted_mask = d['element_status_history'][frame] <= .5
                tri.set_mask(np.tile(deleted_mask, 2))
                artist.set_array(d['umag'][frame])
                deleted_artist.set_verts(vertices(d, frame)[deleted_mask])
            else:
                mesh.set_verts(vertices(d, frame))
                mesh.set_facecolors(field_colors(d, frame, field))
            deleted = int((d['element_status_history'][frame] <= .5).sum())
            label.set_text(f'{deleted} / 1580 elements deleted')
            line.set_data(d['strain'][:frame+1], d['force'][:frame+1])
            dot.set_data([d['strain'][frame]], [d['force'][frame]])

    update(100)
    fig.savefig(out / f'case-{field}-comparison.png', dpi=170)
    if animate:
        frames = []
        for frame in range(101):
            update(frame); fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=192))
            if frame % 25 == 0:
                print(f'Rendered {field} frame {frame}/100', flush=True)
        frames[0].save(out / f'case-{field}-comparison.gif', save_all=True, append_images=frames[1:],
                       duration=[90]*100+[1800], loop=0, optimize=False, disposal=2)
    plt.close(fig)


def make_timeline(cases, out):
    # Damage emerges late in the loading path; show common late-stage frames.
    stages = [40, 60, 75, 90, 100]
    fig, axes = plt.subplots(6, 5, figsize=(15, 12))
    fig.subplots_adjust(left=.15, right=.91, bottom=.08, top=.89, hspace=.20, wspace=.07)
    title_block(fig, 'From localization to element deletion',
                'Shared SDEG scale and physical coordinates | 3x deformation | The same six cases at five loading stages')
    for i, d in enumerate(cases):
        for j, frame in enumerate(stages):
            ax = axes[i, j]
            ax.add_collection(PolyCollection(vertices(d, frame), facecolors=field_colors(d, frame, 'damage'), edgecolors='none'))
            ax.set(xlim=(-130, 140), ylim=(-70, 70), aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if i == 0: ax.set_title(f'p = {frame/100:.2f}', fontsize=11, pad=10)
            if j == 0:
                ax.text(-.07, .5, f"{d['short']}\n{d['title']}", transform=ax.transAxes,
                        ha='right', va='center', fontsize=9, color=COLORS[i])
    cax = fig.add_axes([.94, .33, .012, .35])
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=DAMAGE_CMAP), cax=cax, label='Damage SDEG')
    fig.legend(handles=[Patch(facecolor=DELETED, label='Deleted element (STATUS = 0)')],
               loc='lower center', bbox_to_anchor=(.5, .035), frameon=False, fontsize=9)
    footer(fig)
    fig.savefig(out / 'case-damage-timeline.png', dpi=170)
    plt.close(fig)


def make_geometry(cases, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    fig.subplots_adjust(left=.065, right=.96, bottom=.14, top=.79, hspace=.48, wspace=.2)
    title_block(fig, 'Six geometries / one common mesh topology',
                'Undeformed configuration | 1684 nodes and 1580 CPS4R elements per case | Common coordinates in mm')
    for i, (d, ax) in enumerate(zip(cases, axes.flat)):
        ax.add_collection(PolyCollection(vertices(d, 0, 0), facecolors='#e9eef3', edgecolors='#7a8b9c', linewidths=.22))
        coords = d['node_coordinates']
        ax.scatter(*coords[d['fixed_node_mask'].astype(bool)].T, s=7, color='#38485a', zorder=3)
        ax.scatter(*coords[d['loaded_node_mask'].astype(bool)].T, s=7, color='#d95f02', zorder=3)
        ax.set(xlim=(-130, 140), ylim=(-70, 70), aspect='equal', xlabel='x [mm]', ylabel='y [mm]')
        r = d['row']
        ax.set_title(f"{d['short']}  /  {d['title']}\nL={float(r['L_mm']):.1f}   W={float(r['W_mm']):.1f}   t={float(r['t_mm']):.2f}   D={2*float(r['R_mm']):.1f} mm", fontsize=9.5, loc='left', color=COLORS[i])
        ax.tick_params(labelsize=8)
    fig.text(.065, .07, 'Dark markers: axial restraint at the left edge. Orange markers: prescribed axial displacement along the full right edge.', fontsize=9)
    footer(fig)
    fig.savefig(out / 'case-geometry-comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Path to an extracted full-edge source batch')
    parser.add_argument('--fields', nargs='+', choices=['damage', 'displacement', 'stress'], default=['damage', 'displacement', 'stress'])
    args = parser.parse_args()
    out = ROOT / 'assets'; out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.source)
    records = []
    for d in cases:
        rec = {k: d['row'][k] for k in ['case_id', 'split', 'L_mm', 'W_mm', 't_mm', 'R_mm', 'R_over_W']}
        rec.update(selection=d['title'], npz_sha256=d['sha256'], source_npz=d['npz_path'],
                   frame_count=101, deleted_elements_final=int((d['element_status_history'][-1] <= .5).sum()),
                   peak_force_kN=float(d['tensile_force_magnitude_history_n'].max()/1000),
                   final_stabilization_energy_ratio=d['summary']['final_allsd_allie_ratio'])
        records.append(rec)
    (ROOT / 'selected_cases.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    with (ROOT / 'selected_cases.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print(json.dumps([(r['case_id'], r['selection']) for r in records]), flush=True)
    if 'damage' in args.fields:
        make_geometry(cases, out)
        make_timeline(cases, out)
    for field in args.fields:
        make_dashboard(cases, out, field=field, animate=True)
        with Image.open(out / f'case-{field}-comparison.gif') as im:
            assert im.n_frames == 101, im.n_frames
            im.seek(75); im.convert('RGB').save(out / f'{field}-animation-frame-075.png')
        print(f'Verified {field} GIF: 101 frames.', flush=True)


if __name__ == '__main__':
    main()
