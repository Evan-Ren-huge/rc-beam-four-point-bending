"""Generate compact benchmark figures and a local Markdown preview."""
from pathlib import Path
import argparse
import json
import csv
import html
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import markdown
from render_damage_gallery import ROOT, load_cases, vertices, field_colors, DAMAGE_CMAP, DELETED

FIELDS = [('displacement', '|U| [mm]', 'viridis', 4),
          ('stress', 'von Mises [MPa]', 'turbo', 400),
          ('damage', 'Damage SDEG [-]', DAMAGE_CMAP, 1)]


def draw_field(ax, d, frame, field, cmap, vmax):
    if field == 'displacement':
        xy = d['node_coordinates'] + 3*d['nodal_displacement_history'][frame]
        triangles = np.concatenate([d['conn'][:, [0, 1, 2]], d['conn'][:, [0, 2, 3]]])
        tri = Triangulation(xy[:, 0], xy[:, 1], triangles)
        deleted = d['element_status_history'][frame] <= .5
        tri.set_mask(np.tile(deleted, 2))
        artist = TriMesh(tri, cmap=cmap, norm=Normalize(0, vmax))
        artist.set_array(np.linalg.norm(d['nodal_displacement_history'][frame], axis=1))
        ax.add_collection(artist)
        ax.add_collection(PolyCollection(vertices(d, frame)[deleted], facecolors=DELETED, edgecolors='none'))
    else:
        ax.add_collection(PolyCollection(vertices(d, frame), facecolors=field_colors(d, frame, field), edgecolors='none'))
    ax.set(xlim=(-125, 135), ylim=(-53, 53), aspect='equal')
    ax.set_axis_off()


def compact_figures(d):
    out = ROOT/'assets'
    fig = plt.figure(figsize=(12, 5.0), dpi=110)
    fig.subplots_adjust(left=.065, right=.965, top=.83, bottom=.16, hspace=.75)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (_, label, cmap, vmax) in enumerate(FIELDS):
        cax = fig.add_axes([.079+i*.312, .475, .255, .014])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=cmap), cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=8, length=2)
    force_ax = fig.add_subplot(gs[1, :])
    u = d['prescribed_displacement_history_mm']
    force = d['tensile_force_magnitude_history_n']/1000
    force_ax.plot(u, force, color='#c8d2dc', lw=1.4)
    line, = force_ax.plot([], [], color='#2367a1', lw=2)
    dot, = force_ax.plot([], [], 'o', color='#c13a43', ms=5)
    force_ax.set(xlabel='Prescribed displacement [mm]', ylabel='Force [kN]', xlim=(0,float(u.max())*1.02), ylim=(0,float(force.max())*1.15))
    force_ax.grid(alpha=.16); force_ax.tick_params(labelsize=8)
    fig.text(.065,.94,'HolePlate2D-Damage  /  finite-element trajectory', fontsize=16, weight='bold')
    label = fig.text(.065,.88,'', fontsize=10)
    fig.text(.065,.035,'3x displayed deformation | fixed color scales | navy: deleted elements (STATUS = 0)', fontsize=9, color='#586777')
    def update(k):
        for ax, (field,title,cmap,vmax) in zip(axes,FIELDS):
            ax.clear(); draw_field(ax,d,k,field,cmap,vmax); ax.set_title(title, fontsize=11,pad=6)
        label.set_text(f"{d['short']}_DMG_FULL   |   p = {k/100:.2f}   |   u = {u[k]:.3f} mm")
        line.set_data(u[:k+1],force[:k+1]); dot.set_data([u[k]],[force[k]])
    frames=[]
    for k in range(101):
        update(k); fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).quantize(colors=256))
        if k%25==0: print(f'Benchmark hero {k}/100',flush=True)
    fig.savefig(out/'benchmark-response-final.png',dpi=180)
    frames[0].save(out/'benchmark-response.gif',save_all=True,append_images=frames[1:],duration=[90]*100+[1800],loop=0,disposal=2)
    plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(14,5.5))
    fig.subplots_adjust(left=.105,right=.91,bottom=.10,top=.84,hspace=.24,wspace=.07)
    for i,(field,title,cmap,vmax) in enumerate(FIELDS):
        for j,k in enumerate([20,40,60,80,100]):
            ax=axes[i,j];draw_field(ax,d,k,field,cmap,vmax)
            if i==0:ax.set_title(f'p = {k/100:.2f}',fontsize=10)
            if j==0:ax.text(-.06,.5,title.replace(' [','\n['),transform=ax.transAxes,ha='right',va='center',fontsize=10)
        box=axes[i,-1].get_position();cax=fig.add_axes([.93,box.y0,.01,box.height])
        fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap=cmap),cax=cax).ax.tick_params(labelsize=8)
    fig.text(.105,.94,'Displacement, stress and damage across the loading path',fontsize=17,weight='bold')
    fig.text(.105,.045,f"{d['short']}_DMG_FULL | 3x deformation | shared scale within each row | navy: deleted elements",fontsize=9)
    fig.savefig(out/'benchmark-field-snapshots.png',dpi=180);plt.close(fig)


def schematic(d):
    fig,ax=plt.subplots(figsize=(11,4.1))
    xy=d['node_coordinates'];L=float(d['row']['L_mm']);W=float(d['row']['W_mm']);R=float(d['row']['R_mm'])
    ax.add_collection(PolyCollection(xy[d['conn']],facecolors='#edf2f7',edgecolors='#a3b2c0',linewidths=.25))
    ax.scatter(*xy[d['fixed_node_mask'].astype(bool)].T,s=10,color='#334b63',zorder=3)
    for y in np.linspace(-W*.42,W*.42,5):
        ax.annotate('',xy=(L/2+25,y),xytext=(L/2,y),arrowprops={'arrowstyle':'->','lw':1.6,'color':'#c64333'})
    ax.text(L/2+10,W/2+10,'u(p) = 0.015 L p',ha='center',fontsize=11,color='#b0392c')
    ax.text(-L/2-5,W/2+10,'Left edge: U1 = 0',ha='center',fontsize=11)
    ax.annotate('Central hole, D = 2R',xy=(R*.7,R*.7),xytext=(15,60),arrowprops={'arrowstyle':'->','color':'#536477'},fontsize=11)
    ax.annotate('',xy=(-L/2,-W/2-15),xytext=(L/2,-W/2-15),arrowprops={'arrowstyle':'<->','color':'#536477'})
    ax.text(0,-W/2-23,'L',ha='center',fontsize=12)
    ax.annotate('',xy=(-L/2-20,-W/2),xytext=(-L/2-20,W/2),arrowprops={'arrowstyle':'<->','color':'#536477'})
    ax.text(-L/2-28,0,'W',ha='center',fontsize=12)
    ax.set(xlim=(-170,175),ylim=(-80,83),aspect='equal');ax.set_axis_off()
    fig.text(.1,.93,'Problem setup / perforated plate in uniaxial tension',fontsize=16,weight='bold')
    fig.text(.1,.045,'Plane stress, thickness t | CPS4R mesh | E = 210 GPa, nu = 0.30 | one transverse anchor prevents rigid translation',fontsize=10)
    fig.savefig(ROOT/'assets/problem-hole-plate-damage.png',dpi=170);plt.close(fig)


def preview():
    css='''body{margin:0;background:#fff;color:#1f2328;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}header{border-bottom:1px solid #d1d9e0;background:#f6f8fa;padding:17px 5%;font-size:14px}header strong{font-size:18px}header span{color:#59636e;margin-left:18px}header nav{float:right}a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}.layout{display:grid;grid-template-columns:205px minmax(0,930px);gap:38px;max-width:1210px;margin:36px auto;padding:0 28px}aside{font-size:13px;position:sticky;top:24px;align-self:start;color:#59636e}aside ul{padding-left:17px}aside li{margin:9px 0}aside>div>ul{list-style:none;padding:0}article{min-width:0;border:1px solid #d1d9e0;border-radius:6px;padding:30px 38px}h1{font-size:30px;line-height:1.3;border-bottom:1px solid #d1d9e0;padding-bottom:14px;margin-top:0}h2{font-size:23px;border-bottom:1px solid #d1d9e0;padding-bottom:8px;margin-top:32px}h3{font-size:18px}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto;margin:18px 0}th,td{padding:8px 13px;border:1px solid #d1d9e0}th{background:#f6f8fa}tr:nth-child(even){background:#f6f8fa}blockquote{margin:18px 0;border-left:4px solid #9aadb9;padding:1px 16px;color:#59636e}code{background:#eff1f3;border-radius:4px;padding:2px 5px;font-size:85%}pre{background:#f6f8fa;border-radius:6px;padding:16px;overflow:auto;font-size:14px}pre code{padding:0;background:none}em{color:#59636e;font-size:14px}details{margin:20px 0}summary{cursor:pointer;font-weight:600}article li{margin:5px 0}.toclink{color:inherit}.language{margin:0 14px}.top{font-size:12px;color:#59636e} @media(max-width:900px){.layout{display:block;padding:0 12px;margin:14px auto}aside{display:none}article{padding:24px 18px}header span{display:none}header nav{float:none;margin-top:6px}h1{font-size:26px}}'''
    pages=[('en','README.md','preview.html'),('en','GALLERY.md','gallery.html'),('en','docs/DATA_SCHEMA.md','schema.html')]
    for lang,src,dest in pages:
        md=markdown.Markdown(extensions=['tables','fenced_code','toc','md_in_html'])
        body=md.convert((ROOT/src).read_text(encoding='utf-8-sig'))
        for _,mdpath,htmlpath in pages:
            body=body.replace(f'href="{mdpath}"',f'href="{htmlpath}"')
        body=body.replace('href="../README.md"','href="preview.html"')
        doc=f'''<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HolePlate2D-Damage | Dataset card</title><style>{css}</style></head><body><header><strong>HolePlate2D-Damage</strong><span>Dataset card · Local preview</span><nav><a href="{src}">Markdown</a></nav></header><div class="layout"><aside><strong>ON THIS PAGE</strong>{md.toc}<p><a href="https://github.com/qilinli/StructBench">Style reference: StructBench ↗</a></p></aside><article>{body}</article></div></body></html>'''
        (ROOT/dest).write_text(doc,encoding='utf-8')
    print('Local previews generated.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preview-only',action='store_true');p.add_argument('--source',type=Path,help='Path to an extracted hole-plate source batch');a=p.parse_args()
    if not a.preview_only:
        if a.source is None: p.error('--source is required unless --preview-only is set')
        cases=load_cases(a.source);d=next(d for d in cases if d['short']=='HP2D_0471')
        compact_figures(d);schematic(d)
    preview()


if __name__=='__main__':main()
