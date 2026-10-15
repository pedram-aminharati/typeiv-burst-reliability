"""Paper 2: section renders coloured by the weakest-link risk density at the
burst requirement (H5 reference and Paper-1 transition)."""
import json, sys
import numpy as np
import pyvista as pv
sys.path.insert(0, '.'); sys.path.insert(0, './schem_ijhe')
sys.path.insert(0, './rel')
import vessel_tr as vt
import p1_study as PS
from rel import runner as RN, weakest as W
from rel import study_full as F
from cutlib import Vessel, Cutaway, polyline, project
import matplotlib
pv.OFF_SCREEN = True
M = 25.0
COL = {'liner': '#ece6d6', 'H': '#2f6db8', 'F0': '#6f7680', 'F1': '#a3aab2'}
SURF = '#9ba2aa'


def density(c, wb, vessel_meta, V, p):
    """log10 risk density per (station, slot) at pressure p"""
    d = RN.data(c)
    sp = np.clip(d['sig'], 0, None)
    dens = np.mean((p*sp/wb.s0)**M, axis=1)/wb.V0          # per mm^3
    # map entries -> (row, slot)
    rows = vessel_meta['rowinfo']
    out = np.full((len(rows), len(V.slots)), np.nan)
    k = 0
    for er, info in enumerate(rows):
        cnt = {}
        for a, t, nm in info['plies']:
            fam = nm.rstrip('+-')
            if nm.endswith('-'):
                j = V.slots.index((fam, cnt[fam]))
            else:
                cnt[fam] = cnt.get(fam, 0) + 1
                j = V.slots.index((fam, cnt[fam]))
            # entries in d follow the order of meta['elems'] rows with ej == 0
            k += 1
    # direct mapping: d['row'] gives er; ply order within row preserved
    for er in range(len(rows)):
        idx = np.where(d['row'] == er)[0]
        cnt = {}
        for q, (a, t, nm) in zip(idx, rows[er]['plies']):
            fam = nm.rstrip('+-')
            if nm.endswith('-'):
                j = V.slots.index((fam, cnt[fam]))
            else:
                cnt[fam] = cnt.get(fam, 0) + 1
                j = V.slots.index((fam, cnt[fam]))
            out[er, j] = np.nanmax([out[er, j], dens[q]]) if np.isfinite(out[er, j]) else dens[q]
    return out


def render(c, vn, fn, vmin, vmax, size=(2400, 1500), show_bar=False):
    meta = vt.build(c, './scratch/p2geo')
    g = meta['g']
    V = Vessel(meta['rowinfo'], g, X=1.0, TL=4.0, capf=1.8, sigma=6.0)
    wb = F.weibull(vn)
    P = PS.VESSELS[vn]['P']
    D = density(c, wb, meta, V, 2.25*P)
    # Vessel sorts rows by zc; rowinfo is already ordered by zc
    order = np.argsort([r['zc'] for r in meta['rowinfo']])
    D = D[order]
    zones = np.array([r['zone'] for r in meta['rowinfo']])[order]
    D[zones == 'boss', :] = np.nan
    C = Cutaway(V, xcut=0.0, wa=np.radians(90), wb=np.radians(180))
    p = pv.Plotter(off_screen=True, window_size=size, lighting='three lights')
    p.set_background('white')
    kw = dict(smooth_shading=True, specular=0.25, specular_power=18, ambient=0.18, diffuse=0.8)
    p.add_mesh(C.surface(0), color='#d9d2bf', **kw)
    from hero_render import CMAP as cmap
    for jb in range(1, V.nl + 1):
        name = 'liner' if jb == 1 else V.slots[jb - 2][0]
        s = C.surface(jb, exposed_only=(jb > 1))
        if s is not None and s.n_cells:
            p.add_mesh(s, color=(COL['liner'] if jb == 1 else SURF), **kw)
        if jb == 1:
            for m in C.cut_faces(jb):
                p.add_mesh(m, color=COL['liner'], ambient=0.45, diffuse=0.6)
        else:
            with np.errstate(invalid='ignore', divide='ignore'):
                sc = np.log10(np.clip(D[:, jb - 2], 10**vmin, None))
            for m in C.cut_faces(jb, scal=sc):
                p.add_mesh(m, scalars='v', cmap=cmap, clim=(vmin, vmax), ambient=0.5, diffuse=0.55,
                           specular=0.0, show_scalar_bar=False, interpolate_before_map=True,
                           nan_color='#c3c7cc')
            for m in C.end_faces(jb):
                p.add_mesh(m, color=SURF, ambient=0.35, diffuse=0.7)
    for sg in (-1, 1):
        xa = sg*(g['L']/2 + g['hd'])
        p.add_mesh(pv.Cylinder(center=(xa + sg*62, 0, 0), direction=(1, 0, 0), radius=0.97*g['rb'],
                               height=124, resolution=120), color='#b9bec5', smooth_shading=True,
                   specular=0.7, specular_power=40, ambient=0.2)
    try:
        p.enable_ssao(radius=10, bias=0.5, kernel_size=128)
    except Exception:
        pass
    p.enable_anti_aliasing('ssaa')
    L, hd, R = g['L'], g['hd'], g['R']
    focal = (L/2 - 0.05*hd, 0, 0.72*R)
    D_ = 4.6*R
    a = np.radians(172)
    p.camera.focal_point = focal
    p.camera.position = (focal[0] - 0.10*D_, D_*np.cos(a), focal[2] + D_*np.sin(a))
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 15
    p.render()
    A = {}
    pr = lambda q: project(p, q, size)
    xi = lambda x: int(np.argmin(abs(V.xs - x)))
    A['tl_in'] = pr(V.pt(xi(L/2), 0, np.radians(90)))
    A['tl_out'] = pr(V.pt(xi(L/2), V.nl, np.radians(90)))
    p.screenshot(fn, transparent_background=False)
    json.dump(dict(anchors=A, vmin=vmin, vmax=vmax), open(fn.replace('.png', '.json'), 'w'))
    p.close()
    return D


if __name__ == '__main__':
    out = {}
    for name in sys.argv[1:] or ('H5-ref', 'H5-e0125'):
        spec = F.designs()[name]
        c = F.nominal(spec, name)
        RN.run_many([c])
        Dm = render(c, spec['vessel'], f'risk_{name}.png', vmin=-15, vmax=-8)
        print(name, np.nanmax(np.log10(Dm)))
