"""Paper 2 hero render: whole vessel with a wedge removed; section faces and the
outer surface coloured by the weakest-link risk density at 2.25 NWP."""
import json, sys
import numpy as np
import pyvista as pv
sys.path.insert(0, '.'); sys.path.insert(0, './schem_ijhe')
sys.path.insert(0, './schem_p2')
import vessel_tr as vt
import p1_study as PS
from rel import runner as RN, study_full as F
from cutlib import Vessel, Cutaway, project
import risk_render as RR
import matplotlib
pv.OFF_SCREEN = True
SCR = './scratch/p2hero'
SURF = '#9ba2aa'
LAYER = {'H': '#2f6db8', 'F0': '#6f7680', 'F1': '#a3aab2'}
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list('risk', ['#e9ecef', '#d5dbe3', '#f3d58a', '#ee8a3c', '#cf3a2a', '#7a0f2f', '#2b0a1e'])


class WB:
    pass


def risk(name, p_fac=2.25):
    spec = F.designs()[name]
    c = F.nominal(spec, name)
    RN.run_many([c])
    meta = vt.build(c, SCR)
    V = Vessel(meta['rowinfo'], meta['g'], X=1.0, TL=4.0, capf=1.8, sigma=6.0)
    wb = F.weibull(spec['vessel'])
    P = PS.VESSELS[spec['vessel']]['P']
    D = RR.density(c, wb, meta, V, p_fac*P)
    order = np.argsort([r['zc'] for r in meta['rowinfo']])
    D = D[order]
    zones = np.array([r['zone'] for r in meta['rowinfo']])[order]
    D[zones == 'boss', :] = np.nan
    Q = origin_density(c, meta, wb, p_fac*P)
    return c, meta, V, D, Q


def origin_density(c, meta, wb, p):
    """burst-origin probability per unit mid-surface area [mm^-2] per row (rows sorted by zc)"""
    d = RN.data(c)
    sp = np.clip(d['sig'], 0, None)
    term = d['vol']/wb.V0*np.mean((p*sp/wb.s0)**wb.m, axis=1)
    rows = meta['rowinfo']
    B = np.array([term[d['row'] == er].sum() for er in range(len(rows))])
    area = np.array([2*np.pi*r['rc']*r['seglen'] for r in rows])
    zone = np.array([r['zone'] for r in rows])
    B[zone == 'boss'] = 0.0
    q = B/area/B.sum()
    order = np.argsort([r['zc'] for r in rows])
    q = q[order]; q[zone[order] == 'boss'] = np.nan
    return q


def scal_surface(C, jb, scal, exposed_only=True):
    V = C.V
    TH = C.TH
    P = np.array([[V.pt(i, jb, t) for t in TH] for i in range(len(V.xs))])
    grid = pv.StructuredGrid(P[:, :, 0], P[:, :, 1], P[:, :, 2])
    grid['v'] = np.tile(scal, len(TH))
    xm = 0.5*(V.xs[:-1] + V.xs[1:]); tm = 0.5*(TH[:-1] + TH[1:])
    keep = ~((xm > C.xcut)[:, None] & C.in_wedge(tm)[None, :])
    if jb >= 2:
        tt = 0.5*(V.T[:-1, jb - 1] + V.T[1:, jb - 1])
        keep &= (tt > V.eps)[:, None]
    if exposed_only:
        above = 0.5*(V.T[:-1, jb:].sum(1) + V.T[1:, jb:].sum(1))
        keep &= (above < V.eps)[:, None]
    ids = np.where(keep.T.ravel())[0]
    if len(ids) == 0:
        return None
    return grid.extract_cells(ids).extract_surface()


def render(name, fn, vmin=-13, vmax=-3, size=(3600, 1700), wa=55, wb=145, cam=(28, 100, 20), view=17,
           fx=0.0, xcut=-1e9, dist=3.2, fz=0.0, qlim=(-9, -5)):
    c, meta, V, D, Q = risk(name)
    g = meta['g']
    L, hd, R = g['L'], g['hd'], g['R']
    C = Cutaway(V, xcut=xcut, wa=np.radians(wa), wb=np.radians(wb), nt=360)
    p = pv.Plotter(off_screen=True, window_size=size, lighting='three lights')
    p.set_background('white')
    kw = dict(smooth_shading=True, specular=0.2, specular_power=18, ambient=0.2, diffuse=0.8)
    cmap = CMAP
    p.add_mesh(C.surface(0), color='#d9d2bf', **kw)
    with np.errstate(invalid='ignore', divide='ignore'):
        LD = np.log10(np.clip(D, 10**vmin, None))
        LQ = np.log10(np.clip(Q, 10**qlim[0], None))
    for jb in range(1, V.nl + 1):
        if jb == 1:
            s = C.surface(1, exposed_only=False)
            p.add_mesh(s, color='#ece6d6', **kw)
            for m in C.cut_faces(1, transverse=False):
                p.add_mesh(m, color='#ece6d6', ambient=0.45, diffuse=0.6)
            continue
        sc = LD[:, jb - 2]
        s = scal_surface(C, jb, LQ)
        if s is not None and s.n_cells:
            p.add_mesh(s, scalars='v', cmap=cmap, clim=qlim, show_scalar_bar=False,
                       nan_color='#c3c7cc', interpolate_before_map=True, **kw)
        fam = V.slots[jb - 2][0]
        for m in C.cut_faces(jb, transverse=False):
            p.add_mesh(m, color=LAYER.get(fam, '#a3aab2'), ambient=0.5, diffuse=0.6, specular=0.0)
        for m in C.end_faces(jb):
            p.add_mesh(m, color=SURF, ambient=0.35, diffuse=0.7)
    for sg in (-1, 1):
        xa = sg*(L/2 + hd)
        p.add_mesh(pv.Cylinder(center=(xa + sg*45, 0, 0), direction=(1, 0, 0), radius=0.97*g['rb'],
                               height=90, resolution=120), color='#b9bec5', smooth_shading=True,
                   specular=0.7, specular_power=40, ambient=0.2)
    try:
        p.enable_ssao(radius=15, bias=0.5, kernel_size=128)
    except Exception:
        pass
    p.enable_anti_aliasing('ssaa')
    az, th, el = [np.radians(a) for a in cam]
    Dc = dist*(L/2 + hd + 90)
    focal = (fx*L, 0, fz*R)
    dirv = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az)*np.cos(th), np.cos(el)*np.sin(az)*np.sin(th)])
    dirv = dirv/np.linalg.norm(dirv)
    p.camera.focal_point = focal
    p.camera.position = tuple(np.array(focal) + Dc*dirv)
    p.camera.up = (0, np.sin(th), -np.cos(th))      # screen-up tangential: vessel axis horizontal
    p.camera.view_angle = view
    p.render()
    A = {}
    # anchors at the cylinder-dome tangent line (right) and mid-barrel
    i_tl = int(np.argmin(abs(V.xs - L/2)))
    tm = np.radians(0.5*(wa + wb))
    for nm, x in (('tl', L/2), ('barrel', L/4), ('dome_mid', L/2 + 0.5*hd), ('pole', L/2 + 0.9*hd)):
        i = int(np.argmin(abs(V.xs - x)))
        for ta in (200, 225, 250, 300):
            A[f'{nm}_{ta}'] = project(p, V.pt(i, V.nl, np.radians(ta)), size)
    A['xs_tl'] = float(L/2); A['hd'] = float(hd); A['R'] = float(R)
    p.screenshot(fn, transparent_background=False)
    json.dump(dict(anchors=A, vmin=vmin, vmax=vmax, qlim=qlim, name=name), open(fn.replace('.png', '.json'), 'w'))
    p.close()
    return D, Q


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'H5-ref'
    D, Q = render(name, f'./schem_p2/hero_{name}.png')
    print(name, np.nanmax(np.log10(D)), np.nanmin(np.log10(Q)), np.nanmax(np.log10(Q)))
