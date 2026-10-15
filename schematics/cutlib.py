"""Generic 3D cutaway of a Type IV vessel built from the CalculiX row data
(build_inp rowinfo): every layer thickness is the one used in the FE model."""
import json
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


class Vessel:
    def __init__(self, rows, g, X=1.0, TL=4.0, eps=0.05, capf=None, sigma=0.0, TLF=None):
        rows = sorted(rows, key=lambda r: r['zc'])
        # slots from the mid-barrel row: merge +/- plies of a family into one layer
        mid = min(rows, key=lambda r: abs(r['zc']))
        slots, seen = [], {}
        for a, t, n in mid['plies']:
            fam = n.rstrip('+-')
            if n.endswith('-'):
                continue
            seen[fam] = seen.get(fam, 0) + 1
            slots.append((fam, seen[fam]))
        self.slots = slots
        self.g, self.X = g, X
        ns = len(rows)
        T = np.zeros((ns, len(slots)))
        ANG = np.full((ns, len(slots)), np.nan)
        for i, r in enumerate(rows):
            cnt = {}
            for a, t, n in r['plies']:
                fam = n.rstrip('+-')
                if n.endswith('-'):
                    j = slots.index((fam, cnt[fam]))
                    T[i, j] += t
                    continue
                cnt[fam] = cnt.get(fam, 0) + 1
                j = slots.index((fam, cnt[fam]))
                T[i, j] += t
                ANG[i, j] = abs(a)
        xs = np.array([r['zc'] for r in rows])
        rs = np.array([r['rc'] for r in rows])
        self.Tfe = T.copy()
        if capf is not None:                         # display cap of the polar build-up
            j0 = int(np.argmin(abs(xs)))
            for j, (fam, _) in enumerate(slots):
                if fam != 'H':
                    T[:, j] = np.minimum(T[:, j], capf*T[j0, j])
        if sigma > 0:                                # display smoothing along the arc length
            sarc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(rs)))])
            W = np.exp(-0.5*((sarc[:, None] - sarc[None, :])/sigma)**2)
            for j, (fam, _) in enumerate(slots):
                if fam == 'H':
                    continue
                m = T[:, j] > eps
                Ts = (W*(T[:, j]*m)[None, :]).sum(1)/np.maximum((W*m[None, :]).sum(1), 1e-12)
                T[:, j] = np.where(m, Ts, 0.0)
        nr, nx = [], []
        for r in rows:
            tr, tz = r['ts']
            a, b = tz, -tr
            if a + b*np.sign(r['zc']) < 0:
                a, b = -a, -b
            nr.append(a); nx.append(b)
        self.xs, self.rs = xs, rs
        self.nrs, self.nxs = np.array(nr), np.array(nx)
        self.zone = [r['zone'] for r in rows]
        self.Tply = T
        self.ANG = ANG
        tl = np.full(ns, TL/X)
        self.flange = np.zeros(ns, bool)
        if TLF is not None:                          # boss flange under the polar build-up
            self.flange = rs <= g['rf']
            tl[self.flange] = TLF/X
        Tl = np.c_[tl, T]*X                      # liner first
        self.T = Tl
        self.OFF = np.concatenate([-Tl[:, :1], np.zeros((ns, 1)), np.cumsum(Tl[:, 1:], axis=1)], axis=1)
        self.nl = Tl.shape[1]                                      # layers incl. liner
        self.eps = eps

    # ------------------------------------------------------------- geometry helpers
    def pt(self, i, jb, th):
        rr = self.rs[i] + self.nrs[i]*self.OFF[i, jb]
        xx = self.xs[i] + self.nxs[i]*self.OFF[i, jb]
        return np.array([xx, rr*np.cos(th), rr*np.sin(th)])

    def cyl(self, i, jb):
        return self.xs[i] + self.nxs[i]*self.OFF[i, jb], self.rs[i] + self.nrs[i]*self.OFF[i, jb]


class Cutaway:
    def __init__(self, V, xcut, wa, wb, nt=240):
        self.V, self.xcut, self.wa, self.wb = V, xcut, wa, wb
        self.TH = np.linspace(0, 2*np.pi, nt)

    def in_wedge(self, t):
        t = np.mod(np.asarray(t) - self.wa, 2*np.pi)
        return t <= (self.wb - self.wa)

    def surface(self, jb, cut=True, exposed_only=False, sel=None):
        V = self.V
        TH = self.TH
        P = np.array([[V.pt(i, jb, t) for t in TH] for i in range(len(V.xs))])
        grid = pv.StructuredGrid(P[:, :, 0], P[:, :, 1], P[:, :, 2])
        xm = 0.5*(V.xs[:-1] + V.xs[1:]); tm = 0.5*(TH[:-1] + TH[1:])
        keep = np.ones((len(xm), len(tm)), bool)
        if cut:
            keep &= ~((xm > self.xcut)[:, None] & self.in_wedge(tm)[None, :])
        if jb >= 2:
            tt = 0.5*(V.T[:-1, jb - 1] + V.T[1:, jb - 1])
            keep &= (tt > V.eps)[:, None]
        if sel is not None:
            keep &= (sel[:-1] & sel[1:])[:, None]
        if exposed_only:
            above = 0.5*(V.T[:-1, jb:].sum(1) + V.T[1:, jb:].sum(1))
            keep &= (above < V.eps)[:, None]
        ids = np.where(keep.T.ravel())[0]
        if len(ids) == 0:
            return None
        return grid.extract_cells(ids).extract_surface()

    def cut_faces(self, jb, sel=None, transverse=True, scal=None):
        """section faces of the layer between boundaries jb-1 and jb; scal: optional
        per-station scalar attached as point data 'v'"""
        V = self.V
        out = []
        m = V.xs >= self.xcut
        if sel is not None:
            m &= sel
        sel = np.where(m)[0]
        for ang in (self.wa, self.wb):
            pts, faces = [], []
            for i in sel:
                for jj in (jb - 1, jb):
                    pts.append(V.pt(i, jj, ang))
            for k in range(len(sel) - 1):
                faces += [4, 2*k, 2*k + 1, 2*k + 3, 2*k + 2]
            mesh = pv.PolyData(np.array(pts), np.array(faces))
            if scal is not None:
                mesh['v'] = np.repeat(np.asarray(scal)[sel], 2)
            out.append(mesh)
        i0 = sel[0]
        x0, r0 = V.cyl(i0, jb - 1); _, r1 = V.cyl(i0, jb)
        if transverse and r1 - r0 > 1e-3:
            ang = np.linspace(self.wa, self.wb, 90)
            pts = [(x0, r*np.cos(a), r*np.sin(a)) for a in ang for r in (r0, r1)]
            faces = []
            for k in range(len(ang) - 1):
                faces += [4, 2*k, 2*k + 1, 2*k + 3, 2*k + 2]
            mesh = pv.PolyData(np.array(pts), np.array(faces))
            if scal is not None:
                mesh['v'] = np.full(len(pts), np.asarray(scal)[i0])
            out.append(mesh)
        return out

    def end_faces(self, jb):
        """close a layer where it ends with finite thickness (turnaround, pole)"""
        V = self.V
        tt = V.T[:, jb - 1]
        out = []
        n = len(V.xs)
        for i in range(n):
            if tt[i] <= V.eps:
                continue
            for nb in (i - 1, i + 1):
                last = nb < 0 or nb >= n
                if not last and tt[nb] > V.eps:
                    continue
                if tt[i] < 0.25*V.X:
                    continue
                th = self.TH
                if V.xs[i] > self.xcut:
                    th = th[~self.in_wedge(th)]
                pts = []
                for t in th:
                    pts.append(V.pt(i, jb - 1, t)); pts.append(V.pt(i, jb, t))
                faces = []
                for k in range(len(th) - 1):
                    if abs(th[k + 1] - th[k]) > 2*(self.TH[1] - self.TH[0]):
                        continue
                    faces += [4, 2*k, 2*k + 1, 2*k + 3, 2*k + 2]
                out.append(pv.PolyData(np.array(pts), np.array(faces)))
        return out

    def fibre_lines(self, jb, rt, n=40, lift=0.3, xmin=-1e9, xmax=1e9):
        """geodesic (Clairaut) paths r sin(a) = rt on the exposed part of boundary jb"""
        V = self.V
        lines = []
        exposed = (V.T[:, jb - 1] > V.eps) & (V.T[:, jb:].sum(1) < V.eps)
        for sign in (1, -1):
            for t0 in np.linspace(0, 2*np.pi, n, endpoint=False):
                th = [t0]
                for i in range(1, len(V.xs)):
                    rm = 0.5*(V.rs[i] + V.rs[i - 1])
                    sa = min(rt/rm, 0.999)
                    ds = np.hypot(V.xs[i] - V.xs[i - 1], V.rs[i] - V.rs[i - 1])
                    th.append(th[-1] + sign*np.tan(np.arcsin(sa))/rm*ds)
                th = np.array(th)
                rr = V.rs + V.nrs*(V.OFF[:, jb] + lift); xx = V.xs + V.nxs*(V.OFF[:, jb] + lift)
                P = np.c_[xx, rr*np.cos(th), rr*np.sin(th)]
                vis = exposed & ~((V.xs > self.xcut) & self.in_wedge(th)) & (V.xs >= xmin) & (V.xs <= xmax)
                idx = np.where(vis)[0]
                if len(idx) < 2:
                    continue
                for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                    if len(seg) > 1:
                        lines.append(P[seg])
        return lines

    def hoop_lines(self, jb, pitch=4.0, lift=0.3):
        V = self.V
        lines = []
        exposed = (V.T[:, jb - 1] > V.eps) & (V.T[:, jb:].sum(1) < V.eps)
        for xq in np.arange(V.xs.min(), V.xs.max(), pitch):
            i = np.argmin(abs(V.xs - xq))
            if not exposed[i]:
                continue
            th = self.TH if V.xs[i] <= self.xcut else self.TH[~self.in_wedge(self.TH)]
            _, rr = V.cyl(i, jb)
            rr += lift
            lines.append(np.c_[np.full(len(th), xq), rr*np.cos(th), rr*np.sin(th)])
        return lines


def polyline(arrs):
    if not arrs:
        return None
    pts, cells, off = [], [], 0
    for a in arrs:
        pts.append(a); cells += [len(a)] + list(range(off, off + len(a))); off += len(a)
    m = pv.PolyData(np.vstack(pts)); m.lines = np.array(cells)
    return m


def project(p, pt, size):
    ren = p.renderer
    ren.SetWorldPoint(*pt, 1.0); ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    return (x/size[0], 1 - y/size[1])
