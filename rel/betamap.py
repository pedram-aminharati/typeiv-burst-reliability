"""Nominal reliability maps over the Paper-1 hoop-end grid (e/sqrt(Rt), l/sqrt(Rt)):
burst scale p0 and P_f(2.25 NWP) with strength scatter only."""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
import p1_study as PS
from rel import study_full as F, runner as RN

VES = {'H5': None, 'H1': None, 'H7': 'H block outside', 'S70': 'H block outside', 'H3': 'H block outside'}

def main():
    fn = f'{F.OUT}/betamap.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for vn, order in VES.items():
        if vn in res: continue
        cs, idx = [], []
        for e in PS.E_GRID:
            for l in PS.L_GRID:
                spec = dict(vessel=vn, order=order, e_w=e, l_w=l)
                cs.append(F.nominal(spec, f'{vn}-map_e{e}_l{l}')); idx.append((e, l))
        RN.run_many(cs)
        P = PS.VESSELS[vn]['P']
        rows = []
        for (e, l), c in zip(idx, cs):
            b = F.evaluate(c, vn)
            rows.append(dict(e=e, l=l, p0=b['p0'], pf=float(-np.expm1(-(2.25*P/b['p0'])**F.M)),
                             shares=b['shares']))
        b0 = F.design(dict(vessel=vn, order=order, mu_d=0.2, l_w=0.5))
        res[vn] = dict(order=order, rows=rows, e_crit02=b0['e_d']/PS.wj(b0['c']),
                       e_crit=[[mu, PS.e_crit(b0['c'], mu)] for mu in (0.1, 0.15, 0.2, 0.25, 0.3)])
        json.dump(res, open(fn, 'w'), indent=1)
        print('map', vn, flush=True)

if __name__ == '__main__':
    main()
