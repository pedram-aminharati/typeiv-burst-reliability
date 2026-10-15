"""Cure cool-down residual stress (T0 = 120 C to 20 C) for the nominal designs:
P_f(2.25 NWP) with and without the residual fibre stress (same anchor)."""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, '.')
import vessel_tr as vt
import p1_study as PS
from rel import study_full as F, thermal as TH, runner as RN

NAMES = ('H5-ref', 'H5-e0125', 'H7-ref', 'H7-out20', 'H1-ref', 'H1-e0125', 'S70-ref', 'S70-out20')

def main():
    fn = f'{F.OUT}/thermal.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in NAMES:
        if nm in res: continue
        spec = F.designs()[nm]
        c = F.nominal(spec, nm)
        path = f'./runs/th_{nm}'
        meta = TH.build(c, path)
        ok, log = vt.run(path, ncpu=2)
        d = TH.extract(path, meta, c)
        wb = F.weibull(spec['vessel'])
        P = PS.VESSELS[spec['vessel']]['P']
        pr = 2.25*P
        d0 = dict(d, sig_th=0*d['sig_th'])
        sth = d['sig_th']
        res[nm] = dict(pf=float(TH.pf(d0, wb, pr)), pf_th=float(TH.pf(d, wb, pr)),
                       sth_min=float(sth.min()), sth_max=float(sth.max()),
                       pf125=float(TH.pf(d0, wb, 1.25*P)), pf125_th=float(TH.pf(d, wb, 1.25*P)))
        shutil.rmtree(path, ignore_errors=True)
        json.dump(res, open(fn, 'w'), indent=1)
        print(nm, res[nm], flush=True)

if __name__ == '__main__':
    main()
