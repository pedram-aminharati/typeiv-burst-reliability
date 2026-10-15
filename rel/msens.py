"""Sensitivity of the nominal burst-origin probabilities and P_f(2.25 NWP) to the
Weibull modulus m (anchor re-computed for every m; with the anchor, V0 cancels out)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
import p1_study as PS
from rel import study_full as F, runner as RN

def main():
    R = json.load(open(f'{F.OUT}/full.json'))
    out = {}
    for name, r in R.items():
        spec = r['spec']; vn = spec['vessel']; P = PS.VESSELS[vn]['P']
        d = RN.data(F.nominal(spec, name))
        rec = {}
        for m in (15, 20, 25, 30, 40):
            wb = F.weibull(vn, m=m)            # anchored: V0 cancels out
            b = wb.burst(d)
            rec[str(m)] = dict(pf=float(-np.expm1(-(2.25*P/b['p0'])**m)), shares=b['shares'], p50=b['p50']/P)
        out[name] = rec
        print(name, {k: round(v['shares']['junction'] + v['shares']['dome'], 3) for k, v in rec.items()}, flush=True)
    json.dump(out, open(f'{F.OUT}/msens.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
