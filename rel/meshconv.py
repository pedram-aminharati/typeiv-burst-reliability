import json, sys
sys.path.insert(0, '.')
import numpy as np
from rel import runner as RN, study2 as S2
MESH = [None, dict(nb=36, nd=75, ref=1.5), dict(nb=48, nd=100, ref=2.0)]
out = {}
cs = []
for name in ('H5-ref', 'H5-best', 'H7-out20'):
    for mz in MESH:
        c = S2.nominal(S2.DESIGNS[name], name)
        if mz:
            c = dict(c, mesh=mz, name=c['name'] + f"_m{mz['ref']}")
        cs.append((name, mz, c))
RN.run_many([c for _, _, c in cs])
for name, mz, c in cs:
    vn = S2.DESIGNS[name]['vessel']
    res = {}
    for m in (15, 25, 40):
        b = S2.evaluate(c, vn, m=m)
        res[m] = dict(p0=b['p0'], tr=b['shares']['junction'] + b['shares']['dome'])
    out.setdefault(name, []).append(dict(mesh=mz or dict(nb=24, nd=50, ref=1.0), res=res))
    print(name, mz, {m: (round(r['p0'], 3), round(r['tr'], 4)) for m, r in res.items()}, flush=True)
json.dump(out, open('./rel_results/meshconv.json', 'w'), indent=1)
