"""Direct Monte Carlo verification of the surrogate: N CalculiX realisations of
the manufacturing/friction scatter for one design; the fibre-strength scatter is
integrated exactly per realisation (conditional weakest-link probability)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from rel import study_full as F, runner as RN, scatter as SC

def main(name='H7-out20', N=300, seed=2026):
    spec = F.designs()[name]
    D = F.design(spec); sc = F.Scatter(D['c']['fam'][1])
    U = np.random.default_rng(seed).standard_normal((N, 5))
    cs = [SC.realize(D, x, f'{name}_dmc{i}') for i, x in enumerate(sc.to_x(U))]
    RN.run_many(cs)
    bs = [F.evaluate(c, spec['vessel']) for c in cs]
    out = dict(name=name, U=U.tolist(), lnp0=[float(np.log(b['p0'])) for b in bs],
               share=[float(b['shares']['junction'] + b['shares']['dome']) for b in bs],
               ok=[json.load(open(f"{RN.STORE}/{RN.key(c)}.json"))['ok'] for c in cs])
    json.dump(out, open(f'{F.OUT}/direct_mc_{name}.json', 'w'))
    print('direct MC done', name, flush=True)

if __name__ == '__main__':
    main(*sys.argv[1:2])
