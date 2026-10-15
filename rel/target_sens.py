"""Wall factor k_req for alternative targets P_f,T = 1e-3, 1e-5 (from the stored surrogates)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from rel import study_full as F, surrogate as SG
import figs_full as FF

def main():
    R = json.load(open(f'{F.OUT}/full.json'))
    out = {}
    for k, r in R.items():
        S, sc = FF.surrogate(k, r)
        pr = 2.25*r['P']
        out[k] = {}
        for pt in (1e-3, 1e-4, 1e-5):
            try:
                out[k][f'{pt:g}'] = F.k_required(S, r['gamma'], pr, pf_t=pt)
            except ValueError:
                out[k][f'{pt:g}'] = None
        print(k, out[k], flush=True)
    json.dump(out, open(f'{F.OUT}/target_sens.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
