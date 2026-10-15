# Where do Type IV hydrogen vessels burst? Weakest-link reliability of the cylinder–dome transition

Code and result data for

> P. Aminharati, M. Shirinbayan, K. Benfriha, J. Fitoussi, *Where do Type IV hydrogen vessels
> burst? Weakest-link reliability of the cylinder–dome transition under manufacturing and
> winding-friction scatter*, International Journal of Hydrogen Energy, submitted, 2026.

A weakest-link (Weibull volume-integral) burst model is evaluated on the ply-level fibre
stresses of CalculiX composite-shell models. Manufacturing scatter (layer thickness, winding
angle, fibre modulus) and the tow–surface slippage coefficient, which limits the windable hoop
end, are propagated with a Gaussian-process surrogate and strength-integrated active learning,
verified by 300 direct Monte Carlo analyses.

## Requirements
* [CalculiX](http://www.calculix.de/) `ccx` 2.21 on the `PATH`
* Python ≥ 3.10 with the packages in `requirements.txt`

Run all scripts from the repository root, e.g. `python -m rel.study_full`.

## Contents
| File | Purpose |
|---|---|
| `core.py`, `vessel_gen.py`, `vessel_tr.py`, `p1_study.py`, `sizing.json` | vessel geometry, dome-consistent walls, transition designs, CalculiX input generator |
| `rel/runner.py` | hash-cached, verified CalculiX job runner (equilibrium, symmetry and Lekhnitskii checks) |
| `rel/weakest.py` | weakest-link burst model, origin probabilities, strength anchoring |
| `rel/scatter.py` | random inputs and friction-limited hoop end |
| `rel/surrogate.py` | Gaussian-process surrogate, active learning, Monte Carlo, FORM, Sobol indices |
| `rel/study_full.py` | full study (34 designs); `run_full.sh`, `chain2.sh`, `chain3.sh` reproduce the batch |
| `rel/direct_mc.py`, `rel/betamap.py`, `rel/meshconv.py`, `rel/msens.py`, `rel/target_sens.py`, `rel/thermal*.py` | verification and sensitivity studies |
| `figs_full.py`, `figs_ga_p2.py`, `figs_p2.py`, `schematics/` | figures, renders and graphical abstract |
| `make_tables_p2.py`, `make_refs_p2_ieee.py` | manuscript tables and bibliography |

## Data (`rel_results/`)
* `full.json` – per design: nominal burst scale and origin shares, P_f and beta (Monte Carlo and FORM), Sobol indices, wall factor, and the surrogate training data
* `direct_mc_H7-out20.json` – 300 direct Monte Carlo realisations
* `betamap.json`, `meshconv.json`, `msens.json`, `target_sens.json`, `thermal.json`

## Licence
MIT (see `LICENSE`). Please cite the article above when using the code or data.
