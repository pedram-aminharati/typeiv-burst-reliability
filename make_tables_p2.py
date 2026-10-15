"""Tables of Paper 2, generated from rel_results/full.json."""
import json, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, '.')
import p1_study as PS

R = json.load(open('./rel_results/full.json'))
VES = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'S65', 'S70', 'S75']
COMPACT = ('H3', 'H4', 'H7', 'H8', 'S65', 'S70', 'S75')
p1 = lambda v: f'{v}-out20' if v in COMPACT else f'{v}-e0125'
beta = lambda pf: -norm.ppf(np.clip(pf, 1e-300, 1))


def fmt_pf(x):
    if x > 0.5:
        return r'$\approx1$'
    e = int(np.floor(np.log10(x)))
    return rf'${x/10**e:.1f}\times10^{{{e}}}$'


def fmt_b(b):
    return rf'${b:.2f}$' if b > -8 else r'$<-8$'


def table_designs():
    L = [r'\begin{table}[!t]\centering', r'\caption{Vessels and the two transition designs compared. '
         r"Hoop end $e$ in units of $\sqrt{Rt}$; ``out'' denotes the two hoop groups "
         r"combined into one block outside the helical layers, ending at the friction limit $e_\mathrm{crit}(0.2)$; "
         r"``ref.'' is the reference stacking. $\ell_h=0.5\sqrt{Rt}$ throughout.}\label{tab:designs}", r'\footnotesize',
         r'\begin{tabular}{lccccll}', r'\toprule',
         r'Vessel & NWP (MPa) & $h_d/R$ & $r_b/R$ & $\sqrt{Rt}$ (mm) & Reference & Paper-1 transition\\', r'\midrule']
    for v in VES:
        if p1(v) not in R:
            continue
        r = R[p1(v)]; V = PS.VESSELS[v]
        e = r['e_d']/r['wj']
        des = (rf'$e={e:.3f}$, out' if v in COMPACT else rf'$e={e:.3f}$, ref.')
        L.append(rf"{v} & {V['P']:.0f} & {V['hd_R']:.2f} & {V['rb_R']:.2f} & {r['wj']:.1f} & "
                 rf"$e=0$, ref. & {des}\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def table_scatter():
    return r'''\begin{table}[!t]\centering
\caption{Random inputs. The fibre strength enters the weakest-link model analytically; all other
inputs are sampled. CoV: coefficient of variation.}\label{tab:scatter}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llllp{3.6cm}}
\toprule
Input & Symbol & Distribution & Median / mean & Scatter, source\\
\midrule
Hoop layer thickness factor & $k_h$ & normal & 1 & CoV 5\% \cite{azizian2023}\\
Helical layer thickness factor & $k_\mathrm{hel}$ & normal & 1 & CoV 5\% \cite{azizian2023}\\
Dome-family angle error & $\Delta\alpha_2$ & normal & $0^\circ$ & SD $0.6^\circ$ (CoV 2\% of $\alpha_2$) \cite{azizian2023}\\
Slippage coefficient & $\mu$ & lognormal & 0.20 & CoV 25\%; 95\% in 0.12--0.32 \cite{jiao2012,jois2023,p1}\\
Fibre modulus factor & $E_1$ & normal & 1 & CoV 3\% (assumed)\\
Ply tensile strength & $\sigma$ & Weibull & $m=25$ & CoV 5.0\%; $m=15$--40 studied \cite{wisnom2017}\\
\bottomrule
\end{tabular}
\end{table}'''


def table_results():
    L = [r'\begin{table}[!t]\centering', r'\caption{Reliability at the burst requirement $2.25\,$NWP '
         r'($P_{f,T}=10^{-4}$, $\beta_T=3.72$). $P_f^\mathrm{nom}$: nominal geometry, strength scatter only; '
         r'$P_f$, $\beta$: with manufacturing and friction scatter (surrogate Monte Carlo), $\beta_\mathrm{FORM}$: '
         r'first-order estimate; $\pi_\mathrm{tr}$: probability that burst starts at the transition '
         r'(junction or dome); $k_\mathrm{req}$: wall factor that restores the target; $\beta_{1.25}$: at '
         r'$1.25\,$NWP.}\label{tab:results}', r'\scriptsize', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular}{llcccccccc}', r'\toprule',
         r'Vessel & Transition & $p_{50}/$NWP & $P_f^\mathrm{nom}$ & $P_f$ & $\beta$ & $\beta_\mathrm{FORM}$ & '
         r'$\pi_\mathrm{tr}$ & $k_\mathrm{req}$ & $\beta_{1.25}$\\', r'\midrule']
    for v in VES:
        for k, lab in ((f'{v}-ref', 'reference'), (p1(v), 'Paper 1')):
            if k not in R:
                continue
            r = R[k]
            L.append(rf"{v if lab == 'reference' else ''} & {lab} & {r['nominal']['p50']/r['P']:.2f} & "
                     rf"{fmt_pf(r['pf_nom'])} & {fmt_pf(r['pf_req'])} & {fmt_b(r['beta'])} & "
                     rf"{fmt_b(r['form']['beta'])} & {r['share_transition']:.2f} & {r['k_req']:.3f} & "
                     rf"{fmt_b(r['beta_125'])}\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


if __name__ == '__main__':
    open('./paper2/tables.tex', 'w').write(table_designs() + '\n\n' + table_scatter() + '\n')
    open('./paper2/table_results.tex', 'w').write(table_results() + '\n')
    print('ok')
