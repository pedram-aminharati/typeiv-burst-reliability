"""Paper 2 figures (scoping set), Elsevier widths, vector PDF + 600 dpi PNG."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
sys.path.insert(0, '.')
import pubstyle as P
P.use()
import tr_eval as T
import bench_agne as BA
from rel import runner as RN, weakest as W

OUT = './figs_p2'
os.makedirs(OUT, exist_ok=True)
INK = P.INK


def crop(fn, pad=10, box=None):
    im = np.asarray(Image.open(fn).convert('RGB'))
    H, Wd = im.shape[:2]
    if box:
        x0, y0, x1, y1 = int(box[0]*Wd), int(box[1]*H), int(box[2]*Wd), int(box[3]*H)
        im = im[y0:y1, x0:x1]
    m = (im < 245).any(axis=2)
    ys, xs = np.where(m)
    return im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def fig_risk():
    """section renders coloured by the weakest-link risk density at 2.25 NWP"""
    ims = [crop(f'./schem_p2/risk_{n}.png', box=(0, 0, 1, 0.98)) for n in ('H5-ref', 'H5-best')]
    J = json.load(open('./schem_p2/risk_H5-ref.json'))
    fig = plt.figure(figsize=(P.W2, 0.42*P.W2))
    for k, (im, lab, txt) in enumerate(zip(ims, 'ab', ('reference transition: hoop end at the tangent line',
                                                       'Paper-1 transition: hoop end 0.125$\\sqrt{Rt}$ onto the dome'))):
        ax = fig.add_axes([0.0 + 0.47*k, 0.09, 0.46, 0.88])
        ax.imshow(im, interpolation='antialiased'); ax.axis('off')
        ax.set_aspect('equal')
        P.label(ax, lab, x=0.03, y=0.97)
        ax.text(0.5, -0.02, txt, transform=ax.transAxes, ha='center', va='top', fontsize=7)
    cax = fig.add_axes([0.955, 0.2, 0.012, 0.66])
    sm = ScalarMappable(Normalize(J['vmin'], J['vmax']), plt.get_cmap('inferno'))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('log$_{10}$ risk density at 2.25 NWP (mm$^{-3}$)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    P.save(fig, f'{OUT}/fig_risk_sections')


def fig_agne():
    rows = []
    for nm in ('GFRP', 'CFRP'):
        pexp = BA.LAYUP[nm][3]
        for v in ('V1', 'V3'):
            c = T.agne_case(nm); c['variant'] = v
            if v == 'V1':
                c = T.agne_case(nm); c['name'] = 'agne_' + nm
            else:
                c['name'] = f'agne_{nm}_{v}'
            d = RN.data(c)
            Xt = c['mat']['Xt']
            base = W.Weibull(Xt, 25, 2000)
            b = base.burst(d)
            ff = base.first_failure(d)
            span = [W.Weibull(Xt, m, v0).burst(d)['p50'] for m in (15, 25, 40) for v0 in (500, 2000, 8000)]
            q = lambda pr: b['p0']*(-np.log(1 - pr))**(1/25)
            rows.append(dict(nm=nm, v=v, pexp=pexp, p50=b['p50'], q05=q(0.05), q95=q(0.95),
                             lo=min(span), hi=max(span), ff=ff['p_ff'], ffz=ff['zone'], sh=b['shares']))
    json.dump(rows, open(f'{OUT}/agne_weakest.json', 'w'), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(P.W2, 0.36*P.W2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.32))
    x = np.arange(len(rows))
    a = ax[0]
    for i, r in enumerate(rows):
        a.plot([i - 0.32, i + 0.32], [r['pexp']]*2, color=INK, lw=1.2, zorder=4)
        a.vlines(i - 0.1, r['lo'], r['hi'], color=P.MUTED, lw=3.0, alpha=0.45, zorder=2)
        a.errorbar(i - 0.1, r['p50'], yerr=[[r['p50'] - r['q05']], [r['q95'] - r['p50']]], fmt='o', color=P.C['blue'],
                   ms=4, capsize=2.2, lw=0.9, zorder=5, label='weakest link, median and 5-95%' if i == 0 else None)
        a.plot(i + 0.12, r['ff'], marker='s', color=P.C['orange'], ms=4, ls='none', zorder=5,
               label='first fibre failure, mean strength' if i == 0 else None)
    a.plot([], [], color=INK, lw=1.2, label='burst test')
    a.plot([], [], color=P.MUTED, lw=3, alpha=0.45, label='$m$ = 15-40, $V_0$ = 0.5-8 cm$^3$')
    a.set_xticks(x); a.set_xticklabels([f"{r['nm']}\n{r['v']}" for r in rows])
    a.set_ylabel('Burst pressure (MPa)'); a.set_ylim(12, 36); a.set_xlim(-0.6, len(rows) - 0.4)
    a.xaxis.set_minor_locator(plt.NullLocator())
    a.legend(loc='lower left', fontsize=6.2, ncol=2, columnspacing=1.0)
    P.label(a, 'a')
    a = ax[1]
    bottom = np.zeros(len(rows))
    for z in ('barrel', 'junction', 'dome'):
        v = np.array([r['sh'][z] for r in rows])
        a.bar(x, v, 0.6, bottom=bottom, color=P.ZONE[z], label=z, edgecolor='white', lw=0.4)
        bottom += v
    a.set_xticks(x); a.set_xticklabels([f"{r['nm']}\n{r['v']}" for r in rows])
    a.xaxis.set_minor_locator(plt.NullLocator())
    a.set_ylabel('Probability of burst origin'); a.set_ylim(0, 1.18)
    a.legend(loc='upper center', ncol=3, fontsize=6.4, handlelength=1.0, columnspacing=0.8)
    P.label(a, 'b')
    P.save(fig, f'{OUT}/fig_agne_weakest')
    return rows


if __name__ == '__main__':
    what = sys.argv[1:] or ['risk', 'agne']
    if 'risk' in what:
        fig_risk()
    if 'agne' in what:
        for r in fig_agne():
            print(r)


# ---------------------------------------------------------------- reliability figures
from rel import surrogate as SG, scatter as SC
LAB = {'H5-ref': 'H5, reference', 'H5-best': 'H5, hoop end on shoulder',
       'H7-out20': 'H7, hoop outside, $e_\\mathrm{crit}(0.20)$', 'H7-out15': 'H7, hoop outside, $e_\\mathrm{crit}(0.15)$'}
DSTY = {'H5-ref': (P.C['blue'], '--', 'o'), 'H5-best': (P.C['blue'], '-', 's'),
        'H7-out20': (P.C['orange'], '-', '^'), 'H7-out15': (P.C['orange'], '--', 'D')}
M = 25.0
NWP = 70.0


def surrogates():
    from rel import post2
    S = json.load(open('./rel_results/scoping_v2.json'))
    out = {}
    for k, v in S.items():
        out[k] = (post2.build(k, v)[0], v)
    return out


def fig_cdf(SU):
    fig, ax = plt.subplots(figsize=(P.W1, 0.85*P.W1))
    rng = np.random.default_rng(5)
    U = rng.standard_normal((60000, 5))
    p = np.linspace(1.6, 3.4, 300)*NWP
    for k, (S, v) in SU.items():
        col, ls, mk = DSTY[k]
        lp = S.predict(U)
        F = np.array([np.mean(-np.expm1(-(pp/np.exp(lp))**M)) for pp in p])
        ok = (F > 1e-6) & (F < 0.999)
        ax.plot(p[ok]/NWP, np.log(-np.log1p(-F[ok])), color=col, ls=ls, label=LAB[k])
        Fr = v['pf_req']
        ax.plot(2.25, np.log(-np.log1p(-Fr)), marker=mk, color=col, ms=4, mec='white', mew=0.5, zorder=5)
    ax.axvline(2.25, color=INK, lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([1.8, 2.0, 2.25, 2.5, 3.0]); ax.set_xticklabels(['1.8', '2.0', '2.25', '2.5', '3.0'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    yt = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9]
    ax.set_yticks([np.log(-np.log1p(-q)) for q in yt]); ax.set_yticklabels([f'{q:g}' for q in yt])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(np.log(-np.log1p(-1e-5)), np.log(-np.log1p(-0.95)))
    ax.set_xlabel('Burst pressure / NWP'); ax.set_ylabel('Probability of burst (Weibull scale)')
    ax.legend(loc='upper left', fontsize=6.0)
    P.save(fig, f'{OUT}/fig_burst_cdf')


def fig_mu(SU):
    mus = np.linspace(0.08, 0.40, 33)
    sc = SC.Scatter(mu_med=0.2)
    rng = np.random.default_rng(6)
    U = rng.standard_normal((20000, 5))
    fig, ax = plt.subplots(1, 2, figsize=(P.W2*0.75, 0.32*P.W2), gridspec_kw=dict(wspace=0.38))
    for k, (S, v) in SU.items():
        col, ls, mk = DSTY[k]
        pf, sh = [], []
        for mu in mus:
            U2 = U.copy(); U2[:, 3] = np.log(mu/0.2)/sc.s_mu
            lp = S.predict(U2)
            pf.append(np.mean(-np.expm1(-(2.25*NWP/np.exp(lp))**M)))
            sh.append(np.mean(S.share(U2)))
        ax[0].semilogy(mus, pf, color=col, ls=ls, label=LAB[k])
        ax[1].plot(mus, sh, color=col, ls=ls)
    ax[0].axhline(1e-3, color=INK, lw=0.6, ls=':')
    ax[0].text(0.398, 0.93e-3, 'target', ha='right', va='top', fontsize=6.5); ax[0].set_ylim(6e-4, 0.3)
    for a in ax:
        a.axvspan(0.1, 0.3, color=P.LIGHT, lw=0, zorder=0)
        a.set_xlabel('Actual slippage coefficient $\\mu$')
    ax[0].set_ylabel('$P$(burst < 2.25 NWP)')
    ax[1].set_ylabel('Probability that burst starts\nat the transition')
    ax[1].set_ylim(-0.02, 1.02)
    ax[0].legend(loc='upper right', fontsize=5.8)
    P.label(ax[0], 'a'); P.label(ax[1], 'b')
    P.save(fig, f'{OUT}/fig_friction_reliability')


def fig_sobol(SU):
    names = ['$k_h$', '$k_{hel}$', '$\\Delta\\alpha_2$', '$\\mu$', '$E_1$', 'strength']
    fig, ax = plt.subplots(figsize=(P.W1*1.15, 0.7*P.W1))
    keys = list(SU)
    w = 0.8/len(keys)
    vW = np.pi**2/(6*M**2)
    for i, k in enumerate(keys):
        S, v = SU[k]
        st = np.clip(np.array(v['sobol_ST']), 0, None)
        Vu = None
        s1, stt, Vu = SG.sobol(S, n=8000)
        frac = np.append(np.clip(stt, 0, None)*Vu, vW)/(Vu + vW)
        col, ls, mk = DSTY[k]
        ax.bar(np.arange(6) + (i - 1.5)*w, frac, w*0.95, color=col, alpha=0.95 if ls == '-' else 0.5,
               edgecolor='white', lw=0.3, label=LAB[k])
    ax.set_xticks(np.arange(6)); ax.set_xticklabels(names)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel('Share of the variance of ln(burst pressure)')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5.8, loc='upper left')
    P.save(fig, f'{OUT}/fig_sobol')


if __name__ == '__main__' and 'rel' in sys.argv:
    SU = surrogates()
    fig_cdf(SU); fig_mu(SU); fig_sobol(SU)
