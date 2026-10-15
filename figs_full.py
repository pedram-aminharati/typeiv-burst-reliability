"""Paper 2 figures from the full study (rel_results/full.json etc.).
Elsevier widths, vector PDF + 600 dpi PNG, Paper-1 style (pubstyle)."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import norm
sys.path.insert(0, '.')
import pubstyle as P
P.use()

RES = './rel_results'
OUT = './figs_p2full'
os.makedirs(OUT, exist_ok=True)
INK = P.INK
VES = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'S65', 'S70', 'S75']
COMPACT = ('H3', 'H4', 'H7', 'H8', 'S65', 'S70', 'S75')
PF_T = 1e-4
B_T = -norm.ppf(PF_T)
M = 25.0


def load():
    return json.load(open(f'{RES}/full.json'))


def p1name(v):
    return f'{v}-out20' if v in COMPACT else f'{v}-e0125'


def beta(pf):
    return -norm.ppf(np.clip(pf, 1e-300, 1))


def vlabel(v):
    if v.startswith('S'):
        return f'S{v[1:]}'
    return v


# ------------------------------------------------------------------ origin + reliability overview
def fig_overview(R):
    ves = [v for v in VES if f'{v}-ref' in R and p1name(v) in R]
    n = len(ves)
    fig, ax = plt.subplots(3, 1, figsize=(P.W2, 0.62*P.W2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.0, 1.15, 0.8], hspace=0.12))
    x = np.arange(n)
    w = 0.36
    # (a) probability of burst origin, nominal design (strength scatter only)
    a = ax[0]
    for j, (suf, dx) in enumerate((('ref', -w/2 - 0.02), ('p1', w/2 + 0.02))):
        bottom = np.zeros(n)
        for z in ('barrel', 'junction', 'dome'):
            v = np.array([R[f'{vv}-ref' if suf == 'ref' else p1name(vv)]['nominal']['shares'][z] for vv in ves])
            a.bar(x + dx, v, w, bottom=bottom, color=P.ZONE[z], edgecolor='white', lw=0.4,
                  alpha=1.0 if suf == 'ref' else 0.55, hatch=None if suf == 'ref' else '////')
            bottom += v
    a.set_ylim(0, 1.0); a.set_ylabel('Probability of\nburst origin')
    hs = [Patch(fc=P.ZONE[z], ec='white', label=z) for z in ('barrel', 'junction', 'dome')]
    hs += [Patch(fc='#999999', ec='white', label='reference (left bar)'),
           Patch(fc='#cccccc', ec='white', hatch='////', label='rule-based transition (right bar)')]
    a.legend(handles=hs, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
             handlelength=1.3, columnspacing=1.0)
    P.label(a, 'a', x=-0.055)
    # (b) reliability index at 2.25 NWP with manufacturing + friction scatter
    a = ax[1]
    BLO = 1.6
    a.axhspan(B_T, 6, color=P.LIGHT, lw=0, zorder=0)
    a.axhline(B_T, color=INK, lw=0.6, ls=':')
    a.text(n - 0.45, B_T + 0.04, r'target $\beta_T=3.72$ ($P_f=10^{-4}$)', ha='right', va='bottom', fontsize=6.3)
    for i, vv in enumerate(ves):
        r0, r1 = R[f'{vv}-ref'], R[p1name(vv)]
        x0, x1 = i - w/2 - 0.02, i + w/2 + 0.02
        b0 = max(r0['beta'], BLO + 0.12)
        a.plot([x0, x1], [b0, r1['beta']], color=P.MUTED, lw=0.6, zorder=3)
        if r0['beta'] < BLO + 0.12:
            a.plot(x0, BLO + 0.12, 'v', color=P.C['orange'], ms=4.6, mec='white', mew=0.5, zorder=5, clip_on=False)
            a.text(x0 - 0.08, BLO + 0.12, f"{r0['beta']:.1f}".replace('-', '\u2212'), fontsize=5.6, va='center',
                   ha='right', color=P.C['orange'])
        else:
            a.plot(x0, r0['beta'], 'o', color=P.C['orange'], ms=4.2, mec='white', mew=0.5, zorder=5)
        a.plot(x1, r1['beta'], 's', color=P.C['blue'], ms=4.2, mec='white', mew=0.5, zorder=5)
        for r, c, xx in ((r0, P.C['orange'], x0), (r1, P.C['blue'], x1)):
            if beta(r['pf_nom']) > BLO + 0.12:
                a.plot(xx, beta(r['pf_nom']), marker='_', color=c, ms=7, mew=1.0, zorder=4)
    a.set_ylabel(r'Reliability index $\beta$' + '\nat 2.25 NWP')
    a.set_ylim(BLO, 4.05)
    a.legend(handles=[Line2D([], [], marker='o', color=P.C['orange'], ls='none', ms=4, label='reference transition'),
                      Line2D([], [], marker='s', color=P.C['blue'], ls='none', ms=4, label='rule-based transition'),
                      Line2D([], [], marker='_', color=P.MUTED, ls='none', ms=6, mew=1.0,
                             label='strength scatter only (nominal geometry)')],
             loc='lower left', bbox_to_anchor=(0.0, 0.16), ncol=1, fontsize=6.0, handletextpad=0.3,
             frameon=True, framealpha=0.9, edgecolor='none')
    P.label(a, 'b', x=-0.055)
    # (c) wall factor that restores the target
    a = ax[2]
    KMAX = 30
    for i, vv in enumerate(ves):
        k0, k1 = R[f'{vv}-ref']['k_req'], R[p1name(vv)]['k_req']
        a.bar(i - w/2 - 0.02, min(100*(k0 - 1), KMAX), w, color=P.C['orange'], edgecolor='white', lw=0.4)
        if 100*(k0 - 1) > KMAX:
            a.text(i - w/2 - 0.02, KMAX*0.97, f'{100*(k0 - 1):.0f}', rotation=90, ha='center', va='top', fontsize=5.5,
                   color='white')
        a.bar(i + w/2 + 0.02, 100*(k1 - 1), w, color=P.C['blue'], edgecolor='white', lw=0.4)
    a.axhline(0, color=INK, lw=0.5)
    a.set_ylabel('Extra laminate\nfor target (%)'); a.set_ylim(0, KMAX)
    a.set_xticks(x); a.set_xticklabels([vlabel(v) for v in ves])
    a.xaxis.set_minor_locator(plt.NullLocator())
    P.label(a, 'c', x=-0.055)
    for aa in ax:
        aa.set_xlim(-0.6, n - 0.4)
    P.save(fig, f'{OUT}/fig_overview')


# ------------------------------------------------------------------ surrogate reconstruction
def surrogate(name, rec):
    from rel import study_full as F, surrogate as SG
    spec = rec['spec']
    D = F.design(spec)
    sc = F.Scatter(D['c']['fam'][1])
    S = SG.Surrogate(np.array(rec['U']), rec['y'], rec['sh'], fmap=F.phys_fmap(D, sc))
    return S, sc


LAB = {'H5-ref': 'H5 reference', 'H5-e0125': r'H5, $e=0.125\sqrt{Rt}$',
       'H7-ref': 'H7 reference', 'H7-out20': r'H7, outer hoop block, $e_d=e_\mathrm{crit}(0.20)$'}
DSTY = {'H5-ref': (P.C['blue'], '--', 'o'), 'H5-e0125': (P.C['blue'], '-', 's'),
        'H7-ref': (P.C['orange'], '--', '^'), 'H7-out20': (P.C['orange'], '-', 'D')}


def wscale(F):
    return np.log(-np.log1p(-F))


def fig_weibull(R, names=('H5-ref', 'H5-e0125', 'H7-ref', 'H7-out20'), dmc=None):
    """burst-pressure distribution on Weibull probability paper"""
    fig, ax = plt.subplots(figsize=(P.W1, 0.9*P.W1))
    U = np.random.default_rng(5).standard_normal((60000, 5))
    p = np.geomspace(1.7, 3.6, 300)
    for k in names:
        if k not in R:
            continue
        S, sc = surrogate(k, R[k])
        col, ls, mk = DSTY[k]
        lp = S.predict(U)
        NWP = R[k]['P']
        Fp = np.array([np.mean(-np.expm1(-(pp*NWP/np.exp(lp))**M)) for pp in p])
        ok = (Fp > 2e-7) & (Fp < 0.999)
        ax.plot(p[ok], wscale(Fp[ok]), color=col, ls=ls, label=LAB[k])
        ax.plot(2.25, wscale(R[k]['pf_req']), marker=mk, color=col, ms=4.2, mec='white', mew=0.5, zorder=5)
        # strength-only distribution of the nominal design (thin)
        p0n = R[k]['nominal']['p0']/NWP
        Fn = -np.expm1(-(p/p0n)**M)
        okn = (Fn > 2e-7) & (Fn < 0.999)
        ax.plot(p[okn], wscale(Fn[okn]), color=col, ls=ls, lw=0.5, alpha=0.6)
    if dmc:
        for k, d in dmc.items():
            col, ls, mk = DSTY[k]
            NWP = R[k]['P']
            p0s = np.exp(np.array(d['lnp0']))
            Fd = np.array([np.mean(-np.expm1(-(pp*NWP/p0s)**M)) for pp in p])
            ok = (Fd > 2e-7) & (Fd < 0.999)
            ax.plot(p[ok][::12], wscale(Fd[ok][::12]), ls='none', marker='o', mfc='none', mec=col, ms=3.2, mew=0.6,
                    label='direct Monte Carlo (300 analyses)')
    ax.axvline(2.25, color=INK, lw=0.6, ls=':')
    ax.axhline(wscale(PF_T), color=INK, lw=0.6, ls=':')
    ax.text(1.72, wscale(PF_T) + 0.15, r'$P_{f,T}=10^{-4}$', ha='left', va='bottom', fontsize=6.3)
    ax.set_xscale('log')
    xt = [1.8, 2.0, 2.25, 2.5, 3.0, 3.5]
    ax.set_xticks(xt); ax.set_xticklabels([f'{v:g}' for v in xt]); ax.xaxis.set_minor_locator(plt.NullLocator())
    yt = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99]
    ax.set_yticks([wscale(q) for q in yt])
    ax.set_yticklabels([f'$10^{{{int(np.log10(q))}}}$' if q < 0.01 else f'{q:g}' for q in yt])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(wscale(2e-7), wscale(0.995)); ax.set_xlim(1.7, 3.6)
    ax.set_xlabel('Burst pressure / NWP'); ax.set_ylabel('Probability of burst')
    ax.legend(loc='lower right', fontsize=5.8, handlelength=2.2)
    ax.text(0.98, 0.47, 'thin lines: strength scatter only\nthick lines: with manufacturing\nand friction scatter',
            transform=ax.transAxes, ha='right', va='top', fontsize=5.8, color=P.INK2)
    P.save(fig, f'{OUT}/fig_weibull')


# ------------------------------------------------------------------ friction: programmed hoop end
def fig_friction(R):
    from rel import study_full as F
    fig, ax = plt.subplots(1, 3, figsize=(P.W2, 0.31*P.W2), gridspec_kw=dict(wspace=0.42))
    mus = [0.10, 0.15, 0.20, 0.25, 0.30]
    reftxt = []
    for i, vn in enumerate(('H3', 'H7', 'S70')):
        col, mk, ls = P.SERIES[i], P.MARKERS[i], P.LINES[i]
        names = [f'{vn}-out{int(round(100*m))}' for m in mus]
        ok = [n in R for n in names]
        if not any(ok):
            continue
        mm = np.array(mus)[ok]
        rr = [R[n] for n, o in zip(names, ok) if o]
        ax[0].plot(mm, [r['beta'] for r in rr], color=col, ls=ls, marker=mk, label=vn)
        ax[0].plot(mm, [beta(r['pf_nom']) for r in rr], color=col, ls='none', marker=mk, mfc='none', ms=3.4)
        ax[1].plot(mm, [r['share_transition'] for r in rr], color=col, ls=ls, marker=mk)
        ax[2].plot(mm, [100*(r['k_req'] - 1) for r in rr], color=col, ls=ls, marker=mk)
        if f'{vn}-ref' in R:
            r0 = R[f'{vn}-ref']
            reftxt.append((col, rf"{vn}: $\beta={r0['beta']:.2f}$, $+{100*(r0['k_req'] - 1):.0f}\%$"))
    ax[0].axhline(B_T, color=INK, lw=0.6, ls=':')
    ax[0].set_ylabel(r'$\beta$ at 2.25 NWP')
    ax[1].set_ylabel('Probability that burst\nstarts at the transition'); ax[1].set_ylim(-0.02, 1.02)
    ax[2].set_ylabel('Extra laminate for target (%)')
    for a in ax:
        a.set_xlabel(r'Design slippage coefficient $\mu_d$')
        a.axvspan(0.175, 0.225, color=P.LIGHT, lw=0, zorder=0)
        a.set_xticks(mus)
    ax[0].legend(fontsize=6.3, loc='lower left')
    ax[0].text(0.98, 0.03, 'open: nominal geometry', transform=ax[0].transAxes, ha='right', va='bottom',
               fontsize=5.8, color=P.INK2)
    ax[0].set_ylim(1.5, 4.2); ax[2].set_ylim(0, None)
    ax[2].text(0.97, 0.97, 'reference transition ($e=0$):', transform=ax[2].transAxes, ha='right', va='top',
               fontsize=5.8, color=P.INK2)
    for j, (c, t) in enumerate(reftxt):
        ax[2].text(0.97, 0.89 - 0.075*j, t, transform=ax[2].transAxes, ha='right', va='top', fontsize=5.8, color=c)
    for a, l in zip(ax, 'abc'):
        P.label(a, l)
    P.save(fig, f'{OUT}/fig_friction')


# ------------------------------------------------------------------ sensitivity
NAMES = [r'$k_h$', r'$k_\mathrm{hel}$', r'$\Delta\alpha_2$', r'$\mu$', r'$E_1$', 'strength']


def fig_sobol(R, names=('H5-ref', 'H5-e0125', 'H7-ref', 'H7-out20')):
    fig, ax = plt.subplots(1, 3, figsize=(P.W2, 0.3*P.W2), gridspec_kw=dict(wspace=0.36, width_ratios=[1.25, 1.25, 1]))
    names = [k for k in names if k in R]
    w = 0.8/max(len(names), 1)
    vW = np.pi**2/(6*M**2)
    for i, k in enumerate(names):
        r = R[k]
        st = np.clip(np.array(r['sobol_ST']), 0, None)
        Vu = r['var_lnp0']
        frac = np.append(st*Vu, vW)/(Vu + vW)
        col, ls, mk = DSTY[k]
        kw = dict(color=col, alpha=0.95 if ls == '-' else 0.45, edgecolor='white', lw=0.3)
        ax[0].bar(np.arange(6) + (i - (len(names) - 1)/2)*w, frac, w*0.95, label=LAB[k], **kw)
        a2 = np.array(r['form']['alpha'])**2
        ax[1].bar(np.arange(6) + (i - (len(names) - 1)/2)*w, a2, w*0.95, **kw)
    for a in ax[:2]:
        a.set_xticks(np.arange(6)); a.set_xticklabels(NAMES, fontsize=6.3); a.xaxis.set_minor_locator(plt.NullLocator())
        a.set_ylim(0, 1.0)
    ax[0].set_ylabel('Share of Var[ln burst pressure]')
    ax[1].set_ylabel(r'FORM importance $\alpha_j^2$')
    ax[0].legend(fontsize=5.4, loc='upper left')
    # (c) origin probability vs Weibull modulus
    Ms = json.load(open(f'{RES}/msens.json'))
    a = ax[2]
    for k in names:
        if k not in Ms:
            continue
        col, ls, mk = DSTY[k]
        mm = sorted(int(x) for x in Ms[k])
        tr = [Ms[k][str(m)]['shares']['junction'] + Ms[k][str(m)]['shares']['dome'] for m in mm]
        a.plot(mm, tr, color=col, ls=ls, marker=mk, ms=3.2, mec='white', mew=0.4)
    a.axvline(25, color=INK, lw=0.5, ls=':')
    a.set_xlabel('Weibull modulus $m$'); a.set_ylabel('Probability of transition origin')
    a.set_ylim(-0.02, 1.02); a.set_xticks([15, 20, 25, 30, 40])
    for a, l in zip(ax, 'abc'):
        P.label(a, l)
    P.save(fig, f'{OUT}/fig_sobol')


def fig_sections():
    from PIL import Image
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sys.path.insert(0, './schem_p2')
    from hero_render import CMAP
    def crop(fn, pad=10):
        im = np.asarray(Image.open(fn).convert('RGB'))
        m = (im < 245).any(axis=2); ys, xs = np.where(m)
        return im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    ims = [crop(f'./schem_p2/risk_{n}.png') for n in ('H5-ref', 'H5-e0125')]
    J = json.load(open('./schem_p2/risk_H5-ref.json'))
    fig = plt.figure(figsize=(P.W2, 0.36*P.W2))
    txt = ('reference: hoops end at the tangent line', r'rule-based: hoops end $0.125\sqrt{Rt}$ onto the dome')
    for k, im in enumerate(ims):
        ax = fig.add_axes([0.0 + 0.465*k, 0.08, 0.455, 0.86])
        ax.imshow(im, interpolation='antialiased'); ax.axis('off'); ax.set_aspect('equal')
        P.label(ax, 'ab'[k], x=0.03, y=0.97)
        ax.text(0.5, -0.02, txt[k], transform=ax.transAxes, ha='center', va='top', fontsize=7)
    cax = fig.add_axes([0.95, 0.18, 0.012, 0.66])
    cb = fig.colorbar(ScalarMappable(Normalize(J['vmin'], J['vmax']), CMAP), cax=cax)
    cb.set_label('log$_{10}$ risk density at 2.25 NWP (mm$^{-3}$)', fontsize=6.6)
    cb.ax.tick_params(labelsize=6.3)
    P.save(fig, f'{OUT}/fig_sections')


# ------------------------------------------------------------------ mass penalty vs transition risk
def fig_mass(R):
    fig, ax = plt.subplots(figsize=(P.W1, 0.8*P.W1))
    for k, r in R.items():
        v = r['vessel']
        ref = k.endswith('-ref')
        p1 = k == p1name(v)
        if not (ref or p1):
            continue
        col = P.C['orange'] if ref else P.C['blue']
        mk = 'o' if v in ('H1', 'H2', 'H5', 'H6') else ('^' if v.startswith('S') else 's')
        ax.plot(r['share_transition'], 100*(r['k_req'] - 1), marker=mk, color=col, ls='none', ms=4, mec='white', mew=0.4)
    ax.set_xlabel('Probability that burst starts at the transition')
    ax.set_ylabel('Extra laminate for target (%)')
    ax.set_xlim(-0.02, 1.02)
    hs = [Line2D([], [], marker='o', color=P.C['orange'], ls='none', label='reference'),
          Line2D([], [], marker='o', color=P.C['blue'], ls='none', label='rule-based transition'),
          Line2D([], [], marker='o', color=P.MUTED, ls='none', label=r'$h_d/R=0.6$'),
          Line2D([], [], marker='s', color=P.MUTED, ls='none', label=r'$h_d/R=0.8$'),
          Line2D([], [], marker='^', color=P.MUTED, ls='none', label=r'$h_d/R=0.65$-0.75')]
    ax.legend(handles=hs, fontsize=6.0, loc='upper left')
    P.save(fig, f'{OUT}/fig_mass')


# ------------------------------------------------------------------ direct MC verification
def fig_dmc(R, name='H7-out20'):
    d = json.load(open(f'{RES}/direct_mc_{name}.json'))
    r = R[name]
    S, sc = surrogate(name, r)
    U = np.array(d['U']); y = np.array(d['lnp0'])
    yp, sd = S.predict(U, std=True)
    NWP = r['P']; pr = 2.25*NWP
    fig, ax = plt.subplots(1, 3, figsize=(P.W2, 0.3*P.W2), gridspec_kw=dict(wspace=0.42))
    a = ax[0]
    a.plot(np.exp(y)/NWP, np.exp(yp)/NWP, 'o', ms=2.4, mfc='none', mec=P.C['blue'], mew=0.5)
    lo, hi = np.exp(min(y.min(), yp.min()))/NWP, np.exp(max(y.max(), yp.max()))/NWP
    a.plot([lo, hi], [lo, hi], color=INK, lw=0.6)
    err = np.exp(yp - y) - 1
    a.text(0.04, 0.96, f'rms error {100*np.sqrt(np.mean(err**2)):.2f}%\nmax {100*np.max(np.abs(err)):.2f}%',
           transform=a.transAxes, va='top', fontsize=6.3)
    a.set_xlabel(r'$p_0$/NWP, CalculiX'); a.set_ylabel(r'$p_0$/NWP, surrogate')
    a = ax[1]
    pc = lambda p0: -np.expm1(-(pr/p0)**M)
    c = pc(np.exp(y))
    n = np.arange(1, len(c) + 1)
    run = np.cumsum(c)/n
    rng = np.random.default_rng(1)
    bs = np.array([np.mean(rng.choice(c, len(c))) for _ in range(4000)])
    sdn = np.array([np.std(c[:k], ddof=1) if k > 1 else np.nan for k in n])/np.sqrt(n)
    a.fill_between(n, np.clip(run - 1.96*sdn, 1e-6, None), run + 1.96*sdn, color=P.C['blue'], alpha=0.15, lw=0)
    a.plot(n, run, color=P.C['blue'], lw=0.9, label='direct MC, running mean and 95% CI')
    a.axhline(r['pf_req'], color=P.C['orange'], lw=0.9, ls='--', label='surrogate (28 analyses)')
    a.axhspan(*r['pf_band'], color=P.C['orange'], alpha=0.15, lw=0)
    a.set_yscale('log'); a.set_ylim(1e-4, 3e-3); a.set_xlim(0, len(c)); a.set_xlabel('Number of CalculiX analyses'); a.set_ylabel(r'$P_f$(2.25 NWP)')
    a.legend(fontsize=5.8, loc='upper right')
    sci = lambda x: f'{x/10**np.floor(np.log10(x)):.2f}' + r'$\times10^{' + f'{int(np.floor(np.log10(x)))}' + '}$'
    a.text(0.97, 0.05, f'direct MC: {sci(np.mean(c))}\n95% CI [{sci(np.percentile(bs, 2.5))}, {sci(np.percentile(bs, 97.5))}]',
           transform=a.transAxes, ha='right', va='bottom', fontsize=5.8)
    a = ax[2]
    sh = np.array(d['share'])
    sp = S.share(U)
    a.plot(sh, sp, 'o', ms=2.4, mfc='none', mec=P.C['aqua'], mew=0.5)
    a.plot([0, 1], [0, 1], color=INK, lw=0.6)
    a.set_xlabel('Transition share, CalculiX'); a.set_ylabel('Transition share, surrogate')
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    for a, l in zip(ax, 'abc'):
        P.label(a, l)
    P.save(fig, f'{OUT}/fig_dmc')
    return dict(pf_dmc=float(np.mean(c)), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                rms=float(np.sqrt(np.mean(err**2))), max=float(np.max(np.abs(err))), pf_sur=r['pf_req'],
                band=r['pf_band'], share_dmc=float(np.mean(sh)), share_sur=r['share_transition'],
                ok=int(sum(d['ok'])), n=len(c))


# ------------------------------------------------------------------ nominal beta maps
def fig_betamap():
    """nominal reliability over the hoop-end grid (strength scatter only), with friction limits"""
    import matplotlib.colors as mcolors
    B = json.load(open(f'{RES}/betamap.json'))
    ves = [v for v in ('H1', 'H5', 'S70', 'H7', 'H3') if v in B]
    fig, ax = plt.subplots(1, len(ves), figsize=(P.W2, 0.36*P.W2), sharey=True, gridspec_kw=dict(wspace=0.06))
    ax = np.atleast_1d(ax)
    cmap = plt.get_cmap('RdYlBu')
    norm_ = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=B_T, vmax=4.1)
    for a, v in zip(ax, ves):
        rows = B[v]['rows']
        es = sorted({r['e'] for r in rows}); ls = sorted({r['l'] for r in rows})
        Z = np.full((len(ls), len(es)), np.nan)
        for r in rows:
            Z[ls.index(r['l']), es.index(r['e'])] = np.clip(beta(r['pf']), -9, 9)
        a.pcolormesh(np.arange(len(es) + 1) - 0.5, np.arange(len(ls) + 1) - 0.5, Z, cmap=cmap, norm=norm_,
                     edgecolors='white', linewidth=0.6)
        for ii in range(len(ls)):
            for jj in range(len(es)):
                z = Z[ii, jj]
                t = f'{z:.1f}' if z > -0.05 else '<0'
                a.text(jj, ii, t.replace('-', '\u2212'), ha='center', va='center', fontsize=4.8,
                       color='white' if z < 1.2 else INK)
        # friction limits (interpolated position on the column axis)
        for mu, ec in B[v]['e_crit']:
            if mu not in (0.1, 0.2, 0.3):
                continue
            xq = np.interp(ec, es, np.arange(len(es))) if ec <= es[-1] else None
            if xq is None:
                continue
            a.plot([xq, xq], [-0.5, len(ls) - 0.5], color=INK, lw=1.0 if mu == 0.2 else 0.5,
                   ls='-' if mu == 0.2 else (0, (2, 1.2)))
            if mu == 0.2:
                a.text(xq, len(ls) - 0.45, r'$e_\mathrm{crit}(0.2)$', ha='center', va='bottom', fontsize=5.4)
        a.set_xticks(range(len(es))); a.set_xticklabels([f'{e:g}' for e in es], rotation=90, fontsize=5.8)
        a.set_yticks(range(len(ls))); a.set_yticklabels([f'{l:g}' for l in ls])
        a.tick_params(which='both', length=0); a.xaxis.set_minor_locator(plt.NullLocator())
        a.yaxis.set_minor_locator(plt.NullLocator())
        a.set_xlabel(r'$e/\sqrt{Rt}$')
        a.set_title(v + (' (outer hoop block)' if B[v]['order'] else ''), fontsize=6.8, pad=11)
    ax[0].set_ylabel(r'$\ell_h/\sqrt{Rt}$')
    cax = fig.add_axes([0.915, 0.2, 0.01, 0.62])
    sm = plt.cm.ScalarMappable(norm=norm_, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, extend='both'); cb.set_label(r'$\beta$ at 2.25 NWP (nominal geometry)', fontsize=6.8)
    cb.ax.axhline(B_T, color=INK, lw=0.8)
    cb.ax.tick_params(labelsize=6.2)
    P.save(fig, f'{OUT}/fig_betamap')

# ------------------------------------------------------------------ hero figure (renders)
def fig_hero(R, pair=('H5-ref', 'H5-e0125'), vessel_title='H5 (70 MPa, $h_d/R=0.6$)'):
    from PIL import Image
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sys.path.insert(0, './schem_p2')
    from hero_render import CMAP
    ims = [np.asarray(Image.open(f'./schem_p2/hero_{n}.png').convert('RGB')) for n in pair]
    J = [json.load(open(f'./schem_p2/hero_{n}.json')) for n in pair]
    m = np.zeros(ims[0].shape[:2], bool)
    for im in ims:
        m |= (im < 245).any(axis=2)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = max(ys.min() - 10, 0), ys.max() + 10, max(xs.min() - 10, 0), xs.max() + 10
    H0, W0 = ims[0].shape[:2]
    ims = [im[y0:y1, x0:x1] for im in ims]
    h, w = ims[0].shape[:2]
    fig = plt.figure(figsize=(P.W2, 0.36*P.W2))
    axw = 0.445
    titles = ['reference transition: hoops end at the tangent line',
              r'rule-based transition: hoops end $0.125\sqrt{Rt}$ onto the dome']
    for k, (im, n, jj) in enumerate(zip(ims, pair, J)):
        ax = fig.add_axes([0.005 + 0.47*k, 0.30, axw, 0.62])
        ax.imshow(im, interpolation='antialiased'); ax.axis('off'); ax.set_aspect('equal')
        P.label(ax, 'ab'[k], x=0.02, y=0.98)
        ax.set_title(titles[k], fontsize=7, pad=2)
        ax.text(0.02, 0.02, vessel_title if k == 0 else '', transform=ax.transAxes, fontsize=6.3, color=P.INK2)
        A = jj['anchors']
        px = lambda q: ((q[0]*W0 - x0), (q[1]*H0 - y0))
        lk = dict(arrowstyle='-', lw=0.5, color=INK, shrinkA=0, shrinkB=1.5)
        ax.annotate('tangent line', xy=px(A['tl_250']), xytext=(px(A['tl_250'])[0] - 0.12*w, h*1.08),
                    ha='center', va='top', fontsize=6.3, arrowprops=lk, annotation_clip=False)
        ax.annotate('barrel', xy=px(A['barrel_250']), xytext=(px(A['barrel_250'])[0] - 0.05*w, h*1.08),
                    ha='center', va='top', fontsize=6.3, arrowprops=lk, annotation_clip=False)
        ax.annotate('dome', xy=px(A['dome_mid_300']), xytext=(px(A['dome_mid_300'])[0] + 0.12*w, h*1.08),
                    ha='center', va='top', fontsize=6.3, arrowprops=lk, annotation_clip=False)
        # origin bar
        r = R[n]
        bx = fig.add_axes([0.06 + 0.47*k, 0.085, 0.36, 0.05])
        left = 0
        for z in ('barrel', 'junction', 'dome'):
            v = r['nominal']['shares'][z]
            bx.barh(0, v, left=left, color=P.ZONE[z], edgecolor='white', lw=0.5, height=0.8)
            if v > 0.08:
                bx.text(left + v/2, 0, f'{z} {100*v:.0f}%', ha='center', va='center', fontsize=6.0, color='white')
            left += v
        bx.set_xlim(0, 1); bx.set_ylim(-0.5, 0.5); bx.axis('off')
        bx.text(0, 0.75, 'probability of burst origin (nominal)', fontsize=6.2, va='bottom', transform=bx.transData)
        bx.text(1.0, -1.2, rf"$\beta$ = {r['beta']:.2f} with scatter, $k_\mathrm{{req}}$ = {r['k_req']:.3f}",
                ha='right', va='top', fontsize=6.2, transform=bx.transData)
    cax = fig.add_axes([0.945, 0.34, 0.011, 0.58])
    sm = ScalarMappable(Normalize(*J[0]['qlim']), CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('log$_{10}$ burst-origin density (mm$^{-2}$)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6.3)
    P.save(fig, f'{OUT}/fig_hero')


if __name__ == '__main__':
    R = load()
    what = sys.argv[1:] or ['overview']
    if 'overview' in what:
        fig_overview(R)
    if 'weibull' in what:
        dm = {}
        if os.path.exists(f'{RES}/direct_mc_H7-out20.json'):
            dm['H7-out20'] = json.load(open(f'{RES}/direct_mc_H7-out20.json'))
        fig_weibull(R, dmc=dm)
    if 'friction' in what:
        fig_friction(R)
    if 'sobol' in what:
        fig_sobol(R)
    if 'mass' in what:
        fig_mass(R)
    if 'dmc' in what:
        print(fig_dmc(R))
    if 'betamap' in what:
        fig_betamap()
    if 'hero' in what:
        fig_hero(R)
    if 'sections' in what:
        fig_sections()

