"""Paper 2 workflow figure (vector layout in mm, Elsevier double column)."""
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, './rel')
import pubstyle as P
P.use()
import p1_study as PS
from rel import runner as RN, weakest as W, study_full as SF, scatter as SC
import figs_full as F

FW, FH = 190.0, 138.0
STRIP = 8.0
COLW, GAPX = 53.0, 9.25
X0 = [STRIP + 2.0 + i*(COLW + GAPX) for i in range(3)]
ROWH, GAPY = 60.0, 11.0
TOP = 136.0
Y0 = [TOP - ROWH, TOP - 2*ROWH - GAPY]
HEAD = 7.0


def mm_axes(fig, x, y, w, h):
    return fig.add_axes([x/FW, y/FH, w/FW, h/FH])


def box(ax, x, y, w, h, title, text, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.0', fc='white', ec=col, lw=0.9))
    ax.add_patch(FancyBboxPatch((x, y + h - HEAD), w, HEAD, boxstyle='round,pad=0,rounding_size=2.0', fc=col, ec=col, lw=0.9))
    ax.add_patch(Rectangle((x, y + h - HEAD), w, 3.0, fc=col, ec=col, lw=0))
    ax.text(x + 2.8, y + h - HEAD/2, title, color='white', fontsize=7.3, fontweight='bold', va='center')
    ax.text(x + 2.8, y + h - HEAD - 2.0, text, fontsize=6.2, va='top', ha='left', linespacing=1.3, color=P.INK)


def arrow(ax, p0, p1, label=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=10, lw=1.2, color=P.INK2, shrinkA=1.2, shrinkB=1.2))
    if label:
        ax.text(0.5*(p0[0] + p1[0]), 0.5*(p0[1] + p1[1]) + 1.4, label, fontsize=5.6, ha='center', va='bottom', color=P.INK2)


def mini(a):
    a.tick_params(labelsize=5.4, length=1.8, pad=1.2)
    for sp in a.spines.values():
        sp.set_linewidth(0.5)
    a.xaxis.label.set_size(5.8); a.yaxis.label.set_size(5.8)


def picture(fig, fn, x, y, w, h, box_=None):
    im = np.asarray(Image.open(fn).convert('RGB'))
    if box_:
        H, Wd = im.shape[:2]
        im = im[int(box_[1]*H):int(box_[3]*H), int(box_[0]*Wd):int(box_[2]*Wd)]
    m = (im < 245).any(axis=2); ys, xs = np.where(m)
    im = im[ys.min():ys.max(), xs.min():xs.max()]
    ih, iw = im.shape[:2]
    if w/h > iw/ih:
        w2 = h*iw/ih; x += (w - w2)/2; w = w2
    else:
        h2 = w*ih/iw; y += (h - h2)/2; h = h2
    a = mm_axes(fig, x, y, w, h); a.imshow(im, interpolation='antialiased'); a.axis('off')


def main():
    RR = json.load(open('./rel_results/full.json'))
    NAMES = ('H5-ref', 'H5-e0125', 'H7-out15', 'H7-out20')
    SU = {n: (F.surrogate(n, RR[n])[0], RR[n]) for n in NAMES}
    STY = dict(F.DSTY); STY['H7-out15'] = (P.C['orange'], '--', 'D')
    fig = plt.figure(figsize=(FW/25.4, FH/25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
    cA, cB, cC = '#4d4d4d', P.C['blue'], P.C['orange']
    for (y, t, c) in ((Y0[0], 'Per realisation (CalculiX)', '#4d4d4d'), (Y0[1], 'On the surrogate (statistics)', P.C['orange'])):
        ax.add_patch(Rectangle((1.0, y), 4.2, ROWH, fc=c, ec='none', alpha=0.10))
        ax.text(3.1, y + ROWH/2, t, rotation=90, ha='center', va='center', fontsize=6.6, fontweight='bold', color=c)
    tx = [('1  Sources of scatter', 'Weibull fibre strength (analytical);\nthickness, angle, $E_1$, friction $\\mu$', cA),
          ('2  CalculiX realisation', 'dome-resolved shell, rule-based transition;\nfriction-limited hoop end', cA),
          ('3  Weakest-link integral', '$P_f(p|\\mathbf{u})=1-\\exp[-(p/p_0(\\mathbf{u}))^m]$;\nshare of risk = origin probability', cB),
          ('4  Surrogate + active learning', 'Gaussian process on $\\ln p_0$; points added\nwhere $\\sigma_{GP}\\,\\partial P_f/\\partial\\ln p_0$ is largest', cB),
          ('5  Reliability measures', '$P_f$, $\\beta$ (MC and FORM), origin\nprobability, Sobol indices', cC),
          ('6  Design decisions', 'wall factor for the target $P_f$;\ntransition and programmed hoop end', cC)]
    k = 0
    for row in range(2):
        for col in range(3):
            t, body, c = tx[k]
            box(ax, X0[col], Y0[row], COLW, ROWH, t, body, c); k += 1

    def area(col, row):
        return X0[col] + 2.5, Y0[row] + 2.5, COLW - 5, ROWH - HEAD - 14

    # 1: friction distribution and the friction-limited hoop end (H7, programmed at e_crit(0.20))
    x, y, w, h = area(0, 0)
    a = mm_axes(fig, x + 9, y + 6.5, w - 20, h - 7)
    D = SF.design(SF.designs()['H7-out20'])
    sc = SF.Scatter(D['c']['fam'][1])
    fm = SF.phys_fmap(D, sc)
    mu = np.linspace(0.06, 0.45, 300)
    from scipy.stats import lognorm
    a.fill_between(mu, lognorm.pdf(mu, sc.s_mu, scale=0.2), color=P.LIGHT, lw=0)
    a.plot(mu, lognorm.pdf(mu, sc.s_mu, scale=0.2), color=P.INK2, lw=0.8)
    a.set_xlabel('slippage coefficient $\\mu$', labelpad=1); a.set_ylabel('density', labelpad=1)
    a2 = a.twinx()
    U = np.zeros((len(mu), 5)); U[:, 3] = np.log(mu/0.2)/sc.s_mu
    a2.plot(mu, fm(U)[:, 3], color=P.C['orange'], lw=1.1)
    a2.set_ylabel('hoop end $e/\\sqrt{Rt}$', labelpad=1, color=P.C['orange'])
    mini(a); mini(a2); a2.tick_params(colors=P.C['orange'])
    # 2: render
    x, y, w, h = area(1, 0)
    picture(fig, './schem_p2/risk_H5-ref.png', x - 1, y - 1, w + 2, h + 6, box_=(0.0, 0.05, 0.85, 0.62))
    # 3: cumulative risk along the meridian
    x, y, w, h = area(2, 0)
    a = mm_axes(fig, x + 9, y + 6.5, w - 11, h - 7)
    for name, ls in (('H5-ref', '--'), ('H5-e0125', '-')):
        c = SF.nominal(SF.designs()[name], name); d = RN.data(c); wb = SF.weibull('H5')
        t = wb.risk_terms(d); sel = d['side'] > 0
        wj = PS.wj(c)
        o = np.argsort(d['s'][sel]); s = d['s'][sel][o]/wj; cum = np.cumsum(t[sel][o])/t[sel].sum()
        a.plot(s, cum, color=P.C['blue'], ls=ls, lw=1.0, label={'H5-ref': 'H5 reference', 'H5-e0125': 'H5 rule-based'}[name])
    a.axvspan(-1, 1, color=P.ZONE['junction'], alpha=0.12, lw=0)
    a.set_xlim(-12, 6); a.set_ylim(0, 1.02)
    a.set_xlabel('$s/\\sqrt{Rt}$ from the tangent line', labelpad=1); a.set_ylabel('cumulative risk', labelpad=1)
    a.legend(fontsize=5.0, loc='upper left', handlelength=1.6)
    mini(a)
    # 4: out-of-sample parity: 300 direct Monte Carlo analyses (H7, Paper-1 transition)
    x, y, w, h = area(0, 1)
    a = mm_axes(fig, x + 9, y + 6.5, w - 11, h - 7)
    dm = json.load(open('./rel_results/direct_mc_H7-out20.json'))
    S7 = SU['H7-out20'][0]
    yt_ = np.exp(np.array(dm['lnp0']))/70; yp_ = np.exp(S7.predict(np.array(dm['U'])))/70
    a.plot(yt_, yp_, 'o', color=P.C['orange'], ms=1.8, mew=0, alpha=0.8)
    lo, hi = a.get_xlim()
    a.plot([lo, hi], [lo, hi], color=P.INK, lw=0.5, ls='--')
    a.set_xlabel('$p_0$/NWP, CalculiX', labelpad=1); a.set_ylabel('$p_0$/NWP, surrogate', labelpad=1)
    a.text(0.04, 0.96, '300 unseen\nrealisations', transform=a.transAxes, va='top', fontsize=5.2, color=P.INK2)
    mini(a)
    # 5: burst probability on Weibull paper
    x, y, w, h = area(1, 1)
    a = mm_axes(fig, x + 9, y + 6.5, w - 11, h - 7)
    U = np.random.default_rng(5).standard_normal((30000, 5))
    p = np.linspace(1.8, 3.2, 120)*70
    for name, (S, v) in SU.items():
        col, ls, mk = STY[name]
        lp = S.predict(U)
        Fp = np.array([np.mean(-np.expm1(-(pp/np.exp(lp))**25)) for pp in p])
        ok = (Fp > 1e-6) & (Fp < 0.99)
        a.plot(p[ok]/70, np.log(-np.log1p(-Fp[ok])), color=col, ls=ls, lw=0.9)
    a.axvline(2.25, color=P.INK, lw=0.5, ls=':')
    yt = [1e-6, 1e-4, 1e-2, 0.5]
    a.set_yticks([np.log(-np.log1p(-q)) for q in yt]); a.set_yticklabels(['$10^{-6}$', '$10^{-4}$', '$10^{-2}$', '0.5'])
    a.axhline(np.log(-np.log1p(-1e-4)), color=P.INK, lw=0.5, ls=':')
    a.yaxis.set_minor_locator(plt.NullLocator())
    a.set_xlabel('burst pressure / NWP', labelpad=1); a.set_ylabel('$P$(burst)', labelpad=1)
    mini(a)
    # 6: wall factor for the target
    x, y, w, h = area(2, 1)
    a = mm_axes(fig, x + 9, y + 6.5, w - 11, h - 7)
    names = list(SU)
    kk = [100*(SU[n][1]['k_req'] - 1) for n in names]
    for i, n in enumerate(names):
        col, ls, mk = STY[n]
        a.bar(i, kk[i], 0.62, color=col, alpha=1.0 if ls == '-' else 0.5, edgecolor='white', lw=0.3)
    a.set_xticks(range(len(names))); a.set_xticklabels(['H5\nref.', 'H5\nrule-based', 'H7\n$\\mu_d$=0.15', 'H7\n$\\mu_d$=0.20'])
    a.xaxis.set_minor_locator(plt.NullLocator())
    a.set_ylabel('extra wall for $P_f=10^{-4}$ (%)', labelpad=1)
    mini(a); a.tick_params(axis='x', labelsize=5.0)
    # arrows
    for row in range(2):
        yc = Y0[row] + ROWH/2
        arrow(ax, (X0[0] + COLW, yc), (X0[1], yc))
        arrow(ax, (X0[1] + COLW, yc), (X0[2], yc))
    for col in (0, 2):
        xc = X0[col] + COLW/2
    xm = X0[2] + COLW/2
    hw = 1.2/72*25.4/2
    ymid = 0.5*(Y0[0] + Y0[1] + ROWH)
    ax.plot([xm, xm, X0[0] + COLW/2 + hw], [Y0[0] - 1.2, ymid, ymid], color=P.INK2, lw=1.2, solid_joinstyle='miter',
            solid_capstyle='butt')
    ax.add_patch(FancyArrowPatch((X0[0] + COLW/2, ymid + hw), (X0[0] + COLW/2, Y0[1] + ROWH), arrowstyle='-|>',
                                 mutation_scale=10, lw=1.2, color=P.INK2, shrinkA=0, shrinkB=1.2, capstyle='butt'))
    ax.text(xm + 1.3, 0.5*(Y0[0] + ymid), '$p_0(\\mathbf{u})$, origin shares', fontsize=5.6, color=P.INK2, va='center')
    P.save(fig, './figs_p2full/fig_workflow')


if __name__ == '__main__':
    main()
