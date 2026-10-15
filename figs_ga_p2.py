"""Graphical abstract, Paper 2 (Elsevier: >= 1328 x 531 px, readable at 5 x 13 cm)."""
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, '.'); sys.path.insert(0, './schem_p2')
import pubstyle as P
P.use()
from hero_render import CMAP
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
R = json.load(open('./rel_results/full.json'))


def crop(fn, box):
    im = np.asarray(Image.open(fn).convert('RGB'))
    H, W = im.shape[:2]
    return im[int(box[1]*H):int(box[3]*H), int(box[0]*W):int(box[2]*W)]


fig = plt.figure(figsize=(13/2.54*1.6, 5/2.54*1.6))
fig.text(0.012, 0.95, 'Where do Type IV hydrogen vessels burst?', fontsize=11, fontweight='bold', va='top')
box = (0.30, 0.12, 0.88, 1.0)
for k, (n, t) in enumerate((('H5-ref', 'hoops end at the tangent line'), ('H5-e0125', 'rule-based transition'))):
    ax = fig.add_axes([0.005 + 0.235*k, 0.13, 0.23, 0.66])
    ax.imshow(crop(f'./schem_p2/hero_{n}.png', box)); ax.axis('off')
    sh = R[n]['nominal']['shares']
    ax.set_title(t, fontsize=7.2, pad=2)
    ax.text(0.5, -0.03, f"burst starts at the transition: {100*(sh['junction'] + sh['dome']):.0f}%",
            transform=ax.transAxes, ha='center', va='top', fontsize=7, color=P.C['orange'] if k == 0 else P.C['blue'],
            fontweight='bold')
cax = fig.add_axes([0.475, 0.2, 0.006, 0.5])
cb = fig.colorbar(ScalarMappable(Normalize(-9, -4), CMAP), cax=cax); cb.set_ticks([])
cb.set_label('burst-origin\ndensity', fontsize=6)
# right: extra laminate for Pf = 1e-4
ax = fig.add_axes([0.575, 0.2, 0.41, 0.58])
ves = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'S65', 'S70', 'S75']
COMPACT = ('H3', 'H4', 'H7', 'H8', 'S65', 'S70', 'S75')
x = np.arange(len(ves))
k0 = [100*(R[f'{v}-ref']['k_req'] - 1) for v in ves]
k1 = [100*(R[(f'{v}-out20' if v in COMPACT else f'{v}-e0125')]['k_req'] - 1) for v in ves]
ax.bar(x - 0.2, np.minimum(k0, 40), 0.38, color=P.C['orange'], label='reference transition')
ax.bar(x + 0.2, k1, 0.38, color=P.C['blue'], label='rule-based transition')
for i, v in enumerate(k0):
    if v > 40:
        ax.text(i - 0.2, 39, f'{v:.0f}', rotation=90, ha='center', va='top', fontsize=6, color='white')
ax.axhline(3.0, color=P.INK, lw=0.7, ls=':', label='floor from manufacturing scatter')
ax.set_xticks(x); ax.set_xticklabels(ves, fontsize=6.5); ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_ylim(0, 40); ax.set_ylabel('extra laminate for\n$P_f(2.25\\,$NWP$)=10^{-4}$ (%)', fontsize=7)
ax.legend(fontsize=6.3, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=0.8, handlelength=1.4)
fig.text(0.575, 0.95, 'weakest-link burst model + manufacturing and winding-friction scatter', fontsize=7.4, va='top')
P.save(fig, './figs_p2full/graphical_abstract')
im = Image.open('./figs_p2full/graphical_abstract.png'); print(im.size)
