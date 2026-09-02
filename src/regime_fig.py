"""Figure for the regime battery: ETA vs its same-objective baseline per
cell (Table 1 of the paper), from results/regime_battery.json.

Writes results/fig_regime.png (and blog/figs/10_regime_battery.png if
that directory exists). Usage: python3 src/regime_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COL = {'ETA': '#2a78d6', 'base': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e5e4e0'
CAP = 600.0

# (cell, dataset, ETA solver, baseline solver, baseline label)
ROWS = [
    ('LIN', 'gisette', 'ETA', 'LIBSVM', 'LIBSVM'),
    ('KHM', 'gisette', 'K-ETA', 'LIBSVM', 'LIBSVM'),
    ('KHM', 'ijcnn1', 'K-ETA', 'LIBSVM', 'LIBSVM'),
    ('KHM', 'a9a', 'K-ETA', 'LIBSVM', 'LIBSVM'),
    ('KHM', 'covtype', 'K-ETA', 'LIBSVM', 'LIBSVM'),
    ('KL2', 'gisette', 'K-ETA', 'K-SMO', 'kernel SMO'),
    ('KL2', 'w8a', 'K-ETA', 'K-SMO', 'kernel SMO'),
    ('KL2', 'a9a', 'K-ETA', 'K-SMO', 'kernel SMO'),
    ('KL2', 'covtype', 'K-ETA', 'K-SMO', 'kernel SMO'),
    ('KL2', 'ijcnn1', 'K-ETA', 'K-SMO', 'kernel SMO'),
]
CELL_NAME = {'LIN': 'linear hard margin', 'KHM': 'RBF hard margin',
             'KL2': 'RBF L2 margin, C = 1'}


def main():
    runs = json.loads((ROOT / 'results/regime_battery.json').read_text())

    def sel(cell, ds, solver):
        return [r for r in runs
                if (r['cell'], r['dataset'], r['solver']) == (cell, ds, solver)]

    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=160)
    h = 0.36
    y = 0
    ticks, labels, seps = [], [], []
    last_cell = None
    for cell, ds, e, b, blab in ROWS:
        if last_cell is not None and cell != last_cell:
            seps.append(y - 0.5)
        last_cell = cell
        for k, (solver, col, name) in enumerate(
                ((e, COL['ETA'], 'ETA'), (b, COL['base'], blab))):
            rs = sel(cell, ds, solver)
            t = np.array([r['time'] for r in rs])
            bad = sum(r['status'] != 'converged' for r in rs)
            yy = y + (k - 0.5) * h
            ax.barh(yy, t.mean(), height=h * 0.88, color=col,
                    edgecolor='white', linewidth=1)
            ci = t.std(ddof=1) / np.sqrt(len(t)) * 2.78 if len(t) > 1 else 0.0
            if ci > 0:
                ax.errorbar(t.mean(), yy, xerr=ci, fmt='none', ecolor=INK2,
                            elinewidth=0.8, capsize=2)
            note = ''
            if bad == len(rs) and t.mean() >= CAP - 5:
                note = 'timeout, all seeds'
            elif bad:
                note = f'{bad}/{len(rs)} not certified'
            if rs and rs[0]['status'] == 'maxiter':
                note = 'iteration cap'
            if note:
                ax.text((t.mean() + ci) * 1.12, yy, note, va='center',
                        ha='left', fontsize=7, color=INK2)
        ticks.append(y)
        labels.append(f'{ds}  ({CELL_NAME[cell]})')
        y += 1
    for s in seps:
        ax.axhline(s, color=GRID, lw=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(0.7, 3000)
    ax.axvline(CAP, color=INK2, lw=0.8, ls=':')
    ax.text(CAP, -0.9, '600 s cap', ha='center', fontsize=7, color=INK2)
    ax.set_xlabel('time to certified solution, s (log); mean of 5 seeds, bars = 95% CI',
                  color=INK2)
    ax.set_title('Regime battery: ETA vs a baseline solving the same objective',
                 loc='left', fontsize=10, color=INK)
    ax.grid(True, axis='x', color=GRID, lw=0.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL['ETA'], label='ETA (this paper)'),
                       Patch(color=COL['base'], label='baseline: LIBSVM (hard margins) / kernel SMO (L2)')],
              frameon=False, fontsize=8, labelcolor=INK2, loc='lower right')
    fig.tight_layout()
    out = ROOT / 'results/fig_regime.png'
    fig.savefig(out, facecolor='white')
    print('wrote', out)
    blog = ROOT / 'blog/figs'
    if blog.is_dir():
        fig.savefig(blog / '10_regime_battery.png', facecolor='white')
        print('wrote', blog / '10_regime_battery.png')


if __name__ == '__main__':
    main()
