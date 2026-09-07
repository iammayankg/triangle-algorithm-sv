"""Figure for the journal section "When does aggregation pay?":
along-the-run profiles from block_diagnostics.py records (k = 16):
drop fraction, column misses per call, eta estimate and support size per
decile of the run, one line per cell.  Also writes the per-cell fit table
(eta, a, b, c_col, k*) as LaTeX rows.

Usage:
  python3 src/block_diag_fig.py --inp results/block_diag2.json \
      --fig paper/figs/fig_diag.pdf --table paper/aor/tab_diag_rows.tex
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CELLS = [('gisette-l2', 'gisette $L_2$'), ('ijcnn1-kl2', 'ijcnn1 KL2'),
         ('w8a-kl2', 'w8a KL2'), ('a9a-kl2', 'a9a KL2'), ('covtype-kl2', 'covtype KL2')]


def deciles(rec, nd=10):
    its = np.array([x['it'] for x in rec]); T = its.max()
    out = []
    for d in range(nd):
        lo, hi = d * T / nd, (d + 1) * T / nd
        sub = [x for x in rec if lo < x['it'] <= hi]
        unc = [x for x in sub if not x['cap1'] and x['kp'] > 1 and x['S'] > 0 and x['d1'] > 0]
        out.append(dict(
            drop=np.mean([x['kind'] == 'drop' for x in sub]) if sub else np.nan,
            miss=np.mean([x['misses'] for x in sub]) if sub else np.nan,
            eta=np.median([(x['D'] / x['S'] - 1) / (x['kp'] - 1) for x in unc]) if unc else np.nan,
            supp=np.median([x['support'] for x in sub]) if sub else np.nan))
    return out


def fit(rows):
    ks = np.array([r['k'] for r in rows], float)
    y = np.array([(r['time'] - r['t_split']['t_cols']) / r['iters'] * 1e3 for r in rows])
    A = np.vstack([np.ones_like(ks), ks]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    ccol = np.mean([r['t_split']['t_cols'] / r['oracle'] * 1e3 for r in rows])
    etas = [(r['kappa']['p50'] - 1) / (r['kp_mean'] - 1) for r in rows if r['kp_mean'] > 1]
    eta = float(np.median(etas))
    return dict(a=a, b=b, ccol=ccol, eta=eta, etas=etas,
                kstar=float(np.sqrt(a * (1 - eta) / (b * eta))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='results/block_diag2.json')
    ap.add_argument('--fig', default='paper/figs/fig_diag.pdf')
    ap.add_argument('--table', default='paper/aor/tab_diag_rows.tex')
    ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()
    R = json.load(open(a.inp))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6))
    x = np.arange(1, 11)
    for cell, label in CELLS:
        run = [r for r in R if r['cell'] == cell and r['k'] == a.k and 'records' in r]
        if not run:
            continue
        D = deciles(run[0]['records'])
        axes[0].plot(x, [100 * d['drop'] for d in D], marker='o', ms=3, label=label)
        axes[1].plot(x, [d['miss'] for d in D], marker='o', ms=3, label=label)
        axes[2].plot(x, [d['eta'] for d in D], marker='o', ms=3, label=label)
    axes[0].set_ylabel('drop steps (% of block calls)')
    axes[1].set_ylabel('column misses per block call')
    axes[1].set_yscale('symlog', linthresh=0.1)
    axes[2].set_ylabel(r'$\hat\eta$ (median over decile)')
    axes[2].set_ylim(0, 0.35)
    for ax in axes:
        ax.set_xlabel('decile of the run (iterations)')
        ax.set_xticks(x)
        ax.grid(alpha=.3)
    axes[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(a.fig)
    print('wrote', a.fig)

    lines = []
    for cell, label in CELLS:
        rows = sorted([r for r in R if r['cell'] == cell], key=lambda r: r['k'])
        f = fit(rows)
        r16 = [r for r in rows if r['k'] == 16][0]
        times = ' / '.join(f"{r['time']:.1f}" for r in rows)
        best = min(rows, key=lambda r: r['time'])
        g1 = [r for r in rows if r['k'] == 1][0]
        lines.append(f"{label} & {f['eta']:.2f} & {f['a']:.1f} & {f['b']:.3f} & {f['ccol']:.2f} & "
                     f"{f['kstar']:.0f} & {times} & {100 * r16['t_split']['t_cols'] / r16['time']:.0f}\\% & "
                     f"{g1['time'] / best['time']:.1f}$\\times$ ({best['k']}) \\\\")
    open(a.table, 'w').write('\n'.join(lines) + '\n')
    print('wrote', a.table); print('\n'.join(lines))


if __name__ == '__main__':
    main()
