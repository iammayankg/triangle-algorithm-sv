"""Consolidated benchmark figure (paper Fig. 3, blog Fig. 8) from the
JSON written by src/final_benchmark.py.

Per problem class: the optimised ETA, the paper-style / MDM
configuration, and the best standard solver. Non-converged bars are
hatched; the nu-SVM row has no paper-style bar because the paper-style
(toward-step) algorithm has no reduced-hull variant, and the slot says
so.

Usage: python3 src/final_benchmark_fig.py [--json results/final_benchmark.json]
           [--out paper/figs/fig_final.png results/fig_final.png blog/figs/08_regimes.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
COL = {'opt': '#2a78d6', 'paper': '#1baf7a', 'std': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e5e4e0'

# bench key -> (row label, optimised solver, paper-style solver or None)
ROWS = [
    ('A hard d=1000', 'hard margin, d=1000 (eps 1e-3)', 'ETA block+shrink', 'ETA paper-style'),
    ('B hard d=10000', 'hard margin, d=10000 (eps 1e-3)', 'ETA block+shrink', 'ETA paper-style'),
    ('C hard d=1000 tight', 'hard margin, d=1000 (eps 1e-5)', 'ETA block+shrink', 'ETA paper-style'),
    ('D soft d=1000', 'L2 soft margin, d=1000', 'ETA block(32)', 'ETA mdm'),
    ('E mnist-oe', 'L2 soft, MNIST odd-vs-even', 'ETA block(8)', 'ETA mdm'),
    ('F rbf d=1000', 'RBF hard margin, d=1000', 'K-ETA block', 'K-ETA pairwise'),
    ('G nu=0.3 d=1000', 'nu-SVM 0.3, d=1000', 'RCH-TA', None),
]
STD_LABEL = {'LIBSVM (SVC linear)': 'LIBSVM', 'LIBSVM (SVC rbf)': 'LIBSVM rbf',
             'LIBLINEAR sq-hinge': 'LIBLINEAR', 'NuSVC (LIBSVM)': 'NuSVC',
             'SMO (uncached)': 'SMO'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=str(ROOT / 'results/final_benchmark.json'))
    ap.add_argument('--out', nargs='*', default=[
        str(ROOT / 'paper/figs/fig_final.png'), str(ROOT / 'results/fig_final.png'),
        str(ROOT / 'blog/figs/08_regimes.png')])
    args = ap.parse_args()
    raw = json.loads(Path(args.json).read_text())
    # average over trials: one row per (bench, solver) with mean time, a
    # 95% CI half-width, and status 'converged' only if every trial converged
    groups = {}
    for r in raw:
        groups.setdefault((r['bench'], r['solver']), []).append(r)
    runs = []
    for (b, sv), rs in groups.items():
        t = np.array([r['time'] for r in rs])
        half = 0.0
        if len(t) > 1:
            from scipy import stats
            half = float(stats.t.ppf(0.975, len(t) - 1) * t.std(ddof=1) / np.sqrt(len(t)))
        runs.append(dict(bench=b, solver=sv, time=float(t.mean()), ci=half,
                         n=len(t), status='converged' if all(
                             r['status'] == 'converged' for r in rs) else 'maxiter'))
    ntrials = max(r['n'] for r in runs)

    def get(bench, solver):
        return next(r for r in runs if r['bench'] == bench and r['solver'] == solver)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    h = 0.25
    labels = []
    for y, (bench, label, opt, paper) in enumerate(ROWS):
        labels.append(label)
        rows = [r for r in runs if r['bench'] == bench]
        std = min((r for r in rows if r['solver'] not in (opt, paper)),
                  key=lambda r: r['time'])
        series = [(get(bench, opt), COL['opt'], None),
                  (get(bench, paper) if paper else None, COL['paper'], None),
                  (std, COL['std'], STD_LABEL.get(std['solver'], std['solver']))]
        for k, (r, col, tag) in enumerate(series):
            yy = y + (k - 1) * h
            if r is None:
                ax.text(0.105, yy, 'no paper-style variant (reduced hulls need capped transfers)',
                        va='center', ha='left', fontsize=7, color=INK2, style='italic')
                continue
            bad = r['status'] != 'converged'
            ax.barh(yy, r['time'], height=h * 0.9, color=col, alpha=0.45 if bad else 1.0,
                    hatch='///' if bad else None, edgecolor='white', linewidth=0.8)
            if r['ci'] > 0:
                ax.errorbar(r['time'], yy, xerr=r['ci'], fmt='none', ecolor=INK2,
                            elinewidth=0.8, capsize=2)
            note = tag or ''
            if bad:
                note = (note + '  ' if note else '') + 'did not converge'
            if note:
                ax.text((r['time'] + r['ci']) * 1.12, yy, note, va='center', ha='left', fontsize=8,
                        color=INK2, style='italic' if bad else 'normal')
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(0.1, 600)
    ax.set_xlabel('time to certified solution, seconds (log)'
                  + (f'; mean of {ntrials} trials, bars = 95% CI' if ntrials > 1 else ''),
                  color=INK2)
    ax.set_title('Consolidated benchmark: time to solution across all problem classes',
                 loc='left', fontsize=11, color=INK, pad=28)
    ax.grid(True, axis='x', color=GRID, lw=0.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(handles=[Patch(color=COL['opt'], label='ETA optimised'),
                       Patch(color=COL['paper'], label='ETA paper-style / MDM'),
                       Patch(color=COL['std'], label='best standard solver')],
              frameon=False, fontsize=9, labelcolor=INK2, ncol=3,
              loc='lower left', bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    for out in args.out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor='white')
        print('wrote', out)


if __name__ == '__main__':
    main()
