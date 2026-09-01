"""Generate replication figures from results/*.json."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# dataviz reference palette (light mode): color follows the solver
C_TA = '#2a78d6'      # blue   - Triangle Algorithm
C_SMO = '#eb6834'     # orange - SMO (uncached, MATLAB-like baseline)
C_SMOC = '#1baf7a'    # aqua   - SMO (optimized, LRU row cache)
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e5e4e0'

PAPER_T3 = {  # dim: (ta_time, smo_time) from the paper's Table 3
    3: (0.64, 3.13), 10: (1.11, 6.48), 50: (1.42, 29.52), 100: (1.77, 44.03),
    300: (2.37, 85.29), 500: (3.20, 90.69), 1000: (4.74, 84.67),
    2000: (5.57, 117.08), 5000: (15.49, 138.74), 10000: (36.39, 165.99),
}

RES = Path(__file__).resolve().parents[1] / 'results'


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc='left', fontsize=11, pad=10)
    ax.grid(True, which='major', color=GRID, linewidth=0.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def fig_time_vs_dim():
    rows = json.loads((RES / 'table3.json').read_text())
    dims = [r['dim'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=160)
    ax.plot(dims, [r['ta_time_s'] for r in rows], '-o', color=C_TA, lw=2,
            ms=4, label='Triangle Algorithm (ours)')
    ax.plot(dims, [r['smo_nc_time_s'] for r in rows], '-o', color=C_SMO,
            lw=2, ms=4, label='SMO, uncached (ours)')
    ax.plot(dims, [r['smo_time_s'] for r in rows], '-o', color=C_SMOC,
            lw=2, ms=4, label='SMO, row cache (ours)')
    pd = sorted(PAPER_T3)
    ax.plot(pd, [PAPER_T3[d][0] for d in pd], '--s', color=C_TA, lw=1.4,
            ms=4, alpha=0.55, label='Triangle Algorithm (paper, MATLAB)')
    ax.plot(pd, [PAPER_T3[d][1] for d in pd], '--s', color=C_SMO, lw=1.4,
            ms=4, alpha=0.55, label='SMO (paper, MATLAB)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    _style(ax, 'dimension d (log)', 'runtime, seconds (log)',
           'Convex-hull distance: runtime vs dimension  '
           '(n = 5000 per set, eps = 1e-3)')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(RES / 'fig_time_vs_dim.png', facecolor='white')
    plt.close(fig)


def fig_time_vs_k():
    rows = json.loads((RES / 'table4.json').read_text())
    ks = [r['k'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    ax.plot(ks, [r['ta_time_s'] for r in rows], '-o', color=C_TA, lw=2,
            ms=4, label='Triangle Algorithm')
    ax.plot(ks, [r['smo_nc_time_s'] for r in rows], '-o', color=C_SMO, lw=2,
            ms=4, label='SMO, uncached')
    ax.plot(ks, [r['smo_time_s'] for r in rows], '-o', color=C_SMOC, lw=2,
            ms=4, label='SMO, row cache')
    ax.set_ylim(bottom=0)
    _style(ax, 'separation parameter k', 'runtime, seconds',
           'Distance sensitivity at d = 1000: TA speeds up as hulls separate')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(RES / 'fig_time_vs_k.png', facecolor='white')
    plt.close(fig)


def fig_agreement():
    rows = json.loads((RES / 'table3.json').read_text())
    dims = [r['dim'] for r in rows]
    rel = [abs(r['ta_dist'] - r['smo_dist']) / r['smo_dist'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    ax.plot(dims, rel, '-o', color=C_TA, lw=2, ms=4)
    ax.axhline(1e-3, color=INK2, lw=1, ls=':')
    ax.annotate('eps = 1e-3 tolerance', xy=(dims[1], 1e-3),
                xytext=(0, 6), textcoords='offset points',
                color=INK2, fontsize=8.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    _style(ax, 'dimension d (log)', 'relative difference (log)',
           'TA vs SMO distance agreement stays inside the tolerance')
    fig.tight_layout()
    fig.savefig(RES / 'fig_agreement.png', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    fig_time_vs_dim()
    fig_time_vs_k()
    fig_agreement()
    print('figures written to', RES)
