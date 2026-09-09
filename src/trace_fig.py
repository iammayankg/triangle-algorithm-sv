"""Figure 1 of the journal draft: certified relative gap of ETA versus
iteration and wall-clock on the uncapped seed-0 runs (a9a and w8a, L2,
C=0.1) from the shards written by

  python3 src/full_battery.py --time-cap 0 --solvers ETA --seeds 1 \
      --datasets a9a w8a --Cs 0.1 --out results/full_battery_uncapped.json

Usage: python3 src/trace_fig.py [--shards results/full_battery_uncapped.shards]
"""
from __future__ import annotations

import argparse
import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', default='results/full_battery_uncapped.shards')
    ap.add_argument('--C', type=float, default=0.1)
    ap.add_argument('--out', nargs='*',
                    default=['paper/figs/fig_realdata_trace.pdf'])
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for p in sorted(glob.glob(f'{a.shards}/*_0.json')):
        for r in json.load(open(p)):
            if r['solver'] != 'ETA' or r['C'] != a.C or not r.get('trace'):
                continue
            it = [x[0] for x in r['trace']]
            t = [x[1] for x in r['trace']]
            gap = [max((ub - lb) / ub, 1e-12) for _, _, ub, lb in r['trace']]
            lab = f"{r['dataset']} ({r['status']}, {r['time']:.0f} s)"
            axes[0].semilogy(it, gap, label=lab)
            axes[1].semilogy(t, gap, label=lab)
    axes[0].set_xlabel('iteration')
    axes[1].set_xlabel('wall-clock (s)')
    axes[0].set_ylabel('certified relative gap')
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    for o in a.out:
        fig.savefig(o)
        print('wrote', o)


if __name__ == '__main__':
    main()
