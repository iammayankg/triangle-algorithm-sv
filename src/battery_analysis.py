"""Analysis and figures for the full battery results.

Reads results/full_battery.json (or its shard directory if the combined
file is absent) and produces:

  results/full_battery_summary.md   mean +/- 95% CI table (paper-ready)
  results/fig_battery_times.png     time per dataset, one panel per C
  results/fig_battery_gap.png       certified gap vs time (seed-0 traces)
  results/fig_battery_oracle.png    oracle calls, ETA vs SMO

Usage: python3 src/battery_analysis.py [--out results/full_battery.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# dataviz reference palette (light mode); color follows the solver
COL = {'ETA': '#2a78d6', 'SMO': '#eb6834', 'LIBLIN': '#1baf7a'}
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e5e4e0'
SOLVERS = ['ETA', 'SMO', 'LIBLIN']


def load(out):
    p = Path(out)
    if p.exists():
        return json.loads(p.read_text())
    shards = Path(str(out) + '.shards')
    runs = []
    for f in sorted(shards.glob('*.json')):
        runs.extend(json.loads(f.read_text()))
    if not runs:
        raise SystemExit(f'no results at {out} or {shards}')
    return runs


def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    from scipy import stats
    half = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return float(xs.mean()), float(half)


def cells(runs):
    ds = sorted({r['dataset'] for r in runs})
    Cs = sorted({r['C'] for r in runs})
    return ds, Cs


def summary_md(runs, path):
    ds, Cs = cells(runs)
    lines = ['# Full battery summary',
             '',
             '| dataset | C | solver | time (s) | accuracy | oracle calls'
             ' | non-converged |',
             '|--|--:|--|--:|--:|--:|--:|']
    for d in ds:
        for C in Cs:
            for sv in SOLVERS:
                sel = [r for r in runs if (r['dataset'], r['C'],
                                           r['solver']) == (d, C, sv)]
                if not sel:
                    continue
                tm, th = ci95([r['time'] for r in sel])
                am, ah = ci95([r['acc'] for r in sel])
                om = np.mean([r.get('oracle', float('nan'))
                              for r in sel])
                bad = sum(r['status'] != 'converged' for r in sel)
                lines.append(
                    f"| {d} | {C:g} | {sv} | {tm:.2f} ± {th:.2f} | "
                    f"{am:.4f} ± {ah:.4f} | "
                    f"{'-' if np.isnan(om) else f'{om:,.0f}'} | "
                    f"{bad}/{len(sel)} |")
    Path(path).write_text('\n'.join(lines) + '\n')
    print('wrote', path)


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc='left', fontsize=10)
    ax.grid(True, axis='x', color=GRID, lw=0.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def fig_times(runs, path):
    ds, Cs = cells(runs)
    fig, axes = plt.subplots(1, len(Cs), figsize=(3.2 * len(Cs) + 1, 4),
                             dpi=160, sharey=True)
    axes = np.atleast_1d(axes)
    h = 0.26
    ypos = np.arange(len(ds))
    for ax, C in zip(axes, Cs):
        for k, sv in enumerate(SOLVERS):
            ts, errs = [], []
            for d in ds:
                sel = [r for r in runs if (r['dataset'], r['C'],
                                           r['solver']) == (d, C, sv)]
                tm, th = ci95([r['time'] for r in sel]) if sel \
                    else (np.nan, 0)
                ts.append(max(tm, 1e-3))
                errs.append(th)
            ax.barh(ypos + (k - 1) * h, ts, xerr=errs, height=h * 0.9,
                    color=COL[sv], error_kw=dict(ecolor=INK2, lw=0.8),
                    label=sv if C == Cs[0] else None)
        ax.set_xscale('log')
        ax.set_title(f'C = {C:g}', color=INK, loc='left', fontsize=10)
        ax.grid(True, axis='x', color=GRID, lw=0.7)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
        ax.set_xlabel('time, s (log)', color=INK2)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ds, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Full battery: training time, mean ± 95% CI',
                 x=0.02, ha='left', fontsize=11, color=INK)
    fig.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3,
               loc='upper right', bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print('wrote', path)


def fig_gap(runs, path, C_show=1.0):
    traces = [(r['dataset'], r['trace']) for r in runs
              if r['solver'] == 'ETA' and r.get('trace')
              and r['C'] == C_show and r['seed'] == 0]
    if not traces:
        print('no traces found; skipping gap figure')
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    # sequential single-hue ramp over datasets (ordered by final time)
    traces.sort(key=lambda t: t[1][-1][1])
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, len(traces)))
    for (name, tr), col in zip(traces, blues):
        t = np.array([x[1] for x in tr])
        ub = np.array([x[2] for x in tr])
        lb = np.array([x[3] for x in tr])
        gap = np.maximum((ub - lb) / np.maximum(ub, 1e-30), 1e-12)
        ax.plot(t, gap, lw=1.8, color=col, label=name)
    ax.set_yscale('log')
    ax.set_xscale('log')
    _style(ax, 'time, s (log)', 'certified relative gap (log)',
           f'ETA convergence traces (C = {C_show:g}, seed 0): '
           'certified gap vs time')
    ax.axhline(1e-3, color=INK2, lw=1, ls=':')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print('wrote', path)


def fig_oracle(runs, path, C_show=1.0):
    ds, _ = cells(runs)
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=160)
    h = 0.38
    ypos = np.arange(len(ds))
    for k, sv in enumerate(('ETA', 'SMO')):
        os_ = []
        for d in ds:
            sel = [r.get('oracle', np.nan) for r in runs
                   if (r['dataset'], r['C'], r['solver'])
                   == (d, C_show, sv)]
            os_.append(max(np.nanmean(sel) if sel else np.nan, 1.0))
        ax.barh(ypos + (k - 0.5) * h, os_, height=h * 0.9,
                color=COL[sv], label=f'{sv} ({"columns" if sv == "ETA" else "rows"})')
    ax.set_xscale('log')
    ax.set_yticks(ypos)
    ax.set_yticklabels(ds, fontsize=8)
    ax.invert_yaxis()
    _style(ax, 'O(nd) oracle evaluations (log)', '',
           f'Implementation-independent cost (C = {C_show:g}): '
           'kernel/Gram evaluations')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/full_battery.json')
    args = ap.parse_args()
    runs = load(args.out)
    base = Path(args.out).parent
    summary_md(runs, base / 'full_battery_summary.md')
    fig_times(runs, base / 'fig_battery_times.png')
    fig_gap(runs, base / 'fig_battery_gap.png')
    fig_oracle(runs, base / 'fig_battery_oracle.png')


if __name__ == '__main__':
    main()
