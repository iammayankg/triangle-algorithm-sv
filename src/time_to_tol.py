"""Time to 1% of the optimum (review question 6).

NOTE: the ETA columns measure DUAL progress (certified lower bound
2/UB_t^2 vs the best known primal) while LIBLINEAR's column measures
PRIMAL progress; this is not a matched primal comparison.  The primal of
ETA's returned (w, b) is not in the traces and would need a rerun.

ETA: from the seed-0 (time, UB, LB) traces in the L2-battery shards.
The dual value 2/UB_t^2 is a lower bound on the primal that increases
to the optimum P*, so the relative suboptimality of the iterate's
certified value is s_t = 1 - (2/UB_t^2)/P*.  P* is taken as the best
primal known for the cell: the smaller of LIBLINEAR's tol=1e-6 primal
and ETA's certified value 2/UB_final^2 where ETA converged.  We report
the first time with s_t <= 1% (and 0.1%), the first time the
certificate itself proves 1% (1 - LB_t^2/UB_t^2 <= 1%), and s at the
cap for cells that timed out.

LIBLINEAR: from results/liblin_tol_sweep.json if present (rows with
dataset, C, tol, seed, time, primal from src/liblin_default_tol.py run
with several --tols): time-to-1% is the mean time of the coarsest
tolerance whose primal is within 1% of P* in every seed.

Usage: python3 src/time_to_tol.py [--shards results/full_battery.json.shards]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def first_time(trace, pred):
    for it, t, ub, lb in trace:
        if pred(ub, lb):
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', default='results/full_battery.json.shards')
    ap.add_argument('--uncapped', default='results/full_battery_uncapped.shards')
    ap.add_argument('--sweep', default='results/liblin_tol_sweep.json')
    args = ap.parse_args()

    cells = {}
    for d in (args.shards, args.uncapped):
        for p in sorted(glob.glob(f'{d}/*_0.json')):
            recs = json.load(open(p))
            eta = next((r for r in recs if r['solver'] == 'ETA'), None)
            lib = next((r for r in recs if r['solver'] == 'LIBLIN'), None)
            if eta is None or eta.get('trace') is None:
                continue
            key = (eta['dataset'], eta['C'])
            cells.setdefault(key, []).append((d, eta, lib))

    sweep = {}
    base = Path(args.sweep)
    for f in sorted(base.parent.glob(base.stem + '*.json')):
        for r in json.load(open(f)):
            sweep.setdefault((r['dataset'], r['C']), []).append(r)

    print('| dataset | C | P* | ETA: t to 1% | t to 0.1% | certified 1% | '
          'converged (1e-3) | LIBLINEAR: t to 1% |')
    print('|--|--:|--:|--:|--:|--:|--:|--:|')
    for key in sorted(cells):
        runs = cells[key]
        p_star = min([lib['primal'] for _, _, lib in runs if lib is not None] +
                     [2.0 / eta['trace'][-1][2] ** 2 for _, eta, _ in runs
                      if eta['status'] == 'converged'])
        # prefer the uncapped trace when both exist (same split, same optimum)
        d, eta, lib = sorted(runs, key=lambda x: x[0] != args.uncapped)[0]
        tr = eta['trace']
        sub = lambda ub: 1.0 - (2.0 / ub ** 2) / p_star
        t1 = first_time(tr, lambda ub, lb: sub(ub) <= 0.01)
        t01 = first_time(tr, lambda ub, lb: sub(ub) <= 0.001)
        tc = first_time(tr, lambda ub, lb: lb > 0 and 1 - (lb / ub) ** 2 <= 0.01)
        conv = f"{eta['time']:.0f} s" if eta['status'] == 'converged' \
            else f"timeout, s={sub(tr[-1][2]):.1%} at {eta['time']:.0f} s"
        fmt = lambda t: f'{t:.0f} s' if t is not None else '-'
        lib_t = '-'
        if key in sweep:
            by_tol = {}
            for r in sweep[key]:
                by_tol.setdefault(r['tol'], []).append(r)
            ok = [(tol, rs) for tol, rs in by_tol.items()
                  if all(r['primal'] <= 1.01 * p_star for r in rs)]
            if ok:
                tol, rs = max(ok, key=lambda x: x[0])
                lib_t = f"{sum(r['time'] for r in rs) / len(rs):.0f} s (tol {tol:g})"
            else:
                lib_t = 'not within 1% at any tolerance run'
        print(f"| {key[0]} | {key[1]:g} | {p_star:.5g} | {fmt(t1)} | {fmt(t01)} | "
              f"{fmt(tc)} | {conv} | {lib_t} |")


if __name__ == '__main__':
    main()
