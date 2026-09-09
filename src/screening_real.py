"""Safe screening on real data (journal review, item 3): does the
gap-certified screening of Theorem 8 activate on a real cell, and what
does it buy?

Cells: the gisette linear hard margin (LIN of Table 1) and the gisette
L2 soft margin (C=1), both with d=5000 and a sparse support; tolerances
eps in {1e-3, 1e-5}; screening off / on (every 25 iterations, compaction
when >= 5% of a class is removable, the setting of the regime battery).
Each (cell, eps, screening, seed) runs in its own fresh subprocess, one
at a time.  Per run: wall-clock, iterations, columns, distance, gap,
accuracy, survivors (points alive at the end, of n+m), number of
compactions and the iteration / time of the first one.

Usage (sequential, roughly an hour):
  OMP_NUM_THREADS=1 python3 src/screening_real.py --data-dir data --seeds 3 \
      --out results/screening_real.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CAP = 3600.0
CELLS = ['gisette-lin', 'gisette-l2']
EPSS = [1e-3, 1e-5]


def run_one(cell, eps, shrink, seed, data_dir):
    from full_battery import _load_cached, acc, sep_from_soft
    from soft_margin import SoftMarginTA
    from triangle_algorithm import EnhancedTriangleAlgorithm

    name, kind = cell.rsplit('-', 1)
    X, y, Xt, yt = _load_cached(name, data_dir, 100_000, seed)
    V, W = X[y == 1], X[y == 0]
    kw = dict(step_mode='block', block_size=16, zigzag_strategy='pairwise',
              seed=0, shrink=shrink, shrink_every=25)
    ta = SoftMarginTA(V, W, C=1.0, **kw) if kind == 'l2' \
        else EnhancedTriangleAlgorithm(V, W, **kw)
    n0 = ta.n + ta.m
    events = []
    t0 = time.perf_counter()
    orig_compact = ta._compact

    def compact(keepV, keepW):
        events.append(dict(t=time.perf_counter() - t0,
                           alive_before=ta.n + ta.m,
                           alive_after=int(keepV.sum() + keepW.sum())))
        return orig_compact(keepV, keepW)
    ta._compact = compact
    r = ta.solve_distance(eps=eps, max_iter=2_000_000, time_cap=CAP)
    if kind == 'l2':
        w, b = sep_from_soft(ta, r)
    else:
        d2 = r.distance ** 2
        w = 2.0 * (r.p - r.q) / d2
        b = (float(r.q @ r.q) - float(r.p @ r.p)) / d2
    return dict(cell=cell, eps=eps, shrink=shrink, seed=seed, time=r.time,
                iters=r.iterations, oracle=ta.col_evals, dist=r.distance,
                gap=(r.distance - r.lower_bound) / r.distance,
                status=r.status, acc=acc(w, b, Xt, yt), sv=r.sparsity,
                n_total=n0, alive=ta.n + ta.m, n_compact=len(events),
                first_compact_t=events[0]['t'] if events else None,
                compactions=events)


def summarise(rows):
    from full_battery import ci95
    print(f"{'cell':12s} {'eps':>6s} {'screen':>6s} {'time (s)':>14s} "
          f"{'iters':>7s} {'cols':>6s} {'alive':>12s} {'compact':>7s} "
          f"{'first (s)':>9s} {'dist':>10s} {'acc':>6s} status")
    for cell in CELLS:
        for eps in EPSS:
            for sh in (False, True):
                rs = [r for r in rows if (r['cell'], r['eps'], r['shrink'])
                      == (cell, eps, sh)]
                if not rs:
                    continue
                m, h = ci95([r['time'] for r in rs])
                ft = [r['first_compact_t'] for r in rs if r['first_compact_t']]
                print(f"{cell:12s} {eps:6.0e} {str(sh):>6s} {m:7.1f} ± {h:4.1f} "
                      f"{np.mean([r['iters'] for r in rs]):7.0f} "
                      f"{np.mean([r['oracle'] for r in rs]):6.0f} "
                      f"{np.mean([r['alive'] for r in rs]):6.0f}/{rs[0]['n_total']:<5d} "
                      f"{np.mean([r['n_compact'] for r in rs]):7.1f} "
                      f"{(np.mean(ft) if ft else float('nan')):9.1f} "
                      f"{np.mean([r['dist'] for r in rs]):10.6f} "
                      f"{np.mean([r['acc'] for r in rs]):6.3f} "
                      f"{','.join(sorted({r['status'] for r in rs}))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--cells', nargs='*', default=CELLS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results/screening_real.json')
    ap.add_argument('--single', nargs=4, metavar=('CELL', 'EPS', 'SHRINK', 'SEED'),
                    help='internal: run one configuration and print JSON')
    a = ap.parse_args()
    if a.single:
        cell, eps, sh, seed = a.single
        print(json.dumps(run_one(cell, float(eps), sh == '1', int(seed),
                                 a.data_dir)))
        return
    out = Path(a.out)
    rows = json.load(open(out)) if out.exists() else []
    done = {(r['cell'], r['eps'], r['shrink'], r['seed']) for r in rows}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    for cell in a.cells:
        for eps in EPSS:
            for sh in (False, True):
                for seed in range(a.seeds):
                    if (cell, eps, sh, seed) in done:
                        continue
                    cmd = [sys.executable, str(HERE / 'screening_real.py'),
                           '--single', cell, str(eps), '1' if sh else '0',
                           str(seed), '--data-dir', a.data_dir]
                    p = subprocess.run(cmd, env=env, capture_output=True,
                                       text=True)
                    if p.returncode != 0:
                        print(p.stderr, file=sys.stderr)
                        raise SystemExit(f'{cell} {eps} {sh} {seed} failed')
                    row = json.loads(p.stdout.strip().splitlines()[-1])
                    rows.append(row)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(rows, indent=1))
                    print(f"[{cell} eps={eps:g} screen={sh} seed={seed}] "
                          f"{row['time']:.1f}s iters={row['iters']} "
                          f"alive={row['alive']}/{row['n_total']} "
                          f"compactions={row['n_compact']} {row['status']}",
                          flush=True)
    summarise(rows)


if __name__ == '__main__':
    main()
