"""BPCG (Tsuji et al. 2022) versus the guarded block-transfer ETA on the
real-data cells of the paper, same objective, same certificate, same
column-caching cost model:

  gisette-l2     linear L2 soft margin, C=1 (full data, d=5000)
  <name>-kl2     RBF L2 margin K+I/C, C=1, class-balanced 10k subsample
                 (the KL2 cells of Table 1: gisette, w8a, a9a, covtype, ijcnn1)

Records wall-clock, iterations, column evaluations, distance, certified
gap, test accuracy and status per (cell, solver, seed).  Resumes.

Usage:
  OMP_NUM_THREADS=1 python3 src/bpcg_baseline.py --data-dir data --seeds 3 \
      --out results/bpcg_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpcg import BPCG                                              # noqa: E402
from full_battery import _load_cached, acc, ci95, sep_from_soft    # noqa: E402
from regime_battery import _subsample                              # noqa: E402
from soft_margin import SoftMarginTA                               # noqa: E402
from kernel_ta import KernelETA                                    # noqa: E402

EPS, CAP, C = 1e-3, 600.0, 1.0
DATASETS = ['gisette', 'w8a', 'a9a', 'covtype', 'ijcnn1']
CELLS = ['gisette-l2'] + [f'{d}-kl2' for d in DATASETS]


def _pack(r, cols, accv, extra=None):
    d = dict(time=r.time, iters=r.iterations, oracle=cols, dist=r.distance,
             gap=(r.distance - r.lower_bound) / r.distance if r.distance > 0
             else float('inf'), acc=accv, status=r.status, sv=r.sparsity)
    if extra:
        d.update(extra)
    return d


def run_cell(cell, solver, X, y, Xt, yt, seed):
    name, kind = cell.rsplit('-', 1)
    if kind == 'l2':
        V, W = X[y == 1], X[y == 0]
        if solver == 'eta':
            ta = SoftMarginTA(V, W, C=C, step_mode='block', block_size=16,
                              zigzag_strategy='pairwise', seed=0)
            r = ta.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
            w, b = sep_from_soft(ta, r)
            return _pack(r, ta.col_evals, acc(w, b, Xt, yt),
                         dict(n_drops=ta.n_drops))
        s = BPCG(V, W, kernel='linear', reg_C=C)
        r = s.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
        pred = (s.decision_function(Xt) > 0).astype(int)
        return _pack(r, s.col_evals, float(np.mean(pred == yt)),
                     dict(n_fw=r.n_fw, n_pw=r.n_pw, n_drops=r.n_drops))
    Xk, yk = _subsample(X, y, 10_000, seed)
    V, W = Xk[yk == 1], Xk[yk == 0]
    gamma = 1.0 / X.shape[1]
    if solver == 'eta':
        kt = KernelETA(V, W, kernel='rbf', gamma=gamma, reg_C=C,
                       step_mode='block', block_size=16,
                       zigzag_strategy='pairwise', seed=0)
        r = kt.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
        pred = (kt.decision_function(Xt, r) > 0).astype(int)
        return _pack(r, kt.col_evals, float(np.mean(pred == yt)),
                     dict(n_drops=kt.n_drops, n_kernel=len(Xk)))
    s = BPCG(V, W, kernel='rbf', gamma=gamma, reg_C=C)
    r = s.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    pred = (s.decision_function(Xt) > 0).astype(int)
    return _pack(r, s.col_evals, float(np.mean(pred == yt)),
                 dict(n_fw=r.n_fw, n_pw=r.n_pw, n_drops=r.n_drops,
                      n_kernel=len(Xk)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--cells', nargs='*', default=CELLS)
    ap.add_argument('--solvers', nargs='*', default=['bpcg', 'eta'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results/bpcg_baseline.json')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['cell'], r['solver'], r['seed']) for r in rows}
    if done:
        print(f"{len(done)} cells already done in {out}", flush=True)
    for cell in args.cells:
        name = cell.rsplit('-', 1)[0]
        for seed in range(args.seeds):
            todo = [s for s in args.solvers if (cell, s, seed) not in done]
            if not todo:
                continue
            X, y, Xt, yt = _load_cached(name, args.data_dir, 100_000, seed)
            for solver in todo:
                r = run_cell(cell, solver, X, y, Xt, yt, seed)
                r.update(cell=cell, solver=solver, seed=seed)
                rows.append(r)
                print(f"[{cell} {solver} seed={seed}] {r['time']:.1f}s/"
                      f"{r['status']} iters={r['iters']} cols={r['oracle']} "
                      f"dist={r['dist']:.5f} gap={r['gap']:.1e} "
                      f"acc={r['acc']:.4f}", flush=True)
                out.write_text(json.dumps(rows, indent=1))

    print('\n| cell | solver | time (s) | iterations | columns | dist | '
          'accuracy | converged |')
    print('|--|--|--:|--:|--:|--:|--:|--:|')
    for cell in args.cells:
        for solver in args.solvers:
            sel = [r for r in rows if (r['cell'], r['solver']) == (cell, solver)]
            if not sel:
                continue
            tm, th = ci95([r['time'] for r in sel])
            print(f"| {cell} | {solver} | {tm:.1f} ± {th:.1f} | "
                  f"{np.mean([r['iters'] for r in sel]):,.0f} | "
                  f"{np.mean([r['oracle'] for r in sel]):,.0f} | "
                  f"{np.mean([r['dist'] for r in sel]):.5f} | "
                  f"{np.mean([r['acc'] for r in sel]):.4f} | "
                  f"{sum(r['status'] == 'converged' for r in sel)}/{len(sel)} |")


if __name__ == '__main__':
    main()
