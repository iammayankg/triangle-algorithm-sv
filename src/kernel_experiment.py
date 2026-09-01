"""Kernelized study: RBF hard-margin hull distance, TA vs SMO vs LIBSVM.

The kernel analogue of the paper's Table 3: separated Gaussian clouds
(translation k = 1.2 x max diameter), RBF kernel with gamma = 1/d, all
three solvers computing the feature-space hull distance:

  K-ETA  - KernelETA (pairwise zig-zag strategy)
  K-SMO  - KernelSMO (our SMO, kernel rows, LRU cache)
  LIBSVM - sklearn SVC(kernel='rbf', C = 1e8): hard margin;
           delta = 2/||w_H||, ||w_H||^2 = (alpha y)' K (alpha y)

Also one kernel L2-soft-margin cell (K + I/C on overlapping data) as an
agreement check between K-ETA(reg_C) and K-SMO(reg_C).

Usage: python3 src/kernel_experiment.py [--trials 2] [--dims 10 100 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import generate_two_balls, generate_overlap   # noqa: E402
from kernel_ta import KernelETA, KernelSMO              # noqa: E402

from sklearn.metrics.pairwise import rbf_kernel         # noqa: E402
from sklearn.svm import SVC                             # noqa: E402

DIMS = [10, 100, 1000]
N = 3000
EPS = 1e-3


def run_hard(d, seed):
    gamma = 1.0 / d
    V, W = generate_two_balls(d, N, 1.2, rng=np.random.default_rng(seed))
    out = {}
    ta = KernelETA(V, W, kernel='rbf', gamma=gamma,
                   zigzag_strategy='pairwise', seed=0)
    r = ta.solve_distance(eps=EPS, max_iter=200_000)
    out['K-ETA'] = dict(time=r.time, dist=r.distance, iters=r.iterations,
                        sv=r.sparsity, status=r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    s = KernelSMO(X, y, kernel='rbf', gamma=gamma, tol=EPS,
                  max_iter=500_000, time_cap=300).solve()
    out['K-SMO'] = dict(time=s.time, dist=s.hull_distance,
                        iters=s.iterations, sv=s.sparsity, status=s.status)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = SVC(kernel='rbf', gamma=gamma, C=1e8, tol=EPS,
                cache_size=500).fit(X, y)
        el = time.perf_counter() - t0
    sv = m.support_
    ay = m.dual_coef_.ravel()
    Ksv = rbf_kernel(X[sv], X[sv], gamma=gamma)
    wn2 = float(ay @ Ksv @ ay)
    out['LIBSVM'] = dict(time=el, dist=2.0 / np.sqrt(wn2),
                         sv=int(m.n_support_.sum()), status='converged')
    return out


def run_soft_check(d=100, C=1.0, seed=123):
    gamma = 1.0 / d
    V, W = generate_overlap(d, N, delta=4.0, rng=np.random.default_rng(seed))
    ta = KernelETA(V, W, kernel='rbf', gamma=gamma, reg_C=C,
                   zigzag_strategy='pairwise', step_mode='mdm', seed=0)
    r = ta.solve_distance(eps=EPS, max_iter=200_000)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    s = KernelSMO(X, y, kernel='rbf', gamma=gamma, reg_C=C, tol=EPS,
                  max_iter=500_000, time_cap=300).solve()
    rel = abs(r.distance - s.hull_distance) / s.hull_distance
    print(f"[ksoft] d={d} C={C} K-ETA=({r.time:.1f}s,d={r.distance:.6f},"
          f"{r.status}) K-SMO=({s.time:.1f}s,d={s.hull_distance:.6f},"
          f"{s.status}) rel={rel:.1e}", flush=True)
    return {'d': d, 'C': C, 'ta_time': r.time, 'ta_dist': r.distance,
            'smo_time': s.time, 'smo_dist': s.hull_distance,
            'rel': rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--dims', type=int, nargs='*', default=DIMS)
    ap.add_argument('--out', default='results/kernel_experiment.json')
    args = ap.parse_args()
    solvers = ['K-ETA', 'K-SMO', 'LIBSVM']
    rows = []
    for d in args.dims:
        recs = []
        for t in range(args.trials):
            r = run_hard(d, seed=11000 + 31 * t + d)
            recs.append(r)
            print(f"[khard] d={d} trial={t} " + " ".join(
                f"{s}=({r[s]['time']:.1f}s,d={r[s]['dist']:.5f},"
                f"sv={r[s]['sv']})" for s in solvers), flush=True)
        row = {'dim': d}
        for s in solvers:
            row[f'{s}_time'] = float(np.mean([r[s]['time'] for r in recs]))
            row[f'{s}_dist'] = float(np.mean([r[s]['dist'] for r in recs]))
            row[f'{s}_sv'] = float(np.mean([r[s]['sv'] for r in recs]))
        row['rel_smo'] = float(np.mean(
            [abs(r['K-ETA']['dist'] - r['K-SMO']['dist'])
             / r['K-SMO']['dist'] for r in recs]))
        row['rel_libsvm'] = float(np.mean(
            [abs(r['K-ETA']['dist'] - r['LIBSVM']['dist'])
             / r['LIBSVM']['dist'] for r in recs]))
        rows.append(row)
        Path(args.out).write_text(json.dumps(rows, indent=2))
    soft = [run_soft_check(C=c) for c in (0.1, 1.0)]
    Path(args.out.replace('.json', '_soft.json')).write_text(
        json.dumps(soft, indent=2))
    print(f"\n{'d':>5} | " + " | ".join(f"{s:>22}" for s in solvers)
          + " |  rel(SMO)  rel(LIBSVM)")
    for row in rows:
        cells = [f"{row[f'{s}_time']:7.1f}s d={row[f'{s}_dist']:.5f}"
                 for s in solvers]
        print(f"{row['dim']:>5} | " + " | ".join(f"{c:>22}" for c in cells)
              + f" | {row['rel_smo']:.1e}  {row['rel_libsvm']:.1e}")


if __name__ == '__main__':
    main()
