"""Optimized ETA vs standard solvers on the paper's Table-3 protocol.

Solvers:
  ETA-pw   - Enhanced Triangle Algorithm, pairwise (MDM) zig-zag strategy
  ETA-mid  - Enhanced Triangle Algorithm, paper's midpoint strategy
  SMO      - our hard-margin SMO (LRU row cache)
  LIBSVM   - sklearn.svm.SVC(kernel='linear', C=1e6)  [libsvm SMO]
  LIBLIN   - sklearn.svm.LinearSVC(C=1e6)             [liblinear dual CD]
  SGD      - sklearn SGDClassifier hinge loss          [stochastic subgradient]

All solve (or approximate) the hard-margin separator on separable data;
the geometric margin 2/||w|| is compared against ETA's certified interval
[LB, UB] computed at eps = 1e-6.  Hard margin is approximated by the
soft-margin solvers with C = 1e6 (no violations at the optimum on these
separable instances).

Usage: python3 src/solver_comparison.py [--trials 3] [--dims 100 1000 ...]
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

from data import generate_two_balls              # noqa: E402
from smo import SMO                              # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402

from sklearn.svm import SVC, LinearSVC           # noqa: E402
from sklearn.linear_model import SGDClassifier   # noqa: E402

EPS = 1e-3
DIMS = [100, 300, 1000, 2000, 5000, 10000]


def margin_from_w(w):
    n = float(np.linalg.norm(w))
    return 2.0 / n if n > 0 else np.inf


def run_one(d, n, k, seed):
    V, W = generate_two_balls(d, n, k=k, rng=np.random.default_rng(seed))
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    out = {}

    # certified reference interval from ETA at tight tolerance
    ref = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise',
                                    seed=0).solve_distance(eps=1e-6,
                                                           max_iter=200_000)
    out['ref_lb'], out['ref_ub'] = ref.lower_bound, ref.distance

    t0 = time.perf_counter()
    r = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise',
                                  seed=0).solve_distance(eps=EPS,
                                                         max_iter=10_000)
    out['ETA-pw'] = (time.perf_counter() - t0, r.distance)

    t0 = time.perf_counter()
    r = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='midpoint',
                                  seed=0).solve_distance(eps=EPS,
                                                         max_iter=10_000)
    out['ETA-mid'] = (time.perf_counter() - t0, r.distance)

    t0 = time.perf_counter()
    r = SMO(X, y, C=1e12, tol=EPS, max_iter=200_000, time_cap=300).solve()
    out['SMO'] = (time.perf_counter() - t0, r.hull_distance)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = SVC(kernel='linear', C=1e6, tol=1e-3)
        m.fit(X, y)
        out['LIBSVM'] = (time.perf_counter() - t0,
                         margin_from_w(m.coef_.ravel()))

        # hinge loss + moderate C + large intercept_scaling recovers the
        # geometric margin accurately (squared hinge / huge C do not)
        t0 = time.perf_counter()
        m = LinearSVC(loss='hinge', C=1e4, tol=1e-6, max_iter=100_000,
                      intercept_scaling=100.0)
        m.fit(X, y)
        out['LIBLIN'] = (time.perf_counter() - t0,
                         margin_from_w(m.coef_.ravel()))

        # stochastic subgradient: finds a separator but does not recover
        # the maximal margin at any alpha (reported as-is)
        t0 = time.perf_counter()
        m = SGDClassifier(loss='hinge', alpha=1e-4, max_iter=1000, tol=1e-4)
        m.fit(X, y)
        out['SGD'] = (time.perf_counter() - t0,
                      margin_from_w(m.coef_.ravel()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--dims', type=int, nargs='*', default=DIMS)
    ap.add_argument('--out', default='results/solver_comparison.json')
    args = ap.parse_args()

    solvers = ['ETA-pw', 'ETA-mid', 'SMO', 'LIBSVM', 'LIBLIN', 'SGD']
    rows = []
    for d in args.dims:
        recs = []
        for t in range(args.trials):
            r = run_one(d, 5000, 1.2, seed=2000 + 31 * t + d)
            recs.append(r)
            msg = ' '.join(f"{s}={r[s][0]:.2f}s/{r[s][1]:.3f}"
                           for s in solvers)
            print(f"[cmp] d={d} trial={t} ref=[{r['ref_lb']:.4f},"
                  f"{r['ref_ub']:.4f}] {msg}", flush=True)
        row = {'dim': d,
               'ref_ub': float(np.mean([r['ref_ub'] for r in recs]))}
        for s in solvers:
            row[f'{s}_time'] = float(np.mean([r[s][0] for r in recs]))
            row[f'{s}_dist'] = float(np.mean([r[s][1] for r in recs]))
            row[f'{s}_relerr'] = float(np.mean(
                [abs(r[s][1] - r['ref_ub']) / r['ref_ub'] for r in recs]))
        rows.append(row)
        Path(args.out).write_text(json.dumps(rows, indent=2))

    print(f"\n{'dim':>6} | " + " | ".join(f"{s:>16}" for s in solvers))
    print(f"{'':>6} | " + " | ".join(f"{'time / rel.err':>16}"
                                     for _ in solvers))
    for row in rows:
        cells = [f"{row[f'{s}_time']:6.2f}s {row[f'{s}_relerr']:8.1e}"
                 for s in solvers]
        print(f"{row['dim']:>6} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == '__main__':
    main()
