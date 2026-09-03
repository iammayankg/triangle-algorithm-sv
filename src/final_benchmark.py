"""Consolidated performance: best ETA configuration vs the strongest
standard solver, across every problem class in this repository.

  A  hard margin, synthetic Table-3 (d=1000,  eps=1e-3)
  B  hard margin, synthetic Table-3 (d=10000, eps=1e-3)
  C  hard margin, tight tolerance   (d=1000,  eps=1e-5)
  D  L2 soft margin, overlap, C=1   (d=1000)
  E  L2 soft margin, mnist5k odd-vs-even, C=1
  F  RBF hard margin, synthetic     (d=1000)
  G  nu-SVM (nu=0.3), overlap       (d=1000)

Per benchmark: the paper-style ETA, the optimised ETA (block / shrink /
mdm as appropriate), and the standard solver, with distance agreement.

Usage: python3 src/final_benchmark.py [--trials N] [--benches A B ...]
           [--out results/final_benchmark.json]
           [--parallel P]
Trial t draws the synthetic instance (or the MNIST split) with seed
offset t; trial 0 reproduces the original single-instance run. Each
(bench, trial) job writes a shard under <out>.shards/ and the shards
are merged into <out>; a rerun skips jobs whose shard (or rows in
<out>) already exist. --parallel P runs P jobs concurrently, one
single-threaded process each (set OMP_NUM_THREADS=1); keep P well
below the core count so timings are not distorted by contention. The
summary printed at the end (and src/final_benchmark_fig.py) averages
over trials.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import generate_two_balls, generate_overlap        # noqa: E402
from kernel_ta import KernelETA                              # noqa: E402
from reduced_hull import ReducedHullTA                       # noqa: E402
from smo import SMO                                          # noqa: E402
from soft_margin import SoftMarginTA                         # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm     # noqa: E402

from sklearn.metrics.pairwise import rbf_kernel              # noqa: E402
from sklearn.svm import SVC, LinearSVC, NuSVC                # noqa: E402

RESULTS = []
OUT = Path('results/final_benchmark.json')
TRIAL = 0
STEP = 100_000          # seed offset per trial


def rec(bench, solver, t, dist, status='converged', extra=''):
    RESULTS.append(dict(bench=bench, solver=solver, time=float(t),
                        dist=float(dist), status=status, extra=extra,
                        trial=TRIAL))
    print(f"[{bench} trial {TRIAL}] {solver:22s} t={t:8.2f}s d={dist:.6f} "
          f"{status} {extra}", flush=True)


def hard(d, eps, bench):
    V, W = generate_two_balls(d, 5000, 1.2,
                              rng=np.random.default_rng(2000 + d + STEP * TRIAL))
    r = EnhancedTriangleAlgorithm(V, W, seed=0).solve_distance(eps=eps)
    rec(bench, 'ETA paper-style', r.time, r.distance, r.status)
    r = EnhancedTriangleAlgorithm(V, W, seed=0, step_mode='block',
                                  zigzag_strategy='pairwise', shrink=True,
                                  shrink_every=25).solve_distance(eps=eps)
    rec(bench, 'ETA block+shrink', r.time, r.distance, r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(5000), -np.ones(5000)])
    s = SMO(X, y, C=1e12, tol=eps, max_iter=200_000, cache_rows=2,
            time_cap=300).solve()
    rec(bench, 'SMO (uncached)', s.time, s.hull_distance, s.status)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = SVC(kernel='linear', C=1e6, tol=eps).fit(X, y)
        el = time.perf_counter() - t0
    wn = np.linalg.norm(m.coef_.ravel())
    rec(bench, 'LIBSVM (SVC linear)', el, 2.0 / wn)


def soft_synth():
    V, W = generate_overlap(1000, 5000, delta=4.0,
                            rng=np.random.default_rng(7 + STEP * TRIAL))
    r = SoftMarginTA(V, W, C=1.0, step_mode='mdm',
                     zigzag_strategy='pairwise', seed=0).solve_distance(
        eps=1e-3, max_iter=200_000)
    rec('D soft d=1000', 'ETA mdm', r.time, r.distance, r.status)
    r = SoftMarginTA(V, W, C=1.0, step_mode='block', block_size=32,
                     zigzag_strategy='pairwise', seed=0).solve_distance(
        eps=1e-3, max_iter=200_000)
    rec('D soft d=1000', 'ETA block(32)', r.time, r.distance, r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(5000), -np.ones(5000)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=0.5, tol=1e-6,
                      max_iter=100_000, intercept_scaling=100.0).fit(X, y)
        el = time.perf_counter() - t0
    wv, bv = m.coef_.ravel(), float(m.intercept_[0])
    xi = np.maximum(0.0, 1.0 - y * (X @ wv + bv))
    P = 0.5 * float(wv @ wv) + 0.5 * float(xi @ xi)
    rec('D soft d=1000', 'LIBLINEAR sq-hinge', el, np.sqrt(2.0 / P),
        extra='(dist from primal)')


def soft_mnist():
    from mlxtend.data import mnist_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = mnist_data()
    y = (y % 2).astype(int)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, stratify=y,
                                      random_state=TRIAL)
    Xtr = StandardScaler().fit_transform(Xtr)
    V, W = Xtr[ytr == 1], Xtr[ytr == 0]
    r = SoftMarginTA(V, W, C=1.0, step_mode='mdm',
                     zigzag_strategy='pairwise', seed=0).solve_distance(
        eps=1e-3, max_iter=200_000)
    rec('E mnist-oe', 'ETA mdm', r.time, r.distance, r.status)
    r = SoftMarginTA(V, W, C=1.0, step_mode='block', block_size=8,
                     zigzag_strategy='pairwise', seed=0).solve_distance(
        eps=1e-3, max_iter=200_000)
    rec('E mnist-oe', 'ETA block(8)', r.time, r.distance, r.status)
    Xs = np.vstack([V, W])
    ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=0.5, tol=1e-6,
                      max_iter=100_000, intercept_scaling=100.0).fit(Xs, ys)
        el = time.perf_counter() - t0
    wv, bv = m.coef_.ravel(), float(m.intercept_[0])
    xi = np.maximum(0.0, 1.0 - ys * (Xs @ wv + bv))
    P = 0.5 * float(wv @ wv) + 0.5 * float(xi @ xi)
    rec('E mnist-oe', 'LIBLINEAR sq-hinge', el, np.sqrt(2.0 / P),
        extra='(dist from primal)')


def kernel_bench():
    gamma = 1.0 / 1000
    V, W = generate_two_balls(1000, 3000, 1.2,
                              rng=np.random.default_rng(12000 + STEP * TRIAL))
    r = KernelETA(V, W, kernel='rbf', gamma=gamma,
                  zigzag_strategy='pairwise', seed=0).solve_distance(eps=1e-3)
    rec('F rbf d=1000', 'K-ETA pairwise', r.time, r.distance, r.status)
    r = KernelETA(V, W, kernel='rbf', gamma=gamma, step_mode='block',
                  zigzag_strategy='pairwise', seed=0).solve_distance(eps=1e-3)
    rec('F rbf d=1000', 'K-ETA block', r.time, r.distance, r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(3000), -np.ones(3000)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = SVC(kernel='rbf', gamma=gamma, C=1e8, tol=1e-3,
                cache_size=500).fit(X, y)
        el = time.perf_counter() - t0
    sv = m.support_
    ay = m.dual_coef_.ravel()
    K = rbf_kernel(X[sv], X[sv], gamma=gamma)
    rec('F rbf d=1000', 'LIBSVM (SVC rbf)', el,
        2.0 / np.sqrt(float(ay @ K @ ay)))


def nu_bench():
    V, W = generate_overlap(1000, 5000, delta=4.0,
                            rng=np.random.default_rng(9100 + STEP * TRIAL))
    rh = ReducedHullTA(V, W, nu=0.3)
    r = rh.solve_distance(eps=1e-3, max_iter=200_000)
    rec('G nu=0.3 d=1000', 'RCH-TA', r.time, r.distance, r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(5000), -np.ones(5000)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = NuSVC(nu=0.3, kernel='linear', tol=1e-3,
                  max_iter=5_000_000).fit(X, y)
        el = time.perf_counter() - t0
    dc = m.dual_coef_.ravel()
    S = float(dc[dc > 0].sum())
    rec('G nu=0.3 d=1000', 'NuSVC (LIBSVM)', el,
        float(np.linalg.norm(m.coef_.ravel())) / S)


BENCHES = {
    'A': lambda: hard(1000, 1e-3, 'A hard d=1000'),
    'B': lambda: hard(10000, 1e-3, 'B hard d=10000'),
    'C': lambda: hard(1000, 1e-5, 'C hard d=1000 tight'),
    'D': soft_synth,
    'E': soft_mnist,
    'F': kernel_bench,
    'G': nu_bench,
}
N_SOLVERS = {'A hard d=1000': 4, 'B hard d=10000': 4, 'C hard d=1000 tight': 4,
             'D soft d=1000': 3, 'E mnist-oe': 3, 'F rbf d=1000': 3,
             'G nu=0.3 d=1000': 2}
BENCH_NAME = {'A': 'A hard d=1000', 'B': 'B hard d=10000',
              'C': 'C hard d=1000 tight', 'D': 'D soft d=1000',
              'E': 'E mnist-oe', 'F': 'F rbf d=1000', 'G': 'G nu=0.3 d=1000'}


def summarize(rows):
    from scipy import stats
    keys = sorted({(r['bench'], r['solver']) for r in rows},
                  key=lambda k: (k[0], [r['solver'] for r in rows].index(k[1])))
    print(f"\n{'bench':22s} {'solver':22s} {'time mean ± CI95':>20s}  conv")
    for b, sv in keys:
        sel = [r for r in rows if (r['bench'], r['solver']) == (b, sv)]
        t = np.array([r['time'] for r in sel])
        half = (stats.t.ppf(0.975, len(t) - 1) * t.std(ddof=1) / np.sqrt(len(t))
                if len(t) > 1 else 0.0)
        conv = sum(r['status'] == 'converged' for r in sel)
        print(f"{b:22s} {sv:22s} {t.mean():10.2f} ± {half:6.2f} s  {conv}/{len(sel)}")


def _shard(out, key, trial):
    return Path(str(out) + '.shards') / f"{key}_{trial}.json"


def _run_job(job):
    """Worker: one (bench, trial) -> shard file; returns its rows."""
    global RESULTS, TRIAL
    key, trial, out = job
    RESULTS, TRIAL = [], trial
    BENCHES[key]()
    shard = _shard(out, key, trial)
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(RESULTS, indent=2))
    return RESULTS


def main():
    import argparse
    import os
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--benches', nargs='*', default=list(BENCHES))
    ap.add_argument('--parallel', type=int, default=1,
                    help='concurrent (bench, trial) jobs; set OMP_NUM_THREADS=1')
    ap.add_argument('--out', default='results/final_benchmark.json')
    args = ap.parse_args()
    if args.parallel > 1 and os.environ.get('OMP_NUM_THREADS') != '1':
        print('WARNING: set OMP_NUM_THREADS=1 when using --parallel', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(out.read_text()) if out.exists() else []
    for r in rows:
        r.setdefault('trial', 0)      # rows from the single-run version
    # a (bench, trial) pair counts as done only with its full solver set;
    # partial rows (an interrupted sequential run) are dropped and rerun
    counts = {}
    for r in rows:
        counts[(r['bench'], r['trial'])] = counts.get((r['bench'], r['trial']), 0) + 1
    done = {k for k, c in counts.items()
            if c >= N_SOLVERS[k[0]]}
    partial = {k for k in counts if k not in done}
    if partial:
        print(f"dropping partial rows for {sorted(partial)}", flush=True)
        rows = [r for r in rows if (r['bench'], r['trial']) not in partial]
    jobs = []
    for trial in range(args.trials):
        for key in args.benches:
            shard = _shard(out, key, trial)
            if (BENCH_NAME[key], trial) in done:
                continue
            if shard.exists():
                rows.extend(json.loads(shard.read_text()))
                continue
            jobs.append((key, trial, out))
    print(f"{len(done)} (bench, trial) pairs already in {out}, "
          f"{len(jobs)} to run, parallel={args.parallel}", flush=True)
    if args.parallel > 1 and jobs:
        with Pool(args.parallel) as pool:
            for recs in pool.imap_unordered(_run_job, jobs):
                rows.extend(recs)
    else:
        for job in jobs:
            rows.extend(_run_job(job))
    rows.sort(key=lambda r: (r['trial'], r['bench']))
    out.write_text(json.dumps(rows, indent=2))
    summarize(rows)
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
