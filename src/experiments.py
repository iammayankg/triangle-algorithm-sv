"""Replication experiments for the Enhanced Triangle Algorithm paper.

Protocols (following the source study):
  table2 : intersection testing.  Two 5000-point clouds, translation
           k = 0.9 x max empirical diameter (overlapping), eps = 1e-3,
           max 10^4 iterations, dimensions 3 .. 10000.
  table3 : TA II vs SMO distance computation on separated clouds
           (k = 1.2), same dimension sweep.  Reports iterations, time,
           sparsity and distance for both solvers.
  table4 : distance sensitivity at d = 1000; separation parameter
           k in {1.2, 1.3, ..., 1.9}.

Usage:
  python3 src/experiments.py --exp table2 --trials 3
  python3 src/experiments.py --exp all --trials 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import generate_two_balls              # noqa: E402
from smo import SMO                              # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402

DIMS = [3, 10, 50, 100, 300, 500, 1000, 2000, 5000, 10000]
N_POINTS = 5000
EPS = 1e-3
MAX_ITER = 10_000
SMO_TIME_CAP = 300.0


def _dtype_for(d):
    # float64 throughout, as in the original MATLAB study; the 7 GB of
    # RAM available comfortably fits the d = 10000 instances
    return np.float64


def run_table2(trials, out_dir, dims=DIMS, n=N_POINTS):
    rows = []
    for d in dims:
        recs = []
        for t in range(trials):
            V, W = generate_two_balls(d, n, k=0.9, mode='ball',
                                      rng=np.random.default_rng(1000 + 17 * t + d),
                                      dtype=_dtype_for(d))
            ta = EnhancedTriangleAlgorithm(V, W, seed=0)
            r = ta.solve_intersection(eps=EPS, max_iter=MAX_ITER)
            recs.append((r.iterations, r.time, r.status, r.distance))
            print(f"[table2] d={d} trial={t} status={r.status} "
                  f"it={r.iterations} t={r.time:.3f}s dist={r.distance:.4g}",
                  flush=True)
            del V, W, ta
        rows.append({
            'dim': d,
            'iterations': float(np.mean([x[0] for x in recs])),
            'time_s': float(np.mean([x[1] for x in recs])),
            'intersect_rate': float(np.mean([x[2] == 'intersect' for x in recs])),
        })
        _dump(out_dir / 'table2.json', rows)
    _write_csv(out_dir / 'table2.csv', rows)
    return rows


def _one_distance_comparison(d, n, k, seed):
    """Run TA II, a cached (optimized) SMO and an uncached SMO.

    The uncached variant recomputes both kernel rows every iteration,
    matching the cost profile of a plain MATLAB SMO implementation like
    the baseline of the source study.
    """
    dtype = _dtype_for(d)
    V, W = generate_two_balls(d, n, k=k, rng=np.random.default_rng(seed),
                              dtype=dtype)
    ta = EnhancedTriangleAlgorithm(V, W, seed=0)
    r_ta = ta.solve_distance(eps=EPS, max_iter=MAX_ITER)
    del ta
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    del V, W
    r_smo = SMO(X, y, C=1e12, tol=EPS, max_iter=200_000,
                time_cap=SMO_TIME_CAP).solve()
    r_smo_nc = SMO(X, y, C=1e12, tol=EPS, max_iter=200_000,
                   time_cap=SMO_TIME_CAP, cache_rows=2).solve()
    del X, y
    return r_ta, r_smo, r_smo_nc


def run_table3(trials, out_dir, dims=DIMS, n=N_POINTS, k=1.2):
    rows = []
    for d in dims:
        recs = []
        for t in range(trials):
            r_ta, r_smo, r_nc = _one_distance_comparison(d, n, k,
                                                         seed=2000 + 31 * t + d)
            recs.append((r_ta, r_smo, r_nc))
            print(f"[table3] d={d} trial={t} "
                  f"TA(it={r_ta.iterations},t={r_ta.time:.2f}s,"
                  f"sp={r_ta.sparsity},dist={r_ta.distance:.4f},{r_ta.status}) "
                  f"SMO(it={r_smo.iterations},t={r_smo.time:.2f}s,"
                  f"sp={r_smo.sparsity},dist={r_smo.hull_distance:.4f},"
                  f"{r_smo.status}) "
                  f"SMOnc(t={r_nc.time:.2f}s,{r_nc.status})", flush=True)
        rows.append({
            'dim': d,
            'ta_iter': float(np.mean([a.iterations for a, _, _ in recs])),
            'ta_time_s': float(np.mean([a.time for a, _, _ in recs])),
            'ta_sparsity': float(np.mean([a.sparsity for a, _, _ in recs])),
            'ta_dist': float(np.mean([a.distance for a, _, _ in recs])),
            'smo_iter': float(np.mean([b.iterations for _, b, _ in recs])),
            'smo_time_s': float(np.mean([b.time for _, b, _ in recs])),
            'smo_sparsity': float(np.mean([b.sparsity for _, b, _ in recs])),
            'smo_dist': float(np.mean([b.hull_distance for _, b, _ in recs])),
            'smo_nc_iter': float(np.mean([c.iterations for _, _, c in recs])),
            'smo_nc_time_s': float(np.mean([c.time for _, _, c in recs])),
            'smo_nc_dist': float(np.mean([c.hull_distance for _, _, c in recs])),
        })
        _dump(out_dir / 'table3.json', rows)
    _write_csv(out_dir / 'table3.csv', rows)
    return rows


def run_table4(trials, out_dir, d=1000, n=N_POINTS,
               ks=(1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)):
    rows = []
    for k in ks:
        recs = []
        for t in range(trials):
            r_ta, r_smo, r_nc = _one_distance_comparison(
                d, n, k, seed=3000 + 53 * t + int(k * 100))
            recs.append((r_ta, r_smo, r_nc))
            print(f"[table4] k={k} trial={t} "
                  f"TA(it={r_ta.iterations},t={r_ta.time:.2f}s,"
                  f"dist={r_ta.distance:.4f}) "
                  f"SMO(it={r_smo.iterations},t={r_smo.time:.2f}s,"
                  f"dist={r_smo.hull_distance:.4f},{r_smo.status}) "
                  f"SMOnc(t={r_nc.time:.2f}s)", flush=True)
        rows.append({
            'k': k,
            'ta_iter': float(np.mean([a.iterations for a, _, _ in recs])),
            'ta_time_s': float(np.mean([a.time for a, _, _ in recs])),
            'ta_dist': float(np.mean([a.distance for a, _, _ in recs])),
            'smo_iter': float(np.mean([b.iterations for _, b, _ in recs])),
            'smo_time_s': float(np.mean([b.time for _, b, _ in recs])),
            'smo_dist': float(np.mean([b.hull_distance for _, b, _ in recs])),
            'smo_nc_iter': float(np.mean([c.iterations for _, _, c in recs])),
            'smo_nc_time_s': float(np.mean([c.time for _, _, c in recs])),
            'smo_nc_dist': float(np.mean([c.hull_distance for _, _, c in recs])),
        })
        _dump(out_dir / 'table4.json', rows)
    _write_csv(out_dir / 'table4.csv', rows)
    return rows


def _dump(path, rows):
    path.write_text(json.dumps(rows, indent=2))


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='all',
                    choices=['table2', 'table3', 'table4', 'all'])
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--out', default='results')
    ap.add_argument('--dims', type=int, nargs='*', default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dims = args.dims or DIMS
    t0 = time.time()
    if args.exp in ('table2', 'all'):
        run_table2(args.trials, out, dims=dims)
    if args.exp in ('table3', 'all'):
        run_table3(args.trials, out, dims=dims)
    if args.exp in ('table4', 'all'):
        run_table4(args.trials, out)
    print(f"total wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
