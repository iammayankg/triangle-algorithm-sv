"""n-scaling of the dense-support regime (journal review: the O(n^2)-type
heuristic of Section 6 needs a measurement).  Synthetic L2 overlap
instances (generate_overlap, delta=4, C=1) at d in {100, 1000} and
n in {1000, 2000, 5000, 10000, 20000} points per class, guarded block
transfers k=16, eps=1e-3, one process at a time; records wall-clock,
iterations, columns, support size and the log-log slopes of iterations
and time against n.

Usage:
  OMP_NUM_THREADS=1 python3 src/n_scaling.py --out results/n_scaling.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import generate_overlap                     # noqa: E402
from soft_margin import SoftMarginTA                  # noqa: E402

NS = [1000, 2000, 5000, 10000, 20000]
DS = [100, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/n_scaling.json')
    ap.add_argument('--ns', type=int, nargs='*', default=NS)
    ap.add_argument('--ds', type=int, nargs='*', default=DS)
    ap.add_argument('--time-cap', type=float, default=3600.0)
    a = ap.parse_args()
    out = Path(a.out)
    rows = json.load(open(out)) if out.exists() else []
    done = {(r['d'], r['n']) for r in rows}
    for d in a.ds:
        for n in a.ns:
            if (d, n) in done:
                continue
            V, W = generate_overlap(d, n, delta=4.0, rng=np.random.default_rng(7))
            ta = SoftMarginTA(V, W, C=1.0, step_mode='block', block_size=16,
                              zigzag_strategy='pairwise', seed=0)
            r = ta.solve_distance(eps=1e-3, max_iter=5_000_000,
                                  time_cap=a.time_cap)
            row = dict(d=d, n=n, time=r.time, iters=r.iterations,
                       oracle=ta.col_evals, dist=r.distance, status=r.status,
                       sv=r.sparsity, sv_frac=r.sparsity / (2 * n),
                       n_drops=ta.n_drops,
                       gap=(r.distance - r.lower_bound) / r.distance)
            rows.append(row)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=1))
            print(json.dumps(row), flush=True)
    for d in a.ds:
        rs = sorted([r for r in rows if r['d'] == d and r['status'] == 'converged'],
                    key=lambda r: r['n'])
        if len(rs) >= 2:
            ln = np.log([r['n'] for r in rs])
            si = np.polyfit(ln, np.log([r['iters'] for r in rs]), 1)[0]
            st = np.polyfit(ln, np.log([r['time'] for r in rs]), 1)[0]
            print(f"d={d}: iterations ~ n^{si:.2f}, time ~ n^{st:.2f}, "
                  f"support fraction {min(r['sv_frac'] for r in rs):.2f}-"
                  f"{max(r['sv_frac'] for r in rs):.2f}")


if __name__ == '__main__':
    main()
