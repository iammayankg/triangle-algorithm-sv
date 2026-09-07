"""Adaptive block size versus fixed k on a synthetic sweep in the
directional diversity eta (journal step 5).  eta is controlled through the
dimension of the L2-overlap instances: low d gives correlated transfer
directions (large eta), high d nearly orthogonal ones.

For each d: guarded k = 1, fixed k = 4, 16, 32 and block_size='auto'
(k_init 16, cost_ratio 20), one process, sequential.  Reports time,
iterations, columns, drops, the eta the auto rule measured and the k it
settled on, next to the model optimum k* = sqrt(cost_ratio (1-eta)/eta).

Usage:  python3 src/adaptive_sweep.py --dims 20 50 100 300 1000 --n 2000 \
            --out results/adaptive_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import generate_overlap          # noqa: E402
from soft_margin import SoftMarginTA       # noqa: E402

CONFIGS = [('block', 1), ('block', 4), ('block', 16), ('block', 32), ('block', 'auto')]


def run(V, W, k, cap, cost_ratio):
    ta = SoftMarginTA(V, W, C=1.0, step_mode='block', block_size=k,
                      zigzag_strategy='pairwise', seed=0, cost_ratio=cost_ratio)
    r = ta.solve_distance(eps=1e-3, max_iter=5_000_000, time_cap=cap)
    out = dict(k=k, time=r.time, iters=r.iterations, status=r.status,
               dist=r.distance, cols=ta.col_evals, drops=ta.n_drops)
    if k == 'auto':
        hist = ta.k_history
        etas = [e for _, _, e, _ in hist if np.isfinite(e)]
        out.update(k_final=ta.block_size,
                   k_median=float(np.median([kk for _, kk, _, _ in hist])) if hist else np.nan,
                   eta_hat=float(np.median(etas)) if etas else np.nan,
                   k_history=[(it, kk, (None if not np.isfinite(e) else round(e, 4)), round(f, 3))
                              for it, kk, e, f in hist])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', nargs='*', type=int, default=[20, 50, 100, 300, 1000])
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--delta', type=float, default=4.0)
    ap.add_argument('--cap', type=float, default=180.0)
    ap.add_argument('--cost-ratio', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/adaptive_sweep.json')
    a = ap.parse_args()
    res = []
    for d in a.dims:
        V, W = generate_overlap(d, a.n, delta=a.delta, rng=np.random.default_rng(a.seed))
        rows = []
        for _, k in CONFIGS:
            rows.append(run(V, W, k, a.cap, a.cost_ratio))
            r = rows[-1]
            extra = (f" k_final {r['k_final']} k_med {r['k_median']:.0f} eta_hat {r['eta_hat']:.3f}"
                     if k == 'auto' else '')
            print(f"d={d:5d} k={str(k):>4s}: {r['time']:7.1f} s {r['iters']:7d} it "
                  f"{r['cols']:5d} cols {r['drops']:5d} drops {r['status']}{extra}", flush=True)
        auto = rows[-1]
        eta = auto['eta_hat']
        kstar = np.sqrt(a.cost_ratio * (1 - eta) / eta) if np.isfinite(eta) and eta > 0 else np.nan
        best = min(rows[:-1], key=lambda r: r['time'])
        print(f"   -> eta_hat {eta:.3f}, model k* {kstar:.1f}; best fixed k = {best['k']} "
              f"({best['time']:.1f} s); auto {auto['time']:.1f} s", flush=True)
        res.append(dict(d=d, n=a.n, rows=rows, eta_hat=eta, kstar=kstar))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
