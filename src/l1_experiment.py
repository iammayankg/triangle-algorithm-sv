"""L1 (nu-SVM) study: reduced-hull Triangle Algorithm vs sklearn NuSVC.

Overlapping Gaussian classes (means 4 sigma apart), n = 5000 per class.
Both solvers target the same nu-SVM solution: RCH-TA solves the distance
between reduced hulls with cap mu = 2/(nu l); NuSVC (LIBSVM) solves the
nu-SVC dual.  Agreement is checked through delta = ||w_svc|| / S with S
the per-class sum of NuSVC's rescaled dual coefficients, plus held-out
accuracy of both separators.

Usage: python3 src/l1_experiment.py [--trials 2] [--dims 100 1000]
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

from data import generate_overlap        # noqa: E402
from reduced_hull import ReducedHullTA   # noqa: E402

from sklearn.svm import NuSVC            # noqa: E402

NUS = [0.1, 0.3, 0.5]
DIMS = [100, 1000]
N = 5000


def run_one(d, nu, seed):
    rng = np.random.default_rng(seed)
    Vall, Wall = generate_overlap(d, N + 2000, delta=4.0, rng=rng)
    V, Vt = Vall[:N], Vall[N:]
    W, Wt = Wall[:N], Wall[N:]
    Xt = np.vstack([Vt, Wt])
    yt = np.concatenate([np.ones(2000), -np.ones(2000)])
    out = {}

    ta = ReducedHullTA(V, W, nu=nu)
    r = ta.solve_distance(eps=1e-3, max_iter=200_000)
    w, b = ta.separator(r)
    acc = float(np.mean(np.sign(Xt @ w + b) == yt))
    out['TA'] = dict(time=r.time, dist=r.distance, status=r.status,
                     iters=r.iterations, sv=r.n_support, cap=r.n_at_cap,
                     acc=acc)

    X = np.vstack([V, W])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = NuSVC(nu=nu, kernel='linear', tol=1e-3,
                  max_iter=5_000_000, cache_size=500).fit(X, y)
        el = time.perf_counter() - t0
    wv = m.coef_.ravel()
    dc = m.dual_coef_.ravel()
    S = float(dc[dc > 0].sum())
    dist = float(np.linalg.norm(wv)) / S if S > 0 else float('nan')
    acc = float(np.mean(np.sign(Xt @ wv + m.intercept_[0]) == yt))
    out['NuSVC'] = dict(time=el, dist=dist, status='converged',
                        sv=int(m.n_support_.sum()), acc=acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--dims', type=int, nargs='*', default=DIMS)
    ap.add_argument('--out', default='results/l1_experiment.json')
    args = ap.parse_args()
    rows = []
    for d in args.dims:
        for nu in NUS:
            recs = []
            for t in range(args.trials):
                r = run_one(d, nu, seed=9000 + 31 * t + d + int(nu * 100))
                recs.append(r)
                ta, sv = r['TA'], r['NuSVC']
                rel = abs(ta['dist'] - sv['dist']) / max(sv['dist'], 1e-12)
                print(f"[l1] d={d} nu={nu} trial={t} "
                      f"TA=({ta['time']:.1f}s,d={ta['dist']:.5f},"
                      f"sv={ta['sv']},acc={ta['acc']:.4f},{ta['status']}) "
                      f"NuSVC=({sv['time']:.1f}s,d={sv['dist']:.5f},"
                      f"sv={sv['sv']},acc={sv['acc']:.4f}) rel={rel:.1e}",
                      flush=True)
            row = {'dim': d, 'nu': nu}
            for s in ('TA', 'NuSVC'):
                for kk in ('time', 'dist', 'acc'):
                    row[f'{s}_{kk}'] = float(np.mean([r[s][kk]
                                                      for r in recs]))
                row[f'{s}_sv'] = float(np.mean([r[s]['sv'] for r in recs]))
            row['rel_dist'] = float(np.mean(
                [abs(r['TA']['dist'] - r['NuSVC']['dist'])
                 / max(r['NuSVC']['dist'], 1e-12) for r in recs]))
            rows.append(row)
            Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n{'d':>5} {'nu':>5} | {'TA time':>9} {'NuSVC time':>11} | "
          f"{'rel dist':>9} | {'TA acc':>7} {'NuSVC acc':>9}")
    for row in rows:
        print(f"{row['dim']:>5} {row['nu']:>5} | {row['TA_time']:>8.1f}s "
              f"{row['NuSVC_time']:>10.1f}s | {row['rel_dist']:>9.1e} | "
              f"{row['TA_acc']:>7.4f} {row['NuSVC_acc']:>9.4f}")


if __name__ == '__main__':
    main()
