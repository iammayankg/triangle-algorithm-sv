"""Soft-margin (L2) study: ETA vs SMO vs LIBLINEAR on overlapping data.

Data: two Gaussian classes with means delta = 4 sigma apart (overlapping
distributions at every dimension; Bayes error ~2.3%).  All solvers target
the same L2 soft-margin objective

    P(w, b) = 1/2 ||w||^2 + C/2 sum_i xi_i^2,

whose optimal value equals 2 / delta_C^2 for the augmented hull distance
delta_C.  Compared:

  ETA-mdm  - SoftMarginTA, MDM step mode (pairwise transfers primary)
  ETA-twd  - SoftMarginTA, paper-style toward steps (pairwise remedy)
  SMO      - SoftMarginSMO on the augmented kernel K + I/C
  LIBLIN   - sklearn LinearSVC(loss='squared_hinge', C_sk = C/2), primal CD

Reported per solver: time, primal objective (from 2/delta^2 for the dual
solvers, and from (w, b) directly for LIBLINEAR), and held-out accuracy.

Usage: python3 src/soft_experiment.py [--trials 2] [--dims 100 1000]
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

from data import generate_overlap                 # noqa: E402
from soft_margin import SoftMarginTA, SoftMarginSMO  # noqa: E402

from sklearn.svm import LinearSVC                 # noqa: E402

CS = [0.1, 1.0, 10.0]
DIMS = [100, 1000]
EPS = 1e-3
N = 5000
DELTA = 4.0


def _ta_separator(ta_solver, r):
    """(w, b) of the induced separator from the witness pair, using the
    augmented norms (slack coordinates do not affect test predictions)."""
    d2 = r.distance ** 2
    w = 2.0 * (r.p - r.q) / d2
    invC = 1.0 / ta_solver.C
    pp_a = float(r.p @ r.p) + invC * sum(v * v for v in r.weights_V.values())
    qq_a = float(r.q @ r.q) + invC * sum(v * v for v in r.weights_W.values())
    b = (qq_a - pp_a) / d2
    return w, b


def _primal_from_wb(w, b, X, y, C):
    xi = np.maximum(0.0, 1.0 - y * (X @ w + b))
    return 0.5 * float(w @ w) + 0.5 * C * float(xi @ xi)


def _accuracy(w, b, Xt, yt):
    return float(np.mean(np.sign(Xt @ w + b) == yt))


def run_one(d, C, seed):
    rng = np.random.default_rng(seed)
    # one draw, split into train and held-out so both share the same
    # class-mean direction
    Vall, Wall = generate_overlap(d, N + 2000, delta=DELTA, rng=rng)
    V, Vt = Vall[:N], Vall[N:]
    W, Wt = Wall[:N], Wall[N:]
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    Xt = np.vstack([Vt, Wt])
    yt = np.concatenate([np.ones(2000), -np.ones(2000)])
    out = {}

    for mode, tag in (('mdm', 'ETA-mdm'), ('toward', 'ETA-twd')):
        ta = SoftMarginTA(V, W, C=C, step_mode=mode,
                          zigzag_strategy='pairwise', seed=0)
        t0 = time.perf_counter()
        r = ta.solve_distance(eps=EPS, max_iter=150_000)
        el = time.perf_counter() - t0
        w, b = _ta_separator(ta, r)
        out[tag] = dict(time=el, primal=2.0 / r.distance ** 2,
                        primal_lb=2.0 / max(r.lower_bound, 1e-12) ** 2,
                        status=r.status, iters=r.iterations,
                        sparsity=r.sparsity, acc=_accuracy(w, b, Xt, yt))

    t0 = time.perf_counter()
    s = SoftMarginSMO(X, y, C_soft=C, tol=EPS, max_iter=500_000,
                      time_cap=180).solve()
    el = time.perf_counter() - t0
    out['SMO'] = dict(time=el, primal=2.0 / s.hull_distance ** 2,
                      status=s.status, iters=s.iterations,
                      sparsity=s.sparsity,
                      acc=_accuracy(s.w, s.b, Xt, yt))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=C / 2.0, tol=1e-6,
                      max_iter=100_000, intercept_scaling=100.0)
        m.fit(X, y)
        el = time.perf_counter() - t0
        w = m.coef_.ravel()
        b = float(m.intercept_[0]) * 1.0
        out['LIBLIN'] = dict(time=el, primal=_primal_from_wb(w, b, X, y, C),
                             status='converged', iters=int(m.n_iter_),
                             sparsity=int(np.sum(
                                 1.0 - y * (X @ w + b) > 1e-6)),
                             acc=_accuracy(w, b, Xt, yt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--dims', type=int, nargs='*', default=DIMS)
    ap.add_argument('--out', default='results/soft_experiment.json')
    args = ap.parse_args()
    solvers = ['ETA-mdm', 'ETA-twd', 'SMO', 'LIBLIN']
    rows = []
    for d in args.dims:
        for C in CS:
            recs = []
            for t in range(args.trials):
                r = run_one(d, C, seed=7000 + 31 * t + d + int(C * 10))
                recs.append(r)
                print(f"[soft] d={d} C={C} trial={t} " + " ".join(
                    f"{s}=({r[s]['time']:.1f}s,P={r[s]['primal']:.4f},"
                    f"acc={r[s]['acc']:.4f},{r[s]['status']})"
                    for s in solvers), flush=True)
            row = {'dim': d, 'C': C}
            for s in solvers:
                for kk in ('time', 'primal', 'acc'):
                    row[f'{s}_{kk}'] = float(np.mean([r[s][kk]
                                                      for r in recs]))
                row[f'{s}_status'] = recs[0][s]['status']
                row[f'{s}_sparsity'] = float(np.mean([r[s]['sparsity']
                                                      for r in recs]))
            rows.append(row)
            Path(args.out).write_text(json.dumps(rows, indent=2))
    # summary
    print(f"\n{'d':>5} {'C':>6} | " + " | ".join(f"{s:>26}" for s in solvers))
    for row in rows:
        cells = [f"{row[f'{s}_time']:6.1f}s P={row[f'{s}_primal']:8.4f} "
                 f"a={row[f'{s}_acc']:.3f}" for s in solvers]
        print(f"{row['dim']:>5} {row['C']:>6} | " +
              " | ".join(f"{c:>26}" for c in cells))


if __name__ == '__main__':
    main()
