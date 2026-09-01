"""Full-scale benchmark battery on the standard LIBSVM datasets.

Designed for dedicated hardware (see docs/experiments_protocol.md).
For each dataset x C x seed: L2 soft margin solved by

  ETA      - SoftMarginTA, step_mode='block' (guarded, away fallback)
  SMO      - SoftMarginSMO on K + I/C
  LIBLIN   - sklearn LinearSVC (squared hinge, C/2, tuned settings)

recording wall-clock, iterations, oracle calls (column/row evaluations),
primal objective, certified gap (ETA), test accuracy, and - for the
first seed - the (time, UB, LB) convergence trace. Results are written
incrementally to JSON; a summary table with mean +/- 95% CI over seeds
is printed at the end.

Usage:
  ./scripts/fetch_datasets.sh data
  OMP_NUM_THREADS=<cores> python3 src/full_battery.py \
      --data-dir data --seeds 10 --out results/full_battery.json
Options: --datasets a9a w8a ...   --Cs 0.1 1 10   --max-n 100000
"""

from __future__ import annotations

import argparse
import json
import time
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soft_margin import SoftMarginTA, SoftMarginSMO      # noqa: E402

from sklearn.datasets import load_svmlight_file          # noqa: E402
from sklearn.model_selection import train_test_split     # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402
from sklearn.svm import LinearSVC                        # noqa: E402

# name -> (train file, test file or None, dense_ok)
DATASETS = {
    'a9a':     ('a9a', 'a9a.t', True),
    'w8a':     ('w8a', 'w8a.t', True),
    'ijcnn1':  ('ijcnn1', 'ijcnn1.t', True),
    'gisette': ('gisette_scale', 'gisette_scale.t', True),
    'covtype': ('covtype.libsvm.binary.scale', None, True),
    # sparse-only, pending sparse-matrix support in the solvers:
    # 'real-sim': ('real-sim', None, False),
    # 'rcv1':     ('rcv1_train.binary', None, False),
}


def load(name, data_dir, max_n, seed):
    train, test, dense = DATASETS[name]
    X, y = load_svmlight_file(str(Path(data_dir) / train))
    if test is not None:
        Xt, yt = load_svmlight_file(str(Path(data_dir) / test),
                                    n_features=X.shape[1])
    else:
        X, Xt, y, yt = train_test_split(X, y, test_size=0.25,
                                        stratify=y, random_state=seed)
    if X.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X, y = X[idx], y[idx]
    if not dense:
        raise NotImplementedError('sparse datasets need sparse support')
    X = np.asarray(X.todense(), dtype=np.float64)
    Xt = np.asarray(Xt.todense(), dtype=np.float64)
    y = (y > 0).astype(int)
    yt = (yt > 0).astype(int)
    sc = StandardScaler().fit(X)
    return sc.transform(X), y, sc.transform(Xt), yt


def sep_from_soft(ta, r):
    d2 = r.distance ** 2
    w = 2.0 * (r.p - r.q) / d2
    inv = 1.0 / ta.C
    pp = float(r.p @ r.p) + inv * sum(v * v for v in r.weights_V.values())
    qq = float(r.q @ r.q) + inv * sum(v * v for v in r.weights_W.values())
    return w, (qq - pp) / d2


def acc(w, b, X, y01):
    return float(np.mean((X @ w + b > 0).astype(int) == y01))


def run_cell(X, y, Xt, yt, C, seed, trace):
    V, W = X[y == 1], X[y == 0]
    out = {}

    ta = SoftMarginTA(V, W, C=C, step_mode='block',
                      zigzag_strategy='pairwise', seed=0)
    if trace:
        ta.trace = []
    r = ta.solve_distance(eps=1e-3, max_iter=500_000)
    w, b = sep_from_soft(ta, r)
    out['ETA'] = dict(time=r.time, iters=r.iterations,
                      oracle=ta.col_evals, primal=2.0 / r.distance ** 2,
                      gap=(r.distance - r.lower_bound) / r.distance,
                      acc=acc(w, b, Xt, yt), status=r.status,
                      trace=ta.trace if trace else None)

    Xs = np.vstack([V, W])
    ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    s = SoftMarginSMO(Xs, ys, C_soft=C, tol=1e-3, max_iter=2_000_000,
                      time_cap=600)
    rs = s.solve()
    out['SMO'] = dict(time=rs.time, iters=rs.iterations,
                      oracle=s.row_evals,
                      primal=2.0 / rs.hull_distance ** 2,
                      acc=acc(rs.w, rs.b, Xt, yt), status=rs.status)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=C / 2.0, tol=1e-6,
                      max_iter=200_000, intercept_scaling=100.0).fit(Xs, ys)
        el = time.perf_counter() - t0
    wv, bv = m.coef_.ravel(), float(m.intercept_[0])
    xi = np.maximum(0.0, 1.0 - ys * (Xs @ wv + bv))
    out['LIBLIN'] = dict(time=el, iters=int(np.ravel(m.n_iter_)[0]),
                         primal=0.5 * float(wv @ wv)
                         + 0.5 * C * float(xi @ xi),
                         acc=acc(wv, bv, Xt, yt), status='converged')
    return out


def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    from scipy import stats
    half = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return float(xs.mean()), float(half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
    ap.add_argument('--Cs', type=float, nargs='*', default=[0.1, 1.0, 10.0])
    ap.add_argument('--max-n', type=int, default=100_000)
    ap.add_argument('--out', default='results/full_battery.json')
    args = ap.parse_args()

    all_runs = []
    for name in args.datasets:
        for C in args.Cs:
            for seed in range(args.seeds):
                X, y, Xt, yt = load(name, args.data_dir, args.max_n, seed)
                cell = run_cell(X, y, Xt, yt, C, seed, trace=(seed == 0))
                for solver, rec in cell.items():
                    rec.update(dataset=name, C=C, seed=seed, solver=solver,
                               n=len(X), d=X.shape[1])
                    all_runs.append(rec)
                msg = ' '.join(f"{s}={cell[s]['time']:.1f}s" for s in cell)
                print(f"[{name} C={C} seed={seed}] {msg}", flush=True)
                Path(args.out).write_text(json.dumps(all_runs, indent=1))

    # summary: mean +/- 95% CI over seeds
    print(f"\n{'dataset':>10} {'C':>5} {'solver':>7} | "
          f"{'time (mean±CI95)':>20} | {'acc':>7} | {'oracle':>10}")
    keys = sorted({(r['dataset'], r['C'], r['solver']) for r in all_runs})
    for ds, C, sv in keys:
        sel = [r for r in all_runs
               if (r['dataset'], r['C'], r['solver']) == (ds, C, sv)]
        tm, th = ci95([r['time'] for r in sel])
        am, _ = ci95([r['acc'] for r in sel])
        om = np.mean([r.get('oracle', -1) for r in sel])
        print(f"{ds:>10} {C:>5} {sv:>7} | {tm:9.2f} ± {th:6.2f} s | "
              f"{am:7.4f} | {om:10.0f}")


if __name__ == '__main__':
    main()
