"""Full-scale benchmark battery on the standard LIBSVM datasets.

Designed for dedicated hardware (see docs/experiments_protocol.md).
For each dataset x C x seed: L2 soft margin solved by

  ETA      - SoftMarginTA, step_mode='block' (guarded, away fallback)
  SMO      - SoftMarginSMO on K + I/C
  LIBLIN   - sklearn LinearSVC (squared hinge, C/2, tuned settings)

recording wall-clock, iterations, oracle calls (column/row evaluations),
primal objective, certified gap (ETA), test accuracy, and - for seed 0 -
the (time, UB, LB) convergence trace.

Each finished cell is written to its own shard
(<out>.shards/<dataset>_<C>_<seed>.json); rerunning skips existing
shards, so interrupted runs resume for free. `--parallel N` runs N cells
concurrently (set OMP_NUM_THREADS=1 so workers do not fight over BLAS
threads). The combined JSON and a mean +/- 95% CI summary are produced
at the end (or by src/battery_analysis.py at any time).

Usage:
  ./scripts/fetch_datasets.sh data
  OMP_NUM_THREADS=1 python3 src/full_battery.py --data-dir data \
      --seeds 10 --parallel 12 --out results/full_battery.json
Smoke test (no downloads): --datasets synthetic --seeds 2 --parallel 2
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from soft_margin import SoftMarginTA, SoftMarginSMO      # noqa: E402

from sklearn.model_selection import train_test_split     # noqa: E402
from sklearn.preprocessing import StandardScaler         # noqa: E402
from sklearn.svm import LinearSVC                        # noqa: E402

# name -> (train file, test file or None, dense_ok)
DATASETS = {
    'a9a':       ('a9a', 'a9a.t', True),
    'w8a':       ('w8a', 'w8a.t', True),
    'ijcnn1':    ('ijcnn1', 'ijcnn1.t', True),
    'gisette':   ('gisette_scale', 'gisette_scale.t', True),
    'covtype':   ('covtype.libsvm.binary.scale', None, True),
    'synthetic': (None, None, True),      # generated; for smoke tests
    'synthetic-sep': (None, None, True),  # generated, separable (smoke)
    # sparse-only, pending sparse-matrix support in the solvers:
    # 'real-sim': ('real-sim', None, False),
    # 'rcv1':     ('rcv1_train.binary', None, False),
}


@functools.lru_cache(maxsize=2)
def _load_cached(name, data_dir, max_n, seed):
    if name in ('synthetic', 'synthetic-sep'):
        from data import generate_overlap, generate_two_balls
        rng = np.random.default_rng(seed)
        if name == 'synthetic':
            V, W = generate_overlap(50, 1200, delta=4.0, rng=rng)
        else:
            V, W = generate_two_balls(50, 1200, 1.2, rng=rng)
        X = np.vstack([V, W])
        y = np.concatenate([np.ones(1200, int), np.zeros(1200, int)])
        X, Xt, y, yt = train_test_split(X, y, test_size=0.25,
                                        stratify=y, random_state=seed)
    else:
        from sklearn.datasets import load_svmlight_file
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


def run_cell(X, y, Xt, yt, C, trace):
    V, W = X[y == 1], X[y == 0]
    out = {}

    ta = SoftMarginTA(V, W, C=C, step_mode='block',
                      zigzag_strategy='pairwise', seed=0)
    if trace:
        ta.trace = []
    # same 600 s budget as SMO: non-converging cells (dense-support /
    # near-touching-hull regime) report 'timeout' with their achieved
    # certified gap instead of running to the iteration cap
    r = ta.solve_distance(eps=1e-3, max_iter=2_000_000, time_cap=600)
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


def _shard_path(out, name, C, seed):
    return Path(str(out) + '.shards') / f"{name}_{C:g}_{seed}.json"


def _run_one(job):
    """Worker entry: one (dataset, C, seed) cell -> shard file."""
    name, C, seed, data_dir, max_n, out = job
    shard = _shard_path(out, name, C, seed)
    t0 = time.time()
    X, y, Xt, yt = _load_cached(name, data_dir, max_n, seed)
    cell = run_cell(X, y, Xt, yt, C, trace=(seed == 0))
    recs = []
    for solver, rec in cell.items():
        rec.update(dataset=name, C=C, seed=seed, solver=solver,
                   n=len(X), d=X.shape[1])
        recs.append(rec)
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(recs))
    def _tag(rec):
        st = rec.get('status', '')
        flag = '' if st == 'converged' else f"/{st}"
        gap = f" gap={rec['gap']:.1e}" if 'gap' in rec and st != 'converged' \
            else ''
        return f"{rec['time']:.1f}s{flag}{gap}"
    msg = ' '.join(f"{s}={_tag(cell[s])}" for s in cell)
    print(f"[{name} C={C:g} seed={seed}] {msg} "
          f"(cell {time.time() - t0:.0f}s)", flush=True)
    return recs


def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    from scipy import stats
    half = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return float(xs.mean()), float(half)


def summarize(all_runs):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--datasets', nargs='*',
                    default=[d for d in DATASETS
                             if not d.startswith('synthetic')])
    ap.add_argument('--Cs', type=float, nargs='*', default=[0.1, 1.0, 10.0])
    ap.add_argument('--max-n', type=int, default=100_000)
    ap.add_argument('--parallel', type=int, default=1,
                    help='concurrent cells; set OMP_NUM_THREADS=1')
    ap.add_argument('--out', default='results/full_battery.json')
    args = ap.parse_args()

    if args.parallel > 1 and os.environ.get('OMP_NUM_THREADS') != '1':
        print('WARNING: set OMP_NUM_THREADS=1 when using --parallel '
              '(workers will oversubscribe BLAS threads otherwise)')

    # cells grouped so a worker reuses its cached dataset across C values
    jobs, done = [], []
    for name in args.datasets:
        for seed in range(args.seeds):
            for C in args.Cs:
                shard = _shard_path(args.out, name, C, seed)
                if shard.exists():
                    done.append(json.loads(shard.read_text()))
                else:
                    jobs.append((name, C, seed, args.data_dir,
                                 args.max_n, args.out))
    print(f"{len(done)} cells already done (shards), {len(jobs)} to run, "
          f"parallel={args.parallel}", flush=True)

    results = [r for shard in done for r in shard]
    if args.parallel > 1:
        with Pool(args.parallel) as pool:
            for recs in pool.imap_unordered(_run_one, jobs,
                                            chunksize=len(args.Cs)):
                results.extend(recs)
    else:
        for job in jobs:
            results.extend(_run_one(job))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    summarize(results)


if __name__ == '__main__':
    main()
