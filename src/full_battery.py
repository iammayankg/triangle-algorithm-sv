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
import ctypes
import functools
import gc
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
        # binarise on the dataset's own two label values: LIBSVM files use
        # {-1,+1} for most sets but {1,2} for covtype.binary - 'y > 0'
        # would put every covtype point in one class (empty other class)
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f'{name}: expected 2 classes, got {labels}')
        pos = labels.max()
        y = (y == pos).astype(int)
        yt = (yt == pos).astype(int)
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


def run_cell(X, y, Xt, yt, C, trace, liblin_cap=None,
             liblin_child_args=None, time_cap=600.0, solvers=('ETA', 'SMO', 'LIBLIN')):
    """One cell: ETA, our SMO (both with the `time_cap` budget, 600 s in
    the paper; None = unbounded, used for the seed-0 traces of Figure 1)
    and LIBLINEAR (unbounded unless liblin_cap is given, in which case it
    runs in a child process killed after liblin_cap seconds;
    liblin_child_args = (name, C, seed, data_dir, max_n) lets the child
    reload the data).  `solvers` selects which of the three to run."""
    V, W = X[y == 1], X[y == 0]
    out = {}
    if 'ETA' in solvers:
        ta = SoftMarginTA(V, W, C=C, step_mode='block',
                          zigzag_strategy='pairwise', seed=0)
        if trace:
            ta.trace = []
        # same 600 s budget as SMO: non-converging cells (dense-support /
        # near-touching-hull regime) report 'timeout' with their achieved
        # certified gap instead of running to the iteration cap
        r = ta.solve_distance(eps=1e-3, max_iter=2_000_000, time_cap=time_cap)
        w, b = sep_from_soft(ta, r)
        out['ETA'] = dict(time=r.time, iters=r.iterations,
                          oracle=ta.col_evals, primal=2.0 / r.distance ** 2,
                          gap=(r.distance - r.lower_bound) / r.distance,
                          acc=acc(w, b, Xt, yt), status=r.status,
                          trace=ta.trace if trace else None)
        # release ETA's Gram-column cache before SMO builds its own: on
        # covtype each can reach ~45 GB, and holding both at once exceeded
        # a 64 GB machine.  Freed after ETA's result is recorded and before
        # SMO's clock starts, so no timing changes.
        del ta, r
        gc.collect()
        try:   # glibc: hand the freed pages back to the OS as well
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass

    Xs = np.vstack([V, W])
    ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    if 'SMO' in solvers:
        s = SoftMarginSMO(Xs, ys, C_soft=C, tol=1e-3, max_iter=2_000_000,
                          time_cap=time_cap, cache_rows=CACHE_ROWS or len(Xs))
        rs = s.solve()
        out['SMO'] = dict(time=rs.time, iters=rs.iterations,
                          oracle=s.row_evals,
                          primal=2.0 / rs.hull_distance ** 2,
                          acc=acc(rs.w, rs.b, Xt, yt), status=rs.status)

    if 'LIBLIN' in solvers:
        if liblin_cap is None:
            out['LIBLIN'] = fit_liblin(Xs, ys, Xt, yt, C)
        else:
            out['LIBLIN'] = _liblin_capped(liblin_cap, *liblin_child_args)
    return out


CACHE_ROWS: int | None = None      # our SMO's kernel-row cache; None = all rows
LIBLIN_DUAL = 'auto'               # LinearSVC dual= ('auto' = primal Newton here)


def fit_liblin(Xs, ys, Xt, yt, C, tol=1e-6, dual=None):
    """LIBLINEAR (LinearSVC, squared hinge, parameter C/2 so that the
    objective is 1/2||w||^2 + C/2 sum xi^2); returns the LIBLIN record.
    dual='auto' (scikit-learn's default) selects the primal trust-region
    Newton solver whenever n_samples >= n_features, i.e. on every dataset
    of the paper; dual=True runs the dual coordinate-descent solver."""
    dual = LIBLIN_DUAL if dual is None else dual
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=C / 2.0, tol=tol, dual=dual,
                      max_iter=200_000, intercept_scaling=100.0).fit(Xs, ys)
        el = time.perf_counter() - t0
    wv, bv = m.coef_.ravel(), float(m.intercept_[0])
    xi = np.maximum(0.0, 1.0 - ys * (Xs @ wv + bv))
    return dict(time=el, iters=int(np.ravel(m.n_iter_)[0]),
                primal=0.5 * float(wv @ wv) + 0.5 * C * float(xi @ xi),
                acc=acc(wv, bv, Xt, yt), status='converged',
                solver_dual=str(dual))


def _liblin_capped(cap, name, C, seed, data_dir, max_n):
    """Run fit_liblin in a child process with a wall-clock cap.
    LinearSVC trains inside compiled code that cannot be interrupted, so
    the only way to cap it is to kill the process; the child reloads the
    (deterministic) cell data and prints the LIBLIN record as JSON.  On
    timeout the record is status='timeout' with time=cap and no model."""
    import subprocess
    cmd = [sys.executable, str(Path(__file__).resolve()), '--liblin-child',
           name, str(C), str(seed), str(data_dir), str(max_n)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
    except subprocess.TimeoutExpired:
        return dict(time=float(cap), iters=None, primal=float('nan'),
                    acc=float('nan'), status='timeout')
    if p.returncode != 0:
        raise RuntimeError(f'LIBLINEAR child failed:\n{p.stderr[-2000:]}')
    return json.loads(p.stdout.strip().splitlines()[-1])


def _liblin_child(name, C, seed, data_dir, max_n):
    X, y, Xt, yt = _load_cached(name, data_dir, max_n, seed)
    V, W = X[y == 1], X[y == 0]
    Xs = np.vstack([V, W])
    ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    print(json.dumps(fit_liblin(Xs, ys, Xt, yt, C)), flush=True)


def _shard_path(out, name, C, seed):
    return Path(str(out) + '.shards') / f"{name}_{C:g}_{seed}.json"


def _run_one(job):
    """Worker entry: one (dataset, C, seed) cell -> shard file."""
    name, C, seed, data_dir, max_n, out, liblin_cap, time_cap, solvers = job
    shard = _shard_path(out, name, C, seed)
    t0 = time.time()
    X, y, Xt, yt = _load_cached(name, data_dir, max_n, seed)
    cell = run_cell(X, y, Xt, yt, C, trace=(seed == 0),
                    liblin_cap=liblin_cap,
                    liblin_child_args=(name, C, seed, data_dir, max_n),
                    time_cap=time_cap, solvers=tuple(solvers))
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
    ap.add_argument('--liblin-cap', type=float, default=None,
                    help='wall-clock cap in seconds for LIBLINEAR (runs it '
                         'in a killable child process); default unbounded, '
                         'which is what the paper\'s Table 2 shards used')
    ap.add_argument('--liblin-child', nargs=5, metavar='ARG',
                    help=argparse.SUPPRESS)   # internal: name C seed dir max_n
    ap.add_argument('--time-cap', type=float, default=600.0,
                    help='wall-clock budget in seconds for ETA and our SMO '
                         '(paper: 600); 0 = unbounded, for the seed-0 '
                         'traces of Figure 1 (use a separate --out)')
    ap.add_argument('--solvers', nargs='*', default=['ETA', 'SMO', 'LIBLIN'],
                    help='subset of ETA SMO LIBLIN to run per cell')
    ap.add_argument('--cache-rows', type=int, default=0,
                    help='kernel-row cache of our SMO (0 = all rows; the '
                         'batteries of Tables 2 and 5 used the default 512)')
    ap.add_argument('--liblin-dual', default='auto',
                    help="LinearSVC dual=: 'auto' (primal Newton here), "
                         "'true' (dual coordinate descent)")
    args = ap.parse_args()
    global CACHE_ROWS, LIBLIN_DUAL
    CACHE_ROWS = None if args.cache_rows <= 0 else args.cache_rows
    LIBLIN_DUAL = {'auto': 'auto', 'true': True, 'false': False}[args.liblin_dual.lower()]
    time_cap = None if args.time_cap <= 0 else args.time_cap

    if args.liblin_child:
        name, C, seed, data_dir, max_n = args.liblin_child
        _liblin_child(name, float(C), int(seed), data_dir, int(max_n))
        return

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
                                 args.max_n, args.out, args.liblin_cap,
                                 time_cap, args.solvers))
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
