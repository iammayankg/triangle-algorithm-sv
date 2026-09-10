"""Regime battery: real-data cells in the Triangle Algorithm's own regime.

Companion to full_battery.py (L2 soft margin, where dense-support data
favours primal solvers). This battery covers the sparse-support regime
the theory identifies as ETA's: hard margins and kernels. Every
comparison is against a solver of the SAME objective, so times are
comparable; a differently-regularised standard solver is added only as
a labelled accuracy/time reference.

Per dataset x seed:

  FEAS   Triangle Algorithm I on the training hulls: linearly separable?
         (a certificate the baselines cannot produce; decides cell LIN)
  LIN    linear hard margin (only if FEAS = separated):
           ETA (block, guarded, shrink) vs LIBSVM SVC(linear, C=1e6)
           vs our SMO - distance agreement via 2/||w||
  KHM    RBF hard margin in feature space (kernel subsample):
           KernelETA (block, guarded) vs SVC(rbf, C=1e6) - agreement via
           ||w_H||^2 = (alpha y)' K (alpha y)
  KL2    RBF L2 soft margin K + I/C, C=1 (kernel subsample):
           KernelETA(reg_C) vs KernelSMO(reg_C) [same objective];
           SVC(rbf, C=1) as L1 reference (accuracy/time only)

All solvers carry a 600 s budget; timeouts report their achieved gap.
Kernel cells run on a class-balanced subsample (--max-n-kernel, default
10,000 points) because O(n^2) kernel methods are the baseline there.

Usage:
  OMP_NUM_THREADS=1 python3 src/regime_battery.py --data-dir data \
      --seeds 5 --parallel 8 --out results/regime_battery.json
  python3 src/regime_battery.py --datasets synthetic --seeds 2   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_battery import _load_cached, DATASETS, ci95, acc   # noqa: E402
from kernel_ta import KernelETA, KernelSMO                    # noqa: E402
from smo import SMO                                           # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm      # noqa: E402

from sklearn.metrics.pairwise import rbf_kernel               # noqa: E402
from sklearn.svm import SVC                                   # noqa: E402

CAP = 600.0
EPS = 1e-3


def _gap(r):
    """Certified relative gap; inf when the distance is zero (the hulls
    intersect - e.g. identical points with opposite labels, which makes a
    hard margin infeasible even in RBF feature space)."""
    return (r.distance - r.lower_bound) / r.distance if r.distance > 0 \
        else float('inf')


def _subsample(X, y, max_n, seed):
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx1 = np.flatnonzero(y == 1)
    idx0 = np.flatnonzero(y == 0)
    k = max_n // 2
    sel = np.concatenate([rng.choice(idx1, min(k, len(idx1)), replace=False),
                          rng.choice(idx0, min(k, len(idx0)), replace=False)])
    return X[sel], y[sel]


def _svc(X, y, Xt, yt, **kw):
    """Fit sklearn SVC with a timing; returns (model, elapsed, status)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = SVC(tol=EPS, max_iter=2_000_000, cache_size=1000, **kw).fit(X, y)
        el = time.perf_counter() - t0
    status = 'converged' if getattr(m, 'fit_status_', 0) == 0 else 'maxiter'
    return m, el, status


# kernel-row cache of our SMO baselines: the paper's Table 2 ran with the
# solver default (512 rows, LRU), so its row counts include re-computed
# rows; CACHE_ROWS=None (the default now) caches every row, like ETA
CACHE_ROWS: int | None = None


def _cache_rows(n):
    return n if CACHE_ROWS is None else int(CACHE_ROWS)


def cell_feas(V, W):
    ta = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise', seed=0)
    r = ta.solve_intersection(eps=EPS, max_iter=50_000, time_cap=CAP)
    return dict(status=r.status, time=r.time, iters=r.iterations,
                dist=r.distance)


def cell_lin(V, W, Xt, yt):
    out = {}
    ta = EnhancedTriangleAlgorithm(V, W, step_mode='block',
                                   zigzag_strategy='pairwise', shrink=True,
                                   shrink_every=25, seed=0)
    r = ta.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    if r.distance > 0:
        d2 = r.distance ** 2
        w = 2.0 * (r.p - r.q) / d2
        b = (float(r.q @ r.q) - float(r.p @ r.p)) / d2
        a_eta = acc(w, b, Xt, yt)
    else:
        a_eta = float('nan')
    out['ETA'] = dict(time=r.time, iters=r.iterations, oracle=ta.col_evals,
                      dist=r.distance, gap=_gap(r),
                      sv=r.sparsity, acc=a_eta, status=r.status,
                      alive=ta.n + ta.m)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    s = SMO(X, y, C=1e12, tol=EPS, max_iter=2_000_000, time_cap=CAP,
            cache_rows=_cache_rows(len(X)))
    rs = s.solve()
    out['SMO'] = dict(time=rs.time, iters=rs.iterations, oracle=s.row_evals,
                      dist=rs.hull_distance, sv=rs.sparsity,
                      acc=acc(rs.w, rs.b, Xt, yt), status=rs.status)
    m, el, st = _svc(X, y, Xt, yt, kernel='linear', C=1e6)
    wv = m.coef_.ravel()
    out['LIBSVM'] = dict(time=el, dist=2.0 / float(np.linalg.norm(wv)),
                         sv=int(m.n_support_.sum()),
                         acc=acc(wv, float(m.intercept_[0]), Xt, yt),
                         status=st)
    return out


def cell_khm(V, W, Xt, yt, gamma):
    out = {}
    kt = KernelETA(V, W, kernel='rbf', gamma=gamma, step_mode='block',
                   zigzag_strategy='pairwise', seed=0)
    r = kt.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    if r.status == 'intersect' or r.distance <= 1e-12:
        # feature-space hulls intersect (conflicting duplicate points):
        # no hard margin exists; report and skip the infeasible baseline
        out['K-ETA'] = dict(time=r.time, iters=r.iterations,
                            oracle=kt.col_evals, dist=0.0, gap=float('inf'),
                            sv=r.sparsity, acc=float('nan'),
                            status='intersect')
        return out
    pred = (kt.decision_function(Xt, r) > 0).astype(int)
    out['K-ETA'] = dict(time=r.time, iters=r.iterations, oracle=kt.col_evals,
                        dist=r.distance,
                        gap=_gap(r),
                        sv=r.sparsity, acc=float(np.mean(pred == yt)),
                        status=r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    m, el, st = _svc(X, y, Xt, yt, kernel='rbf', gamma=gamma, C=1e6)
    sv = m.support_
    ay = m.dual_coef_.ravel()
    K = rbf_kernel(X[sv], X[sv], gamma=gamma)
    wn2 = float(ay @ K @ ay)
    pred = (m.decision_function(Xt) > 0).astype(int)
    out['LIBSVM'] = dict(time=el, dist=2.0 / np.sqrt(wn2) if wn2 > 0 else
                         float('inf'), sv=int(len(sv)),
                         acc=float(np.mean(pred == yt)), status=st)
    return out


def cell_kl2(V, W, Xt, yt, gamma, C=1.0):
    out = {}
    kt = KernelETA(V, W, kernel='rbf', gamma=gamma, reg_C=C,
                   step_mode='block', zigzag_strategy='pairwise', seed=0)
    r = kt.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    pred = (kt.decision_function(Xt, r) > 0).astype(int)
    out['K-ETA'] = dict(time=r.time, iters=r.iterations, oracle=kt.col_evals,
                        dist=r.distance,
                        gap=_gap(r),
                        sv=r.sparsity, acc=float(np.mean(pred == yt)),
                        status=r.status)
    X = np.vstack([V, W])
    y = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    s = KernelSMO(X, y, kernel='rbf', gamma=gamma, reg_C=C, tol=EPS,
                  max_iter=2_000_000, time_cap=CAP, cache_rows=_cache_rows(len(X)))
    rs = s.solve()
    # kernel SMO classifier: dual expansion with a kernel-space intercept
    # (the base SMO's b is computed from a linear w - meaningless here).
    # KKT at a support vector i in the augmented space K + I/C:
    #   y_i ( sum_j ay_j K(x_j, x_i) + ay_i / C + b ) = 1
    ay = rs.alpha * y
    sv = np.flatnonzero(rs.alpha > 1e-8 * max(rs.alpha.max(), 1e-300))
    if len(sv):
        Ks = rbf_kernel(X[sv], X, gamma=gamma)
        b = float(np.mean(y[sv] - Ks @ ay - ay[sv] / C))
    else:
        b = 0.0
    Kt = rbf_kernel(Xt, X, gamma=gamma)
    pred = ((Kt @ ay + b) > 0).astype(int)
    out['K-SMO'] = dict(time=rs.time, iters=rs.iterations,
                        oracle=s.row_evals, dist=rs.hull_distance,
                        sv=rs.sparsity, acc=float(np.mean(pred == yt)),
                        status=rs.status)
    m, el, st = _svc(X, y, Xt, yt, kernel='rbf', gamma=gamma, C=C)
    pred = (m.decision_function(Xt) > 0).astype(int)
    out['SVC-L1-ref'] = dict(time=el, sv=int(m.n_support_.sum()),
                             acc=float(np.mean(pred == yt)), status=st)
    return out


def run_cells(name, data_dir, max_n, max_n_kernel, seed):
    X, y, Xt, yt = _load_cached(name, data_dir, max_n, seed)
    recs = []

    def emit(cell, solver, rec):
        rec = dict(rec)
        rec.update(dataset=name, seed=seed, cell=cell, solver=solver,
                   n=len(X), d=X.shape[1])
        recs.append(rec)

    V, W = X[y == 1], X[y == 0]
    f = cell_feas(V, W)
    emit('FEAS', 'TA-I', f)
    if f['status'] == 'separated':
        for sv, rec in cell_lin(V, W, Xt, yt).items():
            emit('LIN', sv, rec)

    Xk, yk = _subsample(X, y, max_n_kernel, seed)
    Vk, Wk = Xk[yk == 1], Xk[yk == 0]
    gamma = 1.0 / X.shape[1]
    for sv, rec in cell_khm(Vk, Wk, Xt, yt, gamma).items():
        rec['n_kernel'] = len(Xk)
        emit('KHM', sv, rec)
    for sv, rec in cell_kl2(Vk, Wk, Xt, yt, gamma).items():
        rec['n_kernel'] = len(Xk)
        emit('KL2', sv, rec)
    return recs


def _shard(out, name, seed):
    return Path(str(out) + '.shards') / f"{name}_{seed}.json"


def _run_one(job):
    name, seed, data_dir, max_n, max_n_kernel, out = job
    t0 = time.time()
    recs = run_cells(name, data_dir, max_n, max_n_kernel, seed)
    sh = _shard(out, name, seed)
    sh.parent.mkdir(parents=True, exist_ok=True)
    sh.write_text(json.dumps(recs))
    parts = []
    for r in recs:
        st = '' if r['status'] in ('converged', 'separated') else \
            f"/{r['status']}"
        parts.append(f"{r['cell']}:{r['solver']}={r['time']:.0f}s{st}")
    print(f"[{name} seed={seed}] " + ' '.join(parts) +
          f" (total {time.time() - t0:.0f}s)", flush=True)
    return recs


def summarize(runs, path):
    keys = sorted({(r['dataset'], r['cell'], r['solver']) for r in runs})
    lines = ['# Regime battery summary', '',
             '| dataset | cell | solver | time (s) | dist | accuracy | '
             'oracle | non-converged |', '|--|--|--|--:|--:|--:|--:|--:|']
    for d, c, s in keys:
        sel = [r for r in runs if (r['dataset'], r['cell'], r['solver'])
               == (d, c, s)]
        tm, th = ci95([r['time'] for r in sel])
        dist = np.nanmean([r['dist'] for r in sel if 'dist' in r]) \
            if any('dist' in r for r in sel) else float('nan')
        am = np.nanmean([r['acc'] for r in sel if 'acc' in r]) \
            if any('acc' in r for r in sel) else float('nan')
        om = np.mean([r['oracle'] for r in sel if 'oracle' in r]) \
            if any('oracle' in r for r in sel) else float('nan')
        bad = sum(r['status'] not in ('converged', 'separated',
                                      'intersect') for r in sel)
        lines.append(
            f"| {d} | {c} | {s} | {tm:.1f} ± {th:.1f} | "
            f"{'-' if np.isnan(dist) else f'{dist:.5f}'} | "
            f"{'-' if np.isnan(am) else f'{am:.4f}'} | "
            f"{'-' if np.isnan(om) else f'{om:,.0f}'} | {bad}/{len(sel)} |")
    for cell, title in (('FEAS', 'Linear hard-margin feasibility (TA I)'),
                        ('KHM', 'RBF feature-space hard-margin feasibility '
                                '(K-ETA; intersect = conflicting duplicates)')):
        sub = [r for r in runs if r['cell'] == cell
               and (cell != 'KHM' or r['solver'] == 'K-ETA')]
        if sub:
            lines += ['', f'## {title}', '']
            for d in sorted({r['dataset'] for r in sub}):
                st = [r['status'] for r in sub if r['dataset'] == d]
                lines.append(f"- {d}: " + ', '.join(
                    f"{s}={st.count(s)}" for s in sorted(set(st))))
    Path(path).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nwrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--datasets', nargs='*',
                    default=[d for d in DATASETS
                             if not d.startswith('synthetic')])
    ap.add_argument('--max-n', type=int, default=100_000)
    ap.add_argument('--max-n-kernel', type=int, default=10_000)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--out', default='results/regime_battery.json')
    ap.add_argument('--cache-rows', type=int, default=0,
                    help='kernel-row cache of our SMO (0 = all rows, no eviction; '
                         'the workshop battery used the solver default 512)')
    args = ap.parse_args()
    global CACHE_ROWS
    CACHE_ROWS = None if args.cache_rows <= 0 else args.cache_rows
    if args.parallel > 1 and os.environ.get('OMP_NUM_THREADS') != '1':
        print('WARNING: set OMP_NUM_THREADS=1 when using --parallel')
    jobs, done = [], []
    for name in args.datasets:
        for seed in range(args.seeds):
            sh = _shard(args.out, name, seed)
            if sh.exists():
                done.append(json.loads(sh.read_text()))
            else:
                jobs.append((name, seed, args.data_dir, args.max_n,
                             args.max_n_kernel, args.out))
    print(f"{len(done)} done (shards), {len(jobs)} to run, "
          f"parallel={args.parallel}", flush=True)
    runs = [r for sh in done for r in sh]
    if args.parallel > 1:
        with Pool(args.parallel) as pool:
            for recs in pool.imap_unordered(_run_one, jobs):
                runs.extend(recs)
    else:
        for job in jobs:
            runs.extend(_run_one(job))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(runs, indent=1))
    summarize(runs, Path(args.out).parent / 'regime_battery_summary.md')


if __name__ == '__main__':
    main()
