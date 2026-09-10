"""Isolated rerun with a timing breakdown (review items 3 and 4): plain
MDM, guarded k=1, block k=16 and BPCG on the two cells of Table 5

  gisette-l2   linear L2 soft margin, C=1 (full data, d=5000)
  ijcnn1-kl2   RBF L2 margin K+I/C, C=1, class-balanced 10k subsample

Every (cell, solver, seed) runs in its own fresh subprocess, one at a
time, so nothing else of this script is on the machine while a run is
timed; check `pgrep -af "src/.*\\.py"` is otherwise empty first.  Per
run it records wall-clock, iterations, columns, drops, distance, gap,
accuracy, peak RSS (ru_maxrss) and, for the ETA solvers, a breakdown of
the solve time into

  scan   time inside _select (the O(n+m) score scan and pair choice)
  column time inside _gram_col_V/_gram_col_W on cache misses
         (Gram or kernel column computation)
  step   time inside _step minus the column time it triggered: block
         assembly (the k' x n copies), the O(k'^2) guard, the line
         search and the cache update
  other  the remainder (bounds, certificate, bookkeeping)

Usage (sequential, ~10 min):
  OMP_NUM_THREADS=1 python3 src/step_profile.py --data-dir data --seeds 3 \
      --out results/step_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EPS, CAP, C = 1e-3, 600.0, 1.0
CELLS = ['gisette-l2', 'ijcnn1-kl2', 'w8a-kl2', 'a9a-kl2', 'covtype-kl2']
SOLVERS = ['mdm', 'guarded1', 'block16', 'auto', 'bpcg']
# added after the 2026-09-10 review: plain MDM and away-step FW with the exact
# receiver search every iteration (prioritized=False), the one-sided
# analogue of what Theorem 6 counts
EXTRA = ['mdm-exact', 'afw-exact', 'guarded1-exact']


class Timers:
    """Wrap an ETA instance's scan, column and step methods with timers."""

    def __init__(self, ta):
        self.ta = ta
        self.scan = self.column = self.step = 0.0
        self._wrap('_select', 'scan')
        self._wrap_col('_gram_col_V')
        self._wrap_col('_gram_col_W')
        self._wrap('_step', 'step')

    def _wrap(self, name, key):
        f = getattr(self.ta, name)

        def g(*a, **kw):
            t = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                setattr(self, key, getattr(self, key) + time.perf_counter() - t)
        setattr(self.ta, name, g)

    def _wrap_col(self, name):
        f = getattr(self.ta, name)

        def g(*a, **kw):
            before = self.ta.col_evals
            t = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                if self.ta.col_evals != before:
                    self.column += time.perf_counter() - t
        setattr(self.ta, name, g)

    def breakdown(self, total):
        # column time triggered inside _step is nested in the step timer
        step = max(self.step - self.column, 0.0)
        return dict(t_scan=self.scan, t_column=self.column, t_step=step,
                    t_other=max(total - self.scan - self.column - step, 0.0))


def run_one(cell, solver, seed, data_dir):
    from bpcg import BPCG
    from full_battery import _load_cached, acc, sep_from_soft
    from regime_battery import _subsample
    from soft_margin import SoftMarginTA
    from kernel_ta import KernelETA

    name, kind = cell.rsplit('-', 1)
    X, y, Xt, yt = _load_cached(name, data_dir, 100_000, seed)
    if kind == 'l2':
        V, W = X[y == 1], X[y == 0]
        kw = dict(C=C)
        Cls, kern = SoftMarginTA, None
    else:
        Xk, yk = _subsample(X, y, 10_000, seed)
        V, W = Xk[yk == 1], Xk[yk == 0]
        gamma = 1.0 / X.shape[1]
        kw = dict(kernel='rbf', gamma=gamma, reg_C=C)
        Cls, kern = KernelETA, gamma
    out = dict(cell=cell, solver=solver, seed=seed)
    if solver == 'bpcg':
        s = BPCG(V, W, kernel='linear' if kind == 'l2' else 'rbf',
                 gamma=kern, reg_C=C)
        r = s.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
        pred = (s.decision_function(Xt) > 0).astype(int)
        out.update(time=r.time, iters=r.iterations, oracle=s.col_evals,
                   dist=r.distance, status=r.status, n_drops=r.n_drops,
                   gap=(r.distance - r.lower_bound) / r.distance,
                   acc=float(np.mean(pred == yt)))
    else:
        mode, k = {'mdm': ('mdm', 1), 'guarded1': ('block', 1),
                   'block16': ('block', 16), 'auto': ('block', 'auto'),
                   'mdm-exact': ('mdm', 1), 'afw-exact': ('away', 1),
                   'guarded1-exact': ('block', 1)}[solver]
        extra = dict(prioritized=False) if solver.endswith('-exact') else {}
        ta = Cls(V, W, step_mode=mode, block_size=k,
                 zigzag_strategy='pairwise', seed=0, **kw, **extra)
        tm = Timers(ta)
        r = ta.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
        if kind == 'l2':
            w, b = sep_from_soft(ta, r)
            a = acc(w, b, Xt, yt)
        else:
            a = float(np.mean((ta.decision_function(Xt, r) > 0).astype(int) == yt))
        out.update(time=r.time, iters=r.iterations, oracle=ta.col_evals,
                   dist=r.distance, status=r.status, n_drops=ta.n_drops,
                   gap=(r.distance - r.lower_bound) / r.distance, acc=a)
        out.update(tm.breakdown(r.time))
        if solver == 'auto':
            hist = ta.k_history
            etas = [e for _, _, e, _ in hist if np.isfinite(e)]
            out.update(k_final=ta.block_size,
                       k_median=float(np.median([kk for _, kk, _, _ in hist])) if hist else float('nan'),
                       eta_hat=float(np.median(etas)) if etas else float('nan'))
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux
    out['peak_rss_gb'] = ru / 1024 ** 3 if sys.platform == 'darwin' else ru / 1024 ** 2
    return out


def summarise(rows):
    from full_battery import ci95
    print(f"{'cell':12s} {'solver':9s} {'time (s)':>14s} {'scan':>6s} {'col':>6s} "
          f"{'step':>6s} {'other':>6s} {'iters':>7s} {'cols':>6s} {'drops':>6s} "
          f"{'peak GB':>8s} status")
    cells = sorted({r['cell'] for r in rows}, key=lambda c: (c not in CELLS, c))
    order = {s: i for i, s in enumerate(SOLVERS)}
    for cell in cells:
        for solver in sorted({r['solver'] for r in rows if r['cell'] == cell},
                             key=lambda s: order.get(s, 99)):
            rs = [r for r in rows if r['cell'] == cell and r['solver'] == solver]
            ts = [r['time'] for r in rs]
            m, h = ci95(ts)
            def mean(key):
                v = [r.get(key) for r in rs if r.get(key) is not None]
                return float(np.mean(v)) if v else float('nan')
            print(f"{cell:12s} {solver:9s} {m:7.1f} ± {h:4.1f} "
                  f"{mean('t_scan'):6.1f} {mean('t_column'):6.1f} "
                  f"{mean('t_step'):6.1f} {mean('t_other'):6.1f} "
                  f"{mean('iters'):7.0f} {mean('oracle'):6.0f} "
                  f"{mean('n_drops'):6.0f} {mean('peak_rss_gb'):8.2f} "
                  f"{','.join(sorted({r['status'] for r in rs}))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--cells', nargs='*', default=CELLS)
    ap.add_argument('--solvers', nargs='*', default=SOLVERS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results/step_profile.json')
    ap.add_argument('--single', nargs=3, metavar=('CELL', 'SOLVER', 'SEED'),
                    help='internal: run one configuration and print JSON')
    a = ap.parse_args()
    if a.single:
        cell, solver, seed = a.single
        print(json.dumps(run_one(cell, solver, int(seed), a.data_dir)))
        return
    out = Path(a.out)
    rows = json.load(open(out)) if out.exists() else []
    done = {(r['cell'], r['solver'], r['seed']) for r in rows}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    for cell in a.cells:
        for solver in a.solvers:
            for seed in range(a.seeds):
                if (cell, solver, seed) in done:
                    continue
                cmd = [sys.executable, str(HERE / 'step_profile.py'), '--single',
                       cell, solver, str(seed), '--data-dir', a.data_dir]
                p = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if p.returncode != 0:
                    print(p.stderr, file=sys.stderr)
                    raise SystemExit(f'{cell} {solver} seed {seed} failed')
                r = json.loads(p.stdout.strip().splitlines()[-1])
                rows.append(r)
                out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(rows, open(out, 'w'), indent=1)
                print(f"{cell:12s} {solver:9s} seed {seed}: {r['time']:.1f} s "
                      f"{r['iters']} it {r['status']} peak {r['peak_rss_gb']:.2f} GB",
                      flush=True)
    summarise(rows)


if __name__ == '__main__':
    main()
