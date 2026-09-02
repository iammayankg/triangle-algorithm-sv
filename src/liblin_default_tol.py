"""LIBLINEAR on gisette at its default tolerance (review item 2).

Table 2 runs LinearSVC at tol=1e-6 so that its primal matches ETA's
certified value. This script reruns the same LinearSVC configuration at
scikit-learn's default tol=1e-4 (and, for reference, at 1e-6 again) on
the same gisette cells, recording wall-clock, iterations, the primal
objective 1/2||w||^2 + C/2 sum xi^2, and test accuracy.

Usage (on the machine that holds data/):
  OMP_NUM_THREADS=1 python3 src/liblin_default_tol.py --data-dir data \
      --seeds 5 --out results/liblin_default_tol.json
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

from full_battery import _load_cached, acc, ci95          # noqa: E402
from sklearn.svm import LinearSVC                          # noqa: E402


def run(X, y, Xt, yt, C, tol):
    ys = np.where(y == 1, 1.0, -1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=C / 2.0, tol=tol,
                      max_iter=200_000, intercept_scaling=100.0).fit(X, ys)
        el = time.perf_counter() - t0
    w, b = m.coef_.ravel(), float(m.intercept_[0])
    xi = np.maximum(0.0, 1.0 - ys * (X @ w + b))
    return dict(time=el, iters=int(np.ravel(m.n_iter_)[0]),
                primal=0.5 * float(w @ w) + 0.5 * C * float(xi @ xi),
                acc=acc(w, b, Xt, yt), tol=tol, C=C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--datasets', nargs='*', default=['gisette'])
    ap.add_argument('--Cs', type=float, nargs='*', default=[0.1, 1.0, 10.0])
    ap.add_argument('--tols', type=float, nargs='*', default=[1e-4, 1e-6])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='results/liblin_default_tol.json')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # resume: cells already in the output file are skipped
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['dataset'], r['C'], r['tol'], r['seed']) for r in rows}
    if done:
        print(f"{len(done)} cells already done in {out}", flush=True)
    for name in args.datasets:
        for seed in range(args.seeds):
            todo = [(C, tol) for C in args.Cs for tol in args.tols
                    if (name, C, tol, seed) not in done]
            if not todo:
                continue
            X, y, Xt, yt = _load_cached(name, args.data_dir, 100_000, seed)
            for C, tol in todo:
                r = run(X, y, Xt, yt, C, tol)
                r.update(dataset=name, seed=seed)
                rows.append(r)
                out.write_text(json.dumps(rows, indent=1))
                print(f"[{name} C={C:g} tol={tol:g} seed={seed}] "
                      f"LIBLIN={r['time']:.1f}s iters={r['iters']} "
                      f"primal={r['primal']:.6g} acc={r['acc']:.4f}",
                      flush=True)

    print('\n| dataset | C | tol | time (s) | iters | primal | accuracy |')
    print('|--|--:|--:|--:|--:|--:|--:|')
    for name in args.datasets:
        for C in args.Cs:
            for tol in args.tols:
                sel = [r for r in rows if (r['dataset'], r['C'], r['tol'])
                       == (name, C, tol)]
                tm, th = ci95([r['time'] for r in sel])
                print(f"| {name} | {C:g} | {tol:g} | {tm:.1f} ± {th:.1f} | "
                      f"{np.mean([r['iters'] for r in sel]):,.0f} | "
                      f"{np.mean([r['primal'] for r in sel]):.5f} | "
                      f"{np.mean([r['acc'] for r in sel]):.4f} |")


if __name__ == '__main__':
    main()
