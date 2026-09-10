"""LIBLINEAR time-to-accuracy curve (journal review, C2): LinearSVC on the
L2 soft margin at tolerances 1e-4 ... 1e-8 with the primal trust-region
Newton solver (dual='auto', what the batteries ran) and the dual
coordinate-descent solver (dual=True), on one overlapping dataset and on
gisette; five seeds.  Writes results/liblin_tol_sweep.json in the format
src/time_to_tol.py reads (dataset, C, tol, seed, time, primal, iters,
solver_dual).

Usage:
  OMP_NUM_THREADS=1 python3 src/liblin_sweep.py --data-dir data \
      --datasets a9a gisette --Cs 0.1 1 --seeds 5
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from full_battery import _load_cached, fit_liblin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--datasets', nargs='*', default=['a9a', 'gisette'])
    ap.add_argument('--Cs', type=float, nargs='*', default=[0.1, 1.0])
    ap.add_argument('--tols', type=float, nargs='*', default=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='results/liblin_tol_sweep.json')
    a = ap.parse_args()
    out = Path(a.out)
    rows = json.load(open(out)) if out.exists() else []
    done = {(r['dataset'], r['C'], r['tol'], r['seed'], r['solver_dual']) for r in rows}
    for name in a.datasets:
        for seed in range(a.seeds):
            X, y, Xt, yt = _load_cached(name, a.data_dir, 100_000, seed)
            V, W = X[y == 1], X[y == 0]
            Xs = np.vstack([V, W]); ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
            for C in a.Cs:
                for tol in a.tols:
                    for dual in ('auto', True):
                        key = (name, C, tol, seed, str(dual))
                        if key in done:
                            continue
                        r = fit_liblin(Xs, ys, Xt, yt, C, tol=tol, dual=dual)
                        r.update(dataset=name, C=C, tol=tol, seed=seed)
                        rows.append(r)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(json.dumps(rows, indent=1))
                        print(f"[{name} C={C:g} tol={tol:g} dual={dual} seed={seed}] "
                              f"{r['time']:.1f}s iters={r['iters']} primal={r['primal']:.5g}", flush=True)


if __name__ == '__main__':
    main()
