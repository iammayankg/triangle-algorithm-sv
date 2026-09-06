"""LIBSVM on the kernel L2 margin: the stronger baseline for the KL2
cells of Table 1.  The L2 soft margin with RBF kernel is the hard margin
on the augmented kernel K + I/C, so SVC(kernel='precomputed') on that
matrix at a large C (the same C=1e6 surrogate the LIN and KHM rows use)
solves exactly the objective K-ETA and our kernel SMO solve.

Same subsample, seeds and gamma as regime_battery.cell_kl2.  Reports
wall-clock (kernel construction counted separately), the augmented hull
distance 2/||w||, support size, test accuracy and status per (dataset,
seed); one shard per pair under <out>.shards/, resumable, --parallel.

Usage:
  OMP_NUM_THREADS=1 python3 src/libsvm_kl2.py --data-dir data --seeds 5 \
      --parallel 5 --out results/libsvm_kl2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_battery import _load_cached, ci95     # noqa: E402
from regime_battery import _subsample           # noqa: E402

EPS, C_SURR, C = 1e-3, 1e6, 1.0
DATASETS = ['gisette', 'w8a', 'a9a', 'covtype', 'ijcnn1']


def _shard(out, name, seed):
    return Path(str(out) + '.shards') / f"{name}_{seed}.json"


def run(job):
    name, seed, data_dir, out = job
    X, y, Xt, yt = _load_cached(name, data_dir, 100_000, seed)
    Xk, yk = _subsample(X, y, 10_000, seed)
    gamma = 1.0 / X.shape[1]
    V, W = Xk[yk == 1], Xk[yk == 0]
    Xs = np.vstack([V, W])
    ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    t0 = time.perf_counter()
    K = rbf_kernel(Xs, Xs, gamma=gamma)
    K[np.diag_indices_from(K)] += 1.0 / C
    t_kernel = time.perf_counter() - t0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = SVC(kernel='precomputed', C=C_SURR, tol=EPS, max_iter=2_000_000,
                cache_size=1000).fit(K, ys)
        t_fit = time.perf_counter() - t0
    status = 'converged' if getattr(m, 'fit_status_', 0) == 0 else 'maxiter'
    sv = m.support_
    ay = m.dual_coef_.ravel()
    wn2 = float(ay @ K[np.ix_(sv, sv)] @ ay)       # augmented ||w||^2
    dist = 2.0 / np.sqrt(wn2) if wn2 > 0 else float('inf')
    # test points have zero augmented coordinates: plain RBF kernel rows
    pred = np.empty(len(Xt), dtype=int)
    for s in range(0, len(Xt), 2000):
        Kt = rbf_kernel(Xt[s:s + 2000], Xs, gamma=gamma)
        pred[s:s + 2000] = (m.decision_function(Kt) > 0).astype(int)
    rec = dict(dataset=name, seed=seed, solver='LIBSVM-precomp',
               n_kernel=len(Xk), d=int(X.shape[1]), time=t_fit,
               time_kernel=t_kernel, dist=dist, sv=int(len(sv)),
               acc=float(np.mean(pred == yt)), status=status)
    sh = _shard(out, name, seed)
    sh.parent.mkdir(parents=True, exist_ok=True)
    sh.write_text(json.dumps(rec))
    print(f"[{name} seed={seed}] fit {t_fit:.1f}s (+{t_kernel:.1f}s kernel) "
          f"{status} dist={dist:.5f} sv={len(sv)} acc={rec['acc']:.4f}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--out', default='results/libsvm_kl2.json')
    args = ap.parse_args()
    out = Path(args.out)
    jobs, rows = [], []
    for name in args.datasets:
        for seed in range(args.seeds):
            sh = _shard(out, name, seed)
            if sh.exists():
                rows.append(json.loads(sh.read_text()))
            else:
                jobs.append((name, seed, args.data_dir, out))
    if rows:
        print(f"{len(rows)} cells already done", flush=True)
    if jobs:
        if args.parallel > 1:
            with Pool(min(args.parallel, len(jobs))) as pool:
                rows += pool.map(run, jobs, chunksize=1)
        else:
            rows += [run(j) for j in jobs]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print('\n| dataset | fit time (s) | kernel (s) | dist | sv | accuracy | '
          'converged |')
    print('|--|--:|--:|--:|--:|--:|--:|')
    for name in args.datasets:
        sel = [r for r in rows if r['dataset'] == name]
        if not sel:
            continue
        tm, th = ci95([r['time'] for r in sel])
        print(f"| {name} | {tm:.1f} ± {th:.1f} | "
              f"{np.mean([r['time_kernel'] for r in sel]):.1f} | "
              f"{np.mean([r['dist'] for r in sel]):.5f} | "
              f"{np.mean([r['sv'] for r in sel]):,.0f} | "
              f"{np.mean([r['acc'] for r in sel]):.4f} | "
              f"{sum(r['status'] == 'converged' for r in sel)}/{len(sel)} |")


if __name__ == '__main__':
    main()
