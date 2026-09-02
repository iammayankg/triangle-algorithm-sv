"""Block-size ablation on real data (review item 3): single MDM (k=1)
versus blocks of k transfers, ETA only, on

  gisette  L2 soft margin, C=1          (expensive columns, d=5000)
  ijcnn1   RBF L2 margin K+I/C, C=1     (cheap columns, d=22; the
                                         class-balanced kernel subsample
                                         of the regime battery)

Records wall-clock, iterations, column evaluations, distance, certified
gap, test accuracy and status per (dataset, k, seed).

Usage (on the machine that holds data/):
  OMP_NUM_THREADS=1 python3 src/k_ablation_real.py --data-dir data \
      --seeds 3 --out results/k_ablation_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_battery import _load_cached, acc, ci95, sep_from_soft   # noqa: E402
from regime_battery import _subsample                             # noqa: E402
from soft_margin import SoftMarginTA                              # noqa: E402
from kernel_ta import KernelETA                                   # noqa: E402

MODES = [('mdm', 1), ('block', 4), ('block', 16)]
EPS, CAP = 1e-3, 600.0


def gisette_l2(X, y, Xt, yt, mode, k, C=1.0):
    V, W = X[y == 1], X[y == 0]
    ta = SoftMarginTA(V, W, C=C, step_mode=mode, block_size=k,
                      zigzag_strategy='pairwise', seed=0)
    r = ta.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    w, b = sep_from_soft(ta, r)
    return dict(time=r.time, iters=r.iterations, oracle=ta.col_evals,
                dist=r.distance, gap=(r.distance - r.lower_bound) / r.distance,
                acc=acc(w, b, Xt, yt), status=r.status, n_drops=ta.n_drops)


def ijcnn1_kl2(X, y, Xt, yt, mode, k, seed, C=1.0, max_n_kernel=10_000):
    Xk, yk = _subsample(X, y, max_n_kernel, seed)
    V, W = Xk[yk == 1], Xk[yk == 0]
    gamma = 1.0 / X.shape[1]
    kt = KernelETA(V, W, kernel='rbf', gamma=gamma, reg_C=C, step_mode=mode,
                   block_size=k, zigzag_strategy='pairwise', seed=0)
    r = kt.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    pred = (kt.decision_function(Xt, r) > 0).astype(int)
    return dict(time=r.time, iters=r.iterations, oracle=kt.col_evals,
                dist=r.distance, gap=(r.distance - r.lower_bound) / r.distance,
                acc=float(np.mean(pred == yt)), status=r.status,
                n_drops=kt.n_drops, n_kernel=len(Xk))


CELLS = {'gisette': gisette_l2, 'ijcnn1': ijcnn1_kl2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--datasets', nargs='*', default=list(CELLS))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default='results/k_ablation_real.json')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # resume: cells already in the output file are skipped
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['dataset'], r['k'], r['seed']) for r in rows}
    if done:
        print(f"{len(done)} cells already done in {out}", flush=True)
    for name in args.datasets:
        for seed in range(args.seeds):
            todo = [(m, k) for m, k in MODES if (name, k, seed) not in done]
            if not todo:
                continue
            X, y, Xt, yt = _load_cached(name, args.data_dir, 100_000, seed)
            for mode, k in todo:
                if name == 'ijcnn1':
                    r = ijcnn1_kl2(X, y, Xt, yt, mode, k, seed)
                else:               # gisette, or synthetic for a smoke test
                    r = gisette_l2(X, y, Xt, yt, mode, k)
                r.update(dataset=name, seed=seed, mode=mode, k=k)
                rows.append(r)
                print(f"[{name} k={k} seed={seed}] {r['time']:.1f}s/"
                      f"{r['status']} iters={r['iters']} cols={r['oracle']} "
                      f"gap={r['gap']:.1e} acc={r['acc']:.4f}", flush=True)
                out.write_text(json.dumps(rows, indent=1))

    print('\n| dataset | k | time (s) | iterations | columns | accuracy | converged |')
    print('|--|--:|--:|--:|--:|--:|--:|')
    for name in args.datasets:
        for mode, k in MODES:
            sel = [r for r in rows if (r['dataset'], r['k']) == (name, k)]
            tm, th = ci95([r['time'] for r in sel])
            print(f"| {name} | {k} | {tm:.1f} ± {th:.1f} | "
                  f"{np.mean([r['iters'] for r in sel]):,.0f} | "
                  f"{np.mean([r['oracle'] for r in sel]):,.0f} | "
                  f"{np.mean([r['acc'] for r in sel]):.4f} | "
                  f"{sum(r['status'] == 'converged' for r in sel)}/{len(sel)} |")


if __name__ == '__main__':
    main()
