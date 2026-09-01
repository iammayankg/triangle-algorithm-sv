"""Real-benchmark study: Triangle Algorithm family vs standard solvers.

Datasets (all bundled in PyPI packages - the sandbox's egress policy
blocks the usual dataset hosts, so these are the reachable standards):

  wdbc         UCI Breast Cancer Wisconsin (Diagnostic), 569 x 30
  digits-oe    UCI handwritten digits, odd vs even, 1797 x 64
  digits-3v8   UCI handwritten digits, 3 vs 8, ~360 x 64
  mnist5k-oe   MNIST (bundled 5000-sample subset), odd vs even, 5000 x 784
  mnist5k-3v8  MNIST subset, 3 vs 8, ~1000 x 784

Per dataset (75/25 stratified split, features standardized on train):

  geometry   TA I on the training hulls: does a linear hard margin exist?
  L2 (C=1)   SoftMarginTA (MDM) vs SoftMarginSMO vs LinearSVC(sq-hinge):
             time, primal objective, test accuracy, support count
  nu-SVM     ReducedHullTA vs NuSVC (nu = 0.2): time, delta agreement, acc
  RBF L2     KernelETA(reg_C=1, MDM) vs KernelSMO(reg_C=1): time, delta
             agreement, accuracy; SVC(rbf, C=1) accuracy as reference

Usage: python3 src/benchmark_real.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_ta import KernelETA, KernelSMO          # noqa: E402
from reduced_hull import ReducedHullTA              # noqa: E402
from soft_margin import SoftMarginTA, SoftMarginSMO  # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402

from sklearn.datasets import load_breast_cancer, load_digits  # noqa: E402
from sklearn.model_selection import train_test_split          # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402
from sklearn.svm import SVC, LinearSVC, NuSVC                 # noqa: E402

EPS = 1e-3
C_L2 = 1.0
NU = 0.2


def datasets():
    d = load_breast_cancer()
    yield 'wdbc', d.data, (d.target == 1).astype(int)
    dg = load_digits()
    yield 'digits-oe', dg.data, (dg.target % 2).astype(int)
    m = np.isin(dg.target, (3, 8))
    yield 'digits-3v8', dg.data[m], (dg.target[m] == 8).astype(int)
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    yield 'mnist5k-oe', X, (y % 2).astype(int)
    m = np.isin(y, (3, 8))
    yield 'mnist5k-3v8', X[m], (y[m] == 8).astype(int)


def _split(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def _sep_from_soft(ta, r):
    d2 = r.distance ** 2
    w = 2.0 * (r.p - r.q) / d2
    inv = 1.0 / ta.C
    pp = float(r.p @ r.p) + inv * sum(v * v for v in r.weights_V.values())
    qq = float(r.q @ r.q) + inv * sum(v * v for v in r.weights_W.values())
    return w, (qq - pp) / d2


def _acc(w, b, X, y01):
    pred = (X @ w + b > 0).astype(int)
    return float(np.mean(pred == y01))


def bench_one(name, X, y):
    Xtr, Xte, ytr, yte = _split(X, y)
    V, W = Xtr[ytr == 1], Xtr[ytr == 0]
    out = {'name': name, 'n_train': len(Xtr), 'd': Xtr.shape[1]}

    # ---- geometry: is a linear hard margin feasible? -----------------
    g = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise',
                                  seed=0).solve_intersection(
        eps=1e-3, max_iter=20_000)
    out['hard_margin'] = g.status
    print(f"[{name}] n={len(Xtr)} d={Xtr.shape[1]} "
          f"hard-margin hulls: {g.status}", flush=True)

    # ---- L2 soft margin, C = 1 --------------------------------------
    ta = SoftMarginTA(V, W, C=C_L2, step_mode='mdm',
                      zigzag_strategy='pairwise', seed=0)
    r = ta.solve_distance(eps=EPS, max_iter=200_000)
    w, b = _sep_from_soft(ta, r)
    out['L2_TA'] = dict(time=r.time, primal=2.0 / r.distance ** 2,
                        acc=_acc(w, b, Xte, yte), sv=r.sparsity,
                        status=r.status)
    Xs = np.vstack([V, W])
    ys = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
    s = SoftMarginSMO(Xs, ys, C_soft=C_L2, tol=EPS, max_iter=500_000,
                      time_cap=300).solve()
    out['L2_SMO'] = dict(time=s.time, primal=2.0 / s.hull_distance ** 2,
                         acc=_acc(s.w, s.b, Xte,
                                  (yte == 1).astype(int)), sv=s.sparsity,
                         status=s.status)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        m = LinearSVC(loss='squared_hinge', C=C_L2 / 2.0, tol=1e-6,
                      max_iter=100_000, intercept_scaling=100.0).fit(Xs, ys)
        el = time.perf_counter() - t0
    wv, bv = m.coef_.ravel(), float(m.intercept_[0])
    xi = np.maximum(0.0, 1.0 - ys * (Xs @ wv + bv))
    out['L2_LIN'] = dict(time=el,
                         primal=0.5 * float(wv @ wv)
                         + 0.5 * C_L2 * float(xi @ xi),
                         acc=_acc(wv, bv, Xte, yte), sv=-1,
                         status='converged')
    print(f"[{name}] L2  TA=({out['L2_TA']['time']:.2f}s,"
          f"P={out['L2_TA']['primal']:.4f},acc={out['L2_TA']['acc']:.4f}) "
          f"SMO=({out['L2_SMO']['time']:.2f}s,P={out['L2_SMO']['primal']:.4f},"
          f"acc={out['L2_SMO']['acc']:.4f}) "
          f"LIN=({out['L2_LIN']['time']:.2f}s,P={out['L2_LIN']['primal']:.4f},"
          f"acc={out['L2_LIN']['acc']:.4f})", flush=True)

    # ---- nu-SVM (reduced hulls), nu = 0.2 ---------------------------
    try:
        rh = ReducedHullTA(V, W, nu=NU)
        rr = rh.solve_distance(eps=EPS, max_iter=200_000)
        if rr.status == 'converged' and rr.distance > 1e-8:
            wr, br = rh.separator(rr)
            acc = _acc(wr, br, Xte, yte)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                t0 = time.perf_counter()
                mn = NuSVC(nu=NU, kernel='linear', tol=EPS,
                           max_iter=5_000_000).fit(Xs, ys)
                el = time.perf_counter() - t0
            dc = mn.dual_coef_.ravel()
            S = float(dc[dc > 0].sum())
            dsvc = float(np.linalg.norm(mn.coef_.ravel())) / S
            out['NU'] = dict(ta_time=rr.time, ta_acc=acc,
                             svc_time=el,
                             svc_acc=_acc(mn.coef_.ravel(),
                                          float(mn.intercept_[0]), Xte, yte),
                             rel=abs(rr.distance - dsvc) / dsvc)
            print(f"[{name}] nu  TA=({rr.time:.2f}s,acc={acc:.4f}) "
                  f"NuSVC=({el:.2f}s,acc={out['NU']['svc_acc']:.4f}) "
                  f"rel={out['NU']['rel']:.1e}", flush=True)
        else:
            out['NU'] = dict(status=rr.status)
            print(f"[{name}] nu  reduced hulls: {rr.status} (nu={NU})",
                  flush=True)
    except ValueError as exc:
        out['NU'] = dict(status=str(exc))

    # ---- RBF, L2 ridge C = 1 ----------------------------------------
    gamma = 1.0 / Xtr.shape[1]
    kta = KernelETA(V, W, kernel='rbf', gamma=gamma, reg_C=1.0,
                    step_mode='mdm', zigzag_strategy='pairwise', seed=0)
    kr = kta.solve_distance(eps=EPS, max_iter=200_000)
    kacc = float(np.mean((kta.decision_function(Xte, kr) > 0).astype(int)
                         == yte))
    ks = KernelSMO(Xs, ys, kernel='rbf', gamma=gamma, reg_C=1.0, tol=EPS,
                   max_iter=500_000, time_cap=300).solve()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t0 = time.perf_counter()
        msvc = SVC(kernel='rbf', gamma=gamma, C=1.0, tol=EPS).fit(Xs, ys)
        el = time.perf_counter() - t0
        svc_acc = float(np.mean((msvc.decision_function(Xte) > 0)
                                .astype(int) == yte))
    out['RBF'] = dict(ta_time=kr.time, ta_acc=kacc, ta_status=kr.status,
                      smo_time=ks.time,
                      rel=abs(kr.distance - ks.hull_distance)
                      / ks.hull_distance,
                      svc_time=el, svc_acc=svc_acc)
    print(f"[{name}] rbf TA=({kr.time:.2f}s,acc={kacc:.4f},{kr.status}) "
          f"K-SMO=({ks.time:.2f}s) rel={out['RBF']['rel']:.1e} "
          f"SVC-L1-ref=({el:.2f}s,acc={svc_acc:.4f})", flush=True)
    return out


def main():
    rows = [bench_one(name, X, y) for name, X, y in datasets()]
    Path('results/benchmark_real.json').write_text(json.dumps(rows, indent=2))
    print("\nsaved results/benchmark_real.json")


if __name__ == '__main__':
    main()
