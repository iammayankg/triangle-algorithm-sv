"""Correctness checks for the kernelized Triangle Algorithm.

1. Linear-kernel consistency: KernelETA('linear') must reproduce the
   euclidean ETA exactly.
2. RBF feature-space hull distance vs the exact kernel QP (SLSQP over the
   two simplices with explicit Gram matrices).
3. Agreement with sklearn SVC (hard margin, RBF) via
   delta = 2 / ||w_H||, ||w_H||^2 = (alpha y)' K (alpha y).
4. KernelSMO agreement with the same QP.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data import generate_two_balls, generate_overlap    # noqa: E402
from kernel_ta import KernelETA, KernelSMO, Kernel       # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402

from sklearn.metrics.pairwise import rbf_kernel          # noqa: E402
from sklearn.svm import SVC                              # noqa: E402


def qp_kernel_distance(KVV, KVW, KWW):
    n, m = KVV.shape[0], KWW.shape[0]

    def obj(z):
        u, v = z[:n], z[n:]
        return float(u @ KVV @ u - 2.0 * u @ KVW @ v + v @ KWW @ v)

    def grad(z):
        u, v = z[:n], z[n:]
        return np.concatenate([2.0 * (KVV @ u - KVW @ v),
                               2.0 * (KWW @ v - KVW.T @ u)])

    cons = [
        {'type': 'eq', 'fun': lambda z: z[:n].sum() - 1.0,
         'jac': lambda z: np.concatenate([np.ones(n), np.zeros(m)])},
        {'type': 'eq', 'fun': lambda z: z[n:].sum() - 1.0,
         'jac': lambda z: np.concatenate([np.zeros(n), np.ones(m)])},
    ]
    z0 = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])
    res = minimize(obj, z0, jac=grad, bounds=[(0.0, 1.0)] * (n + m),
                   constraints=cons, method='SLSQP',
                   options={'maxiter': 1000, 'ftol': 1e-14})
    return float(np.sqrt(max(res.fun, 0.0)))


def test_linear_consistency():
    rng = np.random.default_rng(2)
    fails = []
    for trial in range(5):
        d, n = int(rng.integers(3, 30)), int(rng.integers(20, 80))
        V, W = generate_two_balls(d, n, 1.3, rng=rng)
        r0 = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise',
                                       seed=0).solve_distance(eps=1e-6)
        r1 = KernelETA(V, W, kernel='linear', zigzag_strategy='pairwise',
                       seed=0).solve_distance(eps=1e-6)
        rel = abs(r0.distance - r1.distance) / r0.distance
        ok = rel < 1e-9
        print(f"[lin ] trial={trial} d={d} n={n} eucl={r0.distance:.8f} "
              f"kern={r1.distance:.8f} rel={rel:.1e} -> "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_rbf_matches_qp():
    rng = np.random.default_rng(9)
    fails = []
    for trial in range(8):
        d = int(rng.integers(2, 10))
        n = int(rng.integers(10, 30))
        V, W = generate_overlap(d, n, delta=2.0, rng=rng)
        gamma = 1.0 / d
        KVV = rbf_kernel(V, V, gamma=gamma)
        KVW = rbf_kernel(V, W, gamma=gamma)
        KWW = rbf_kernel(W, W, gamma=gamma)
        truth = qp_kernel_distance(KVV, KVW, KWW)
        r = KernelETA(V, W, kernel='rbf', gamma=gamma,
                      zigzag_strategy='pairwise',
                      seed=0).solve_distance(eps=1e-5, max_iter=200_000)
        rel = abs(r.distance - truth) / max(truth, 1e-12)
        ok = rel < 1e-3
        print(f"[rbf ] trial={trial} d={d} n={n} qp={truth:.6f} "
              f"ta={r.distance:.6f} rel={rel:.1e} it={r.iterations} "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_rbf_matches_svc():
    rng = np.random.default_rng(13)
    fails = []
    for trial in range(4):
        d, n = 10, 100
        V, W = generate_two_balls(d, n, 1.4, rng=rng)   # separated
        gamma = 1.0 / d
        r = KernelETA(V, W, kernel='rbf', gamma=gamma,
                      zigzag_strategy='pairwise',
                      seed=0).solve_distance(eps=1e-6, max_iter=500_000)
        X = np.vstack([V, W])
        y = np.concatenate([np.ones(n), -np.ones(n)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = SVC(kernel='rbf', gamma=gamma, C=1e8, tol=1e-8).fit(X, y)
        sv = m.support_
        ay = np.zeros(2 * n)
        ay[sv] = m.dual_coef_.ravel()
        Ksv = rbf_kernel(X[sv], X[sv], gamma=gamma)
        wn2 = float(ay[sv] @ Ksv @ ay[sv])
        delta_svc = 2.0 / np.sqrt(wn2)
        rel = abs(r.distance - delta_svc) / delta_svc
        ok = rel < 5e-3
        print(f"[svc ] trial={trial} ta={r.distance:.6f} "
              f"svc={delta_svc:.6f} rel={rel:.1e} -> "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_kernel_smo_matches_qp():
    rng = np.random.default_rng(31)
    fails = []
    for trial in range(4):
        d, n = 8, 40
        V, W = generate_two_balls(d, n, 1.3, rng=rng)
        gamma = 1.0 / d
        KVV = rbf_kernel(V, V, gamma=gamma)
        KVW = rbf_kernel(V, W, gamma=gamma)
        KWW = rbf_kernel(W, W, gamma=gamma)
        truth = qp_kernel_distance(KVV, KVW, KWW)
        X = np.vstack([V, W])
        y = np.concatenate([np.ones(n), -np.ones(n)])
        r = KernelSMO(X, y, kernel='rbf', gamma=gamma, tol=1e-6).solve()
        rel = abs(r.hull_distance - truth) / truth
        ok = rel < 1e-2
        print(f"[ksmo] trial={trial} qp={truth:.6f} smo={r.hull_distance:.6f} "
              f"rel={rel:.1e} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


if __name__ == '__main__':
    test_linear_consistency()
    test_rbf_matches_qp()
    test_rbf_matches_svc()
    test_kernel_smo_matches_qp()
    print("\nAll kernel checks passed.")
