"""Correctness checks for the reduced-convex-hull (L1 / nu-SVM) solver.

Ground truth on small instances: the RCH distance QP solved exactly by
SLSQP with box constraints 0 <= a_i <= mu.  Cross-check against sklearn's
NuSVC (LIBSVM), using ||w_nu|| = (nu/2) * delta.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data import generate_overlap                # noqa: E402
from reduced_hull import ReducedHullTA           # noqa: E402

from sklearn.svm import NuSVC                    # noqa: E402


def qp_rch_distance(V, W, mu):
    n, m = V.shape[0], W.shape[0]

    def obj(z):
        diff = V.T @ z[:n] - W.T @ z[n:]
        return float(diff @ diff)

    def grad(z):
        diff = V.T @ z[:n] - W.T @ z[n:]
        return np.concatenate([2.0 * (V @ diff), -2.0 * (W @ diff)])

    cons = [
        {'type': 'eq', 'fun': lambda z: z[:n].sum() - 1.0,
         'jac': lambda z: np.concatenate([np.ones(n), np.zeros(m)])},
        {'type': 'eq', 'fun': lambda z: z[n:].sum() - 1.0,
         'jac': lambda z: np.concatenate([np.zeros(n), np.ones(m)])},
    ]
    z0 = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])
    res = minimize(obj, z0, jac=grad, bounds=[(0.0, mu)] * (n + m),
                   constraints=cons, method='SLSQP',
                   options={'maxiter': 1000, 'ftol': 1e-14})
    return float(np.sqrt(max(res.fun, 0.0)))


def test_rch_matches_qp():
    rng = np.random.default_rng(17)
    fails = []
    for trial in range(10):
        d = int(rng.integers(2, 10))
        n = int(rng.integers(10, 30))
        mu = float(rng.uniform(1.5 / n, 0.25))
        V, W = generate_overlap(d, n, delta=2.5, rng=rng)
        truth = qp_rch_distance(V, W, mu)
        r = ReducedHullTA(V, W, mu=mu).solve_distance(eps=1e-5)
        if truth < 1e-6:
            ok = r.status == 'intersect' or r.distance < 1e-3
            rel = float('nan')
        else:
            rel = abs(r.distance - truth) / truth
            ok = rel < 1e-3 and r.lower_bound <= r.distance + 1e-9
        print(f"[rch ] trial={trial} d={d} n={n} mu={mu:.4f} qp={truth:.6f} "
              f"ta={r.distance:.6f} rel={rel:.1e} cap={r.n_at_cap} "
              f"{r.status} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_rch_matches_nusvc():
    rng = np.random.default_rng(29)
    fails = []
    for trial in range(6):
        d, n = 15, 80
        # nu large enough that the reduced hulls separate for delta = 1.5
        nu = float(rng.uniform(0.55, 0.85))
        V, W = generate_overlap(d, n, delta=1.5, rng=rng)
        mu = 2.0 / (nu * 2 * n)
        if mu * n < 1.02:            # keep clearly feasible
            continue
        r = ReducedHullTA(V, W, mu=mu).solve_distance(eps=1e-6)
        if r.status != 'converged' or r.distance < 1e-6:
            print(f"[nsvc] trial={trial} nu={nu:.3f} degenerate "
                  f"({r.status}) - skipped")
            continue
        X = np.vstack([V, W])
        y = np.concatenate([np.ones(n), -np.ones(n)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = NuSVC(nu=nu, kernel='linear', tol=1e-6,
                      max_iter=2_000_000).fit(X, y)
        # libsvm rescales its nu-SVC solution so that w = S * (p* - q*)
        # with S the per-class sum of the (rescaled) dual coefficients;
        # hence delta = ||w|| / S
        wn = float(np.linalg.norm(m.coef_.ravel()))
        dc = m.dual_coef_.ravel()
        S = float(dc[dc > 0].sum())
        delta_svc = wn / S
        rel = abs(r.distance - delta_svc) / delta_svc
        ok = rel < 5e-3
        print(f"[nsvc] trial={trial} nu={nu:.3f} mu={mu:.4f} "
              f"ta={r.distance:.6f} nusvc={delta_svc:.6f} rel={rel:.1e} "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_mu_path():
    """Shrinking mu (raising nu) must shrink the hulls: distance grows as
    mu decreases on overlapping data; mu = 1 recovers the hard margin."""
    rng = np.random.default_rng(3)
    V, W = generate_overlap(10, 150, delta=3.5, rng=rng)
    prev = -np.inf
    for mu in (0.5, 0.1, 0.05, 0.02, 1.5 / 150):
        r = ReducedHullTA(V, W, mu=mu).solve_distance(eps=1e-5)
        dist = 0.0 if r.status == 'intersect' else r.distance
        print(f"[path] mu={mu:.4f} status={r.status} dist={dist:.6f} "
              f"sv={r.n_support} at_cap={r.n_at_cap}")
        assert dist >= prev - 1e-6
        prev = dist


if __name__ == '__main__':
    test_rch_matches_qp()
    test_rch_matches_nusvc()
    test_mu_path()
    print("\nAll reduced-hull checks passed.")
