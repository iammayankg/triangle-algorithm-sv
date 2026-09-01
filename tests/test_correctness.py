"""Correctness checks for the Enhanced Triangle Algorithm and SMO baseline.

Ground truth on small instances comes from solving the convex-hull
distance QP exactly with scipy (SLSQP over the two probability
simplices):

    min || V' a - W' b ||   s.t.  a, b in simplex.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data import generate_two_balls              # noqa: E402
from smo import SMO                              # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


def qp_hull_distance(V, W):
    """Exact polytope distance via SLSQP (small instances only)."""
    n, m = V.shape[0], W.shape[0]

    def obj(z):
        a, b = z[:n], z[n:]
        diff = V.T @ a - W.T @ b
        return float(diff @ diff)

    def grad(z):
        a, b = z[:n], z[n:]
        diff = V.T @ a - W.T @ b
        return np.concatenate([2.0 * (V @ diff), -2.0 * (W @ diff)])

    cons = [
        {'type': 'eq', 'fun': lambda z: z[:n].sum() - 1.0,
         'jac': lambda z: np.concatenate([np.ones(n), np.zeros(m)])},
        {'type': 'eq', 'fun': lambda z: z[n:].sum() - 1.0,
         'jac': lambda z: np.concatenate([np.zeros(n), np.ones(m)])},
    ]
    z0 = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])
    res = minimize(obj, z0, jac=grad, bounds=[(0.0, 1.0)] * (n + m),
                   constraints=cons, method='SLSQP',
                   options={'maxiter': 500, 'ftol': 1e-14})
    return float(np.sqrt(max(res.fun, 0.0)))


def test_distance_matches_qp():
    rng = np.random.default_rng(7)
    fails = []
    for trial in range(10):
        d = int(rng.integers(2, 12))
        n = int(rng.integers(8, 40))
        k = float(rng.uniform(1.05, 2.0))
        V, W = generate_two_balls(d, n, k, rng=rng)
        truth = qp_hull_distance(V, W)
        ta = EnhancedTriangleAlgorithm(V, W, seed=0)
        r = ta.solve_distance(eps=1e-4, max_iter=100_000)
        rel = abs(r.distance - truth) / max(truth, 1e-12)
        ok = rel < 5e-3 and r.lower_bound <= r.distance + 1e-9
        print(f"[dist] trial={trial} d={d} n={n} k={k:.2f} "
              f"qp={truth:.6f} ta={r.distance:.6f} lb={r.lower_bound:.6f} "
              f"rel_err={rel:.2e} status={r.status} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, f"failed trials: {fails}"


def test_smo_matches_qp():
    rng = np.random.default_rng(11)
    fails = []
    for trial in range(6):
        d = int(rng.integers(2, 12))
        n = int(rng.integers(8, 30))
        k = float(rng.uniform(1.1, 2.0))
        V, W = generate_two_balls(d, n, k, rng=rng)
        truth = qp_hull_distance(V, W)
        X = np.vstack([V, W])
        y = np.concatenate([np.ones(len(V)), -np.ones(len(W))])
        r = SMO(X, y, tol=1e-6).solve()
        rel = abs(r.hull_distance - truth) / max(truth, 1e-12)
        ok = rel < 1e-2
        print(f"[smo ] trial={trial} d={d} n={n} k={k:.2f} "
              f"qp={truth:.6f} smo={r.hull_distance:.6f} rel_err={rel:.2e} "
              f"status={r.status} it={r.iterations} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, f"failed trials: {fails}"


def test_intersection_detection():
    rng = np.random.default_rng(3)
    fails = []
    for trial in range(8):
        d = int(rng.integers(2, 20))
        n = int(rng.integers(20, 100))
        intersecting = trial % 2 == 0
        k = 0.2 if intersecting else 1.5
        V, W = generate_two_balls(d, n, k, rng=rng)
        ta = EnhancedTriangleAlgorithm(V, W, seed=0)
        r = ta.solve_intersection(eps=1e-3, max_iter=200_000)
        # verify separation certificates against the exact QP distance
        truth = None
        want = 'intersect' if intersecting else 'separated'
        ok = r.status == want
        if not ok and n <= 60:
            truth = qp_hull_distance(V, W)
            if r.status == 'separated' and truth > 1e-3:
                ok = True
            if r.status == 'intersect' and truth <= 1e-3:
                ok = True
        print(f"[int ] trial={trial} d={d} n={n} k={k} status={r.status} "
              f"dist={r.distance:.5f} it={r.iterations} truth={truth} "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, f"failed trials: {fails}"


def test_enhancement_ablation():
    """Every enhancement configuration must reach the same distance."""
    rng = np.random.default_rng(5)
    V, W = generate_two_balls(30, 200, 1.3, rng=rng)
    truth = None
    configs = [
        dict(),
        dict(joint_update=False),
        dict(cache_dots=False),
        dict(anti_zigzag=False),
        dict(prioritized=False),
        dict(joint_update=False, cache_dots=False, anti_zigzag=False,
             prioritized=False),
    ]
    vals = []
    for cfg in configs:
        ta = EnhancedTriangleAlgorithm(V, W, seed=0, **cfg)
        r = ta.solve_distance(eps=1e-3, max_iter=200_000)
        vals.append(r.distance)
        print(f"[abl ] cfg={cfg or 'all-on'} dist={r.distance:.8f} "
              f"it={r.iterations} t={r.time:.3f}s status={r.status}")
    spread = (max(vals) - min(vals)) / max(vals)
    print(f"[abl ] relative spread = {spread:.2e}")
    assert spread < 1e-3


if __name__ == '__main__':
    test_distance_matches_qp()
    test_smo_matches_qp()
    test_intersection_detection()
    test_enhancement_ablation()
    print("\nAll correctness checks passed.")
