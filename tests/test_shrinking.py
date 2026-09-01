"""Correctness and effectiveness checks for gap-certified shrinking.

1. Shrunk and unshrunk runs must agree with each other and with the exact
   QP, across step modes and solver variants (euclidean, soft-margin,
   kernel).
2. Safety: no point of the exact QP's optimal support may ever be
   screened out.
3. Effectiveness: the surviving point count and wall-clock time on
   Table-3-style instances.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

from data import generate_two_balls, generate_overlap    # noqa: E402
from kernel_ta import KernelETA                          # noqa: E402
from soft_margin import SoftMarginTA                     # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402
from test_correctness import qp_hull_distance            # noqa: E402


def test_shrunk_matches_unshrunk():
    rng = np.random.default_rng(41)
    fails = []
    cases = []
    for trial in range(6):
        d = int(rng.integers(5, 40))
        n = int(rng.integers(40, 150))
        k = float(rng.uniform(1.1, 1.8))
        V, W = generate_two_balls(d, n, k, rng=rng)
        cases.append(('hard-toward', V, W,
                      dict(zigzag_strategy='pairwise')))
        cases.append(('hard-mdm', V, W,
                      dict(step_mode='mdm', zigzag_strategy='pairwise')))
    for name, V, W, kw in cases:
        truth = qp_hull_distance(V, W)
        r0 = EnhancedTriangleAlgorithm(V, W, seed=0, **kw)\
            .solve_distance(eps=1e-5, max_iter=200_000)
        r1 = EnhancedTriangleAlgorithm(V, W, seed=0, shrink=True,
                                       shrink_every=20, **kw)\
            .solve_distance(eps=1e-5, max_iter=200_000)
        e0 = abs(r0.distance - truth) / truth
        e1 = abs(r1.distance - truth) / truth
        ok = e1 < 1e-3 and abs(r1.distance - r0.distance) / truth < 1e-3
        print(f"[shr ] {name:12s} qp={truth:.6f} plain={r0.distance:.6f} "
              f"shrunk={r1.distance:.6f} err={e1:.1e} -> "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    assert not fails, fails


def test_qp_support_never_screened():
    """Every point carrying weight in the exact QP solution must survive."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(43)
    fails = []
    for trial in range(5):
        d, n = 10, 60
        V, W = generate_two_balls(d, n, 1.2, rng=rng)

        def obj(z):
            diff = V.T @ z[:n] - W.T @ z[n:]
            return float(diff @ diff)

        def grad(z):
            diff = V.T @ z[:n] - W.T @ z[n:]
            return np.concatenate([2.0 * (V @ diff), -2.0 * (W @ diff)])

        cons = [{'type': 'eq', 'fun': lambda z: z[:n].sum() - 1.0},
                {'type': 'eq', 'fun': lambda z: z[n:].sum() - 1.0}]
        z0 = np.concatenate([np.full(n, 1 / n), np.full(n, 1 / n)])
        res = minimize(obj, z0, jac=grad, bounds=[(0, 1)] * 2 * n,
                       constraints=cons, method='SLSQP',
                       options={'maxiter': 500, 'ftol': 1e-14})
        supV = set(np.flatnonzero(res.x[:n] > 1e-6).tolist())
        supW = set(np.flatnonzero(res.x[n:] > 1e-6).tolist())
        ta = EnhancedTriangleAlgorithm(V, W, seed=0, shrink=True,
                                       shrink_every=10,
                                       zigzag_strategy='pairwise')
        ta.solve_distance(eps=1e-6, max_iter=200_000)
        aliveV = set(ta._origV.tolist())
        aliveW = set(ta._origW.tolist())
        ok = supV <= aliveV and supW <= aliveW
        print(f"[safe] trial={trial} qp support {len(supV)}+{len(supW)}, "
              f"alive {len(aliveV)}+{len(aliveW)} of {n}+{n} -> "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_soft_and_kernel_variants():
    rng = np.random.default_rng(47)
    V, W = generate_overlap(30, 300, delta=3.0, rng=rng)
    for name, mk in (
        ('soft C=1', lambda sh: SoftMarginTA(
            V, W, C=1.0, step_mode='mdm', zigzag_strategy='pairwise',
            seed=0, shrink=sh, shrink_every=25)),
        ('rbf', lambda sh: KernelETA(
            V, W, kernel='rbf', gamma=1 / 30, zigzag_strategy='pairwise',
            seed=0, shrink=sh, shrink_every=25)),
    ):
        r0 = mk(False).solve_distance(eps=1e-5, max_iter=300_000)
        s = mk(True)
        r1 = s.solve_distance(eps=1e-5, max_iter=300_000)
        rel = abs(r0.distance - r1.distance) / r0.distance
        ok = rel < 1e-3
        print(f"[var ] {name:10s} plain={r0.distance:.6f} "
              f"shrunk={r1.distance:.6f} rel={rel:.1e} "
              f"alive={s.n}+{s.m} of 300+300 -> {'OK' if ok else 'FAIL'}")
        assert ok, name


def test_effectiveness():
    """Alive counts and timing on a Table-3-style instance."""
    rng = np.random.default_rng(53)
    V, W = generate_two_balls(1000, 5000, 1.2, rng=rng)
    t0 = time.perf_counter()
    r0 = EnhancedTriangleAlgorithm(V, W, seed=0, zigzag_strategy='pairwise')\
        .solve_distance(eps=1e-5, max_iter=100_000)
    t_plain = time.perf_counter() - t0
    ta = EnhancedTriangleAlgorithm(V, W, seed=0, zigzag_strategy='pairwise',
                                   shrink=True, shrink_every=25)
    t0 = time.perf_counter()
    r1 = ta.solve_distance(eps=1e-5, max_iter=100_000)
    t_shr = time.perf_counter() - t0
    rel = abs(r0.distance - r1.distance) / r0.distance
    print(f"[eff ] d=1000 n=5000/set eps=1e-5: plain {t_plain:.2f}s "
          f"({r0.iterations} it) vs shrunk {t_shr:.2f}s ({r1.iterations} it), "
          f"alive at end {ta.n}+{ta.m} of 5000+5000, rel={rel:.1e}")
    assert rel < 1e-3


if __name__ == '__main__':
    test_shrunk_matches_unshrunk()
    test_qp_support_never_screened()
    test_soft_and_kernel_variants()
    test_effectiveness()
    print("\nAll shrinking checks passed.")
