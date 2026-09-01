"""Correctness checks for the L2 soft-margin extension.

Ground truth: the augmentation is materialised explicitly on small
instances and solved (a) by the exact QP and (b) by the *hard-margin*
Triangle Algorithm on the augmented points - both must agree with
SoftMarginTA, which never materialises the augmentation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

from data import generate_overlap                       # noqa: E402
from soft_margin import SoftMarginTA, SoftMarginSMO, augment  # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm      # noqa: E402
from test_correctness import qp_hull_distance           # noqa: E402


def test_soft_matches_augmented_qp():
    rng = np.random.default_rng(21)
    fails = []
    for trial in range(8):
        d = int(rng.integers(2, 10))
        n = int(rng.integers(8, 25))
        C = float(10 ** rng.uniform(-1, 2))
        V, W = generate_overlap(d, n, delta=1.5, rng=rng)
        Va, Wa = augment(V, W, C)
        truth = qp_hull_distance(Va, Wa)
        r = SoftMarginTA(V, W, C=C, zigzag_strategy='pairwise',
                         seed=0).solve_distance(eps=1e-5, max_iter=200_000)
        ra = EnhancedTriangleAlgorithm(Va, Wa, zigzag_strategy='pairwise',
                                       seed=0).solve_distance(
            eps=1e-5, max_iter=200_000)
        e1 = abs(r.distance - truth) / truth
        e2 = abs(r.distance - ra.distance) / truth
        ok = e1 < 1e-3 and e2 < 1e-3
        print(f"[soft] trial={trial} d={d} n={n} C={C:8.3f} qp={truth:.6f} "
              f"soft-ta={r.distance:.6f} aug-ta={ra.distance:.6f} "
              f"err_qp={e1:.1e} err_aug={e2:.1e} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_soft_smo_agrees():
    rng = np.random.default_rng(33)
    fails = []
    for trial in range(5):
        d, n = 20, 60
        C = float(10 ** rng.uniform(-1, 2))
        V, W = generate_overlap(d, n, delta=1.5, rng=rng)
        Va, Wa = augment(V, W, C)
        truth = qp_hull_distance(Va, Wa)
        X = np.vstack([V, W])
        y = np.concatenate([np.ones(n), -np.ones(n)])
        r = SoftMarginSMO(X, y, C_soft=C, tol=1e-6).solve()
        e = abs(r.hull_distance - truth) / truth
        ok = e < 1e-2
        print(f"[ssmo] trial={trial} C={C:8.3f} qp={truth:.6f} "
              f"smo={r.hull_distance:.6f} err={e:.1e} "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails


def test_regularisation_path_monotone():
    """delta_C must decrease as C grows (less slack -> tighter hulls)."""
    rng = np.random.default_rng(5)
    V, W = generate_overlap(30, 300, delta=2.0, rng=rng)
    prev = np.inf
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        r = SoftMarginTA(V, W, C=C, zigzag_strategy='pairwise',
                         seed=0).solve_distance(eps=1e-5, max_iter=200_000)
        print(f"[path] C={C:8.2f} delta_C={r.distance:.6f} it={r.iterations}")
        assert r.distance < prev + 1e-9
        prev = r.distance


if __name__ == '__main__':
    test_soft_matches_augmented_qp()
    test_soft_smo_agrees()
    test_regularisation_path_monotone()
    print("\nAll soft-margin checks passed.")
