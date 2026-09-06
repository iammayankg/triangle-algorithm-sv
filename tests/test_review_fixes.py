"""Regression tests for the numerical / bookkeeping review findings.

1. numerical stalls are never reported as convergence (ETA and
   ReducedHullTA); sub-unit-scale data converges with a certificate;
2. cancellation in pp - 2 pq + qq cannot certify a false intersection;
3. KernelETA.decision_function is correct after screening compacted the
   data (weights are in original indices);
4. the original-index maps survive repeated / warm-started solves;
5. full_scan_every=1 keeps certifying (ETA and ReducedHullTA);
6. KernelSMO returns the kernel-space intercept and dual objective;
7. SoftMarginSMO returns the augmented objective and intercept;
8. drop_skip=False is the unconditional V-then-W block schedule.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from kernel_ta import KernelETA, KernelSMO               # noqa: E402
from reduced_hull import ReducedHullTA                   # noqa: E402
from soft_margin import SoftMarginSMO                    # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


def _certified(r, eps):
    return r.distance - r.lower_bound <= eps * r.distance + 1e-15


def test_tiny_scale_converges_with_certificate():
    V, W = np.array([[0.0], [5e-7]]), np.array([[3e-6]])
    r = EnhancedTriangleAlgorithm(V, W).solve_distance(eps=1e-3)
    assert r.status == 'converged'
    assert abs(r.distance - 2.5e-6) < 1e-12
    assert _certified(r, 1e-3)


def test_success_exits_always_certified():
    """Every 'converged' result must carry the gap certificate, across
    scales, step modes and solver variants."""
    rng = np.random.default_rng(0)
    for scale in (1e-7, 1e-3, 1.0, 1e3):
        for mode in ('toward', 'mdm', 'block'):
            V = scale * (rng.normal(size=(40, 4)) + 3.0)
            W = scale * (rng.normal(size=(40, 4)) - 3.0)
            r = EnhancedTriangleAlgorithm(
                V, W, step_mode=mode, zigzag_strategy='pairwise',
                seed=0).solve_distance(eps=1e-5, max_iter=50_000,
                                       eps_intersect=1e-6 * scale)
            assert r.status in ('converged', 'stalled'), (scale, mode, r)
            if r.status == 'converged':
                assert _certified(r, 1e-5), (scale, mode, r)


def test_cancellation_cannot_certify_intersection():
    V, W = np.array([[1e8]]), np.array([[1e8 + 1.0]])
    r = EnhancedTriangleAlgorithm(V, W).solve_distance(eps=1e-3)
    assert r.status != 'intersect'
    assert abs(r.distance - 1.0) < 1e-6
    r1 = EnhancedTriangleAlgorithm(V, W).solve_intersection(eps=1e-3)
    assert r1.status == 'separated'
    assert abs(r1.distance - 1.0) < 1e-6
    # implicit (kernel) iterates: the cached distance has a noise floor
    k = KernelETA(V, W, kernel='linear')
    rk = k.solve_distance(eps=1e-3)
    assert rk.status != 'intersect'


def test_kernel_decision_function_after_screening():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(200, 5)) + 3.0
    W = rng.normal(size=(200, 5)) - 3.0
    ta = KernelETA(V, W, kernel='rbf', gamma=0.2, zigzag_strategy='pairwise',
                   seed=0, shrink=True, shrink_every=10, shrink_min_frac=0.0)
    r = ta.solve_distance(eps=1e-5, max_iter=100_000)
    assert ta.n < 200 and ta.m < 200          # screening actually happened
    X = np.vstack([V[:20], W[:20]])
    f_res = ta.decision_function(X, r)
    f_live = ta.decision_function(X)
    assert np.all(np.sign(f_res) == np.r_[np.ones(20), -np.ones(20)])
    assert np.allclose(f_res, f_live)
    # reference: unshrunk solve on the same data
    r0 = KernelETA(V, W, kernel='rbf', gamma=0.2, zigzag_strategy='pairwise',
                   seed=0).solve_distance(eps=1e-5, max_iter=100_000)
    assert abs(r.distance - r0.distance) / r0.distance < 1e-3


def test_original_index_maps_survive_repeated_solves():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(200, 5)) + 3.0
    W = rng.normal(size=(200, 5)) - 3.0
    ta = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise', seed=0,
                                   shrink=True, shrink_every=10,
                                   shrink_min_frac=0.0)
    ta.solve_distance(eps=1e-5, max_iter=100_000)
    assert ta.n < 200
    for warm in (True, False):
        r = ta.solve_distance(eps=1e-7, max_iter=100_000, warm_start=warm)
        p = sum(w * V[i] for i, w in r.weights_V.items())
        q = sum(w * W[j] for j, w in r.weights_W.items())
        assert np.linalg.norm(p - r.p) < 1e-9
        assert np.linalg.norm(q - r.q) < 1e-9


def test_full_scan_every_one():
    V, W = np.array([[0.0], [1.0]]), np.array([[3.0]])
    r = EnhancedTriangleAlgorithm(V, W, full_scan_every=1).solve_distance(
        eps=1e-3, max_iter=50)
    assert r.status == 'converged' and abs(r.distance - 2.0) < 1e-12
    rng = np.random.default_rng(0)
    V = rng.normal(size=(30, 3)); W = rng.normal(size=(30, 3)) + 5.0
    for fse in (1, 2, 5):
        r = ReducedHullTA(V, W, mu=0.2, full_scan_every=fse).solve_distance(
            eps=1e-4, max_iter=5000)
        assert r.status == 'converged', (fse, r.status)
        assert _certified(r, 1e-4)


def test_reduced_hull_stall_is_not_convergence():
    V, W = np.array([[1e8], [1e8 + 0.5]]), np.array([[1e8 + 1.0], [1e8 + 1.5]])
    r = ReducedHullTA(V, W, mu=1.0).solve_distance(eps=1e-6)
    assert r.status != 'intersect'
    if r.status == 'converged':
        assert _certified(r, 1e-6)


def test_kernel_smo_intercept_and_objective():
    from sklearn.metrics.pairwise import rbf_kernel
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(size=(15, 3)) + 1.0,
                   rng.normal(size=(15, 3)) - 1.0])
    y = np.r_[np.ones(15), -np.ones(15)]
    for C in (None, 2.0):
        s = KernelSMO(X, y, kernel='rbf', gamma=0.5, reg_C=C,
                      tol=1e-10).solve()
        K = rbf_kernel(X, X, gamma=0.5)
        if C is not None:
            K = K + np.eye(len(X)) / C
        ay = s.alpha * y
        f = K @ ay
        sv = s.alpha > 1e-10 * s.alpha.max()
        b_kkt = float(np.mean(y[sv] - f[sv]))
        obj = 0.5 * float(ay @ K @ ay) - float(s.alpha.sum())
        assert abs(s.b - b_kkt) < 1e-8, (C, s.b, b_kkt)
        assert abs(s.objective - obj) < 1e-8, (C, s.objective, obj)
        assert abs(s.hull_distance - 2.0 / np.sqrt(ay @ K @ ay)) < 1e-8


def test_soft_smo_objective_and_intercept():
    X = np.zeros((3, 2)); y = np.array([1.0, 1.0, -1.0])
    s = SoftMarginSMO(X, y, C_soft=1.0, tol=1e-10).solve()
    assert abs(s.objective + 4.0 / 3.0) < 1e-8
    # general instance: augmented objective and KKT intercept
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(size=(20, 4)) + 0.5,
                   rng.normal(size=(20, 4)) - 0.5])
    y = np.r_[np.ones(20), -np.ones(20)]
    C = 3.0
    s = SoftMarginSMO(X, y, C_soft=C, tol=1e-10).solve()
    ay = s.alpha * y
    Ka = X @ X.T + np.eye(40) / C
    obj = 0.5 * float(ay @ Ka @ ay) - float(s.alpha.sum())
    assert abs(s.objective - obj) < 1e-8
    sv = s.alpha > 1e-8 * s.alpha.max()
    b_kkt = float(np.mean(y[sv] - (Ka @ ay)[sv]))
    assert abs(s.b - b_kkt) < 1e-8


def test_drop_skip_false_is_unconditional_v_then_w():
    rng = np.random.default_rng(3)
    V = rng.normal(size=(50, 4)); W = rng.normal(size=(50, 4)) + 4.0
    steps = []

    class Traced(EnhancedTriangleAlgorithm):
        def _step(self, iV, iW, sV=-np.inf, sW=-np.inf):
            steps.append([iV is not None and iW is not None])
            return super()._step(iV, iW, sV, sW)

        def _block_transfer_V(self, k, s=None):
            steps[-1].append('V'); return super()._block_transfer_V(k, s)

        def _block_transfer_W(self, k, s=None):
            steps[-1].append('W'); return super()._block_transfer_W(k, s)

    ta = Traced(V, W, step_mode='block', drop_skip=False,
                zigzag_strategy='pairwise', seed=0)
    r = ta.solve_distance(eps=1e-4)
    assert r.status == 'converged'
    assert ta.n_drop_skips == 0
    two_sided = [st[1:] for st in steps if st[0]]
    assert two_sided, 'no two-sided iterations exercised'
    # every two-sided iteration steps V first, then W, never skipping W
    assert all(order == ['V', 'W'] for order in two_sided), two_sided[:5]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print(name, 'OK')
