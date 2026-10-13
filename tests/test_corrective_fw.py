"""Correctness checks for the corrective-FW baselines (src/corrective_fw.py):
the three corrective rules against the exact QP on hard-margin, L2 and
RBF instances, the Cholesky updates against a fresh factorisation, and
monotone descent of every step."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))

from corrective_fw import CorrectiveFW                  # noqa: E402
from data import generate_overlap, generate_two_balls   # noqa: E402
from soft_margin import augment                         # noqa: E402
from test_correctness import qp_hull_distance           # noqa: E402
from test_kernel import qp_kernel_distance              # noqa: E402
from sklearn.metrics.pairwise import rbf_kernel         # noqa: E402

MODES = ('lps', 'qcmnp', 'fcfw')


def _check(label, truth, r, tol=1e-3):
    rel = abs(r.distance - truth) / max(truth, 1e-12)
    ok = r.status == 'converged' and rel < tol
    print(f"[{label}] qp={truth:.6f} cfw={r.distance:.6f} rel={rel:.1e} "
          f"it={r.iterations} fw={r.n_fw} lps={r.n_lps} qc={r.n_qc} "
          f"full={r.n_qc_full} trunc={r.n_qc_trunc} -> {'OK' if ok else 'FAIL'}")
    return ok


def test_hard_margin_matches_qp():
    rng = np.random.default_rng(5)
    fails = []
    for trial in range(6):
        d, n = int(rng.integers(3, 12)), int(rng.integers(10, 30))
        V, W = generate_two_balls(d, n, 1.5, rng=rng)
        truth = qp_hull_distance(V, W)
        for mode in MODES:
            r = CorrectiveFW(V, W, corrective=mode).solve_distance(
                eps=1e-6, max_iter=200_000)
            if not _check(f"hm {mode:5s} t{trial}", truth, r):
                fails.append((trial, mode))
    assert not fails, fails


def test_l2_matches_augmented_qp():
    rng = np.random.default_rng(7)
    fails = []
    for trial in range(6):
        d, n = int(rng.integers(2, 10)), int(rng.integers(8, 25))
        C = float(10 ** rng.uniform(-1, 2))
        V, W = generate_overlap(d, n, delta=1.5, rng=rng)
        truth = qp_hull_distance(*augment(V, W, C))
        for mode in MODES:
            r = CorrectiveFW(V, W, reg_C=C, corrective=mode).solve_distance(
                eps=1e-6, max_iter=200_000)
            if not _check(f"l2 {mode:5s} t{trial}", truth, r):
                fails.append((trial, mode))
    assert not fails, fails


def test_rbf_l2_matches_qp():
    rng = np.random.default_rng(11)
    fails = []
    for trial in range(5):
        d, n = int(rng.integers(2, 10)), int(rng.integers(10, 30))
        V, W = generate_overlap(d, n, delta=2.0, rng=rng)
        gamma, C = 1.0 / d, 1.0
        KVV = rbf_kernel(V, V, gamma=gamma) + np.eye(len(V)) / C
        KVW = rbf_kernel(V, W, gamma=gamma)
        KWW = rbf_kernel(W, W, gamma=gamma) + np.eye(len(W)) / C
        truth = qp_kernel_distance(KVV, KVW, KWW)
        for mode in MODES:
            r = CorrectiveFW(V, W, kernel='rbf', gamma=gamma, reg_C=C,
                             corrective=mode).solve_distance(
                eps=1e-6, max_iter=200_000)
            if not _check(f"kl2 {mode:5s} t{trial}", truth, r):
                fails.append((trial, mode))
    assert not fails, fails


def test_factor_updates_match_refactorisation():
    """After a run with many appends and deletes, the updated factor
    equals a fresh Cholesky factor of the face Gram matrix."""
    rng = np.random.default_rng(3)
    V, W = generate_overlap(20, 300, delta=1.0, rng=rng)
    for mode in ('qcmnp', 'fcfw'):
        s = CorrectiveFW(V, W, reg_C=1.0, corrective=mode,
                         refactor_every=10 ** 9)
        r = s.solve_distance(eps=1e-6, max_iter=200_000)
        p = s.p
        L_upd = s.L[:p, :p].copy()
        s._refactor()
        err = np.abs(L_upd - s.L[:p, :p]).max() / np.abs(s.L[:p, :p]).max()
        print(f"[chol {mode}] p={p} drops={r.n_drops} rel.err={err:.1e}")
        assert r.n_drops > 0 and err < 1e-8, (mode, r.n_drops, err)


def test_every_step_descends():
    rng = np.random.default_rng(17)
    V, W = generate_overlap(10, 200, delta=1.2, rng=rng)
    for mode in MODES:
        s = CorrectiveFW(V, W, reg_C=1.0, corrective=mode, refresh_every=1)
        prev = [np.inf]
        bad = []
        orig = s._refresh_scores

        def spy():
            orig()
            fV, fW = s._f()
            SVi, SWi = s._supports()
            zz = float(s.al_arr[SVi] @ fV[SVi] - s.be_arr[SWi] @ fW[SWi])
            if zz > prev[0] * (1 + 1e-12):
                bad.append((prev[0], zz))
            prev[0] = zz
        s._refresh_scores = spy
        s.solve_distance(eps=1e-6, max_iter=20_000)
        print(f"[descent {mode}] increases: {len(bad)}")
        assert not bad, (mode, bad[:3])


if __name__ == '__main__':
    for f in [test_hard_margin_matches_qp, test_l2_matches_augmented_qp,
              test_rbf_l2_matches_qp, test_factor_updates_match_refactorisation,
              test_every_step_descends]:
        f()
