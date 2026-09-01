"""Numerical verification of the screening-activation theorem
(docs/block_convergence.md, Lemmas 9-10 and Theorem 11).

For random separated instances with the optimal face computed by the
exact QP:

1. Lemma 9: at every screening round where the radius satisfies
   r <= tau/(3D), every zero-weight point outside the optimal face is
   removed by that round's screening.
2. Lemma 10 identity: at full scans, UB - LB = gFW(z)/UB to machine
   precision (verified against a direct evaluation).
3. Theorem 11 end state: after the first activating round, the working
   set is contained in (optimal faces) union (current support).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data import generate_two_balls                       # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


def qp_solution(V, W):
    n, m = len(V), len(W)

    def obj(z):
        diff = V.T @ z[:n] - W.T @ z[n:]
        return float(diff @ diff)

    def grad(z):
        diff = V.T @ z[:n] - W.T @ z[n:]
        return np.concatenate([2.0 * (V @ diff), -2.0 * (W @ diff)])

    cons = [{'type': 'eq', 'fun': lambda z: z[:n].sum() - 1.0},
            {'type': 'eq', 'fun': lambda z: z[n:].sum() - 1.0}]
    z0 = np.concatenate([np.full(n, 1 / n), np.full(m, 1 / m)])
    res = minimize(obj, z0, jac=grad, bounds=[(0, 1)] * (n + m),
                   constraints=cons, method='SLSQP',
                   options={'maxiter': 2000, 'ftol': 1e-16})
    p = V.T @ res.x[:n]
    q = W.T @ res.x[n:]
    return p - q


def run_instance(seed, d, n, k):
    rng = np.random.default_rng(seed)
    V, W = generate_two_balls(d, n, k, rng=rng)
    zstar = qp_solution(V, W)
    delta_star = float(np.linalg.norm(zstar))
    if delta_star < 1e-6:
        return None
    # optimal faces and score margins
    sV = V @ zstar
    sW = W @ zstar
    mV, MW = sV.min(), sW.max()
    faceV = set(np.flatnonzero(sV - mV <= 1e-7 * max(1, abs(mV))).tolist())
    faceW = set(np.flatnonzero(MW - sW <= 1e-7 * max(1, abs(MW))).tolist())
    margins = np.concatenate([np.delete(sV - mV, list(faceV)),
                              np.delete(MW - sW, list(faceW))])
    if len(margins) == 0:
        return None
    tau = float(margins.min())
    D = max(
        float(np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2)
                      .sum(-1).max())),
        float(np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2)
                      .sum(-1).max())))
    if tau < 1e-4:
        return None

    events = []

    class Probe(EnhancedTriangleAlgorithm):
        def _screen_and_compact(self, lb_best):
            ub2 = self.dist2()
            lb = max(lb_best, 0.0)
            r = float(np.sqrt(max(ub2 - lb * lb, 0.0)))
            super()._screen_and_compact(lb_best)
            # zero-weight survivors outside the optimal face
            aliveV = set(self._origV.tolist())
            aliveW = set(self._origW.tolist())
            wV_orig = {int(self._origV[i]) for i, w in self.wV.items()
                       if w > 1e-12}
            wW_orig = {int(self._origW[j]) for j, w in self.wW.items()
                       if w > 1e-12}
            badV = [i for i in aliveV
                    if i not in faceV and i not in wV_orig]
            badW = [j for j in aliveW
                    if j not in faceW and j not in wW_orig]
            events.append((r, len(badV) + len(badW)))

    ta = Probe(V, W, zigzag_strategy='pairwise', seed=0, shrink=True,
               shrink_every=10)
    ta.solve_distance(eps=1e-8, max_iter=300_000)

    thr = tau / (3.0 * D)
    lemma_ok = all(bad == 0 for r, bad in events if r <= thr)
    activated = any(r <= thr for r, _ in events)
    # end state: working set inside face union support
    aliveV = set(ta._origV.tolist())
    aliveW = set(ta._origW.tolist())
    wV_orig = {int(ta._origV[i]) for i, w in ta.wV.items() if w > 1e-12}
    wW_orig = {int(ta._origW[j]) for j, w in ta.wW.items() if w > 1e-12}
    end_ok = aliveV <= (faceV | wV_orig) and aliveW <= (faceW | wW_orig) \
        if activated else True
    return dict(tau=tau, D=D, thr=thr, events=len(events),
                activated=activated, lemma_ok=lemma_ok, end_ok=end_ok,
                final_alive=len(aliveV) + len(aliveW),
                face=len(faceV) + len(faceW))


def test_lemma10_identity():
    rng = np.random.default_rng(3)
    V, W = generate_two_balls(15, 80, 1.3, rng=rng)
    ta = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise', seed=0)
    ta._init_state()
    for _ in range(5):
        # take a few steps, then check the identity at this iterate
        fV, fW = ta._scores_ta2()
        iV = int(np.argmax(fV))
        ta._apply_p(iV, ta._alpha_single_p(iV))
        ub = float(np.sqrt(ta.dist2()))
        lb = (float(np.min(ta.a - ta.b)) - float(np.max(ta.c - ta.e))) / ub
        # gFW(z) = <z,z> - min_u <z,u> over Z, computed directly
        z = ta.p - ta.q
        gfw = float(z @ z) - (float(np.min(V @ z)) - float(np.max(W @ z)))
        assert abs((ub - lb) - gfw / ub) < 1e-9 * max(1.0, gfw)
    print("[lem10] certified gap == gFW/UB verified to 1e-9 at 5 iterates")


def test_activation():
    rng = np.random.default_rng(71)
    checked = activated = 0
    fails = []
    for trial in range(12):
        d = int(rng.integers(4, 20))
        n = int(rng.integers(30, 80))
        k = float(rng.uniform(1.15, 1.6))
        out = run_instance(int(rng.integers(1e6)), d, n, k)
        if out is None:
            continue
        checked += 1
        activated += out['activated']
        ok = out['lemma_ok'] and out['end_ok']
        print(f"[act ] d={d} n={n} tau={out['tau']:.4f} thr={out['thr']:.5f} "
              f"rounds={out['events']} activated={out['activated']} "
              f"lemma_ok={out['lemma_ok']} end_ok={out['end_ok']} "
              f"alive={out['final_alive']} (faces={out['face']}) "
              f"-> {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(trial)
    assert not fails, fails
    assert checked >= 6 and activated >= 3, \
        f"too few informative instances ({checked} checked, {activated} activated)"
    print(f"[act ] {checked} instances checked, {activated} reached the "
          f"activation threshold; Lemma 9 and Theorem 11 hold on all")


if __name__ == '__main__':
    test_lemma10_identity()
    test_activation()
    print("\nAll screening-activation checks passed.")
