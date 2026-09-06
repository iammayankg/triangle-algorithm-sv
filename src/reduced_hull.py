"""L1 (hinge) soft-margin SVM via the Triangle Algorithm on reduced
convex hulls.

The nu-SVM dual is exactly the nearest-point problem between the two
reduced convex hulls (Bennett & Bredensteiner; Crisp & Burges)

    R(V, mu) = { sum_i a_i v_i : sum a_i = 1, 0 <= a_i <= mu },

with per-point weight cap mu = 2 / (nu * l), l the total sample count
(balanced classes).  mu = 1 recovers the hard margin; smaller mu shrinks
each hull toward its centroid, which is the slack mechanism in geometric
form.  The optimal separator is w ~ p* - q* for the nearest pair
(p*, q*); against sklearn's NuSVC, ||w_nu|| = (nu/2) * delta where delta
is the reduced-hull distance.

Algorithmic changes relative to the hard-margin ETA:

  * the extreme-point oracle in direction h is a capped top-k blend
    (weight mu on the floor(1/mu) best points, remainder on the next),
    computed with a partial sort - O(n), same order as an argmax scan;
  * the primary step is the capped MDM transfer: weight moves from the
    worst active donor to the best receiver below its cap, clipped to the
    box - the pairwise machinery of the hard-margin solver with a cap;
  * toward-steps move the iterate toward the blended extreme point
    (the reduced hull is convex, so iterates stay feasible);
  * dot-product caching carries over unchanged for the transfers, and
    the lower bound comes from the reduced support function, which is
    the same capped sum the oracle computes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


@dataclass
class RCHResult:
    status: str          # 'converged' | 'intersect' | 'maxiter' | 'stalled'
    distance: float
    lower_bound: float
    iterations: int
    time: float
    n_support: int       # points with weight > 0
    n_at_cap: int        # margin violators (weight == mu)
    p: np.ndarray | None = None
    q: np.ndarray | None = None
    weights_V: np.ndarray | None = None
    weights_W: np.ndarray | None = None


class ReducedHullTA:
    """Triangle Algorithm between reduced convex hulls R(V, muV), R(W, muW)."""

    def __init__(self, V, W, mu=None, nu=None, refresh_every=100,
                 full_scan_every=5):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.W = np.ascontiguousarray(W, dtype=np.float64)
        self.n, self.d = self.V.shape
        self.m = self.W.shape[0]
        if mu is None:
            if nu is None:
                raise ValueError("provide mu or nu")
            mu = 2.0 / (nu * (self.n + self.m))
        self.muV = float(mu) if np.isscalar(mu) else float(mu[0])
        self.muW = float(mu) if np.isscalar(mu) else float(mu[1])
        if self.muV * self.n < 1.0 - 1e-9 or self.muW * self.m < 1.0 - 1e-9:
            raise ValueError("infeasible cap: mu * n must be >= 1")
        self.refresh_every = refresh_every
        self.full_scan_every = full_scan_every
        self.Vsq = np.einsum('ij,ij->i', self.V, self.V)
        self.Wsq = np.einsum('ij,ij->i', self.W, self.W)
        self._colV: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._colW: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        scale = max(1.0, float(self.Vsq.max()), float(self.Wsq.max()))
        self.tol = 16.0 * np.finfo(np.float64).eps * np.sqrt(self.d) * scale
        # denominator guard, scaled with the data below unit scale
        self._eps_den = _EPS * min(1.0, max(float(self.Vsq.max()),
                                            float(self.Wsq.max())))

    # ------------------------------------------------------------------
    def _refresh(self):
        self.p = self.V.T @ self.wV
        self.q = self.W.T @ self.wW
        self.a = self.V @ self.p
        self.b = self.V @ self.q
        self.c = self.W @ self.p
        self.e = self.W @ self.q
        self.pp = float(self.p @ self.p)
        self.qq = float(self.q @ self.q)
        self.pq = float(self.p @ self.q)

    def dist2(self):
        # from the explicit iterates: pp - 2 pq + qq cancels when the
        # hulls are far from the origin relative to their distance
        diff = self.p - self.q
        return float(diff @ diff)

    def _lower_bound(self, ub):
        """Lower bound from the reduced support functions."""
        _, _, vmin = self._capped_extreme(self.a - self.b, self.muV, True)
        _, _, wmax = self._capped_extreme(self.c - self.e, self.muW, False)
        return (vmin - wmax) / ub

    def _col_V(self, i):
        col = self._colV.get(i)
        if col is None:
            x = self.V[i]
            col = (self.V @ x, self.W @ x)
            self._colV[i] = col
        return col

    def _col_W(self, j):
        col = self._colW.get(j)
        if col is None:
            x = self.W[j]
            col = (self.V @ x, self.W @ x)
            self._colW[j] = col
        return col

    # ------------------------------------------------------------------
    @staticmethod
    def _capped_extreme(scores, mu, minimize):
        """Extreme point of the reduced hull for the given per-point scores
        (h . v_i): indices and weights of the capped top-k blend."""
        n = len(scores)
        k = int(np.floor(1.0 / mu + 1e-12))
        s = scores if minimize else -scores
        if k >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(s, k)[:k + 1]
            idx = idx[np.argsort(s[idx])]
        wts = np.full(len(idx), mu)
        rem = 1.0 - mu * k
        if len(idx) > k:
            wts[k:] = 0.0
            if rem > _EPS:
                wts[k] = rem
        value = float(scores[idx] @ wts)
        return idx, wts, value

    # ------------------------------------------------------------------
    # capped MDM transfer on one class; returns True if progress was made
    # ------------------------------------------------------------------
    def _transfer_V(self):
        s = self.b - self.a                      # (q - p) . v_i
        recv_ok = self.wV < self.muV - 1e-14
        don_ok = self.wV > 1e-14
        if not recv_ok.any() or not don_ok.any():
            return False
        v = int(np.argmax(np.where(recv_ok, s, -np.inf)))
        u = int(np.argmin(np.where(don_ok, s, np.inf)))
        num = s[v] - s[u]
        if v == u or num <= self.tol:
            return False
        cva, cvc = self._col_V(v)
        cua, cuc = self._col_V(u)
        den = self.Vsq[v] - 2.0 * cva[u] + self.Vsq[u]
        if den <= self._eps_den:
            return False
        g = min(num / den, self.wV[u], self.muV - self.wV[v])
        if g <= 0.0:
            return False
        self.pq += g * (self.b[v] - self.b[u])
        self.pp += 2.0 * g * (self.a[v] - self.a[u]) + g * g * den
        self.a += g * (cva - cua)
        self.c += g * (cvc - cuc)
        self.p += g * (self.V[v] - self.V[u])
        self.wV[v] += g
        self.wV[u] -= g
        return True

    def _transfer_W(self):
        s = self.c - self.e                      # (p - q) . w_j
        recv_ok = self.wW < self.muW - 1e-14
        don_ok = self.wW > 1e-14
        if not recv_ok.any() or not don_ok.any():
            return False
        v = int(np.argmax(np.where(recv_ok, s, -np.inf)))
        u = int(np.argmin(np.where(don_ok, s, np.inf)))
        num = s[v] - s[u]
        if v == u or num <= self.tol:
            return False
        cwa, cwe = self._col_W(v)
        cua, cue = self._col_W(u)
        den = self.Wsq[v] - 2.0 * cwe[u] + self.Wsq[u]
        if den <= self._eps_den:
            return False
        g = min(num / den, self.wW[u], self.muW - self.wW[v])
        if g <= 0.0:
            return False
        self.pq += g * (self.c[v] - self.c[u])
        self.qq += 2.0 * g * (self.e[v] - self.e[u]) + g * g * den
        self.b += g * (cwa - cua)
        self.e += g * (cwe - cue)
        self.q += g * (self.W[v] - self.W[u])
        self.wW[v] += g
        self.wW[u] -= g
        return True

    # ------------------------------------------------------------------
    # toward-step: move an iterate toward the blended reduced extreme
    # ------------------------------------------------------------------
    def _toward_V(self, idx, wts):
        gvec = self.V[idx].T @ wts
        gp = float(wts @ self.a[idx])
        gq = float(wts @ self.b[idx])
        gg = float(gvec @ gvec)
        den = gg - 2.0 * gp + self.pp             # ||g - p||^2
        num = gq - self.pq - gp + self.pp         # (q - p) . (g - p)
        if den <= self._eps_den or num <= self.tol:
            return False
        lam = min(num / den, 1.0)
        om = 1.0 - lam
        self.pq = om * self.pq + lam * gq
        self.pp = om * om * self.pp + 2.0 * lam * om * gp + lam * lam * gg
        self.a = om * self.a + lam * (self.V @ gvec)
        self.c = om * self.c + lam * (self.W @ gvec)
        self.p = om * self.p + lam * gvec
        self.wV *= om
        self.wV[idx] += lam * wts
        return True

    def _toward_W(self, idx, wts):
        gvec = self.W[idx].T @ wts
        gq = float(wts @ self.e[idx])
        gp = float(wts @ self.c[idx])
        gg = float(gvec @ gvec)
        den = gg - 2.0 * gq + self.qq
        num = gp - self.pq - gq + self.qq         # (p - q) . (g - q)
        if den <= self._eps_den or num <= self.tol:
            return False
        lam = min(num / den, 1.0)
        om = 1.0 - lam
        self.pq = om * self.pq + lam * gp
        self.qq = om * om * self.qq + 2.0 * lam * om * gq + lam * lam * gg
        self.b = om * self.b + lam * (self.V @ gvec)
        self.e = om * self.e + lam * (self.W @ gvec)
        self.q = om * self.q + lam * gvec
        self.wW *= om
        self.wW[idx] += lam * wts
        return True

    # ------------------------------------------------------------------
    def solve_distance(self, eps=1e-3, max_iter=200_000, eps_intersect=1e-8):
        t0 = time.perf_counter()
        # centroid start: always feasible for any valid cap
        self.wV = np.full(self.n, 1.0 / self.n)
        self.wW = np.full(self.m, 1.0 / self.m)
        self._refresh()
        status = 'maxiter'
        lb_best = -np.inf
        stalls = 0
        it = 0
        for it in range(1, max_iter + 1):
            if it % self.refresh_every == 0:
                self._refresh()
            ub = float(np.sqrt(self.dist2()))
            if ub <= eps_intersect:
                status = 'intersect'      # reduced hulls overlap (nu too small)
                break
            if (it - 1) % self.full_scan_every == 0:
                lb_best = max(lb_best, self._lower_bound(ub))
                if ub - lb_best <= eps * ub:
                    status = 'converged'
                    break
            movedV = self._transfer_V()
            movedW = self._transfer_W()
            if not (movedV or movedW):
                # capped MDM exhausted: try blended toward-steps
                iV, wV_, _ = self._capped_extreme(self.a - self.b, self.muV,
                                                  minimize=True)
                iW, wW_, _ = self._capped_extreme(self.c - self.e, self.muW,
                                                  minimize=False)
                tv = self._toward_V(iV, wV_)
                tw = self._toward_W(iW, wW_)
                if not (tv or tw):
                    self._refresh()
                    stalls += 1
                    if stalls > 3:
                        # every step refused: success only with the gap
                        # certificate, otherwise report the stall
                        ub = float(np.sqrt(self.dist2()))
                        if ub <= eps_intersect:
                            status = 'intersect'
                        else:
                            lb_best = max(lb_best, self._lower_bound(ub))
                            status = 'converged' \
                                if ub - lb_best <= eps * ub else 'stalled'
                        break
                else:
                    stalls = 0
            else:
                stalls = 0
        self._refresh()
        ub = float(np.sqrt(self.dist2()))
        if ub > eps_intersect:
            lb_best = max(lb_best, self._lower_bound(ub))
        thr = 1e-10
        return RCHResult(
            status=status, distance=ub, lower_bound=float(lb_best),
            iterations=it, time=time.perf_counter() - t0,
            n_support=int((self.wV > thr).sum() + (self.wW > thr).sum()),
            n_at_cap=int((self.wV > self.muV - 1e-12).sum()
                         + (self.wW > self.muW - 1e-12).sum()),
            p=self.p.copy(), q=self.q.copy(),
            weights_V=self.wV.copy(), weights_W=self.wW.copy())

    def separator(self, r: RCHResult):
        """(w, b) of the induced classifier: bisector of the witness pair."""
        d2 = r.distance ** 2
        w = 2.0 * (r.p - r.q) / d2
        b = (float(r.q @ r.q) - float(r.p @ r.p)) / d2
        return w, b
