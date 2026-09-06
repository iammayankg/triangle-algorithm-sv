"""Blended Pairwise Conditional Gradients (Tsuji, Tanaka, Pokutta, ICML
2022) for the polytope-distance problem  min_{z in Z} 1/2 ||z||^2  on
Z = conv(V) - conv(W), as a baseline for the guarded block transfers.

Same problem classes as the ETA solvers: linear or RBF kernel, hard
margin (reg_C=None) or L2 soft margin through the augmented Gram matrix
K + I/C (never materialised: augmented inner products are the cached
ones plus a sparse diagonal correction, exactly as in Appendix B).

Same cost model as ETA: scores <z, v_i>, <z, w_j> are maintained
incrementally from Gram/kernel columns that are computed on first use
and cached (col_evals counts them, O((n+m)d) each); every iteration is
otherwise O(n + m + |S|) with S the active atom set of Z.

BPCG step (their Algorithm 1): let a_FW be the FW atom (LMO over Z,
separable: argmin_i f_V[i], argmax_j f_W[j]), a_A the away atom and a_S
the local FW atom over the active set S.  If the local pairwise gap
<z, a_A - a_S> is at least the FW gap <z, z - a_FW>, move weight from
a_A to a_S (exact line search, capped at lambda_{a_A}); otherwise take a
FW step towards a_FW.  Certificate: UB = ||z||, LB = <z, a_FW>/||z||.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

_W_MIN = 1e-14
_EPS = 1e-300


@dataclass
class BPCGResult:
    distance: float
    lower_bound: float
    iterations: int
    time: float
    status: str
    sparsity: int
    n_fw: int = 0
    n_pw: int = 0
    n_drops: int = 0
    alpha: dict = field(default_factory=dict)
    beta: dict = field(default_factory=dict)


class BPCG:
    def __init__(self, V, W, *, kernel='linear', gamma=None, reg_C=None,
                 refresh_every=500):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.W = np.ascontiguousarray(W, dtype=np.float64)
        self.n, self.m = len(self.V), len(self.W)
        self.kernel = kernel
        self.gamma = gamma
        self.C = reg_C
        self.refresh_every = refresh_every
        self.col_evals = 0
        self._cV: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cW: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if kernel == 'linear':
            self.Vsq = np.einsum('ij,ij->i', self.V, self.V)
            self.Wsq = np.einsum('ij,ij->i', self.W, self.W)
        elif kernel == 'rbf':
            self.Vsq = np.ones(self.n)
            self.Wsq = np.ones(self.m)
        else:
            raise ValueError(kernel)

    # ----- kernel columns ------------------------------------------------
    def _k(self, X, x):
        if self.kernel == 'linear':
            return X @ x
        d2 = np.einsum('ij,ij->i', X, X) - 2.0 * (X @ x) + float(x @ x)
        return np.exp(-self.gamma * np.maximum(d2, 0.0))

    def colV(self, i):
        c = self._cV.get(i)
        if c is None:
            x = self.V[i]
            c = (self._k(self.V, x), self._k(self.W, x))
            self._cV[i] = c
            self.col_evals += 1
        return c

    def colW(self, j):
        c = self._cW.get(j)
        if c is None:
            x = self.W[j]
            c = (self._k(self.V, x), self._k(self.W, x))
            self._cW[j] = c
            self.col_evals += 1
        return c

    def _kvv(self, i, k):
        return float(self.colV(i)[0][k])

    def _kww(self, j, l):
        return float(self.colW(j)[1][l])

    def _kvw(self, i, j):
        return float(self.colV(i)[1][j])

    def _atom_sq(self, a, b):
        """||a - b||^2 for atoms a=(i,j), b=(k,l) of Z, augmented if C."""
        i, j = a
        k, l = b
        s = (self.Vsq[i] - 2 * self._kvv(i, k) + self.Vsq[k]
             + self.Wsq[j] - 2 * self._kww(j, l) + self.Wsq[l]
             - 2 * (self._kvw(i, j) - self._kvw(i, l)
                    - self._kvw(k, j) + self._kvw(k, l)))
        if self.C is not None:
            s += (2.0 * (i != k) + 2.0 * (j != l)) / self.C
        return float(max(s, 0.0))

    # ----- state -----------------------------------------------------------
    def _refresh(self):
        sV = np.zeros(self.n)
        sW = np.zeros(self.m)
        for i, a in self.alpha.items():
            cv = self.colV(i)
            sV += a * cv[0]
            sW += a * cv[1]
        for j, b in self.beta.items():
            cw = self.colW(j)
            sV -= b * cw[0]
            sW -= b * cw[1]
        self.sV, self.sW = sV, sW

    def _f(self):
        """Augmented scores f_V[i] = <z~, v~_i>, f_W[j] = <z~, w~_j>."""
        if self.C is None:
            return self.sV, self.sW
        return self.sV + self.al_arr / self.C, self.sW - self.be_arr / self.C

    def _add(self, i, j, t):
        self.lam[(i, j)] = self.lam.get((i, j), 0.0) + t
        self.alpha[i] = self.alpha.get(i, 0.0) + t
        self.beta[j] = self.beta.get(j, 0.0) + t
        self.al_arr[i] += t
        self.be_arr[j] += t
        for dct, key in ((self.lam, (i, j)), (self.alpha, i), (self.beta, j)):
            if dct[key] <= _W_MIN:
                del dct[key]

    def solve_distance(self, eps=1e-3, max_iter=2_000_000, time_cap=None,
                       i0=0, j0=0):
        t0 = time.perf_counter()
        self.lam: dict[tuple[int, int], float] = {}
        self.alpha: dict[int, float] = {}
        self.beta: dict[int, float] = {}
        self.al_arr = np.zeros(self.n)
        self.be_arr = np.zeros(self.m)
        self._add(i0, j0, 1.0)
        self._refresh()
        n_fw = n_pw = n_drops = 0
        status = 'maxiter'
        it = 0
        lb = 0.0
        ub = 0.0
        for it in range(1, max_iter + 1):
            fV, fW = self._f()
            keys = list(self.lam.keys())
            I = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
            J = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
            fS = fV[I] - fW[J]                      # <z~, a~> over S
            lamS = np.fromiter((self.lam[k] for k in keys), dtype=np.float64,
                               count=len(keys))
            zz = float(lamS @ fS)                   # <z~, z~>
            if zz <= _EPS:
                status = 'intersect'
                break
            ub = np.sqrt(zz)
            iF = int(np.argmin(fV))
            jF = int(np.argmax(fW))
            f_fw = float(fV[iF] - fW[jF])
            g_fw = zz - f_fw
            lb = max(f_fw, 0.0) / ub
            if g_fw / zz <= eps:
                status = 'converged'
                break
            if time_cap is not None and time.perf_counter() - t0 > time_cap:
                status = 'timeout'
                break
            iA = int(np.argmax(fS))
            iS = int(np.argmin(fS))
            aA, aS = keys[iA], keys[iS]
            g_loc = float(fS[iA] - fS[iS])
            if g_loc >= g_fw and aA != aS:
                # local pairwise step aA -> aS, capped at lambda_{aA}
                den = self._atom_sq(aS, aA)
                cap = self.lam[aA]
                t = cap if den <= _EPS else min(g_loc / den, cap)
                if t >= cap:
                    n_drops += 1
                self._add(aS[0], aS[1], t)
                self._add(aA[0], aA[1], -t)
                cvS, cwS = self.colV(aS[0]), self.colW(aS[1])
                cvA, cwA = self.colV(aA[0]), self.colW(aA[1])
                self.sV += t * (cvS[0] - cwS[0] - cvA[0] + cwA[0])
                self.sW += t * (cvS[1] - cwS[1] - cvA[1] + cwA[1])
                n_pw += 1
            else:
                # FW step towards a_FW = (iF, jF)
                asq = self.Vsq[iF] + self.Wsq[jF] - 2 * self._kvw(iF, jF)
                if self.C is not None:
                    asq += 2.0 / self.C
                den = asq - 2 * f_fw + zz
                t = 1.0 if den <= _EPS else min(g_fw / den, 1.0)
                om = 1.0 - t
                for k in self.lam:
                    self.lam[k] *= om
                for k in self.alpha:
                    self.alpha[k] *= om
                for k in self.beta:
                    self.beta[k] *= om
                self.al_arr *= om
                self.be_arr *= om
                self._add(iF, jF, t)
                cv, cw = self.colV(iF), self.colW(jF)
                self.sV = om * self.sV + t * (cv[0] - cw[0])
                self.sW = om * self.sW + t * (cv[1] - cw[1])
                n_fw += 1
            if it % self.refresh_every == 0:
                self._refresh()
        el = time.perf_counter() - t0
        return BPCGResult(distance=float(ub), lower_bound=float(lb),
                          iterations=it, time=el, status=status,
                          sparsity=len(self.lam), n_fw=n_fw, n_pw=n_pw,
                          n_drops=n_drops, alpha=dict(self.alpha),
                          beta=dict(self.beta))

    # ----- classifier -------------------------------------------------------
    def decision_function(self, Xt, chunk=2000):
        """<z, x> - <z~, (p~ + q~)/2>: the bisector of the closest pair
        (augmented coordinates of test points are zero)."""
        fV, fW = self._f()
        b = -0.5 * (sum(a * fV[i] for i, a in self.alpha.items())
                    + sum(bb * fW[j] for j, bb in self.beta.items()))
        Ia = np.fromiter(self.alpha.keys(), dtype=np.int64)
        aa = np.fromiter(self.alpha.values(), dtype=np.float64)
        Jb = np.fromiter(self.beta.keys(), dtype=np.int64)
        bb = np.fromiter(self.beta.values(), dtype=np.float64)
        Xt = np.asarray(Xt, dtype=np.float64)
        out = np.empty(len(Xt))
        if self.kernel == 'linear':
            w = aa @ self.V[Ia] - bb @ self.W[Jb]
            return Xt @ w + b
        for s in range(0, len(Xt), chunk):
            xt = Xt[s:s + chunk]
            kv = _rbf(xt, self.V[Ia], self.gamma)
            kw = _rbf(xt, self.W[Jb], self.gamma)
            out[s:s + chunk] = kv @ aa - kw @ bb + b
        return out


def _rbf(A, B, gamma):
    d2 = (np.einsum('ij,ij->i', A, A)[:, None] - 2.0 * (A @ B.T)
          + np.einsum('ij,ij->i', B, B)[None, :])
    return np.exp(-gamma * np.maximum(d2, 0.0))
