"""Kernelized Enhanced Triangle Algorithm.

The ETA touches the data only through inner products - the pivot tests,
the joint/toward/away/pairwise steps, the lower bound, and the caches all
run on <x, p>, <x, q> and the scalars pp, pq, qq.  Kernelizing therefore
requires no change to the algorithm itself, only to the plumbing:

  * "Gram columns" become kernel columns  k(V, x_i), k(W, x_i);
  * squared norms become kernel diagonals k(x_i, x_i);
  * the iterates p, q exist only as convex weights (no explicit feature
    vector), and the exact cache refresh reconstructs
        a = sum_i w_i k(V, v_i),  pp = sum_i w_i a[i],  etc.
    from the cached kernel columns of the support;
  * the decision function is the kernel expansion
        f(x) = sum_i wV_i k(v_i, x) - sum_j wW_j k(w_j, x) + b.

Supported kernels: 'linear', 'rbf' (exp(-gamma ||x-y||^2)), 'poly'
((gamma <x,y> + coef0)^degree).  An optional ridge term reg adds 1/C to
the kernel diagonal, giving the kernelized L2 soft margin (K + I/C) with
the same sparse-correction trick as the linear soft-margin solver.

KernelSMO is the matching SMO baseline on the same kernel.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smo import SMO                              # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


class Kernel:
    def __init__(self, kind='rbf', gamma=None, degree=3, coef0=1.0):
        self.kind = kind
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def resolve_gamma(self, d):
        if self.gamma is None:
            self.gamma = 1.0 / d

    def cross(self, A, Asq, x, xsq):
        """k(A, x) for one point x with cached squared norms."""
        if self.kind == 'linear':
            return (A @ x).astype(np.float64)
        if self.kind == 'rbf':
            d2 = Asq - 2.0 * (A @ x) + xsq
            return np.exp(-self.gamma * np.maximum(d2, 0.0))
        if self.kind == 'poly':
            return (self.gamma * (A @ x) + self.coef0) ** self.degree
        raise ValueError(self.kind)

    def diag(self, Asq):
        if self.kind == 'linear':
            return Asq.astype(np.float64)
        if self.kind == 'rbf':
            return np.ones_like(Asq, dtype=np.float64)
        if self.kind == 'poly':
            return (self.gamma * Asq + self.coef0) ** self.degree
        raise ValueError(self.kind)


class KernelETA(EnhancedTriangleAlgorithm):
    """ETA in the kernel feature space; iterates live in weight space."""

    def __init__(self, V, W, kernel='rbf', gamma=None, degree=3, coef0=1.0,
                 reg_C=None, **kw):
        super().__init__(V, W, **kw)
        self.kern = kernel if isinstance(kernel, Kernel) else \
            Kernel(kernel, gamma, degree, coef0)
        self.kern.resolve_gamma(self.d)
        self.reg = 0.0 if reg_C is None else 1.0 / float(reg_C)
        # raw squared norms for kernel evaluation
        self._Vsq_raw = self.Vsq.copy()
        self._Wsq_raw = self.Wsq.copy()
        # the original inputs: screening compacts V / W and the raw norms
        # in place, while result weights are keyed by original indices
        self._V0, self._W0 = self.V, self.W
        self._Vsq0_raw, self._Wsq0_raw = self._Vsq_raw, self._Wsq_raw
        # replace "norms" with kernel diagonals (+ ridge for L2 soft margin)
        self.Vsq = self.kern.diag(self._Vsq_raw) + self.reg
        self.Wsq = self.kern.diag(self._Wsq_raw) + self.reg
        kmax = max(1.0, float(self.Vsq.max()), float(self.Wsq.max()))
        self.tol = 16.0 * np.finfo(np.float64).eps * np.sqrt(self.d) * kmax

    # ---- kernel columns ---------------------------------------------
    def _gram_col_V(self, i):
        col = self._colV.get(i)
        if col is None:
            self.col_evals += 1
            x, xsq = self.V[i], self._Vsq_raw[i]
            cV = self.kern.cross(self.V, self._Vsq_raw, x, xsq)
            if self.reg:
                cV = cV.copy()
                cV[i] += self.reg
            col = (cV, self.kern.cross(self.W, self._Wsq_raw, x, xsq))
            self._colV[i] = col
        return col

    def _gram_col_W(self, j):
        col = self._colW.get(j)
        if col is None:
            self.col_evals += 1
            x, xsq = self.W[j], self._Wsq_raw[j]
            cW = self.kern.cross(self.W, self._Wsq_raw, x, xsq)
            if self.reg:
                cW = cW.copy()
                cW[j] += self.reg
            col = (self.kern.cross(self.V, self._Vsq_raw, x, xsq), cW)
            self._colW[j] = col
        return col

    def _compact_extra(self, idxV, idxW):
        self._Vsq_raw = self._Vsq_raw[idxV]
        self._Wsq_raw = self._Wsq_raw[idxW]

    # ---- weight-space state -----------------------------------------
    def _init_state(self, i0=0, j0=0):
        self.p = None            # feature-space iterates are implicit
        self.q = None
        self.wV = {i0: 1.0}
        self.wW = {j0: 1.0}
        self._refresh_caches()
        self.hist_V = []
        self.hist_W = []
        self.active_V = {i0}
        self.active_W = {j0}

    def _refresh_caches(self):
        """Reconstruct all caches exactly from the support weights using
        cached kernel columns (O(|support| * (n + m)))."""
        n, m = self.n, self.m
        a = np.zeros(n)
        c = np.zeros(m)
        for i, w in self.wV.items():
            if w <= 0.0:
                continue
            cV, cW = self._gram_col_V(i)
            a += w * cV
            c += w * cW
        b = np.zeros(n)
        e = np.zeros(m)
        for j, w in self.wW.items():
            if w <= 0.0:
                continue
            cV, cW = self._gram_col_W(j)
            b += w * cV
            e += w * cW
        self.a, self.b, self.c, self.e = a, b, c, e
        self.pp = float(sum(w * a[i] for i, w in self.wV.items() if w > 0))
        self.qq = float(sum(w * e[j] for j, w in self.wW.items() if w > 0))
        self.pq = float(sum(w * b[i] for i, w in self.wV.items() if w > 0))

    # ---- kernel decision function -----------------------------------
    def decision_function(self, X, r=None):
        """f(x) = (2/d^2) [ sum wV_i k(v_i,x) - sum wW_j k(w_j,x) ] + b,
        the bisector classifier of the final witness pair.  (The ridge
        coordinates of a soft-margin solve do not touch test points.)

        Weights are taken in *original* input indices - the result's
        weights_V / weights_W already are, and the live weights are mapped
        through the screening bookkeeping - and applied to the original
        (uncompacted) inputs, so the expansion is correct after shrinking."""
        if r is None:
            wV, wW = self._orig_weights()
        else:
            wV, wW = r.weights_V, r.weights_W
        X = np.asarray(X, dtype=np.float64)
        Xsq = np.einsum('ij,ij->i', X, X)
        f = np.zeros(len(X))
        for i, w in wV.items():
            if w > 0:
                f += w * self.kern.cross(X, Xsq, self._V0[i],
                                         self._Vsq0_raw[i])
        for j, w in wW.items():
            if w > 0:
                f -= w * self.kern.cross(X, Xsq, self._W0[j],
                                         self._Wsq0_raw[j])
        d2 = self.dist2()
        return (2.0 * f + (self.qq - self.pp)) / d2


class KernelSMO(SMO):
    """SMO with a kernel (and optional ridge 1/C for L2 soft margin)."""

    def __init__(self, X, y, kernel='rbf', gamma=None, degree=3, coef0=1.0,
                 reg_C=None, **kw):
        kw.setdefault('C', 1e12)
        super().__init__(X, y, **kw)
        self.kern = kernel if isinstance(kernel, Kernel) else \
            Kernel(kernel, gamma, degree, coef0)
        self.kern.resolve_gamma(X.shape[1])
        self.reg = 0.0 if reg_C is None else 1.0 / float(reg_C)
        self._Xsq_raw = self.diag.copy()
        self.diag = self.kern.diag(self._Xsq_raw) + self.reg

    def _krow(self, i):
        row = self._cache.get(i)
        if row is None:
            self.row_evals += 1
            row = self.kern.cross(self.X, self._Xsq_raw, self.X[i],
                                  self._Xsq_raw[i])
            if self.reg:
                row = row.copy()
                row[i] += self.reg
            self._cache[i] = row
            if len(self._cache) > self._cache_rows:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(i)
        return row

    def solve(self):
        r = super().solve()
        # The base class derives the distance, the intercept and the
        # objective from the original-space linear w, which is meaningless
        # under a kernel.  Recompute all three from the kernel rows of the
        # support (ridge included when reg > 0):
        #   f_i       = sum_j ay_j k(x_j, x_i)
        #   ||w_H||^2 = ay' K ay = sum_i ay_i f_i
        #   b         = mean over support vectors of y_i - f_i   (KKT)
        #   objective = 1/2 ||w_H||^2 - 1' alpha                 (dual)
        ay = r.alpha * self.y
        sv = np.flatnonzero(r.alpha > 1e-10 * max(1.0, r.alpha.max()))
        f_sv = np.array([float(self._krow(i) @ ay) for i in sv])
        wn2 = float(ay[sv] @ f_sv) if len(sv) else 0.0
        r.hull_distance = 2.0 / np.sqrt(wn2) if wn2 > 0 else np.inf
        r.b = float(np.mean(self.y[sv] - f_sv)) if len(sv) else 0.0
        r.objective = 0.5 * wn2 - float(r.alpha.sum())
        return r
