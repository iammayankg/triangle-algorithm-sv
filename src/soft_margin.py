"""Soft-margin (L2) extension of the Enhanced Triangle Algorithm.

The L2 soft-margin SVM

    min  1/2 ||w||^2 + C/2 sum_i xi_i^2
    s.t. y_i (w.x_i + b) >= 1 - xi_i

is *exactly* a hard-margin problem in the augmented space

    x~_i = [ x_i ; e_i / sqrt(C) ]   (one indicator coordinate per point)

whose Gram matrix is K + I/C.  The distance between the augmented convex
hulls therefore solves the L2 soft-margin problem for any C > 0 (and the
augmented hulls never intersect, since the slack block keeps them apart),
with optimal primal value  1/2 ||w~||^2 = 2 / delta_C^2.

`SoftMarginTA` runs the Enhanced Triangle Algorithm on the augmented
problem WITHOUT materialising the augmentation: the extra coordinates of
the iterates are exactly the convex weights (alpha/sqrt(C)), so every
augmented dot product is a cached original-space dot product plus a sparse
diagonal correction:

    x~_i . p~   = x_i . p + alpha_i / C      (same class; 0 across classes)
    p~ . p~     = p . p  + ||alpha||^2 / C
    x~_i . x~_j = x_i . x_j + delta_ij / C

`SoftMarginSMO` is the SMO baseline on the same reduction (kernel
K + I/C), and `qp_soft_distance` gives exact small-instance ground truth
by materialising the augmentation explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smo import SMO                              # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


class SoftMarginTA(EnhancedTriangleAlgorithm):
    """Enhanced Triangle Algorithm on the L2 soft-margin reduction."""

    def __init__(self, V, W, C=1.0, **kw):
        self.C = float(C)
        super().__init__(V, W, **kw)
        inv = 1.0 / self.C
        self.Vsq = self.Vsq + inv      # ||x~_i||^2 = ||x_i||^2 + 1/C
        self.Wsq = self.Wsq + inv

    # ---- augmented dot-product plumbing ------------------------------
    def _refresh_caches(self):
        super()._refresh_caches()
        inv = 1.0 / self.C
        for i, w in self.wV.items():
            if i >= 0:
                self.a[i] += w * inv          # a_i = x_i.p + alpha_i/C
        for j, w in self.wW.items():
            if j >= 0:
                self.e[j] += w * inv
        self.pp += inv * sum(w * w for w in self.wV.values())
        self.qq += inv * sum(w * w for w in self.wW.values())
        # p~ . q~ has no correction: the slack blocks are disjoint

    def _gram_col_V(self, i):
        col = self._colV.get(i)
        if col is None:
            self.col_evals += 1
            x = self.V[i]
            cV = (self.V @ x).astype(np.float64)
            cV[i] += 1.0 / self.C
            col = (cV, (self.W @ x).astype(np.float64))
            self._colV[i] = col
        return col

    def _gram_col_W(self, j):
        col = self._colW.get(j)
        if col is None:
            self.col_evals += 1
            x = self.W[j]
            cW = (self.W @ x).astype(np.float64)
            cW[j] += 1.0 / self.C
            col = ((self.V @ x).astype(np.float64), cW)
            self._colW[j] = col
        return col

    # primal value of the L2 soft-margin problem
    def primal_objective(self, distance):
        return 2.0 / (distance * distance)


class SoftMarginSMO(SMO):
    """Hard-margin SMO on the augmented kernel K + I/C."""

    def __init__(self, X, y, C_soft=1.0, **kw):
        kw.setdefault('C', 1e12)      # box stays (effectively) infinite
        super().__init__(X, y, **kw)
        self.C_soft = float(C_soft)
        self.diag = self.diag + 1.0 / self.C_soft

    def _krow(self, i):
        row = self._cache.get(i)
        if row is None:
            self.row_evals += 1
            row = (self.X @ self.X[i]).astype(np.float64)
            row[i] += 1.0 / self.C_soft
            self._cache[i] = row
            if len(self._cache) > self._cache_rows:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(i)
        return row

    def solve(self):
        r = super().solve()
        # augmented ||w~||^2 = ||w||^2 + sum alpha_i^2 / C
        wn2 = float(r.w @ r.w) + float(r.alpha @ r.alpha) / self.C_soft
        r.hull_distance = 2.0 / np.sqrt(wn2) if wn2 > 0 else np.inf
        return r


def augment(V, W, C):
    """Materialise the augmentation explicitly (small instances only)."""
    n, m = len(V), len(W)
    s = 1.0 / np.sqrt(C)
    Va = np.hstack([V, s * np.eye(n), np.zeros((n, m))])
    Wa = np.hstack([W, np.zeros((m, n)), s * np.eye(m)])
    return Va, Wa
