"""Sequential Minimal Optimization (SMO) baseline for hard-margin linear SVM.

Solves the standard SVM dual with a linear kernel and (effectively)
C = infinity, as in the source study:

    min_alpha  1/2 sum_ij alpha_i alpha_j y_i y_j <x_i, x_j> - sum_i alpha_i
    s.t.       sum_i y_i alpha_i = 0,   0 <= alpha_i <= C

Working-set selection is the classical maximal-violating-pair rule
(Keerthi et al. / LIBSVM WSS-1), stopping when the KKT violation gap
m(alpha) - M(alpha) <= tol.

For separable data the optimal hard-margin solution satisfies
||w||^2 = sum_i alpha_i and the distance between the two convex hulls is
delta = 2 / ||w||, which is what `hull_distance` reports so results are
directly comparable with the Triangle Algorithm's output.

Kernel rows are computed on demand (X @ x_i) and kept in an LRU cache,
mirroring the caching a practical SMO implementation would use.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class SMOResult:
    status: str          # 'converged' | 'maxiter' | 'timeout'
    hull_distance: float
    iterations: int
    time: float
    sparsity: int        # number of support vectors (alpha_i > 0)
    alpha: np.ndarray
    w: np.ndarray
    b: float
    objective: float


class SMO:
    def __init__(self, X, y, *, C=1e12, tol=1e-3, max_iter=200_000,
                 time_cap=None, cache_rows=512):
        self.X = np.ascontiguousarray(X)
        self.y = np.asarray(y, dtype=np.float64)
        self.n = self.X.shape[0]
        self.C = C
        self.tol = tol
        self.max_iter = int(max_iter)
        self.time_cap = time_cap
        self.diag = np.einsum('ij,ij->i', self.X, self.X).astype(np.float64)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_rows = cache_rows

    def _krow(self, i):
        row = self._cache.get(i)
        if row is None:
            row = (self.X @ self.X[i]).astype(np.float64)
            self._cache[i] = row
            if len(self._cache) > self._cache_rows:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(i)
        return row

    def solve(self):
        t0 = time.perf_counter()
        n, y, C = self.n, self.y, self.C
        alpha = np.zeros(n)
        grad = -np.ones(n)               # grad of 1/2 a'Qa - 1'a at a = 0
        status = 'maxiter'
        it = 0
        for it in range(1, self.max_iter + 1):
            if self.time_cap is not None and \
                    time.perf_counter() - t0 > self.time_cap:
                status = 'timeout'
                break
            F = -y * grad                # optimality scores
            up = ((y > 0) & (alpha < C)) | ((y < 0) & (alpha > 0))
            low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < C))
            if not up.any() or not low.any():
                status = 'converged'
                break
            Fu = np.where(up, F, -np.inf)
            Fl = np.where(low, F, np.inf)
            i = int(np.argmax(Fu))
            j = int(np.argmin(Fl))
            gap = F[i] - F[j]
            if gap <= self.tol:
                status = 'converged'
                break
            Ki = self._krow(i)
            Kj = self._krow(j)
            eta = self.diag[i] + self.diag[j] - 2.0 * Ki[j]
            if eta <= 1e-15:
                eta = 1e-12
            # step in the variable u = y_i alpha_i (and y_j alpha_j -= step)
            delta = gap / eta
            # feasibility clipping: alpha_i + y_i*delta in [0, C],
            #                       alpha_j - y_j*delta in [0, C]
            if y[i] > 0:
                delta = min(delta, C - alpha[i])
            else:
                delta = min(delta, alpha[i])
            if y[j] > 0:
                delta = min(delta, alpha[j])
            else:
                delta = min(delta, C - alpha[j])
            if delta <= 0.0:
                status = 'converged'
                break
            dai = y[i] * delta
            daj = -y[j] * delta
            alpha[i] += dai
            alpha[j] += daj
            grad += (dai * y[i]) * (y * Ki) + (daj * y[j]) * (y * Kj)
        w = self.X.T.astype(np.float64) @ (alpha * y)
        sv = alpha > 1e-8 * max(1.0, alpha.max() if alpha.max() > 0 else 1.0)
        # intercept from support-vector KKT conditions
        if sv.any():
            b = float(np.mean(y[sv] - (self.X[sv] @ w)))
        else:
            b = 0.0
        wn = float(np.linalg.norm(w))
        dist = 2.0 / wn if wn > 0 else np.inf
        obj = 0.5 * wn * wn - float(alpha.sum())
        return SMOResult(status=status, hull_distance=dist, iterations=it,
                         time=time.perf_counter() - t0,
                         sparsity=int(sv.sum()), alpha=alpha, w=w, b=b,
                         objective=obj)
