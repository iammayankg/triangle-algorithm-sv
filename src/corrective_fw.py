"""Corrective Frank-Wolfe (Halbey, Rakotomandimby, Besancon, Designolle,
Pokutta, NeurIPS 2025, Algorithm 1) for the polytope-distance problem,
as a baseline for the guarded block transfers.

Domain and active set.  The problem is min f(alpha, beta) =
1/2 ||V~ alpha - W~ beta||^2 over the product of simplices
Delta_n x Delta_m (V~, W~ augmented for the L2 margin, feature-space for
kernels).  Its vertices are the pairs (e_i, e_j), i.e. the atoms
v_i - w_j of Z, and the face of the product spanned by an active set of
such vertices is Delta_{S_V} x Delta_{S_W}, with S_V = supp alpha and
S_W = supp beta.  We keep the active set as that face (the side
supports), the parametrisation the ETA solvers use, not as a list of
product atoms (the parametrisation of src/bpcg.py).

One iteration (their Algorithm 1):
  away vertex   a = (argmax_{S_V} f_V, argmin_{S_W} f_W)
  local FW      s = (argmin_{S_V} f_V, argmax_{S_W} f_W)
  global FW     v = (argmin_i f_V, argmax_j f_W)          (the TA scan)
  if <grad, a - s> >= <grad, x - v>:  corrective step inside the face
  else:                               FW step towards v (exact line search)
where f_V[i] = <z~, v~_i>, f_W[j] = <z~, w~_j> are maintained from cached
Gram/kernel columns exactly as in BPCG and ETA (col_evals counts the
same O((n+m)d) columns).  Certificate and stopping rule are those of
ETA and BPCG: UB = ||z||, LB = <z, v>/||z||, stop at (UB-LB)/UB <= eps.

Corrective steps (`corrective=`):
  'lps'     local pairwise step (their Algorithm 3): exact line search
            from a to s, capped at min(alpha_{a_V}, beta_{a_W}).  With
            this rule the solver is BPCG (Tsuji, Tanaka, Pokutta 2022) on
            the side parametrisation.
  'qcmnp'   QC-MNP (their Algorithm 6) in the hybrid schedule of their
            experiments: a quadratic correction whenever `qc_every` new
            atoms have entered since the last one, the LPS otherwise.
            The correction minimises f over the affine hull of the face,
            {sum alpha_S = 1, sum beta_S = 1} (their eq. 8, here a
            (|S_V|+|S_W|)-dimensional KKT system with two constraints);
            if the minimiser is nonnegative it is taken (an FCFW step),
            otherwise the iterate moves towards it until the first weight
            hits zero (one ratio-test truncation, at least one drop).
            An infeasible or non-descent system falls back to the LPS.
  'fcfw'    fully-corrective FW: after every FW step, minimise f over the
            face exactly by repeating the truncation until the affine
            minimiser is nonnegative (Wolfe's minor cycles on the side
            supports), and use the same full correction as the corrective
            step.

Linear algebra.  The face Gram matrix G (entries <s_a a~, s_b b~>, sign
+ for V and - for W points; positive definite for the L2 margin, where
it carries I/C) is held as a Cholesky factor updated in O(p^2) when a
support index enters (one triangular solve) or leaves (a rank-one
update), p = |S_V|+|S_W|; the support columns are kept in two stacked
arrays so a correction updates the scores with two matrix-vector
products.  The affine minimiser costs two triangular solves.  Hard
margins (reg_C=None) add a ridge of 1e-10 * max diag(G).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from bpcg import BPCG, BPCGResult

_W_MIN = 1e-14
_EPS = 1e-300


@dataclass
class CFWResult(BPCGResult):
    n_lps: int = 0
    n_qc: int = 0             # corrections attempted (n_drops: indices removed)
    n_qc_full: int = 0        # affine minimiser nonnegative: taken
    n_qc_trunc: int = 0       # truncations (each drops >= 1 index)
    n_qc_fallback: int = 0    # infeasible / non-descent system -> LPS
    n_refactor: int = 0
    max_support: int = 0
    t_linalg: float = 0.0     # factor updates, solves, score updates of QC


class CorrectiveFW(BPCG):
    def __init__(self, V, W, *, kernel='linear', gamma=None, reg_C=None,
                 corrective='qcmnp', qc_every=1, refresh_every=500,
                 refactor_every=2000, ridge=1e-10):
        super().__init__(V, W, kernel=kernel, gamma=gamma, reg_C=reg_C,
                         refresh_every=refresh_every)
        if corrective not in ('lps', 'qcmnp', 'fcfw'):
            raise ValueError(corrective)
        self.corrective = corrective
        self.qc_every = int(qc_every)
        self.use_qc = corrective != 'lps'
        self.refactor_every = refactor_every
        self.ridge_rel = ridge
        self.invC = 0.0 if reg_C is None else 1.0 / reg_C

    # ----- face bookkeeping ------------------------------------------------
    def _alloc(self, cap):
        self.cap = cap
        self.ord_isV = np.zeros(cap, dtype=bool)
        self.ord_idx = np.zeros(cap, dtype=np.int64)
        if self.use_qc:
            self.L = np.zeros((cap, cap))
            self.SVr = np.zeros((self.n, cap))   # signed columns, V rows
            self.SWr = np.zeros((self.m, cap))   # signed columns, W rows

    def _grow(self):
        old, p = self.cap, self.p
        cap = 2 * old
        isV, idx = self.ord_isV, self.ord_idx
        self.ord_isV = np.zeros(cap, dtype=bool)
        self.ord_idx = np.zeros(cap, dtype=np.int64)
        self.ord_isV[:p], self.ord_idx[:p] = isV[:p], idx[:p]
        if self.use_qc:
            L, SVr, SWr = self.L, self.SVr, self.SWr
            self.L = np.zeros((cap, cap))
            self.L[:p, :p] = L[:p, :p]
            self.SVr = np.zeros((self.n, cap))
            self.SWr = np.zeros((self.m, cap))
            self.SVr[:, :p], self.SWr[:, :p] = SVr[:, :p], SWr[:, :p]
        self.cap = cap

    def _ridge(self, diag):
        return self.ridge_rel * diag if self.C is None else 0.0

    def _add_key(self, isV, i):
        """Append support index i of side V (isV) or W to the face."""
        if self.p == self.cap:
            self._grow()
        p = self.p
        self.ord_isV[p], self.ord_idx[p] = isV, i
        (self.posV if isV else self.posW)[i] = p
        if self.use_qc:
            col = self.colV(i) if isV else self.colW(i)   # column cost
            t = time.perf_counter()
            if isV:
                cv = col
                self.SVr[:, p], self.SWr[:, p] = cv[0], cv[1]
                g = self.SVr[i, :p]
                gaa = self.Vsq[i] + self.invC
            else:
                cw = col
                self.SVr[:, p], self.SWr[:, p] = -cw[0], -cw[1]
                g = -self.SWr[i, :p]
                gaa = self.Wsq[i] + self.invC
            gaa += self._ridge(gaa)
            if p:
                c = solve_triangular(self.L[:p, :p], g, lower=True,
                                     check_finite=False)
                d2 = gaa - float(c @ c)
                self.L[p, :p] = c
            else:
                d2 = gaa
            self.L[p, p] = np.sqrt(max(d2, 1e-14 * gaa))
            self.t_linalg += time.perf_counter() - t
        self.p = p + 1

    def _remove_pos(self, k):
        """Remove the support index at face position k (weight ~ 0)."""
        p = self.p
        isV, i = bool(self.ord_isV[k]), int(self.ord_idx[k])
        w = self.al_arr[i] if isV else self.be_arr[i]
        if w != 0.0:          # residual weight below _W_MIN: keep z exact
            cv = self.colV(i) if isV else self.colW(i)
            sgn = 1.0 if isV else -1.0
            self.sV -= sgn * w * cv[0]
            self.sW -= sgn * w * cv[1]
            if isV:
                self.al_arr[i] = 0.0
            else:
                self.be_arr[i] = 0.0
        (self.posV if isV else self.posW)[i] = -1
        sh = slice(k + 1, p)
        movedV = self.ord_idx[sh][self.ord_isV[sh]]
        movedW = self.ord_idx[sh][~self.ord_isV[sh]]
        self.posV[movedV] -= 1
        self.posW[movedW] -= 1
        self.ord_isV[k:p - 1] = self.ord_isV[k + 1:p]
        self.ord_idx[k:p - 1] = self.ord_idx[k + 1:p]
        if self.use_qc:
            t = time.perf_counter()
            self._chol_delete(k)
            self.SVr[:, k:p - 1] = self.SVr[:, k + 1:p]
            self.SWr[:, k:p - 1] = self.SWr[:, k + 1:p]
            self.t_linalg += time.perf_counter() - t
        self.p = p - 1

    def _chol_delete(self, k):
        """Delete row/column k of G = L L^T in O(p^2): shift, then the
        rank-one update L22 L22^T + x x^T of the trailing block."""
        p, L = self.p, self.L
        x = L[k + 1:p, k].copy()
        L[k:p - 1, :p] = L[k + 1:p, :p]
        L[:p - 1, k:p - 1] = L[:p - 1, k + 1:p]
        L[p - 1, :p] = 0.0
        L[:p, p - 1] = 0.0
        q = p - 1 - k
        for j in range(q):
            jj = k + j
            Ljj, xj = L[jj, jj], x[j]
            if xj == 0.0:
                continue
            r = np.hypot(Ljj, xj)
            c, s = r / Ljj, xj / Ljj
            L[jj, jj] = r
            if j + 1 < q:
                col = L[jj + 1:p - 1, jj]
                col += s * x[j + 1:]
                col /= c
                x[j + 1:] = c * x[j + 1:] - s * col

    def _refactor(self):
        """Rebuild the factor from the stacked columns (numerical hygiene)."""
        t = time.perf_counter()
        p = self.p
        isV, idx = self.ord_isV[:p], self.ord_idx[:p]
        G = np.empty((p, p))
        G[isV] = self.SVr[idx[isV], :p]
        G[~isV] = -self.SWr[idx[~isV], :p]
        G = 0.5 * (G + G.T)
        dg = np.diag(G).copy()
        G[np.diag_indices(p)] = dg + self.invC + (
            self.ridge_rel * (dg.max() if p else 0.0) if self.C is None else 0.0)
        self.L[:p, :p] = np.linalg.cholesky(G)
        self.n_refactor += 1
        self.t_linalg += time.perf_counter() - t

    def _gather(self, aV, aW):
        """Per face position: aV[i] for a V index, aW[j] for a W index
        (gathered side by side; the classes may differ in size)."""
        p = self.p
        isV, idx = self.ord_isV[:p], self.ord_idx[:p]
        out = np.empty(p)
        out[isV] = aV[idx[isV]]
        out[~isV] = aW[idx[~isV]]
        return out

    def _x(self):
        return self._gather(self.al_arr, self.be_arr)

    def _set_x(self, x):
        p = self.p
        isV, idx = self.ord_isV[:p], self.ord_idx[:p]
        self.al_arr[idx[isV]] = x[isV]
        self.be_arr[idx[~isV]] = x[~isV]

    def _drop_zeros(self):
        x = self._x()
        dead = np.flatnonzero(x <= _W_MIN)
        for k in dead[::-1]:
            self._remove_pos(int(k))
        return len(dead)

    # ----- steps -------------------------------------------------------------
    def _supports(self):
        p = self.p
        isV, idx = self.ord_isV[:p], self.ord_idx[:p]
        return idx[isV], idx[~isV]

    def _lps(self, fV, fW, SVi, SWi):
        """Local pairwise step a -> s inside the face; returns gain sign."""
        ia = int(SVi[np.argmax(fV[SVi])])
        is_ = int(SVi[np.argmin(fV[SVi])])
        ja = int(SWi[np.argmin(fW[SWi])])
        js = int(SWi[np.argmax(fW[SWi])])
        g_loc = float(fV[ia] - fV[is_] + fW[js] - fW[ja])
        caps = []
        if ia != is_:
            caps.append(self.al_arr[ia])
        if ja != js:
            caps.append(self.be_arr[ja])
        if not caps or g_loc <= 0.0:
            return False
        cap = min(caps)
        den = self._atom_sq((is_, js), (ia, ja))
        t = cap if den <= _EPS else min(g_loc / den, cap)
        if ia != is_:
            self.al_arr[ia] -= t
            self.al_arr[is_] += t
            cs, ca = self.colV(is_), self.colV(ia)
            self.sV += t * (cs[0] - ca[0])
            self.sW += t * (cs[1] - ca[1])
        if ja != js:
            self.be_arr[ja] -= t
            self.be_arr[js] += t
            cs, ca = self.colW(js), self.colW(ja)
            self.sV -= t * (cs[0] - ca[0])
            self.sW -= t * (cs[1] - ca[1])
        if t >= cap:
            if ia != is_ and self.al_arr[ia] <= _W_MIN:
                self.al_arr[ia] = max(self.al_arr[ia], 0.0)
            if ja != js and self.be_arr[ja] <= _W_MIN:
                self.be_arr[ja] = max(self.be_arr[ja], 0.0)
        self.n_lps += 1
        return True

    def _affine_min(self):
        """Minimiser of f over aff(face): x = G^-1 E mu, E^T x = 1."""
        p = self.p
        L = self.L[:p, :p]
        E = np.zeros((p, 2))
        isV = self.ord_isV[:p]
        E[isV, 0] = 1.0
        E[~isV, 1] = 1.0
        Y = solve_triangular(L, E, lower=True, check_finite=False)
        Y = solve_triangular(L.T, Y, lower=False, check_finite=False)
        M = E.T @ Y
        try:
            mu = np.linalg.solve(M, np.ones(2))
        except np.linalg.LinAlgError:
            return None
        xt = Y @ mu
        return xt if np.all(np.isfinite(xt)) else None

    def _qc(self, full):
        """QC-MNP correction (one truncation) or, if full, the exact
        minimum over the face (truncations until nonnegative).  Returns
        False when the system gives no descent (caller takes the LPS)."""
        self.n_qc += 1
        self.added_since_qc = 0
        moved = False
        for _ in range(self.p + 1):
            if self.p < 3:          # a vertex or an edge: nothing to solve
                return moved
            t0 = time.perf_counter()
            x = self._x()
            xt = self._affine_min()
            if xt is None:
                self.t_linalg += time.perf_counter() - t0
                if not moved:
                    self.n_qc_fallback += 1
                return moved
            dx = xt - x
            # descent check with the exact gradient on the face
            p = self.p
            fVc, fWc = self._f()
            grad = self._gather(fVc, -fWc)
            if float(grad @ dx) > 1e-12 * max(abs(float(grad @ x)), 1.0):
                self.t_linalg += time.perf_counter() - t0
                if not moved:
                    self.n_qc_fallback += 1
                return moved
            neg = xt < 0.0
            if not neg.any():
                tau, xn = 1.0, np.maximum(xt, 0.0)
                self.n_qc_full += 1
            else:
                r = x[neg] / (x[neg] - xt[neg])
                tau = float(r.min())
                xn = np.maximum(x + tau * dx, 0.0)
                xn[np.flatnonzero(neg)[np.argmin(r)]] = 0.0
                self.n_qc_trunc += 1
            dxa = xn - x
            self.sV += self.SVr[:, :p] @ dxa
            self.sW += self.SWr[:, :p] @ dxa
            self._set_x(xn)
            self.t_linalg += time.perf_counter() - t0
            moved = True
            self.n_drops += self._drop_zeros()
            if not neg.any() or not full:
                return True
        return moved

    def _fw(self, iF, jF, f_fw, zz, g_fw):
        asq = self.Vsq[iF] + self.Wsq[jF] - 2 * self._kvw(iF, jF) + 2 * self.invC
        den = asq - 2 * f_fw + zz
        t = 1.0 if den <= _EPS else min(g_fw / den, 1.0)
        om = 1.0 - t
        new = self.posV[iF] < 0 or self.posW[jF] < 0
        if self.posV[iF] < 0:
            self._add_key(True, iF)
        if self.posW[jF] < 0:
            self._add_key(False, jF)
        self.added_since_qc += int(new)    # a new atom (vertex) entered
        self.al_arr *= om
        self.be_arr *= om
        self.al_arr[iF] += t
        self.be_arr[jF] += t
        cv, cw = self.colV(iF), self.colW(jF)
        self.sV = om * self.sV + t * (cv[0] - cw[0])
        self.sW = om * self.sW + t * (cv[1] - cw[1])
        self.n_fw += 1

    def _refresh_scores(self):
        p = self.p
        if self.use_qc:
            x = self._x()
            self.sV = self.SVr[:, :p] @ x
            self.sW = self.SWr[:, :p] @ x
            return
        sV, sW = np.zeros(self.n), np.zeros(self.m)
        SVi, SWi = self._supports()
        for i in SVi:
            cv = self.colV(int(i))
            sV += self.al_arr[i] * cv[0]
            sW += self.al_arr[i] * cv[1]
        for j in SWi:
            cw = self.colW(int(j))
            sV -= self.be_arr[j] * cw[0]
            sW -= self.be_arr[j] * cw[1]
        self.sV, self.sW = sV, sW

    # ----- driver ------------------------------------------------------------
    def solve_distance(self, eps=1e-3, max_iter=2_000_000, time_cap=None,
                       i0=0, j0=0):
        t0 = time.perf_counter()
        self.al_arr = np.zeros(self.n)
        self.be_arr = np.zeros(self.m)
        self.posV = np.full(self.n, -1, dtype=np.int64)
        self.posW = np.full(self.m, -1, dtype=np.int64)
        self.p = 0
        self.added_since_qc = 0
        self.n_fw = self.n_lps = self.n_qc = self.n_qc_full = 0
        self.n_qc_trunc = self.n_qc_fallback = self.n_drops = 0
        self.n_refactor = 0
        self.t_linalg = 0.0
        self._alloc(256)
        self.al_arr[i0] = 1.0
        self.be_arr[j0] = 1.0
        self._add_key(True, i0)
        self._add_key(False, j0)
        self.sV = np.zeros(self.n)
        self.sW = np.zeros(self.m)
        self._refresh_scores()
        max_sup = self.p
        status, it, lb, ub = 'maxiter', 0, 0.0, 0.0
        qc_since_refactor = 0
        for it in range(1, max_iter + 1):
            fV, fW = self._f()
            SVi, SWi = self._supports()
            zz = float(self.al_arr[SVi] @ fV[SVi] - self.be_arr[SWi] @ fW[SWi])
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
            g_loc = float(fV[SVi].max() - fV[SVi].min()
                          + fW[SWi].max() - fW[SWi].min())
            if g_loc >= g_fw and g_loc > 0.0:
                done = False
                if self.corrective == 'fcfw' or (
                        self.corrective == 'qcmnp'
                        and self.added_since_qc >= self.qc_every):
                    done = self._qc(full=self.corrective == 'fcfw')
                    qc_since_refactor += 1
                if not done:
                    if self._lps(fV, fW, SVi, SWi):
                        self.n_drops += self._drop_zeros()
            else:
                self._fw(iF, jF, f_fw, zz, g_fw)
                self.n_drops += self._drop_zeros()
                if self.corrective == 'fcfw':
                    self._qc(full=True)
                    qc_since_refactor += 1
            max_sup = max(max_sup, self.p)
            if self.use_qc and qc_since_refactor >= self.refactor_every:
                self._refactor()
                qc_since_refactor = 0
            if it % self.refresh_every == 0:
                self._refresh_scores()
        el = time.perf_counter() - t0
        SVi, SWi = self._supports()
        self.alpha = {int(i): float(self.al_arr[i]) for i in SVi}
        self.beta = {int(j): float(self.be_arr[j]) for j in SWi}
        return CFWResult(distance=float(ub), lower_bound=float(lb),
                         iterations=it, time=el, status=status,
                         sparsity=self.p, n_fw=self.n_fw,
                         n_pw=self.n_lps, n_drops=self.n_drops,
                         alpha=dict(self.alpha), beta=dict(self.beta),
                         n_lps=self.n_lps, n_qc=self.n_qc,
                         n_qc_full=self.n_qc_full,
                         n_qc_trunc=self.n_qc_trunc,
                         n_qc_fallback=self.n_qc_fallback,
                         n_refactor=self.n_refactor, max_support=max_sup,
                         t_linalg=self.t_linalg)
