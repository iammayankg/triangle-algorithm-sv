"""Enhanced Triangle Algorithm (ETA) for convex-hull intersection / distance.

Implementation of the algorithm described in:

    Gupta, M. and Kalantari, B., "An Enhanced Triangle Algorithm for
    Large-Scale Support Vector Machine Optimization: A Comparative Study
    with Classical and Modern Solvers", JIDMIS, Vol 3 Issue 9s (2026).

Given two finite point sets V (rows of matrix V) and V' (rows of matrix W)
with convex hulls K and K', the Triangle Algorithm maintains an iterate
pair (p, q) in K x K'.

Definitions (distance duality):
  * v in V is a pivot for p (w.r.t. q)  iff  d(p, v) >= d(q, v),
    equivalently  2 v.(q - p) >= ||q||^2 - ||p||^2.
  * If neither class contains a pivot, (p, q) is a *witness pair*: the
    perpendicular bisector of segment [p, q] separates K and K'.

Triangle Algorithm I  (intersection / separation):
  Repeatedly move an iterate to the closest point on the segment joining
  it and a pivot; stop when d(p, q) <= eps (intersection, within
  tolerance) or when a witness pair is found (separation certificate).

Triangle Algorithm II (distance / optimal support):
  Starting from a witness pair, shrink the gap between the upper bound
  UB = d(p, q) and the lower bound LB obtained from parallel supporting
  hyperplanes orthogonal to h = p - q:
      LB = ( min_{v in V} h.v  -  max_{w in W} h.w ) / ||h||
  using the extreme points in direction h as (weak) pivots, until
  UB - LB <= eps * UB.

The four enhancements from the paper (all individually switchable):
  1. joint closest-point updates of the two line segments [p, v], [q, w]
  2. dot-product caching (incremental maintenance of V.p, V.q, W.p, W.q
     and lazy caching of Gram columns of used pivots)
  3. anti-zig-zag pivot strategy (midpoint of alternating pivots)
  4. prioritized searches over previously used (bounding) points before
     scanning the full input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

_EPS_NUM = 1e-12  # numerical slack for strict inequalities
# support threshold: a convex weight at or below this is treated as zero
# by every step (pair selection, away/pairwise/block bookkeeping)
_W_MIN = 1e-12


@dataclass
class TAResult:
    status: str            # 'intersect' | 'separated' | 'converged' |
                           # 'maxiter' | 'stalled' | 'timeout'
    distance: float        # final upper bound d(p, q)
    lower_bound: float     # final lower bound (TA II); 0.0 otherwise
    iterations: int
    time: float            # wall-clock seconds
    sparsity: int          # points with nonzero weight in the representation
    p: np.ndarray | None = None
    q: np.ndarray | None = None
    weights_V: dict = field(default_factory=dict)
    weights_W: dict = field(default_factory=dict)


class EnhancedTriangleAlgorithm:
    """Enhanced Triangle Algorithm over two point sets V (n x d), W (m x d)."""

    def __init__(self, V, W, *, joint_update=True, cache_dots=True,
                 anti_zigzag=True, zigzag_strategy='midpoint', prioritized=True,
                 step_mode='toward', refresh_every=100, full_scan_every=5,
                 shrink=False, shrink_every=50, shrink_min_frac=0.05,
                 block_size=16, drop_skip=True, seed=None,
                 k_max=32, adapt_every=50, cost_ratio=20.0,
                 drop_frac_max=0.1, k_init=16):
        """zigzag_strategy: what to do when an i,j,i,j pivot cycle is detected.
          'midpoint' - pivot on the midpoint of the two cycling vertices
                       (the strategy suggested in the paper);
          'away'     - away step: shrink the weight of the worst active
                       vertex (Guelat-Marcotte / away-step Frank-Wolfe);
          'pairwise' - transfer weight from the worst active vertex to the
                       best pivot (MDM / pairwise Frank-Wolfe step);
          None       - no remedy (anti_zigzag=False implies None).

        step_mode: 'toward' (default) uses the paper's toward-steps with the
        zig-zag remedy above; 'mdm' makes the pairwise weight-transfer the
        primary step every iteration (Mitchell-Demyanov-Malozemov), with a
        toward-step fallback - the right mode when the support is dense,
        e.g. soft-margin problems; 'block' amortises the O(n) scan over up
        to `block_size` weight transfers per iteration: the top-k
        receivers are paired with the worst-k active donors, per-pair
        magnitudes are clipped to donor capacity, and one exact line
        search is taken along the aggregated direction (guaranteed
        descent; falls back to single MDM transfers when no block forms).
        In block mode each iteration steps the side with the larger
        pairwise gap first and, with drop_skip=True (the analysed
        two-sided schedule, Lemma 6 of the paper), skips the other side
        whenever that first step is a drop step (an away step that
        removes a support index); drop_skip=False steps V then W
        unconditionally (the workshop-era order, kept as an ablation).

        shrink: gap-certified safe screening (solve_distance only).  With
        the current bounds [LB, UB], strong convexity of 1/2||x||^2 over
        the Minkowski difference gives ||h - h*|| <= r = sqrt(UB^2 - LB^2);
        a zero-weight point is certifiably outside the optimal support when
            (h.v_i - h.v_min) > r ||v_i - v_min||
        (symmetrically on the W side with the max).  Screened points are
        removed and the problem compacted; because the optimum's support
        survives, the reduced problem has the same optimum and later lower
        bounds remain valid.  Screening runs every `shrink_every`
        iterations and compacts only when at least `shrink_min_frac` of a
        class would be removed.
        """
        self.V = np.ascontiguousarray(V)
        self.W = np.ascontiguousarray(W)
        self.n, self.d = self.V.shape
        self.m = self.W.shape[0]
        if self.n == 0 or self.m == 0:
            raise ValueError(f'both point sets must be non-empty '
                             f'(got |V|={self.n}, |W|={self.m})')
        self.joint_update = joint_update
        self.cache_dots = cache_dots
        self.anti_zigzag = anti_zigzag and zigzag_strategy is not None
        self.zigzag_strategy = zigzag_strategy if self.anti_zigzag else None
        self.step_mode = step_mode
        # block_size='auto': adaptive block size (docs/aggregation_model.md
        # section 4).  Every adapt_every iterations the directional
        # diversity eta of the recent blocks is estimated from the exact
        # kappa = D/S the guard already computes, and k is set to the
        # work-optimal sqrt(cost_ratio (1 - eta) / eta), cost_ratio = a/b
        # the ratio of the fixed to the per-pair iteration cost; k is
        # halved instead when more than drop_frac_max of the recent
        # block calls had their leading pair capped.  Guard and fallback
        # are untouched, so Theorem 3 holds with k_max in the drop bound.
        self.block_auto = (block_size == 'auto')
        self.block_size = int(k_init if self.block_auto else block_size)
        self.k_max = int(k_max)
        self.adapt_every = int(adapt_every)
        self.cost_ratio = float(cost_ratio)
        self.drop_frac_max = float(drop_frac_max)
        self._adapt_win = []
        self.k_history = []       # (iteration, k, eta_hat, capped fraction)
        self.drop_skip = bool(drop_skip)
        # instrumentation for the two-sided schedule (block mode)
        self.n_drops = 0          # drop steps taken (either side)
        # per-block-step diagnostics (src/block_diagnostics.py): set to a
        # list to record gains, Q, kappa, candidate choice and a time
        # split for every block transfer; None = off (no overhead)
        self.diag = None
        self.n_drop_skips = 0     # second-side steps skipped after a drop
        self.step_kinds: dict[str, int] = {}
        self._last_drop = False   # set by the last block transfer
        self.last_iter_drop = False  # set by _step: first step was a drop
        self.shrink = shrink
        self.shrink_every = int(shrink_every)
        self.shrink_min_frac = float(shrink_min_frac)
        self.prioritized = prioritized
        self.refresh_every = refresh_every
        self.full_scan_every = full_scan_every
        self.rng = np.random.default_rng(seed)

        # squared row norms (cached once)
        self.Vsq = np.einsum('ij,ij->i', self.V, self.V).astype(np.float64)
        self.Wsq = np.einsum('ij,ij->i', self.W, self.W).astype(np.float64)

        # score tolerance scaled to the data's floating-point precision:
        # dot-product noise is O(eps_mach * sqrt(d) * scale)
        scale = max(1.0, float(self.Vsq.max()), float(self.Wsq.max()))
        self.tol = 16.0 * float(np.finfo(self.V.dtype).eps) \
            * float(np.sqrt(self.d)) * scale
        # denominator guard for the closest-point ratios (squared norms):
        # an absolute 1e-12 at unit scale and above, scaled with the data
        # below unit scale so that small instances are not stalled by an
        # absolute threshold
        scale_raw = max(float(self.Vsq.max()), float(self.Wsq.max()))
        self._eps_den = _EPS_NUM * min(1.0, scale_raw)
        # original-index bookkeeping: screening compacts V / W in place,
        # so these maps must persist across solves (see _orig_weights)
        self._origV = np.arange(self.n)
        self._origW = np.arange(self.m)
        # whether p, q (when present) are the full iterates, so that
        # ||p - q||^2 can be formed directly; subclasses whose iterates
        # carry implicit extra coordinates (soft margin) set this False
        self._explicit_iterates = True

        # lazy Gram-column caches: idx -> (V @ x, W @ x) for x = V[idx] / W[idx]
        self._colV: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._colW: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # oracle counter: O(nd) column evaluations (cache misses) - an
        # implementation-independent cost metric for benchmarking
        self.col_evals = 0
        # optional convergence trace: set to a list before solve_distance
        # to receive (iteration, elapsed_s, UB, LB_best) at every full scan
        self.trace = None

    # ------------------------------------------------------------------
    # state initialisation & cache plumbing
    # ------------------------------------------------------------------
    def _init_state(self, i0=0, j0=0):
        self.p = self.V[i0].astype(np.float64).copy()
        self.q = self.W[j0].astype(np.float64).copy()
        self.wV = {i0: 1.0}
        self.wW = {j0: 1.0}
        self._refresh_caches()
        self.hist_V: list = []   # recent pivot indices (anti-zig-zag)
        self.hist_W: list = []
        self.active_V: set[int] = {i0}   # prioritized-search subsets
        self.active_W: set[int] = {j0}

    def _refresh_caches(self):
        """Exact recomputation of all maintained dot products (O(nd))."""
        dt = self.V.dtype
        p32 = self.p.astype(dt, copy=False)
        q32 = self.q.astype(dt, copy=False)
        self.a = (self.V @ p32).astype(np.float64)   # V . p
        self.b = (self.V @ q32).astype(np.float64)   # V . q
        self.c = (self.W @ p32).astype(np.float64)   # W . p
        self.e = (self.W @ q32).astype(np.float64)   # W . q
        self.pp = float(self.p @ self.p)
        self.qq = float(self.q @ self.q)
        self.pq = float(self.p @ self.q)

    def _gram_col_V(self, i):
        """(V @ V[i], W @ V[i]) with lazy caching."""
        if self.cache_dots:
            col = self._colV.get(i)
            if col is None:
                self.col_evals += 1
                x = self.V[i]
                col = ((self.V @ x).astype(np.float64),
                       (self.W @ x).astype(np.float64))
                self._colV[i] = col
            return col
        x = self.V[i]
        return ((self.V @ x).astype(np.float64),
                (self.W @ x).astype(np.float64))

    def _gram_col_W(self, j):
        if self.cache_dots:
            col = self._colW.get(j)
            if col is None:
                self.col_evals += 1
                x = self.W[j]
                col = ((self.V @ x).astype(np.float64),
                       (self.W @ x).astype(np.float64))
                self._colW[j] = col
            return col
        x = self.W[j]
        return ((self.V @ x).astype(np.float64),
                (self.W @ x).astype(np.float64))

    def dist2(self):
        """||p - q||^2.  Computed from the explicit iterates when they
        exist: the cached-scalar form pp - 2 pq + qq cancels catastrophically
        when ||p||, ||q|| >> ||p - q|| and would certify a false
        intersection.  Implicit iterates (kernel, soft margin) only have
        the cached form; see _ub_certified for the matching noise floor."""
        if self._explicit_iterates and self.p is not None \
                and self.q is not None:
            diff = self.p - self.q
            return float(diff @ diff)
        return max(self.pp - 2.0 * self.pq + self.qq, 0.0)

    def _dist2_noise(self):
        """Rounding-error bound on the cached-scalar pp - 2 pq + qq."""
        return 4.0 * float(np.finfo(np.float64).eps) \
            * (abs(self.pp) + 2.0 * abs(self.pq) + abs(self.qq))

    def _ub_certified(self):
        """Upper bound safe to test against an intersection tolerance:
        with implicit iterates the cached distance is not trusted below
        its cancellation noise floor."""
        d2 = self.dist2()
        if not self._explicit_iterates or self.p is None or self.q is None:
            d2 = max(d2, self._dist2_noise())
        return float(np.sqrt(d2))

    def _lower_bound(self, ub):
        """Parallel-supporting-hyperplane lower bound for h = p - q,
        from the exact caches (call after _refresh_caches or a full scan)."""
        return (float(np.min(self.a - self.b))
                - float(np.max(self.c - self.e))) / ub

    # ------------------------------------------------------------------
    # iterate updates
    # ------------------------------------------------------------------
    def _apply_p(self, i, alpha, colVa=None, colVc=None, vsq=None,
                 aval=None, bval=None, wsplit=None):
        """p <- (1 - alpha) p + alpha x, x = V[i] (or a synthetic point)."""
        if alpha <= 0.0:
            return
        if colVa is None:
            colVa, colVc = self._gram_col_V(i)
            vsq, aval, bval = self.Vsq[i], self.a[i], self.b[i]
        om = 1.0 - alpha
        self.pq = om * self.pq + alpha * bval
        self.pp = om * om * self.pp + 2.0 * alpha * om * aval + alpha * alpha * vsq
        self.a = om * self.a + alpha * colVa
        self.c = om * self.c + alpha * colVc
        if wsplit is None:
            if self.p is not None:
                self.p *= om
                self.p += alpha * self.V[i].astype(np.float64)
            self.wV = {k: om * w for k, w in self.wV.items()}
            self.wV[i] = self.wV.get(i, 0.0) + alpha
            self.active_V.add(i)
        else:  # synthetic midpoint of vertices in wsplit
            if self.p is not None:
                self.p *= om
                for k, share in wsplit:
                    self.p += (alpha * share) * self.V[k].astype(np.float64)
            self.wV = {k: om * w for k, w in self.wV.items()}
            for k, share in wsplit:
                self.wV[k] = self.wV.get(k, 0.0) + alpha * share
                self.active_V.add(k)

    def _apply_q(self, j, beta, colWa=None, colWe=None, wsq=None,
                 eval_=None, cval=None, wsplit=None):
        """q <- (1 - beta) q + beta x, x = W[j] (or a synthetic point)."""
        if beta <= 0.0:
            return
        if colWa is None:
            colWa, colWe = self._gram_col_W(j)
            wsq, eval_, cval = self.Wsq[j], self.e[j], self.c[j]
        om = 1.0 - beta
        self.pq = om * self.pq + beta * cval
        self.qq = om * om * self.qq + 2.0 * beta * om * eval_ + beta * beta * wsq
        self.b = om * self.b + beta * colWa
        self.e = om * self.e + beta * colWe
        if wsplit is None:
            if self.q is not None:
                self.q *= om
                self.q += beta * self.W[j].astype(np.float64)
            self.wW = {k: om * w for k, w in self.wW.items()}
            self.wW[j] = self.wW.get(j, 0.0) + beta
            self.active_W.add(j)
        else:
            if self.q is not None:
                self.q *= om
                for k, share in wsplit:
                    self.q += (beta * share) * self.W[k].astype(np.float64)
            self.wW = {k: om * w for k, w in self.wW.items()}
            for k, share in wsplit:
                self.wW[k] = self.wW.get(k, 0.0) + beta * share
                self.active_W.add(k)

    def _alpha_single_p(self, i):
        """Step to the closest point to q on segment [p, V[i]]."""
        num = self.b[i] - self.pq - self.a[i] + self.pp      # (q-p).(v-p)
        den = self.Vsq[i] - 2.0 * self.a[i] + self.pp        # ||v-p||^2
        if den <= self._eps_den:
            return 0.0
        return float(np.clip(num / den, 0.0, 1.0))

    def _beta_single_q(self, j):
        num = self.c[j] - self.pq - self.e[j] + self.qq      # (p-q).(w-q)
        den = self.Wsq[j] - 2.0 * self.e[j] + self.qq        # ||w-q||^2
        if den <= self._eps_den:
            return 0.0
        return float(np.clip(num / den, 0.0, 1.0))

    def _joint_step(self, i, j):
        """Closest points between segments [p, V[i]] and [q, W[j]].

        Solves min ||(p + a u) - (q + b r)||, u = v - p, r = w - q,
        with the standard two-line closest-point equations clipped to [0,1].
        Returns (alpha, beta).  All dot products come from the caches.
        """
        colVa, colVc = self._gram_col_V(i)
        A = self.Vsq[i] - 2.0 * self.a[i] + self.pp                  # u.u
        C = self.Wsq[j] - 2.0 * self.e[j] + self.qq                  # r.r
        B = colVc[j] - self.b[i] - self.c[j] + self.pq               # u.r
        D = self.a[i] - self.b[i] - self.pp + self.pq                # u.(p-q)
        E = self.c[j] - self.e[j] - self.pq + self.qq                # r.(p-q)
        if A <= self._eps_den and C <= self._eps_den:
            return 0.0, 0.0
        denom = A * C - B * B
        if abs(denom) > _EPS_NUM * max(A * C, 1.0):
            alpha = (B * E - C * D) / denom
            beta = (A * E - B * D) / denom
        else:  # (near-)parallel segments
            alpha = -D / A if A > self._eps_den else 0.0
            beta = 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        # re-optimise beta given clamped alpha, then alpha given beta
        if C > self._eps_den:
            beta = float(np.clip((E + alpha * B) / C, 0.0, 1.0))
        else:
            beta = 0.0
        if A > self._eps_den:
            alpha = float(np.clip((beta * B - D) / A, 0.0, 1.0))
        return alpha, beta

    # ------------------------------------------------------------------
    # pivot searches
    # ------------------------------------------------------------------
    def _scores_ta1(self):
        """Pivot scores.  score > 0  <=>  the point is a pivot."""
        gap = self.qq - self.pp
        fV = 2.0 * (self.b - self.a) - gap        # class V, pivot for p
        fW = 2.0 * (self.c - self.e) + gap        # class W, pivot for q
        return fV, fW

    def _scores_ta2(self):
        """Descent scores: score > 0 <=> moving toward the point reduces d(p,q)."""
        gV = (self.b - self.a) - (self.pq - self.pp)
        gW = (self.c - self.e) - (self.pq - self.qq)
        return gV, gW

    @staticmethod
    def _best_in(scores, subset):
        if not subset:
            return None, -np.inf
        idx = np.fromiter(subset, dtype=np.int64)
        loc = int(np.argmax(scores[idx]))
        return int(idx[loc]), float(scores[idx][loc])

    def _select(self, scores_fn, it, tol):
        """Pick the best pivot per class; prioritized search over active sets.

        Returns (iV, sV, iW, sW, full_scan_done).  Indices are None when the
        class has no pivot with score > tol (as far as the scan looked).
        """
        fV, fW = scores_fn()
        full = (not self.prioritized) or (it % self.full_scan_every == 0)
        if not full:
            iV, sV = self._best_in(fV, self.active_V)
            iW, sW = self._best_in(fW, self.active_W)
            if sV > tol or sW > tol:
                return (iV if sV > tol else None, sV,
                        iW if sW > tol else None, sW, False)
            # nothing in the priority subsets -> fall through to full scan
        iV = int(np.argmax(fV)); sV = float(fV[iV])
        iW = int(np.argmax(fW)); sW = float(fW[iW])
        return (iV if sV > tol else None, sV,
                iW if sW > tol else None, sW, True)

    # ------------------------------------------------------------------
    # anti-zig-zag
    # ------------------------------------------------------------------
    def _zigzag_candidate(self, hist):
        """Detect pivot oscillation in recent history.

        Returns (i, j) when the incoming pivot hist[-1] was already used
        within the previous three steps with some other pivot in between
        (this covers the classic i,j,i,j two-cycle and short cycles).
        """
        if not self.anti_zigzag or len(hist) < 3:
            return None
        cur = hist[-1]
        if cur < 0:
            return None
        for back in (hist[-3], hist[-4] if len(hist) >= 4 else -9):
            if back == cur:
                other = hist[-2]
                if other >= 0 and other != cur:
                    return (cur, other)
        return None

    def _apply_p_midpoint(self, i, j):
        """Move p toward the midpoint of V[i] and V[j] (anti-zig-zag)."""
        ci_a, ci_c = self._gram_col_V(i)
        cj_a, cj_c = self._gram_col_V(j)
        colVa = 0.5 * (ci_a + cj_a)
        colVc = 0.5 * (ci_c + cj_c)
        g_ij = ci_a[j]                                   # V[i] . V[j]
        vsq = 0.25 * (self.Vsq[i] + 2.0 * g_ij + self.Vsq[j])
        aval = 0.5 * (self.a[i] + self.a[j])
        bval = 0.5 * (self.b[i] + self.b[j])
        num = bval - self.pq - aval + self.pp
        den = vsq - 2.0 * aval + self.pp
        if den <= self._eps_den:
            return False
        alpha = float(np.clip(num / den, 0.0, 1.0))
        if alpha <= 0.0:
            return False
        self._apply_p(None, alpha, colVa, colVc, vsq, aval, bval,
                      wsplit=[(i, 0.5), (j, 0.5)])
        return True

    def _apply_q_midpoint(self, i, j):
        ci_a, ci_e = self._gram_col_W(i)
        cj_a, cj_e = self._gram_col_W(j)
        colWa = 0.5 * (ci_a + cj_a)
        colWe = 0.5 * (ci_e + cj_e)
        g_ij = ci_e[j]
        wsq = 0.25 * (self.Wsq[i] + 2.0 * g_ij + self.Wsq[j])
        eval_ = 0.5 * (self.e[i] + self.e[j])
        cval = 0.5 * (self.c[i] + self.c[j])
        num = cval - self.pq - eval_ + self.qq
        den = wsq - 2.0 * eval_ + self.qq
        if den <= self._eps_den:
            return False
        beta = float(np.clip(num / den, 0.0, 1.0))
        if beta <= 0.0:
            return False
        self._apply_q(None, beta, colWa, colWe, wsq, eval_, cval,
                      wsplit=[(i, 0.5), (j, 0.5)])
        return True

    # ------------------------------------------------------------------
    # away and pairwise (MDM) steps
    # ------------------------------------------------------------------
    @staticmethod
    def _active_indices(weights):
        """Indices with weight above the support threshold, as an int64
        array, extracted at C speed: the per-iteration Python loops over
        the weight dicts were the dominant fixed cost on dense supports."""
        n = len(weights)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        keys = np.fromiter(weights.keys(), dtype=np.int64, count=n)
        vals = np.fromiter(weights.values(), dtype=np.float64, count=n)
        return keys[(keys >= 0) & (vals > _W_MIN)]

    def _worst_active_p(self):
        """Active vertex of V most opposed to the descent direction:
        argmin over support of (q - p).u = (b - a)[u]."""
        act = self._active_indices(self.wV)
        if act.size == 0:
            return None
        s = self.b - self.a
        return int(act[np.argmin(s[act])])

    def _worst_active_q(self):
        act = self._active_indices(self.wW)
        if act.size == 0:
            return None
        s = self.c - self.e
        return int(act[np.argmin(s[act])])

    def _away_p(self, u):
        """Away step: p <- p + gamma (p - V[u]), gamma <= w_u / (1 - w_u)."""
        wu = self.wV.get(u, 0.0)
        if wu <= _W_MIN or wu >= 1.0 - 1e-12:
            return False
        num = self.pq - self.b[u] - self.pp + self.a[u]   # (q-p).(p-u)
        den = self.pp - 2.0 * self.a[u] + self.Vsq[u]     # ||p-u||^2
        if den <= self._eps_den or num <= 0.0:
            return False
        g = min(num / den, wu / (1.0 - wu))
        colVa, colVc = self._gram_col_V(u)
        au, bu = self.a[u], self.b[u]
        op = 1.0 + g
        self.pq = op * self.pq - g * bu
        self.pp = op * op * self.pp - 2.0 * g * op * au + g * g * self.Vsq[u]
        self.a = op * self.a - g * colVa
        self.c = op * self.c - g * colVc
        if self.p is not None:
            self.p *= op
            self.p -= g * self.V[u].astype(np.float64)
        self.wV = {k: op * w for k, w in self.wV.items()}
        self.wV[u] = self.wV[u] - g
        if self.wV[u] <= _W_MIN:
            del self.wV[u]
        return True

    def _away_q(self, u):
        wu = self.wW.get(u, 0.0)
        if wu <= _W_MIN or wu >= 1.0 - 1e-12:
            return False
        num = self.pq - self.c[u] - self.qq + self.e[u]   # (p-q).(q-u)
        den = self.qq - 2.0 * self.e[u] + self.Wsq[u]     # ||q-u||^2
        if den <= self._eps_den or num <= 0.0:
            return False
        g = min(num / den, wu / (1.0 - wu))
        colWa, colWe = self._gram_col_W(u)
        eu, cu = self.e[u], self.c[u]
        op = 1.0 + g
        self.pq = op * self.pq - g * cu
        self.qq = op * op * self.qq - 2.0 * g * op * eu + g * g * self.Wsq[u]
        self.b = op * self.b - g * colWa
        self.e = op * self.e - g * colWe
        if self.q is not None:
            self.q *= op
            self.q -= g * self.W[u].astype(np.float64)
        self.wW = {k: op * w for k, w in self.wW.items()}
        self.wW[u] = self.wW[u] - g
        if self.wW[u] <= _W_MIN:
            del self.wW[u]
        return True

    def _pairwise_p(self, v, u):
        """MDM step: transfer weight from active u to pivot v,
        p <- p + gamma (V[v] - V[u]), gamma <= w_u."""
        wu = self.wV.get(u, 0.0)
        if wu <= _W_MIN or v == u:
            return False
        colv_a, colv_c = self._gram_col_V(v)
        colu_a, colu_c = self._gram_col_V(u)
        num = (self.b[v] - self.b[u]) - (self.a[v] - self.a[u])  # (q-p).(v-u)
        den = self.Vsq[v] - 2.0 * colv_a[u] + self.Vsq[u]        # ||v-u||^2
        if den <= self._eps_den or num <= 0.0:
            return False
        g = min(num / den, wu)
        self.pq += g * (self.b[v] - self.b[u])
        self.pp += 2.0 * g * (self.a[v] - self.a[u]) + g * g * den
        self.a += g * (colv_a - colu_a)
        self.c += g * (colv_c - colu_c)
        if self.p is not None:
            self.p += g * (self.V[v].astype(np.float64)
                           - self.V[u].astype(np.float64))
        self.wV[v] = self.wV.get(v, 0.0) + g
        self.wV[u] = wu - g
        if self.wV[u] <= _W_MIN:
            del self.wV[u]
        self.active_V.add(v)
        return True

    def _pairwise_q(self, v, u):
        wu = self.wW.get(u, 0.0)
        if wu <= _W_MIN or v == u:
            return False
        colv_a, colv_e = self._gram_col_W(v)
        colu_a, colu_e = self._gram_col_W(u)
        num = (self.c[v] - self.c[u]) - (self.e[v] - self.e[u])  # (p-q).(v-u)
        den = self.Wsq[v] - 2.0 * colv_e[u] + self.Wsq[u]
        if den <= self._eps_den or num <= 0.0:
            return False
        g = min(num / den, wu)
        self.pq += g * (self.c[v] - self.c[u])
        self.qq += 2.0 * g * (self.e[v] - self.e[u]) + g * g * den
        self.b += g * (colv_a - colu_a)
        self.e += g * (colv_e - colu_e)
        if self.q is not None:
            self.q += g * (self.W[v].astype(np.float64)
                           - self.W[u].astype(np.float64))
        self.wW[v] = self.wW.get(v, 0.0) + g
        self.wW[u] = wu - g
        if self.wW[u] <= _W_MIN:
            del self.wW[u]
        self.active_W.add(v)
        return True

    # ------------------------------------------------------------------
    # block transfers: k paired MDM moves per scan, one exact line search
    # along the aggregated direction
    # ------------------------------------------------------------------
    def _block_pairs(self, s, weights, cap, k):
        """Pair top-k receivers with worst-k active donors under `s`.
        Returns [(recv, donor, gamma)] with per-pair capacity clipping."""
        # active donors as an array (the Python-level comprehension over
        # the weight dict was the dominant per-call cost on dense supports)
        act_arr = self._active_indices(weights)
        if act_arr.size == 0:
            return []
        n = len(s)
        k = max(1, min(k, int(act_arr.size)))
        if k >= n:
            recv = np.argsort(s)[::-1][:k]
        else:
            part = np.argpartition(s, n - k)[n - k:]
            recv = part[np.argsort(s[part])[::-1]]
        if k >= len(act_arr):
            don = act_arr[np.argsort(s[act_arr])]
        else:
            part = np.argpartition(s[act_arr], k)[:k]
            don = act_arr[part[np.argsort(s[act_arr][part])]]
        # keep the worst active donor as don[0] and drop receivers that are
        # donors, so pair 1 is always the MDM pair (global argmax, worst
        # active) that the guard and the analysis refer to
        don = [int(u) for u in don]
        recv = [int(r) for r in recv]
        if recv and recv[0] in don and recv[0] != don[0]:
            # the global argmax is itself a (non-worst) active point: keep
            # it as receiver 1 and remove it from the donor list, so that
            # pair 1 stays the MDM pair (global argmax, worst active)
            don = [u for u in don if u != recv[0]]
        dset = set(don)
        recv = [r for r in recv if r not in dset]
        pairs = []
        for r, u in zip(recv, don):
            num = s[r] - s[u]
            if num <= self.tol:
                continue
            g = weights[u]
            if cap is not None:
                g = min(g, cap - weights.get(r, 0.0))
            if g > 0.0:
                pairs.append((r, u, num, g))
        return pairs

    def _block_transfer_V(self, k, s=None):
        if self.diag is None:
            return self._block_transfer_V_impl(k, s)
        self._diag_rec = None
        t0 = time.perf_counter()
        ce0, nd0, d2_0 = self.col_evals, self.n_drops, self.dist2()
        self._diag_ce = ce0
        ok = self._block_transfer_V_impl(k, s)
        rec = self._diag_rec
        if rec is not None:
            rec.update(side='V', ok=ok, t_total=time.perf_counter() - t0,
                       misses=self.col_evals - ce0, drop=self.n_drops > nd0,
                       d2_before=d2_0, d2_after=self.dist2())
            rec['t_update'] = max(rec['t_total'] - rec['t_pairs'] - rec['t_cols']
                                  - rec['t_assemble'] - rec['t_guard']
                                  - rec['t_single'], 0.0)
            self.diag.append(rec)
        return ok

    def _block_transfer_V_impl(self, k, s=None):
        self._last_drop = False
        dg = self.diag is not None
        if dg:
            t0 = time.perf_counter()
        if s is None:
            s = self.b - self.a
        raw = self._block_pairs(s, self.wV, None, k)
        if dg:
            t1 = time.perf_counter()
            t_col1, miss1, dens = 0.0, 0, []
        if not raw:
            return False
        # per-pair unclipped optimum, clipped to capacity
        pairs = []
        for j_, (r, u, num, gmax) in enumerate(raw):
            colr = self._gram_col_V(r)[0]
            if dg and j_ == 0:
                t_col1 = time.perf_counter() - t1
                miss1 = self.col_evals - self._diag_ce
            den = self.Vsq[r] - 2.0 * colr[u] + self.Vsq[u]
            if den <= self._eps_den:
                continue
            pairs.append((r, u, min(num / den, gmax)))
            if dg:
                dens.append((num, den, gmax))
        if not pairs:
            return False
        # a single surviving pair is handled by the same guard and case-(b)
        # fallback below (the block then coincides with the MDM step); an
        # early pairwise return here would bypass the away fallback that
        # Theorem 3 requires when pair 1 is capacity-clipped
        # exact line search along dvec = sum g_j (v_rj - v_uj)
        R = np.array([p_[0] for p_ in pairs])
        U = np.array([p_[1] for p_ in pairs])
        G = np.array([p_[2] for p_ in pairs])
        num_t = float(G @ (s[R] - s[U]))
        cols = {i: self._gram_col_V(i)[0]
                for pr in pairs for i in pr[:2]}
        if dg:
            t2 = time.perf_counter()
        # Gram matrix of the pair directions from k'^2 cached entries (no
        # k' x n column copies): M_ij = <d_i, d_j>
        CRR = np.array([cols[int(r)][R] for r in R])
        CRU = np.array([cols[int(r)][U] for r in R])
        CUR = np.array([cols[int(u)][R] for u in U])
        CUU = np.array([cols[int(u)][U] for u in U])
        M = (CRR - CRU) - (CUR - CUU)
        den_t = float(G @ M @ G)
        if den_t <= self._eps_den or num_t <= 0.0:
            t, delta_B = 0.0, -np.inf      # degenerate block: never chosen
        else:
            t = min(num_t / den_t, 1.0)
            delta_B = t * num_t - 0.5 * t * t * den_t
        if dg:
            t3 = time.perf_counter()
        # guard (Theorem 3): fall back to the single MDM step whenever its
        # exact gain exceeds the block's - the guarded step never makes
        # less progress than pairwise FW, so the PFW linear rate is
        # inherited with identical constants
        r1, u1, g1 = pairs[0]
        s1 = self.Vsq[r1] - 2.0 * cols[int(r1)][u1] + self.Vsq[u1]
        gap1 = s[r1] - s[u1]
        delta_1 = g1 * gap1 - 0.5 * g1 * g1 * s1
        best, kind = delta_B, 'block'
        if delta_1 > best:
            best, kind = delta_1, 'mdm'
        cap1 = self.wV.get(u1, 0.0)
        if self.block_auto:
            self._adapt_win.append((
                len(pairs),
                den_t / num_t if (num_t > 0.0 and np.isfinite(delta_B)) else np.nan,
                bool(s1 > self._eps_den and gap1 / s1 > cap1 * (1.0 + 1e-12))))
        if dg:
            tiny = max(abs(delta_1), 1e-300)
            self._diag_rec = dict(
                it=getattr(self, '_diag_it', -1), k=k, kp=len(pairs),
                capped=sum(1 for nu_, de_, gm_ in dens if nu_ / de_ > gm_),
                cap1=bool(dens and dens[0][0] / dens[0][1] > dens[0][2]),
                dB=float(delta_B), d1=float(delta_1), S=num_t, D=den_t,
                Q_unc=sum(nu_ * nu_ / (2.0 * de_) for nu_, de_, gm_ in dens) / tiny,
                Q_real=sum(min(nu_ / de_, gm_) * nu_ - 0.5 * min(nu_ / de_, gm_) ** 2 * de_
                           for nu_, de_, gm_ in dens) / tiny,
                # the theorem's assumptions, recorded separately from the
                # aggregate Q and kappa: maximum absolute pairwise cosine of
                # the retained directions (eta of Prop. 12 must dominate it),
                # minimum relative uncapped pair gain (beta-flatness) and
                # whether every retained pair is uncapped
                max_cos=(float(np.max(np.abs(M / np.sqrt(np.outer(np.diag(M), np.diag(M))))
                                      [~np.eye(len(pairs), dtype=bool)])) if len(pairs) > 1 else 0.0),
                beta_min=(min(nu_ * nu_ / (2.0 * de_) for nu_, de_, gm_ in dens)
                          / max(dens[0][0] ** 2 / (2.0 * dens[0][1]), 1e-300)),
                all_uncapped=all(nu_ / de_ <= gm_ for nu_, de_, gm_ in dens),
                support=len(self.wV), n_drops=self.n_drops,
                t_pairs=t1 - t0, t_cols=t2 - t1, t_col1=t_col1, miss1=miss1,
                t_assemble=t3 - t2, t_guard=0.0, t_single=0.0, kind=kind)
        if s1 > self._eps_den and gap1 / s1 > cap1 * (1.0 + 1e-12):
            # Theorem 6 case (b): pair 1 capacity-clipped -> away/toward
            num_a = self.pq - self.b[u1] - self.pp + self.a[u1]
            den_a = self.pp - 2.0 * self.a[u1] + self.Vsq[u1]
            if num_a > 0.0 and den_a > self._eps_den and cap1 < 1.0 - 1e-12:
                if num_a / den_a >= cap1 / (1.0 - cap1):
                    if self._away_p(u1):             # drop step
                        if dg:
                            self._diag_rec['kind'] = 'drop'
                            self._diag_rec['t_guard'] = time.perf_counter() - t3
                        self._last_drop = True
                        self.n_drops += 1
                        self.step_kinds['drop'] = self.step_kinds.get('drop', 0) + 1
                        return True
                    # away step refused (weight at the support threshold):
                    # fall through to the remaining candidates
                else:
                    d_away = num_a * num_a / (2.0 * den_a)
                    if d_away > best:
                        best, kind = d_away, 'away'
            num_f = self.b[r1] - self.pq - self.a[r1] + self.pp
            den_f = self.Vsq[r1] - 2.0 * self.a[r1] + self.pp
            if num_f > 0.0 and den_f > self._eps_den:
                al = min(num_f / den_f, 1.0)
                d_tow = al * num_f - 0.5 * al * al * den_f
                if d_tow > best:
                    best, kind = d_tow, 'toward'
        if dg:
            t4 = time.perf_counter()
            rec = self._diag_rec
            rec['kind'] = kind
            rec['t_guard'] = t4 - t3
            # cost of the single MDM update from this same state, dry run
            ar_, cr_ = self._gram_col_V(r1)
            au_, cu_ = self._gram_col_V(u1)
            _ = self.a + g1 * (ar_ - au_)
            _ = self.c + g1 * (cr_ - cu_)
            rec['t_single'] = time.perf_counter() - t4
        self.step_kinds[kind] = self.step_kinds.get(kind, 0) + 1
        if kind == 'mdm':
            return self._pairwise_p(r1, u1)
        if kind == 'away':
            return self._away_p(u1)
        if kind == 'toward':
            alpha = self._alpha_single_p(r1)
            if alpha > 0.0:
                self._apply_p(r1, alpha)
                return True
        # scalar updates
        self.pq += t * float(G @ (self.b[R] - self.b[U]))
        self.pp += 2.0 * t * float(G @ (self.a[R] - self.a[U])) \
            + t * t * den_t
        # batched cache update via net per-index coefficients (one GEMV)
        coeff: dict[int, float] = {}
        for r, u, g in pairs:
            coeff[r] = coeff.get(r, 0.0) + t * g
            coeff[u] = coeff.get(u, 0.0) - t * g
        idxs = np.fromiter(coeff.keys(), dtype=np.int64)
        cvec = np.fromiter(coeff.values(), dtype=np.float64)
        CA = np.stack([self._gram_col_V(int(i))[0] for i in idxs])
        CC = np.stack([self._gram_col_V(int(i))[1] for i in idxs])
        self.a += cvec @ CA
        self.c += cvec @ CC
        if self.p is not None:
            self.p += cvec @ self.V[idxs].astype(np.float64)
        for i, ci in coeff.items():
            w = self.wV.get(i, 0.0) + ci
            if w <= _W_MIN:
                self.wV.pop(i, None)
            else:
                self.wV[i] = w
            self.active_V.add(i)
        return True

    def _block_transfer_W(self, k, s=None):
        if self.diag is None:
            return self._block_transfer_W_impl(k, s)
        self._diag_rec = None
        t0 = time.perf_counter()
        ce0, nd0, d2_0 = self.col_evals, self.n_drops, self.dist2()
        self._diag_ce = ce0
        ok = self._block_transfer_W_impl(k, s)
        rec = self._diag_rec
        if rec is not None:
            rec.update(side='W', ok=ok, t_total=time.perf_counter() - t0,
                       misses=self.col_evals - ce0, drop=self.n_drops > nd0,
                       d2_before=d2_0, d2_after=self.dist2())
            rec['t_update'] = max(rec['t_total'] - rec['t_pairs'] - rec['t_cols']
                                  - rec['t_assemble'] - rec['t_guard']
                                  - rec['t_single'], 0.0)
            self.diag.append(rec)
        return ok

    def _block_transfer_W_impl(self, k, s=None):
        self._last_drop = False
        dg = self.diag is not None
        if dg:
            t0 = time.perf_counter()
        if s is None:
            s = self.c - self.e
        raw = self._block_pairs(s, self.wW, None, k)
        if dg:
            t1 = time.perf_counter()
            t_col1, miss1, dens = 0.0, 0, []
        if not raw:
            return False
        # per-pair unclipped optimum, clipped to capacity
        pairs = []
        for j_, (r, u, num, gmax) in enumerate(raw):
            colr = self._gram_col_W(r)[1]
            if dg and j_ == 0:
                t_col1 = time.perf_counter() - t1
                miss1 = self.col_evals - self._diag_ce
            den = self.Wsq[r] - 2.0 * colr[u] + self.Wsq[u]
            if den <= self._eps_den:
                continue
            pairs.append((r, u, min(num / den, gmax)))
            if dg:
                dens.append((num, den, gmax))
        if not pairs:
            return False
        # single pair: same guard and case-(b) fallback as below (see V side)
        R = np.array([p_[0] for p_ in pairs])
        U = np.array([p_[1] for p_ in pairs])
        G = np.array([p_[2] for p_ in pairs])
        num_t = float(G @ (s[R] - s[U]))
        cols = {j: self._gram_col_W(j)[1]
                for pr in pairs for j in pr[:2]}
        if dg:
            t2 = time.perf_counter()
        CRR = np.array([cols[int(r)][R] for r in R])
        CRU = np.array([cols[int(r)][U] for r in R])
        CUR = np.array([cols[int(u)][R] for u in U])
        CUU = np.array([cols[int(u)][U] for u in U])
        M = (CRR - CRU) - (CUR - CUU)
        den_t = float(G @ M @ G)
        if den_t <= self._eps_den or num_t <= 0.0:
            t, delta_B = 0.0, -np.inf      # degenerate block: never chosen
        else:
            t = min(num_t / den_t, 1.0)
            delta_B = t * num_t - 0.5 * t * t * den_t
        if dg:
            t3 = time.perf_counter()
        r1, u1, g1 = pairs[0]
        s1 = self.Wsq[r1] - 2.0 * cols[int(r1)][u1] + self.Wsq[u1]
        gap1 = s[r1] - s[u1]
        delta_1 = g1 * gap1 - 0.5 * g1 * g1 * s1
        best, kind = delta_B, 'block'
        if delta_1 > best:
            best, kind = delta_1, 'mdm'
        cap1 = self.wW.get(u1, 0.0)
        if self.block_auto:
            self._adapt_win.append((
                len(pairs),
                den_t / num_t if (num_t > 0.0 and np.isfinite(delta_B)) else np.nan,
                bool(s1 > self._eps_den and gap1 / s1 > cap1 * (1.0 + 1e-12))))
        if dg:
            tiny = max(abs(delta_1), 1e-300)
            self._diag_rec = dict(
                it=getattr(self, '_diag_it', -1), k=k, kp=len(pairs),
                capped=sum(1 for nu_, de_, gm_ in dens if nu_ / de_ > gm_),
                cap1=bool(dens and dens[0][0] / dens[0][1] > dens[0][2]),
                dB=float(delta_B), d1=float(delta_1), S=num_t, D=den_t,
                Q_unc=sum(nu_ * nu_ / (2.0 * de_) for nu_, de_, gm_ in dens) / tiny,
                Q_real=sum(min(nu_ / de_, gm_) * nu_ - 0.5 * min(nu_ / de_, gm_) ** 2 * de_
                           for nu_, de_, gm_ in dens) / tiny,
                # the theorem's assumptions, recorded separately from the
                # aggregate Q and kappa: maximum absolute pairwise cosine of
                # the retained directions (eta of Prop. 12 must dominate it),
                # minimum relative uncapped pair gain (beta-flatness) and
                # whether every retained pair is uncapped
                max_cos=(float(np.max(np.abs(M / np.sqrt(np.outer(np.diag(M), np.diag(M))))
                                      [~np.eye(len(pairs), dtype=bool)])) if len(pairs) > 1 else 0.0),
                beta_min=(min(nu_ * nu_ / (2.0 * de_) for nu_, de_, gm_ in dens)
                          / max(dens[0][0] ** 2 / (2.0 * dens[0][1]), 1e-300)),
                all_uncapped=all(nu_ / de_ <= gm_ for nu_, de_, gm_ in dens),
                support=len(self.wW), n_drops=self.n_drops,
                t_pairs=t1 - t0, t_cols=t2 - t1, t_col1=t_col1, miss1=miss1,
                t_assemble=t3 - t2, t_guard=0.0, t_single=0.0, kind=kind)
        if s1 > self._eps_den and gap1 / s1 > cap1 * (1.0 + 1e-12):
            num_a = self.pq - self.c[u1] - self.qq + self.e[u1]
            den_a = self.qq - 2.0 * self.e[u1] + self.Wsq[u1]
            if num_a > 0.0 and den_a > self._eps_den and cap1 < 1.0 - 1e-12:
                if num_a / den_a >= cap1 / (1.0 - cap1):
                    if self._away_q(u1):             # drop step
                        if dg:
                            self._diag_rec['kind'] = 'drop'
                            self._diag_rec['t_guard'] = time.perf_counter() - t3
                        self._last_drop = True
                        self.n_drops += 1
                        self.step_kinds['drop'] = self.step_kinds.get('drop', 0) + 1
                        return True
                    # away step refused (weight at the support threshold):
                    # fall through to the remaining candidates
                else:
                    d_away = num_a * num_a / (2.0 * den_a)
                    if d_away > best:
                        best, kind = d_away, 'away'
            num_f = self.c[r1] - self.pq - self.e[r1] + self.qq
            den_f = self.Wsq[r1] - 2.0 * self.e[r1] + self.qq
            if num_f > 0.0 and den_f > self._eps_den:
                al = min(num_f / den_f, 1.0)
                d_tow = al * num_f - 0.5 * al * al * den_f
                if d_tow > best:
                    best, kind = d_tow, 'toward'
        if dg:
            t4 = time.perf_counter()
            rec = self._diag_rec
            rec['kind'] = kind
            rec['t_guard'] = t4 - t3
            # cost of the single MDM update from this same state, dry run
            ar_, cr_ = self._gram_col_W(r1)
            au_, cu_ = self._gram_col_W(u1)
            _ = self.b + g1 * (ar_ - au_)
            _ = self.e + g1 * (cr_ - cu_)
            rec['t_single'] = time.perf_counter() - t4
        self.step_kinds[kind] = self.step_kinds.get(kind, 0) + 1
        if kind == 'mdm':
            return self._pairwise_q(r1, u1)
        if kind == 'away':
            return self._away_q(u1)
        if kind == 'toward':
            beta = self._beta_single_q(r1)
            if beta > 0.0:
                self._apply_q(r1, beta)
                return True
        self.pq += t * float(G @ (self.c[R] - self.c[U]))
        self.qq += 2.0 * t * float(G @ (self.e[R] - self.e[U])) \
            + t * t * den_t
        coeff: dict[int, float] = {}
        for r, u, g in pairs:
            coeff[r] = coeff.get(r, 0.0) + t * g
            coeff[u] = coeff.get(u, 0.0) - t * g
        idxs = np.fromiter(coeff.keys(), dtype=np.int64)
        cvec = np.fromiter(coeff.values(), dtype=np.float64)
        CB = np.stack([self._gram_col_W(int(j))[0] for j in idxs])
        CE = np.stack([self._gram_col_W(int(j))[1] for j in idxs])
        self.b += cvec @ CB
        self.e += cvec @ CE
        if self.q is not None:
            self.q += cvec @ self.W[idxs].astype(np.float64)
        for j, cj in coeff.items():
            w = self.wW.get(j, 0.0) + cj
            if w <= _W_MIN:
                self.wW.pop(j, None)
            else:
                self.wW[j] = w
            self.active_W.add(j)
        return True

    # ------------------------------------------------------------------
    # gap-certified safe screening (shrinking)
    # ------------------------------------------------------------------
    def _screen_and_compact(self, lb_best):
        """Remove points certified to be outside the optimal support."""
        ub2 = self.dist2()
        lb = max(lb_best, 0.0)
        r2 = ub2 - lb * lb
        if r2 <= 0.0:
            return
        r = float(np.sqrt(r2)) * (1.0 + 1e-9)
        # V side: active points at the optimum minimise h*.v
        sV = self.a - self.b
        i0 = int(np.argmin(sV))
        colVa, _ = self._gram_col_V(i0)
        dV = np.sqrt(np.maximum(self.Vsq - 2.0 * colVa + self.Vsq[i0], 0.0))
        keepV = (sV - sV[i0]) <= r * dV + self.tol
        for i, w in self.wV.items():
            if w > 0.0:
                keepV[i] = True
        keepV[i0] = True
        # W side: active points at the optimum maximise h*.w
        sW = self.c - self.e
        j0 = int(np.argmax(sW))
        _, colWe = self._gram_col_W(j0)
        dW = np.sqrt(np.maximum(self.Wsq - 2.0 * colWe + self.Wsq[j0], 0.0))
        keepW = (sW[j0] - sW) <= r * dW + self.tol
        for j, w in self.wW.items():
            if w > 0.0:
                keepW[j] = True
        keepW[j0] = True
        if (1.0 - keepV.mean()) < self.shrink_min_frac and \
                (1.0 - keepW.mean()) < self.shrink_min_frac:
            return
        self._compact(keepV, keepW)

    def _compact(self, keepV, keepW):
        idxV = np.flatnonzero(keepV)
        idxW = np.flatnonzero(keepW)
        mapV = {int(o): k for k, o in enumerate(idxV)}
        mapW = {int(o): k for k, o in enumerate(idxW)}
        self.V = self.V[idxV]
        self.W = self.W[idxW]
        self.n, self.m = len(idxV), len(idxW)
        self.Vsq = self.Vsq[idxV]
        self.Wsq = self.Wsq[idxW]
        self.a = self.a[idxV]
        self.b = self.b[idxV]
        self.c = self.c[idxW]
        self.e = self.e[idxW]
        self._origV = self._origV[idxV]
        self._origW = self._origW[idxW]
        # zero-weight entries may have been screened; drop them
        self.wV = {mapV[i]: w for i, w in self.wV.items() if i in mapV}
        self.wW = {mapW[j]: w for j, w in self.wW.items() if j in mapW}
        self.active_V = {mapV[i] for i in self.active_V if keepV[i]}
        self.active_W = {mapW[j] for j in self.active_W if keepW[j]}
        self.hist_V = []
        self.hist_W = []
        self._colV = {mapV[i]: (cV[idxV], cW[idxW])
                      for i, (cV, cW) in self._colV.items() if keepV[i]}
        self._colW = {mapW[j]: (cV[idxV], cW[idxW])
                      for j, (cV, cW) in self._colW.items() if keepW[j]}
        self._compact_extra(idxV, idxW)

    def _compact_extra(self, idxV, idxW):
        """Hook for subclasses holding extra per-point arrays."""

    def _orig_weights(self):
        oV = getattr(self, '_origV', None)
        if oV is None:
            return dict(self.wV), dict(self.wW)
        return ({int(self._origV[i]): w for i, w in self.wV.items()},
                {int(self._origW[j]): w for j, w in self.wW.items()})

    # ------------------------------------------------------------------
    # sparsity
    # ------------------------------------------------------------------
    def _sparsity(self, thresh=1e-9):
        sV = sum(1 for w in self.wV.values() if w > thresh)
        sW = sum(1 for w in self.wW.values() if w > thresh)
        return sV + sW

    # ------------------------------------------------------------------
    # Triangle Algorithm I : intersection / separation
    # ------------------------------------------------------------------
    def solve_intersection(self, eps=1e-3, max_iter=10_000, time_cap=None):
        """Decide (within eps) whether conv(V) and conv(W) intersect.
        time_cap: optional wall-clock budget (seconds) -> status 'timeout'."""
        t0 = time.perf_counter()
        self._init_state()
        tol = self.tol
        status = 'maxiter'
        stalls = 0
        it = 0
        for it in range(1, max_iter + 1):
            if it % self.refresh_every == 0:
                self._refresh_caches()
            if time_cap is not None and it % self.full_scan_every == 0 \
                    and time.perf_counter() - t0 > time_cap:
                status = 'timeout'
                break
            if self._ub_certified() <= eps:
                status = 'intersect'
                break
            iV, sV, iW, sW, full = self._select(self._scores_ta1, it, tol)
            if iV is None and iW is None:
                status = 'separated'          # witness pair certificate
                break
            if not self._step(iV, iW, sV, sW):
                # numerical stall: never a certificate.  Drop the offending
                # indices from the priority sets, restore exact caches and
                # retry; give up only after repeated failures.
                if iV is not None:
                    self.active_V.discard(iV)
                if iW is not None:
                    self.active_W.discard(iW)
                self._refresh_caches()
                stalls += 1
                if stalls > 5:
                    status = 'intersect' if self._ub_certified() <= eps \
                        else 'stalled'
                    break
                continue
            stalls = 0
        self._refresh_caches()
        return TAResult(status=status, distance=self._ub_certified(),
                        lower_bound=0.0, iterations=it,
                        time=time.perf_counter() - t0,
                        sparsity=self._sparsity(),
                        p=None if self.p is None else self.p.copy(),
                        q=None if self.q is None else self.q.copy(),
                        weights_V=self._orig_weights()[0],
                        weights_W=self._orig_weights()[1])

    # ------------------------------------------------------------------
    # Triangle Algorithm II : distance / optimal support
    # ------------------------------------------------------------------
    def solve_distance(self, eps=1e-3, max_iter=10_000, eps_intersect=1e-6,
                       warm_start=False, time_cap=None):
        """Compute the distance between conv(V) and conv(W) to relative
        tolerance eps, i.e. stop when UB - LB <= eps * UB.

        time_cap: optional wall-clock budget in seconds; on expiry the
        solver returns status 'timeout' with its current certified bounds
        (checked at full scans, so the overshoot is at most a few
        iterations)."""
        t0 = time.perf_counter()
        if not warm_start or not hasattr(self, 'p'):
            self._init_state()
        tol = self.tol
        status = 'maxiter'
        lb_best = -np.inf
        stalls = 0
        it = 0
        for it in range(1, max_iter + 1):
            if it % self.refresh_every == 0:
                self._refresh_caches()
            if time_cap is not None and it % self.full_scan_every == 0 \
                    and time.perf_counter() - t0 > time_cap:
                status = 'timeout'
                break
            # certified upper bound: equals d(p, q) for explicit iterates
            # and is floored at the cancellation noise for implicit ones,
            # so neither the intersect test nor the bound below it can be
            # fooled by a cancelled cached distance
            ub = self._ub_certified()
            if ub <= eps_intersect:
                status = 'intersect'
                break
            full = (not self.prioritized) \
                or ((it - 1) % self.full_scan_every == 0)
            if full:
                # lower bound from the parallel supporting hyperplanes
                lb_best = max(lb_best, self._lower_bound(ub))
                if self.trace is not None:
                    self.trace.append((it, time.perf_counter() - t0,
                                       ub, lb_best))
                if ub - lb_best <= eps * ub:
                    status = 'converged'
                    break
            if self.shrink and it % self.shrink_every == 0:
                self._screen_and_compact(lb_best)
            if self.block_auto and it % self.adapt_every == 0:
                self._adapt_k(it)
            iV, sV, iW, sW, _ = self._select(self._scores_ta2, it, tol)
            if iV is None and iW is None:
                # no descent direction over the full input (a None pair
                # always comes from a full scan): success only with the
                # gap certificate, otherwise this is a numerical stall
                lb_best = max(lb_best, self._lower_bound(ub))
                status = 'converged' if ub - lb_best <= eps * ub \
                    else 'stalled'
                break
            if not self._step(iV, iW, sV, sW):
                # numerical stall: never a certificate.  Drop the
                # offending indices from the priority sets, restore exact
                # caches and retry; after repeated failures exit with
                # whatever the current bounds certify
                if iV is not None:
                    self.active_V.discard(iV)
                if iW is not None:
                    self.active_W.discard(iW)
                self._refresh_caches()
                stalls += 1
                if stalls > 5:
                    ub = self._ub_certified()
                    if ub <= eps_intersect:
                        status = 'intersect'
                    else:
                        lb_best = max(lb_best, self._lower_bound(ub))
                        status = 'converged' if ub - lb_best <= eps * ub \
                            else 'stalled'
                    break
                continue
            stalls = 0
        self._refresh_caches()
        ub = self._ub_certified()
        if ub > eps_intersect:
            lb_best = max(lb_best, self._lower_bound(ub))
        return TAResult(status=status, distance=ub,
                        lower_bound=float(lb_best), iterations=it,
                        time=time.perf_counter() - t0,
                        sparsity=self._sparsity(),
                        p=None if self.p is None else self.p.copy(),
                        q=None if self.q is None else self.q.copy(),
                        weights_V=self._orig_weights()[0],
                        weights_W=self._orig_weights()[1])

    # ------------------------------------------------------------------
    # one enhancement-aware update step; returns False if no progress
    # ------------------------------------------------------------------
    def _adapt_k(self, it):
        """Adaptive block size (block_size='auto'): see __init__."""
        win = self._adapt_win
        self._adapt_win = []
        if not win:
            return
        frac = float(np.mean([c for _, _, c in win]))
        etas = [(kap - 1.0) / (kp - 1) for kp, kap, _ in win
                if kp > 1 and np.isfinite(kap)]
        eta = float(np.median(etas)) if etas else float('nan')
        k = self.block_size
        if frac > self.drop_frac_max:
            k = max(1, k // 2)
        elif len(etas) < 5:
            k = min(self.k_max, 2 * k)      # no diversity signal yet: explore
        else:
            eta_c = min(max(eta, 1e-3), 1.0 - 1e-3)
            k = int(round(np.sqrt(self.cost_ratio * (1.0 - eta_c) / eta_c)))
            k = max(1, min(self.k_max, k))
        self.block_size = k
        self.k_history.append((it, k, eta, frac))

    def _step(self, iV, iW, sV=-np.inf, sW=-np.inf):
        d_before = self.dist2()
        moved = False
        # MDM mode: pairwise weight transfer is the primary step,
        # with a toward-step fallback when no transfer is possible.
        # Block mode: k transfers per scan first, then the same fallbacks.
        if self.step_mode in ('mdm', 'block'):
            did = False
            if self.step_mode == 'block':
                self.last_iter_drop = False
                sVa = self.b - self.a if iV is not None else None
                sWa = self.c - self.e if iW is not None else None
                if iV is not None and iW is not None:
                    if self.drop_skip:
                        # analysed schedule (Lemma 6): larger pairwise
                        # side gap (max score - worst active) first
                        gV = float(sVa.max()) - float(
                            sVa[self._active_indices(self.wV)].min())
                        gW = float(sWa.max()) - float(
                            sWa[self._active_indices(self.wW)].min())
                        order = ('V', 'W') if gV >= gW else ('W', 'V')
                    else:
                        # unconditional V-then-W (the batteries' schedule)
                        order = ('V', 'W')
                else:
                    order = ('V',) if iV is not None else ('W',)
                for pos, side in enumerate(order):
                    # the first step changes the other side's scores, so
                    # only the first side may reuse the precomputed array
                    if side == 'V':
                        did |= self._block_transfer_V(
                            self.block_size, sVa if pos == 0 else None)
                    else:
                        did |= self._block_transfer_W(
                            self.block_size, sWa if pos == 0 else None)
                    if pos == 0 and self._last_drop:
                        self.last_iter_drop = True
                        if self.drop_skip:
                            if len(order) > 1:
                                self.n_drop_skips += 1
                            break
                if did:
                    return self.dist2() <= d_before + _EPS_NUM * max(1.0,
                                                                     d_before)
            if iV is not None:
                u = self._worst_active_p()
                if u is not None and self._pairwise_p(iV, u):
                    did = True
                else:
                    alpha = self._alpha_single_p(iV)
                    if alpha > 0.0:
                        self._apply_p(iV, alpha)
                        did = True
            if iW is not None:
                u = self._worst_active_q()
                if u is not None and self._pairwise_q(iW, u):
                    did = True
                else:
                    beta = self._beta_single_q(iW)
                    if beta > 0.0:
                        self._apply_q(iW, beta)
                        did = True
            if not did:
                return False
            return self.dist2() <= d_before + _EPS_NUM * max(1.0, d_before)
        # zig-zag remedies apply on cycle detection, before (and instead
        # of) the regular toward-step, in joint and single branches alike
        if self.zigzag_strategy is not None:
            did = False
            zzV = self._zigzag_candidate(self.hist_V + [iV]) \
                if iV is not None else None
            if zzV is not None:
                if self.zigzag_strategy == 'pairwise':
                    u = self._worst_active_p()
                    did |= u is not None and self._pairwise_p(iV, u)
                elif self.zigzag_strategy == 'away':
                    u = self._worst_active_p()
                    did |= u is not None and self._away_p(u)
                else:  # midpoint
                    did |= self._apply_p_midpoint(*zzV)
            zzW = self._zigzag_candidate(self.hist_W + [iW]) \
                if iW is not None else None
            if zzW is not None:
                if self.zigzag_strategy == 'pairwise':
                    u = self._worst_active_q()
                    did |= u is not None and self._pairwise_q(iW, u)
                elif self.zigzag_strategy == 'away':
                    u = self._worst_active_q()
                    did |= u is not None and self._away_q(u)
                else:  # midpoint
                    did |= self._apply_q_midpoint(*zzW)
            if did:
                self.hist_V.append(-2)
                self.hist_W.append(-2)
                return self.dist2() <= d_before + _EPS_NUM * max(1.0, d_before)
        if iV is not None and iW is not None and not self.joint_update:
            # without joint updates, advance the side with the larger
            # violation so both iterates keep making progress
            if sV >= sW:
                iW = None
            else:
                iV = None
        if iV is not None and iW is not None and self.joint_update:
            alpha, beta = self._joint_step(iV, iW)
            if alpha > 0.0 or beta > 0.0:
                # capture pre-update cached values for the q side
                colWa, colWe = self._gram_col_W(iW)
                wsq, ev, cv = self.Wsq[iW], self.e[iW], self.c[iW]
                # v.q after p-update changes b; use stored values
                self._apply_p(iV, alpha)
                if beta > 0.0:
                    # cv = W[iW].p must reflect the *new* p
                    cv = float(self.c[iW])
                    self._apply_q(iW, beta, colWa, colWe, wsq, ev, cv)
                moved = True
                self.hist_V.append(iV)
                self.hist_W.append(iW)
        elif iV is not None:
            zz = self._zigzag_candidate(self.hist_V + [iV]) \
                if self.zigzag_strategy == 'midpoint' else None
            if zz is not None and self._apply_p_midpoint(*zz):
                self.hist_V.append(-1)
                moved = True
            else:
                alpha = self._alpha_single_p(iV)
                if alpha > 0.0:
                    self._apply_p(iV, alpha)
                    self.hist_V.append(iV)
                    moved = True
        elif iW is not None:
            zz = self._zigzag_candidate(self.hist_W + [iW]) \
                if self.zigzag_strategy == 'midpoint' else None
            if zz is not None and self._apply_q_midpoint(*zz):
                self.hist_W.append(-1)
                moved = True
            else:
                beta = self._beta_single_q(iW)
                if beta > 0.0:
                    self._apply_q(iW, beta)
                    self.hist_W.append(iW)
                    moved = True
        if len(self.hist_V) > 8:
            del self.hist_V[:-8]
        if len(self.hist_W) > 8:
            del self.hist_W[:-8]
        if not moved:
            return False
        return self.dist2() <= d_before + _EPS_NUM * max(1.0, d_before)
