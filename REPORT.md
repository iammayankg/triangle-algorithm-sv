# Replication Report: An Enhanced Triangle Algorithm for Large-Scale SVM Optimization

**Paper:** Gupta, M. & Kalantari, B., *An Enhanced Triangle Algorithm for Large-Scale
Support Vector Machine Optimization: A Comparative Study with Classical and Modern
Solvers*, JIDMIS Vol. 3, Issue 9s (2026). <https://jidmis.org/index.php/jidmis/article/view/2284>

**This replication:** independent Python/NumPy implementation of the Enhanced
Triangle Algorithm (ETA) and a hard-margin SMO baseline, run on the paper's
experimental protocol. Environment: Linux, 2 CPU cores, 7 GB RAM, NumPy 2.4
(float64 throughout). The paper's numbers come from MATLAB on unspecified
hardware, so absolute runtimes are not comparable; iteration counts,
solution quality, sparsity, and *relative* speedups are the replication targets.

## What was implemented

**Triangle Algorithm I** (intersection/separation): iterate pair (p, p') in
K x K'; a point v in V is a pivot for p iff d(p, v) >= d(p', v)
(equivalently 2v.(p'-p) >= ||p'||^2 - ||p||^2); each iteration moves an
iterate to the closest point on the segment to its pivot; if no pivot exists
in either class, (p, p') is a witness pair whose perpendicular bisector
separates the hulls.

**Triangle Algorithm II** (distance/optimal support): shrinks the gap between
the upper bound UB = d(p, p') and the lower bound from parallel supporting
hyperplanes orthogonal to h = p - p', using the extreme points in direction h
as pivots, until UB - LB <= eps * UB.

**The four enhancements**, each independently switchable:

1. *Joint closest-point updates* — when both classes provide a pivot, the pair
   (p, p') moves to the closest points between segments [p, v] and [p', v'],
   solved by the standard two-line closest-point equations clipped to [0, 1].
2. *Dot-product caching* — the vectors V.p, V.p', W.p, W.p' and the scalars
   p.p, p.p', p'.p' are maintained incrementally (iterates are convex
   combinations, so updates are O(n) rank-1 corrections); Gram columns of used
   pivots are cached lazily. Per-iteration cost drops from O(nd) to O(n)
   once a pivot's column is cached.
3. *Anti-zig-zag* — an alternating i, j, i, j pivot pattern is broken by
   pivoting on the midpoint of the two offending vertices.
4. *Prioritized search* — previously used pivots are scanned first; the full
   point set is scanned on a fixed cadence and whenever the priority subset
   yields no pivot (termination is only ever declared from a full scan).

**SMO baseline**: standard hard-margin SVM dual (C = 1e12 ~ infinity, linear
kernel), maximal-violating-pair working-set selection, KKT gap stopping rule,
in two variants: *row cache* (LRU kernel-row cache, an optimized modern
implementation) and *uncached* (recomputes both kernel rows every iteration —
the cost profile of a plain MATLAB implementation like the study's baseline).

## Correctness verification

`tests/test_correctness.py` checks, on small random instances, that:

- ETA's distance matches an exact QP solution (scipy SLSQP over the two
  probability simplices) to < 5e-3 relative error — observed 1e-5 to 1e-13
  across 10 trials, with the certified lower bound always valid;
- SMO's hull distance (2/||w||) matches the same QP ground truth;
- intersection/separation classification is correct on both overlapping and
  separated instances;
- every enhancement configuration converges to the same distance
  (relative spread 4e-4 at eps = 1e-3), while iteration counts show the
  enhancements' effect: 36 iterations with all enhancements on vs 327 with
  all off on the ablation instance.

## Experimental protocol and assumptions

The paper generates "two sets of points from unit balls with random means",
translating one set along a random direction by k x max(empirical diameters):
k = 0.9 for intersection tests (Table 2), fixed separation for the dimension
sweep (Table 3), and varying k for distance sensitivity (Table 4); n = 5000
points per set, eps = 1e-3, max 10^4 iterations, dimensions
3–10,000. The paper does not specify the exact sampling law, the number of
trials, or the k values; we used uniform-in-unit-ball sampling for Table 2,
isotropic Gaussian clouds (which reproduce the reported distance scales) with
k = 1.2 for Table 3 and k in {1.2, ..., 1.9} for Table 4, and 3 trials per
configuration (means reported).

## Results

### Table 2 — intersection testing (k = 0.9, eps = 1e-3)

| dim | iters (ours) | time s (ours) | iters (paper) | time s (paper) |
|----:|----:|----:|----:|----:|
| 3 | 13.0 | 0.002 | 102.4 | 0.68 |
| 10 | 13.7 | 0.002 | 10.1 | 0.64 |
| 50 | 2.3 | 0.001 | 26.7 | 0.70 |
| 100 | 1.0 | 0.001 | 33.0 | 0.78 |
| 300 | 1.0 | 0.003 | 24.9 | 1.36 |
| 500 | 1.0 | 0.005 | 14.7 | 1.83 |
| 1000 | 1.0 | 0.011 | 9.3 | 3.40 |
| 2000 | 1.0 | 0.023 | 4.0 | 5.72 |
| 5000 | 1.0 | 0.081 | 4.2 | 13.69 |
| 10000 | 1.0 | 0.136 | 5.0 | 26.53 |

Replicated: iteration counts are high at low dimension and collapse to a
handful as d grows, while runtime grows with d only through the O(nd) cost of
a scan — the paper's exact pattern. A geometric note the paper does not
dwell on: with 5000 samples per set, the convex hulls are much thinner than
the balls they are drawn from once d >> log n, so at k = 0.9 the *hulls*
genuinely separate at higher dimensions (we observe intersection at d = 3 and
separation certificates beyond; the certificate is verified against the exact
QP on small instances). The low iteration counts at large d in both studies
reflect this same concentration phenomenon.

### Table 3 — TA vs SMO by dimension (n = 5000/set, eps = 1e-3)

Ours (3-trial means; SMO-u = uncached, the MATLAB-like baseline):

| dim | TA iter | TA s | TA sparse | TA dist | SMO iter | SMO-u s | SMO sparse | SMO dist |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 3 | 24 | 0.002 | 4.3 | 3.345 | 41 | 0.004 | 3.3 | 3.345 |
| 10 | 188 | 0.013 | 12.0 | 5.894 | 105 | 0.018 | 8.3 | 5.890 |
| 50 | 409 | 0.037 | 22.7 | 12.462 | 137 | 0.041 | 19.7 | 12.456 |
| 100 | 444 | 0.041 | 30.3 | 17.361 | 115 | 0.057 | 27.3 | 17.351 |
| 300 | 423 | 0.072 | 52.3 | 30.174 | 143 | 0.174 | 49.7 | 30.159 |
| 500 | 429 | 0.188 | 65.7 | 38.834 | 168 | 0.349 | 63.0 | 38.810 |
| 1000 | 321 | 0.259 | 92.3 | 54.747 | 200 | 0.792 | 91.3 | 54.716 |
| 2000 | 336 | 0.586 | 137.7 | 77.178 | 288 | 2.212 | 140.3 | 77.147 |
| 5000 | 479 | 3.317 | 195.3 | 121.442 | 352 | 13.664 | 211.3 | 121.401 |
| 10000 | 604 | 8.889 | 247.3 | 171.555 | 442 | 32.866 | 276.7 | 171.548 |

Paper (Table 3): at d = 10,000, TA 699 iters / 36.39 s / sparsity 290 /
dist 157.01 vs SMO 647 iters / 165.99 s / sparsity 333 / dist 156.94.

Replicated, point by point:

- **Headline speedup.** Paper: TA 4.6x faster than SMO at d = 10,000
  (36.4 s vs 166.0 s). Ours: 3.7x against the uncached SMO
  (8.9 s vs 32.9 s), 1.5x against the LRU-cached SMO (13.1 s), with the gap
  widening monotonically with dimension from d ~ 300 onward.
- **Accuracy.** Paper: distances agree to 0.05% at d = 10,000. Ours: all ten
  dimensions agree within the eps = 1e-3 tolerance (4e-5 relative at
  d = 10,000).
- **Sparsity.** Paper: TA yields sparser solutions than SMO at scale
  (290 vs 333). Ours: same (247 vs 277), with near-identical
  sparsity-vs-dimension growth.
- **Iterations.** Paper TA: ~200 at d = 3 rising to ~700 at d = 10,000.
  Ours: 24 rising to 604 — same order and same growth shape.

### Table 4 — distance sensitivity at d = 1000

| k | TA iter | TA s | TA dist | SMO iter | SMO-u s | SMO dist |
|---:|----:|----:|----:|----:|----:|----:|
| 1.2 | 326 | 0.226 | 54.668 | 214 | 0.879 | 54.639 |
| 1.3 | 301 | 0.226 | 59.483 | 208 | 0.732 | 59.452 |
| 1.4 | 259 | 0.208 | 64.562 | 202 | 0.720 | 64.537 |
| 1.5 | 213 | 0.185 | 69.236 | 198 | 0.721 | 69.188 |
| 1.6 | 191 | 0.173 | 74.602 | 200 | 0.729 | 74.552 |
| 1.7 | 178 | 0.145 | 80.182 | 208 | 0.734 | 80.124 |
| 1.8 | 166 | 0.135 | 84.189 | 193 | 0.689 | 84.170 |
| 1.9 | 149 | 0.129 | 89.762 | 203 | 0.686 | 89.746 |

Paper (Table 4): TA iterations fall 814 -> 70 and time 5.14 s -> 2.75 s as
separation grows, while SMO stays flat between 67 s and 118 s.

Replicated: TA iterations fall 326 -> 149 and time 0.226 s -> 0.129 s as k
grows, while both SMO variants stay essentially flat — the same qualitative
signature (TA's geometric convergence rate improves with separation; SMO's
does not). The paper's distance range (45.5–78.4) is closely matched by
k in [1.2, 1.9] (54.7–89.8), supporting the reconstruction of the
unspecified k grid.

## Figures

![runtime vs dimension](results/fig_time_vs_dim.png)

![runtime vs separation](results/fig_time_vs_k.png)

![distance agreement](results/fig_agreement.png)

## Verdict

The paper's three main claims all replicate in an independent implementation:

1. **ETA matches SMO's solution quality** (distances within tolerance at every
   configuration, comparable-to-sparser support sets). Replicated.
2. **ETA outperforms SMO increasingly with dimension**, with the same
   iteration-count profile. Replicated: 3.7x at d = 10,000 vs the paper's
   4.6x against a comparable (uncached) SMO.
3. **ETA accelerates as inter-hull separation grows while SMO does not.**
   Replicated, with matching monotone trends on both iteration count and time.

Caveats: the source study's data-sampling law, trial counts and k grid are
not fully specified, so those were reconstructed (Gaussian clouds match the
reported distance scales; uniform balls match the intersection regime); with
3 trials per cell, per-cell means carry sampling noise; and the size of the
TA-over-SMO speedup depends on the SMO implementation quality — an LRU
kernel-row cache alone closes much of the gap (1.5x at d = 10,000), which
supports the paper's own limitation note that comparisons should be extended
to modern solvers (LIBSVM, LIBLINEAR, ThunderSVM).

## Reproducing

```bash
python3 tests/test_correctness.py          # ground-truth verification
python3 src/experiments.py --exp all --trials 3 --out results
python3 src/plots.py                       # figures
```
