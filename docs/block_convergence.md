# Linear Convergence of the Block-Transfer Triangle Algorithm

This note proves convergence guarantees for the block-transfer step
(`step_mode='block'`) used in the Enhanced Triangle Algorithm. The main
results:

1. **Lemma 2 (gain bound).** Every block step decreases the objective by
   at least 1/(2k) of the decrease the single MDM (pairwise) step would
   achieve, where k is the block size.
2. **Guarded variant.** With an O(1) guard (take the single MDM step
   whenever its computed gain exceeds the block's), every iteration's
   decrease is at least the MDM step's decrease, so the guarded block
   algorithm inherits the pairwise-Frank-Wolfe linear rate *with
   identical constants* while performing up to k transfers per scan.
3. **Proposition 4 (best case).** When the pair directions are
   near-orthogonal, the block gain approaches the *sum* of the k
   individual pair gains, explaining the observed 4-30x iteration
   reductions.

Throughout, the tool is elementary: exact line search on a quadratic,
the triangle inequality, and Cauchy-Schwarz; the linear rate is then
inherited from Lacoste-Julien & Jaggi's analysis of pairwise Frank-Wolfe
(NeurIPS 2015), whose "pairwise FW pair" is exactly the MDM pair the
block always contains.

## 1. Setting

Let V = {v_1..v_n}, W = {w_1..w_m} in R^d, and consider the polytope
distance problem

    min f(p, q) = 1/2 ||p - q||^2   over  (p, q) in conv(V) x conv(W).

Write x = (p, q), P = conv(V) x conv(W), and note f(x) = g(Ax) with
A(p, q) = p - q and g(z) = 1/2 ||z||^2 strongly convex. f itself is
convex quadratic but not strongly convex on P; it belongs to the class
g(Ax) + b'x for which Lacoste-Julien & Jaggi (2015, Thm. 11 and App. F)
prove linear convergence of away-step and pairwise Frank-Wolfe over
polytopes, with a rate governed by the *pyramidal width* delta of P and
the diameter M of P. All results below apply verbatim to the soft-margin
and kernel variants: the L2 reduction only augments the coordinates
(strengthening convexity), and the kernel case replaces V, W by their
feature-space images, with every inner product below delivered by the
cached kernel columns.

We analyse a V-side step at a fixed q; the W-side is symmetric, and a
two-sided iteration is a sum of two one-sided decreases, so all bounds
add. Let

    g := q - p          (the negative gradient in p),
    s_i := <g, v_i>     (the scores; in the code, s = b - a).

An index r is a *receiver*, and an active index u (weight w_u > 0) a
*donor*. The **MDM pair** is (r*, u*) with r* = argmax_i s_i and
u* = argmin_{u active} s_u; this is precisely the pairwise-FW pair of
Lacoste-Julien & Jaggi (their s_FW and v_away).

## 2. The block step

Given block size k, the algorithm selects the top-k receivers and the
worst-k active donors (disjoint), pairs them best-with-worst into pairs
j = 1..k' (k' <= k) with directions and per-pair steps

    d_j := v_{r_j} - v_{u_j},        s_j := ||d_j||^2,
    g_j := <g, d_j> = s_{r_j} - s_{u_j} > 0,
    gamma_j := min( g_j / s_j , c_j ),

where c_j is the pair's capacity (the donor's weight; additionally the
cap mu - w_{r_j} in the reduced-hull setting). Pair 1 is always the MDM
pair, by construction of the top/bottom selection and best-with-worst
pairing. The update is

    p  <-  p + t d,     d := sum_j gamma_j d_j,     t in [0, 1]

with t chosen by exact line search.

**Lemma 1 (feasibility).** For every t in [0, 1] the update keeps p in
conv(V) (and, in the capped setting, keeps every weight in [0, mu]).

*Proof.* The weight update is w_{r_j} += t gamma_j, w_{u_j} -= t gamma_j
per pair; the total sum is conserved, receivers only gain, and each
donor u_j loses t gamma_j <= gamma_j <= c_j <= w_{u_j} (donors are
distinct, and each index appears in at most one pair). In the capped
setting gamma_j <= mu - w_{r_j} bounds the receiver.  QED

## 3. The gain bound

Write the exact line-search quantities

    S := <g, d> = sum_j gamma_j g_j,      D := ||d||^2,

so f(p + t d, q) = f(p, q) - t S + t^2 D / 2, the unconstrained
minimiser is t* = S / D, and the realised gain of the block step is

    Delta_B = S^2 / (2D)          if S/D <= 1   (interior),
    Delta_B = S - D/2 >= S/2      if S/D >= 1   (boundary t = 1).

Let Delta_1 denote the gain of the exact single MDM step:
Delta_1 = gamma_1 g_1 - gamma_1^2 s_1 / 2, which equals g_1^2 / (2 s_1)
when pair 1 is uncapped and satisfies Delta_1 <= gamma_1 g_1 always.

**Lemma 2 (block gain).** With b_j := gamma_j sqrt(s_j),

    (i)   S >= sum_j b_j^2   and   D <= k' * S;
    (ii)  Delta_B >= S / (2k');
    (iii) Delta_B >= Delta_1 / (2k').

*Proof.* (i) Since gamma_j <= g_j / s_j we have
gamma_j g_j >= gamma_j^2 s_j = b_j^2, and summing gives S >= sum b_j^2.
By the triangle inequality ||d|| <= sum_j gamma_j ||d_j|| = sum_j b_j,
and by Cauchy-Schwarz (sum b_j)^2 <= k' sum b_j^2 <= k' S; hence
D <= k' S.

(ii) Interior case: Delta_B = S^2/(2D) >= S^2/(2 k' S) = S/(2k').
Boundary case: Delta_B >= S/2 >= S/(2k').

(iii) All terms of S are nonnegative, so S >= gamma_1 g_1 >= Delta_1,
and (ii) gives Delta_B >= Delta_1/(2k').  QED

Remark: in the boundary case the bound sharpens to
Delta_B >= S/2 >= (sum_j b_j^2)/2, which for uncapped pairs equals the
*sum* of the individual pair gains.

## 4. The guarded block step

Both Delta_B = t S - t^2 D / 2 and Delta_1 are available in O(1) from
quantities the implementation already computes (S, D, t, and pair 1's
score gap and squared distance). The **guarded block step** takes the
single MDM step on pair 1 whenever Delta_1 > Delta_B, and the block step
otherwise. Its realised gain is

    Delta = max(Delta_B, Delta_1) >= Delta_1 ,

i.e. the guarded step never makes less progress than pairwise FW / MDM
at that iterate.

**Theorem 3 (linear convergence).** Let h_T = f(x_T) - f* denote the
suboptimality after T iterations of the guarded block-transfer Triangle
Algorithm (any block size k) on the polytope distance problem over P,
with exact line search. Then there is rho in (0, 1], depending only on
the pyramidal width and diameter of P (and the generalized strong
convexity constant of f = g(Ax)) - the same constant as for pairwise
Frank-Wolfe - such that every non-swap iteration satisfies

    h_{T+1} <= (1 - rho) h_T ,

and the number of swap iterations (a capacity-clipped pair 1 whose donor
is exhausted without an interior line search) between successive
non-swap iterations is bounded exactly as in the pairwise-FW analysis.
Consequently h_T -> 0 linearly. Without the guard the same statement
holds with rho replaced by rho/(2k), by Lemma 2(iii).

*Proof sketch.* Lacoste-Julien & Jaggi (2015) prove that for f in the
class g(Ax)+b'x over a polytope, the pairwise-FW step on the pair
(argmax_i <g, v_i>, argmin_{active} <g, v_i>) with exact line search
contracts h by (1 - rho) on good steps, where the key inequality lower
bounds the pairwise dual gap g_1 = <g, d_1> against h via the pyramidal
width, and the good-step progress is Delta_1 >= g_1^2 / (2 L M^2)-type.
Pair 1 of the block step is exactly this pair, and the guarded step's
progress is >= Delta_1 by construction, so the same recursion holds for
the guarded algorithm at every good step. Swap steps (progress limited
by an exhausted donor) are identical events in both algorithms and their
counting argument transfers unchanged. The unguarded statement follows
by replacing Delta_1 with Delta_1/(2k) in the recursion.  QED

Two remarks. First, the theorem covers the L2 soft-margin and kernel
solvers unchanged (Section 1). Second, the *certificate* is independent
of all of this: the algorithm stops only when UB - LB <= eps * UB, and
the lower bound is valid whatever steps were taken, so correctness never
rests on the rate.

## 5. Why blocks help: the near-orthogonal regime

**Proposition 4.** Suppose the pair directions are eta-near-orthogonal:
|<d_i, d_j>| <= eta sqrt(s_i s_j) for i != j. Then in the interior case

    Delta_B >= S / (2 (1 + eta (k'-1)))
            >= ( sum_j Delta_j ) / (1 + eta (k'-1))     (uncapped pairs),

where Delta_j = g_j^2/(2 s_j) is pair j's individual gain.

*Proof.* D = sum_{i,j} gamma_i gamma_j <d_i, d_j>
<= sum_j b_j^2 + eta sum_{i != j} b_i b_j
<= (1 + eta(k'-1)) sum_j b_j^2 <= (1 + eta(k'-1)) S, using
Cauchy-Schwarz on the cross terms. Then Delta_B = S^2/(2D) >=
S / (2(1 + eta(k'-1))), and for uncapped pairs S = sum_j 2 Delta_j.  QED

For eta -> 0 the block step realises the sum of all k pair gains for the
price of one scan - a k-fold rate improvement per iteration. In high
dimension, distinct support-vector directions are typically
near-orthogonal, which matches the measured behaviour: iteration
reductions of 4-30x, growing with d (block 32 best at d = 1000, block 8
at d = 100).

## 6. Numerical verification

`tests/test_block_lemma.py` verifies every inequality above on random
ensembles (including capacity-clipped and boundary cases): Lemma 2
(i)-(iii), the boundary sharpening, Proposition 4, and - on instrumented
solver runs - that each realised block-step decrease of 1/2||p - q||^2
matches t S - t^2 D / 2 to machine precision and dominates
max(Delta_B guard bound, Delta_1)/(2k).

## References

- B. Kalantari. *An algorithm for computing the distance between two
  convex hulls* (distance duality; Triangle Algorithm).
- M. Gupta, B. Kalantari. *An Enhanced Triangle Algorithm for
  Large-Scale SVM Optimization.* JIDMIS 3(9s), 2026.
- S. Lacoste-Julien, M. Jaggi. *On the Global Linear Convergence of
  Frank-Wolfe Optimization Variants.* NeurIPS 2015.
- B. F. Mitchell, V. F. Demyanov, V. N. Malozemov. *Finding the point of
  a polyhedron closest to the origin.* SIAM J. Control, 1974.
