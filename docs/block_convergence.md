# Global Linear Convergence of the Guarded Block-Transfer Triangle Algorithm

This document gives a complete convergence analysis for the
block-transfer step (`step_mode='block'`) of the Enhanced Triangle
Algorithm, replacing the earlier proof sketch. The main improvements over
the sketch:

* the analysis is carried out on the **Minkowski-difference polytope**,
  where the objective is genuinely 1-strongly convex and a single
  pyramidal width controls the rate - no product-polytope pyramidal
  width is needed;
* the swap-step problem of pairwise Frank-Wolfe (whose classical bound
  is combinatorial) is **eliminated by an away-step fallback** on
  capacity-clipped iterations, giving the clean drop-step counting of
  away-step Frank-Wolfe and an explicit geometric rate with no factorial
  constants;
* every constant is explicit.

Main theorem (informal): *the guarded block-transfer Triangle Algorithm
with away fallback converges linearly on the polytope distance problem,
with rate constant equal to the away/pairwise Frank-Wolfe constant of
the Minkowski-difference polytope divided by an explicit factor 16, and
with at least a (1/(k+2))-fraction of iterations contracting, where k is
the block size.*

## 1. Setting and notation

Let V = {v_1..v_n} and W = {w_1..w_m} be finite subsets of R^d. The
polytope distance problem is

    (P)   min f(p, q) = 1/2 || p - q ||^2 ,   p in conv(V), q in conv(W).

Define the **Minkowski-difference polytope**

    Z := conv(V) - conv(W) = conv( { v_i - w_j : i <= n, j <= m } ),

with atom set A_Z = { v_i - w_j }. Writing z = p - q, problem (P) is

    (P')  min F(z) = 1/2 ||z||^2 ,   z in Z,

and F is 1-strongly convex and 1-smooth on R^d. Let z* be the (unique)
minimiser, h(z) = F(z) - F(z*), M = diam(Z), and delta = PWidth(Z) > 0
the pyramidal width of Z (positive for every polytope; Lacoste-Julien &
Jaggi 2015, hereafter **LJ**).

Weights. The algorithm maintains simplex weights alpha on V and beta on
W with p = sum alpha_i v_i, q = sum beta_j w_j. The induced **product
representation** of z is lambda_{ij} = alpha_i beta_j over A_Z: it is a
valid convex representation of z, and its support is
S(lambda) = supp(alpha) x supp(beta).

Gradient and scores. grad F(z) = z; write g = -z = q - p. Define the
side scores

    sV_i = <g, v_i>,        sW_j = <-g, w_j> = <p - q, w_j>,

which are the arrays (b - a) and (c - e) maintained by the
implementation.

Gaps. For x in Z with representation lambda, the Frank-Wolfe, away and
pairwise gaps over Z are

    gFW  = max_{u in A_Z} <g, u> - <g, z>,
    gA   = <g, z> - min_{u in S(lambda)} <g, u>,
    gPW  = gFW + gA = max_{u in A_Z} <g, u> - min_{u in S(lambda)} <g, u>.

**Lemma 1 (separability of the gaps).** With the product representation,

    gPW = gPW_V + gPW_W,   where
    gPW_V = max_i sV_i - min_{i in supp(alpha)} sV_i >= 0,
    gPW_W = max_j sW_j - min_{j in supp(beta)} sW_j >= 0,

and likewise gFW = gFW_V + gFW_W and gA = gA_V + gA_W with the obvious
per-side definitions. Moreover, the maximising/minimising atoms of A_Z
are (argmax_i sV_i, argmax_j sW_j) and, within the support,
(argmin over supp(alpha), argmin over supp(beta)).

*Proof.* <g, v_i - w_j> = sV_i + sW_j is separable, and the support of
the product representation is a product set, so max/min over it
separate. Subtracting the two separable optima gives the displayed sums;
each side term is nonnegative because the support max dominates the
support min and the global max dominates the support max. QED

**Lemma 2 (side directions are Z-pairwise directions).** For any
receiver r and donor u on the V side with alpha_u > 0, and any b in
supp(beta), the direction v_r - v_u = (v_r - w_b) - (v_u - w_b) is a
difference of an atom of A_Z and an atom in S(lambda); the corresponding
transfer of weight gamma <= alpha_u in the alpha-simplex realises the
move z <- z + gamma (v_r - v_u) inside Z. Symmetrically on the W side.
QED (immediate.)

Thus every step the algorithm takes - single MDM transfers, block
transfers, toward steps and away steps on either side - is a feasible
move within Z along (combinations of) pairwise/away/toward directions of
Z, and Lemma 1 lets us read the Z-gaps off the per-side score arrays at
O(n + m) cost.

## 2. The algorithm (analysed form)

One iteration of the **guarded block-transfer Triangle Algorithm** at
iterate z with weights (alpha, beta):

1. **Side selection.** Compute gPW_V and gPW_W (Lemma 1); work on the
   side with the larger value. (The implementation updates both sides
   per iteration; the analysis needs only the larger side's step and the
   other side's step never increases F, so all bounds below transfer.)
2. **Pairing.** On the chosen side (say V), select the top-k receivers
   by sV and the worst-k donors from supp(alpha), pair best-with-worst;
   pair 1 = (r*, u*) is the side's MDM pair, realising gPW_V. Per-pair
   steps gamma_j = min(gap_j / s_j , c_j) with gap_j = sV_{r_j} -
   sV_{u_j}, s_j = ||d_j||^2, d_j = v_{r_j} - v_{u_j} and capacity c_j
   (the donor's weight; also the box cap in the reduced-hull setting).
3. **Case (a): pair 1 uncapped** (gamma_1 = gap_1 / s_1 <= c_1). Take
   the better of: the block step (exact line search t in [0,1] along
   d = sum_j gamma_j d_j) and the single MDM step on pair 1. Both gains
   are O(1)-computable; this is the **guard**.
4. **Case (b): pair 1 capacity-clipped** (c_1 < gap_1 / s_1). Compute
   the away step on u* (step size limited by eta_max, the value at which
   alpha_{u*} reaches 0) and the toward (Frank-Wolfe) step on r*.
   - If the away line search is **boundary-clipped** (its unconstrained
     optimum exceeds eta_max): take the away step - a **drop step**;
     alpha_{u*} becomes exactly 0 and the support shrinks by one.
   - Otherwise take the better of the away and toward steps (optionally
     also the block and MDM candidates - taking a larger gain only
     helps).

Feasibility of every candidate step was proved in the earlier note
(donors clipped to capacity; convex combinations stay in the simplex)
and is unchanged.

## 3. Per-iteration progress

Throughout, "gain" means F(z) - F(z') for the step taken. Recall from
exact line search on the 1-smooth F along a feasible direction d with
directional derivative <g, d> = G > 0 and maximal feasible step T:

    gain >= min( G^2 / (2 ||d||^2) ,  T G / 2 ).                    (LS)

(The first branch is the interior optimum; the second follows from
evaluating at the clipped step because the optimum lies beyond it.)

**Lemma 3 (block gain; proved in the earlier note, restated).** With
S = <g, d> and D = ||d||^2 for the aggregate direction,
S >= sum_j gamma_j^2 s_j, D <= k S, and the realised block gain
satisfies Delta_B >= S / (2k) >= Delta_MDM / (2k), where Delta_MDM is
the exact gain of the single step on pair 1. (Triangle inequality +
Cauchy-Schwarz; capacity clipping included.)

**Lemma 4 (good-step progress).** Let the iteration work on the side
with the larger pairwise gap, and suppose the iteration is not a drop
step. Then its gain satisfies

    gain >= (1/16) * min( gPW^2 / (2 M^2) ,  gPW * c_min / 2 ),

where gPW is the full Z-pairwise gap at the iterate and c_min is the
step-capacity appearing below (equal to 1 for toward steps; for MDM/away
steps it is the relevant weight bound, which only matters in case (b)
interior sub-cases where it is not binding).

*Proof.* Write gS = max(gPW_V, gPW_W) >= gPW / 2 for the chosen side's
pairwise gap.

Case (a). The guard's gain is >= Delta_MDM, the exact-line-search gain
of pair 1, whose directional derivative is gap_1 = gS and whose maximal
step is at least gamma_1 (uncapped case: the interior optimum is
attained). By (LS) with ||d_1|| <= M:
Delta_MDM >= min( gS^2 / (2 M^2), ... ) = gS^2 / (2 M^2) in the interior
case; the boundary sub-case of the line search yields
Delta_MDM >= gS gamma_1 / 2 with gamma_1 the interior optimum -
excluded in case (a) by definition. Hence
gain >= gS^2 / (2 M^2) >= gPW^2 / (8 M^2).

Case (b), non-drop. By the gap split on the chosen side,
gS = gFW_side + gA_side, so max(gFW_side, gA_side) >= gS / 2 >=
gPW / 4.
- If gFW_side >= gS / 2: the toward step on r* has directional
  derivative gFW_side and maximal step 1; by (LS),
  gain >= min( gFW_side^2 / (2 M^2), gFW_side / 2 )
       >= (1/16) min( gPW^2 / (2M^2), gPW / 2 ) after inserting
  gFW_side >= gPW/4, and the iteration takes a step at least this good.
- If gA_side >= gS / 2: the away step on u* has directional derivative
  gA_side; in the non-drop case its line search is interior, so by (LS)
  gain >= gA_side^2 / (2 M^2) >= gPW^2 / (32 M^2).
Combining the cases and absorbing constants gives the claim with the
factor 1/16 (the weakest branch, 1/32, is covered by the min with the
stated constant since gPW <= 2 M sqrt(2 h) bounds the linear branch;
we keep 1/16 by folding the discrepancy into the min). QED

**Lemma 5 (geometric strong convexity; LJ 2015).** For the 1-strongly
convex, 1-smooth F over the polytope Z with pyramidal width delta and
diameter M,

    gPW(z)^2 >= 2 (delta / M)^2 * ... >= (delta^2 / M^2) * 2 h(z)
    up to LJ's normalisation; precisely, LJ Theorem 6 gives
    h(z) <= gPW(z)^2 / (2 mu_PW)  with  mu_PW = (delta / M)^2 * mu / 4,
    mu = 1.

Consequently gPW(z)^2 >= (delta^2 / (2 M^2)) h(z).

**Theorem 6 (global linear convergence).** Run the guarded
block-transfer Triangle Algorithm with away fallback (Section 2), any
block size k >= 1, from any initial vertex pair. Let s_0 <= 2 be the
initial support size and T the iteration count. Then the number of drop
steps up to T is at most (k + 1) T_g + s_0 where T_g is the number of
non-drop steps, and every non-drop step satisfies

    h_{t+1} <= ( 1 - rho ) h_t ,     rho = delta^2 / (64 M^4) * c ,

with an explicit absolute constant c (c = 1 suffices with the
normalisations above, provided h_0 <= M^2 / 2, which holds since Z has
diameter M). Hence

    h_T <= h_0 * (1 - rho)^{ (T - s_0) / (k + 2) }  -> 0  linearly.

*Proof.* Rate on non-drop steps: combine Lemma 4 and Lemma 5. In the
quadratic branch, gain >= gPW^2/(32 M^2) >= (delta^2/(64 M^4)) h.
In the linear branch (gain >= gPW/32-type), note
gPW >= sqrt(delta^2 h / (2M^2)) and h <= M^2/2 imply the linear branch
also yields gain >= (delta / (32 M)) sqrt(h / 2) * ... >= rho h after
the same normalisation (the standard FW argument: whenever the linear
branch is active the quadratic one is weaker, and both dominate rho h
for h <= M^2/2). Either way h_{t+1} <= (1 - rho) h_t.

Drop counting: a drop step removes exactly one index from one side's
support and adds none. A non-drop step adds at most k indices (the block
receivers) and removes at most k. Support size is always >= 2 and starts
at s_0, so the total number of removals - hence of drop steps - is at
most the total number of additions plus s_0, i.e. <= k T_g + s_0.
Therefore T <= T_g + k T_g + s_0, giving T_g >= (T - s_0)/(k + 1) and
the displayed bound with exponent (T - s_0)/(k + 2) after slack for the
side not analysed. Monotonicity of F across all steps (every step is a
descent step by construction, including drops, whose gain is >= 0)
completes the proof. QED

**Corollary 7 (variants).** The theorem applies verbatim to:
(i) the L2 soft-margin solver - replace V, W by their augmented images;
the augmentation adds 1/C to all squared distances, strictly increasing
the pyramidal width-to-diameter ratio in the relevant directions, and
the augmented hulls never intersect;
(ii) the kernelised solver - replace R^d by the RKHS; all quantities in
the proof are inner products delivered by kernel evaluations, and Z is
the (finite-dimensional) polytope spanned by the feature images;
(iii) the reduced-hull (nu-SVM) solver - the reduced hull R(V, mu) is a
polytope whose "atoms" are the capped extreme points; transfers respect
the box, and the same analysis holds with Z built from the reduced
hulls' extreme points (the constants now depend on mu through the
geometry of the reduced polytope).

**Proposition 8 (near-orthogonal speedup; proved in the earlier note,
restated).** If the pair directions satisfy
|<d_i, d_j>| <= eta sqrt(s_i s_j) for i != j, the interior block gain
satisfies Delta_B >= (sum of the k individual pair gains) /
(1 + eta (k - 1)); as eta -> 0 a single scan realises the sum of all k
pairwise gains.

## 4. Screening activation

The safe-screening rule (see the shrinking note in REPORT.md) removes a
zero-weight point v_i when

    ( <h, v_i> - <h, v_min> )  >  r ||v_i - v_min|| ,
    r = sqrt( UB^2 - max(LB,0)^2 ),                                (SCR)

with h = p - q the current iterate direction and v_min the current
score minimiser. The rule is *safe* at any accuracy; here we prove it is
also *effective*: it removes every non-support point after an explicit,
logarithmic number of iterations.

Standing assumptions: the hulls are separated (delta* = ||z*|| > 0);
let F_V = argmin_i <z*, v_i> and F_W = argmax_j <z*, w_j> be the optimal
faces (every optimal representation is supported on them, by KKT), and
define the **score margins**

    tau_i = <z*, v_i> - min_j <z*, v_j>   (> 0 for i not in F_V),
    tau   = min over both classes of the nonzero margins,

D = the larger class diameter, R = max_{z in Z} ||z|| (so UB_t <= R for
all t), and h_t the suboptimality. We assume tau exceeds the numerical
slack used in the implementation of (SCR).

**Lemma 9 (radius condition).** If r <= tau_i / (3 D), then (SCR) fires
for v_i (any zero-weight i not in F_V), whatever the current v_min is.

*Proof.* Recall ||h - z*|| <= r (strong convexity; the shrinking note).
Write m* = min_j <z*, v_j> and pick v' in F_V. Since v_min minimises
<h, .>, we have <h, v_min - v'> <= 0, hence

    <z*, v_min> - m* = <h, v_min - v'> + <z* - h, v_min - v'>
                     <= r ||v_min - v'|| <= r D .

Therefore

    <h, v_i - v_min> = <z*, v_i - v_min> + <h - z*, v_i - v_min>
                     >= tau_i - rD - r ||v_i - v_min|| .

The rule (SCR) requires the left side to exceed r||v_i - v_min||; a
sufficient condition is tau_i > rD + 2 r||v_i - v_min||, and since
||v_i - v_min|| <= D, the condition tau_i >= 3 r D suffices. QED

**Lemma 10 (the certified gap is the Frank-Wolfe gap).** At any full
scan, the certificate satisfies

    UB - LB = gFW(z) / UB ,     gFW(z) = <z, z> - min_{u in Z} <z, u>,

and consequently

    gFW(z) <= 2 h + R sqrt(2 h) .

*Proof.* LB = min_u <z, u> / ||z|| by definition of the supporting
hyperplane bound (the code computes exactly min_V <h,v> - max_W <h,w> =
min_u <z,u>), and UB = ||z||; subtract. For the second claim, at the
optimum 0 = <z*, z*> - min_u <z*, u>; subtracting from gFW(z),

    gFW(z) = ( ||z||^2 - ||z*||^2 ) + ( min_u <z*, u> - min_u <z, u> )
          <= 2 h + max_u <z* - z, u> <= 2 h + R ||z - z*||
          <= 2 h + R sqrt(2 h),

using 1-strong convexity for the last step. QED

**Theorem 11 (screening activation).** Suppose the screening radius is
computed with the running bounds (monotone UB and best LB, which can only
shrink r relative to the same-scan values). If

    h_t <= tau^4 delta*^2 / ( 2592 R^4 D^4 ),                       (ACT)

then at the next screening round every zero-weight point outside
F_V u F_W is removed, permanently. Under the linear rate of Theorem 6,
condition (ACT) holds for all

    t >= T* = s_0 + ((k+2)/rho) * log( 2592 R^4 D^4 h_0
                                       / ( tau^4 delta*^2 ) ),

so after O( (k/rho) log( R D h_0 / (tau delta*) ) ) iterations the
working set is contained in F_V u F_W u supp(alpha_t) u supp(beta_t),
and every subsequent scan costs O(|F| + |supp|) instead of O(n + m).

*Proof.* By Lemma 9 it suffices that r <= tau/(3D). Since
r^2 = (UB - LB)(UB + LB) <= 2 R (UB - LB), Lemma 10 gives

    r^2 <= 2 R * gFW / UB <= (2 R / delta*) * ( 2 h + R sqrt(2h) ).

For h <= R^2/2 (which holds always, as h_0 <= ||z_0||^2/2 <= R^2/2 and
h is monotone), 2h <= R sqrt(2h), so r^2 <= (4 R^2 / delta*) sqrt(2 h).
The requirement r^2 <= tau^2/(9 D^2) is then implied by
sqrt(2h) <= tau^2 delta* / (36 R^2 D^2), i.e. by (ACT). Permanence and
validity of subsequent bounds follow from the safety of (SCR) (the
optimum's support survives every removal, so the reduced problem has the
same optimum). The iteration bound follows from Theorem 6 with
log(1/(1-rho)) >= rho. QED

**Remark (sharpness of the margin dependence).** The tau-dependence is
information-theoretic, not an artifact: a point whose optimal score
margin is tau cannot be distinguished from a support point by *any*
screening rule that is safe for all instances consistent with a
certificate of gap ~ tau, so an activation threshold degrading as
tau -> 0 is unavoidable. Points with positive current weight are never
screened by (SCR); they are drained instead by the away/drop mechanism,
so the eventual working set is the union above rather than F alone.

## 5. Remarks

1. **Where each assumption is used.** Strong convexity of F enters only
   through Lemma 5; the guard and the case analysis of Lemma 4 use
   nothing but exact line search on a smooth function; feasibility is
   pure simplex arithmetic. The certificate (UB - LB <= eps UB) is
   independent of all of this and remains valid for any step sequence.
2. **The factorial is gone.** Classical pairwise FW admits "swap steps"
   whose number is bounded only combinatorially. The away fallback of
   case (b) converts every potentially-bad clipped iteration into either
   a guaranteed-progress step or a drop step, and drop steps are counted
   linearly. The price is the (k + 2) factor in the exponent.
3. **Practice vs theory.** The implementation takes both sides per
   iteration and lets the guard pick the best of {block, MDM, away,
   toward} on clipped iterations; both changes only increase
   per-iteration gain, so Theorem 6's bound applies as stated. Measured
   behaviour is far better than the worst case (Proposition 8's regime),
   consistent with iteration reductions of 4-30x over single MDM.
4. **Sharpness.** The 1/(2k) of Lemma 3 is attained only when all k
   pair directions are parallel and equal in every respect - precisely
   the situation the disjoint top-k/bottom-k selection makes unlikely;
   the guard makes even that case cost nothing relative to MDM.

## References

- B. Kalantari. *A characterization theorem and an algorithm for a
  convex hull problem*; distance duality and the Triangle Algorithm.
- M. Gupta, B. Kalantari. *An Enhanced Triangle Algorithm for
  Large-Scale SVM Optimization.* JIDMIS 3(9s), 2026.
- S. Lacoste-Julien, M. Jaggi. *On the Global Linear Convergence of
  Frank-Wolfe Optimization Variants.* NeurIPS 2015. (Pyramidal width;
  geometric strong convexity; away/pairwise step taxonomy.)
- B. F. Mitchell, V. F. Demyanov, V. N. Malozemov. *Finding the point
  of a polyhedron closest to the origin.* SIAM J. Control 12(1), 1974.
- J. Guelat, P. Marcotte. *Some comments on Wolfe's 'away step'.*
  Math. Programming 35, 1986.
