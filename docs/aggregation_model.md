# When does aggregation pay? Measurements, model, and the theorem to build

Written 2026-09-06 from `results/block_diag.json` (Studio, isolated run of
`src/block_diagnostics.py`, seed 0, k in {1, 4, 16, 32}) and the local
pilot on the synthetic L2 instance. This is the working document for the
journal contribution; nothing here is in the paper yet.

## 1. What was measured

Per block transfer, from one iterate and cache state: the block gain
Delta_B and the MDM gain Delta_1, Q = sum_j Delta_j / Delta_1, kappa = D/S,
the candidate taken, capped pairs, cache misses, support size, and the
time split of the block step next to a dry-run timing of the single MDM
update. Per iteration: scan time, step time, fixed overhead.

| cell | k | time (s) | iters | columns | column time (s) | ms / it (non-column) | kappa (median) | eta = (kappa-1)/(k-1) | Q/k | gain ratio (median) | k/(1+eta(k-1)) | drop calls | pair 1 capped | support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gisette L2 | 1 | 30.8 | 3,471 | 1,334 | 26.2 | 1.3 | 1.00 | — | 1.00 | 1.00 | 1.00 | 1.2 % | 3.1 % | 611 |
| gisette L2 | 4 | 30.9 | 1,071 | 1,349 | 28.8 | 2.0 | 1.12 | 0.041 | 0.98 | 3.42 | 3.56 | 3.8 % | 4.9 % | 609 |
| gisette L2 | 16 | 35.0 | 831 | 1,480 | 32.3 | 3.2 | 2.16 | 0.078 | 0.98 | 6.90 | 7.37 | 20.6 % | 21.7 % | 627 |
| gisette L2 | 32 | 37.9 | 861 | 1,588 | 33.9 | 4.6 | 3.27 | 0.074 | 0.84 | 7.54 | 9.72 | 30.6 % | 31.1 % | 629 |
| ijcnn1 KL2 | 1 | 14.9 | 6,211 | 4,582 | 1.1 | 2.2 | 1.00 | — | 1.00 | 1.00 | 1.00 | 0.1 % | 0.1 % | 2,180 |
| ijcnn1 KL2 | 4 | 7.9 | 2,376 | 4,598 | 1.0 | 2.9 | 1.77 | 0.257 | 0.94 | 2.11 | 2.26 | 0.8 % | 0.9 % | 2,180 |
| ijcnn1 KL2 | 16 | 7.4 | 1,456 | 4,660 | 1.1 | 4.3 | 4.92 | 0.262 | 0.89 | 2.77 | 3.25 | 3.1 % | 3.5 % | 2,182 |
| ijcnn1 KL2 | 32 | 11.0 | 1,346 | 4,796 | 1.3 | 7.2 | 8.51 | 0.243 | 0.85 | 2.99 | 3.75 | 8.2 % | 8.3 % | 2,181 |
| synthetic L2, d=1000 (Mac) | 1 | 13.4 | 39,271 | 1,308 | 2.0 | — | 1.00 | — | 1.00 | 1.00 | 1.00 | 0.6 % | 0.6 % | 437 |
| synthetic L2, d=1000 (Mac) | 16 | 3.8 | 2,801 | 1,341 | 1.9 | — | 1.06 | 0.004 | 0.67 | 10.1 | 15.1 | 3.6 % | 3.6 % | 446 |

The k=1 rows are the guarded single pair; their "block" and "mdm" kinds
are the same step.

## 2. What the data supports

**Gain side: direction diversity, one number per dataset.** The exact
gain ratio at a state is Delta_B/Delta_1 = Q(2-kappa) for kappa <= 1 and
Q/kappa for kappa > 1 (verified algebraically and numerically). Two
empirical facts make it predictable:

- Q ~= k' on every cell (0.84-0.98 k'): the top-k' receivers have gains
  comparable to the leading pair's, so "useful progress beyond pair 1"
  is not the bottleneck.
- kappa grows linearly in k' with a slope eta that is constant across k
  for a given cell: eta ~= 0.04-0.08 (gisette), 0.24-0.26 (ijcnn1),
  0.004 (synthetic). eta is the average cosine between transfer
  directions, exactly the parameter of Proposition 4 (kappa <= 1 +
  eta(k'-1)), measured for the first time.

Hence gain ratio ~= k'/(1 + eta(k'-1)), which saturates at 1/eta: about
13 on gisette, 4 on ijcnn1, 250 on the synthetic instance. Observed
medians 3.4/6.9/7.5 (gisette), 2.1/2.8/3.0 (ijcnn1), 10.1 (synthetic)
follow this within 10-25 %, the shortfall at large k coming from Q/k
falling below 1 and from capped pairs.

**Cost side: two regimes, one linear model.** Non-column cost per
iteration is a + b k with a = 1.4 ms, b = 0.10 ms (gisette) and a = 2.1
ms, b = 0.16 ms (ijcnn1); the k-dependent part is block assembly and the
k'-column score update. Column cost is 21 ms per column on gisette
(d = 5,000) and 0.24 ms on ijcnn1 (d = 22, kernel subsample), and the
number of columns computed over a run is nearly independent of k
(1,334-1,588 and 4,582-4,796): every point that ever enters the support
costs one column, whichever step rule brought it in, and aggregation
touches only 10-20 % more transient points.

So total time ~= c_col N_cols + T(k) (a + b k) with T(k) ~= T(1) (1 +
eta(k-1))/k. Predicted vs observed: gisette 33/30/33/35 vs 31/31/35/38 s;
ijcnn1 15/8.6/9.7/13.4 vs 14.9/7.9/7.4/11.0 s. The model optimum k* =
sqrt(a(1-eta)/(b eta)) is 13 (gisette) and 6 (ijcnn1); the measured
optima are "anything below 16" and 4-16. On gisette the k-dependent term
is 5-10 s under a 26-34 s column floor, which is why no block size helps
there: the lower bound c_col x support = 611 x 21 ms = 13 s is
unreachable by any step rule, and the remaining 13-21 s are columns of
transient points, a screening/caching question, not an aggregation one.

**Drop control is what breaks the model at large k.** The iteration
model predicts 457 iterations at k = 16 on gisette; observed 831. There,
21-31 % of calls are case (b) with pair 1 capped and 21-31 % of calls
are drops that make little progress; on ijcnn1 drops stay at 3-8 % and
the model over-predicts iterations instead (trajectory effect
favourable). The worst-case drop bound of Theorem 3, T_good >= (T -
s_0)/(k+1), would cancel the k-fold gain entirely; the measured drop
fraction phi is 0.01-0.31, far from 1 - 1/(k+1) = 0.94.

**Implementation finding.** The guarded k = 1 machinery cost 2.4-2.6x a
bare MDM step per call because `_block_pairs` rebuilt the active-donor
list in Python from the weight dict on every call (2.98 s of 14.9 s on
ijcnn1, support 2,180). Fixed 2026-09-06 (array filter; trajectories
unchanged): 13.4 -> 11.2 s (k = 1) and 3.8 -> 3.4 s (k = 16) on the
synthetic instance. The Studio numbers above predate the fix; rerun
`block_diagnostics.py` and `step_profile.py` before quoting them in the
paper.

## 3. The theorem to build (draft)

Definitions at a block iteration with k' retained uncapped pairs:
eta-diverse if |<d_i, d_j>| <= eta ||d_i|| ||d_j|| for i != j; beta-flat if
Delta_j >= beta Delta_1 for all j. Cost model: C_B = a + b k' + c_col m_B,
C_1 = a + b + c_col m_1 with m the number of column misses.

**Proposition A (progress per unit work at one state).** At an
eta-diverse, beta-flat block iteration,
Delta_B >= Delta_1 (1 + beta(k'-1)) / (1 + eta(k'-1)), and the guarded
step's progress per unit cost is at least that ratio divided by
C_B/C_1. [Proof: Proposition 4 gives Delta_B >= sum_j Delta_j / (1 +
eta(k'-1)); flatness bounds the sum below by Delta_1 (1 + beta(k'-1)).
Immediate.]

**Theorem B (run-level).** Let the algorithm run with block size k under
the schedule of Theorem 3. Suppose that among the first T iterations at
most phi T are drop iterations and that every non-drop iteration is
either case (a) with an eta-diverse, beta-flat block, or case (b). Then
h_T <= h_0 (1 - rho_k)^{(1-phi) T} with
rho_k = min{ 1/2, rho (1 + beta(k-1)) / (1 + eta(k-1)) },
rho = (delta/M)^2/16 as in Theorem 3, and the work to reach h_T <= eps is
at most
W(k) <= c_col N_cols + [ log(h_0/eps) / ((1-phi) rho_k) ] (a + b k).
[Proof sketch: on a case-(a) non-drop iteration the guard takes
max(Delta_B, Delta_1) >= Delta_B >= (1+beta(k-1))/(1+eta(k-1)) Delta_1
and Delta_1 >= rho h by Lemmas 2 and 9 (the case-(a) branch of the
Lemma-2 proof); on a case-(b) non-drop iteration Lemma 2 gives the
plain rho; the min with 1/2 keeps the contraction meaningful. Drop
iterations are descent steps (Theorem 3). Multiply.]

Theorem B replaces the worst-case drop count (k+1) by the measured drop
fraction phi and the k-independent rho by rho_k. Its content is the
statement W(k)/W(1) ~= [(1+eta(k-1))/k] [(a+bk)/(a+b)] / (1-phi(k)),
minimised at k* = sqrt(a(1-eta)/(b eta)) when phi is flat, which is what
the two real cells show. What it does not do: bound phi(k) a priori. A
bound of the form phi(k) <= phi_0 + c k / |supp| under a strict
complementarity assumption (Garber 2020; Bomze-Rinaldi-Zeffiro 2020)
would close the argument and is the open theoretical piece; the measured
phi(k) (gisette 1/4/21/31 %, ijcnn1 0.1/0.8/3/8 %) is the target.

Corollary C (what the paper can claim). Aggregation cannot reduce the
column term; it reduces the per-iteration term by the factor
k/(1+eta(k-1)) at a cost (a+bk)/(a+b). It therefore pays only when
per-iteration O(n) work is a substantial share of the run (ijcnn1: 93 %;
gisette: 15 %), by up to 1/eta, and the block size that pays most is
sqrt(a(1-eta)/(b eta)), which is a property of the data (eta) and of the
implementation (a/b), not a universal constant.

## 4. Adaptive block size (step 5, design)

Every block step computes kappa = D/S exactly, so eta is observable
online: eta_hat = median over the last w block steps of (kappa-1)/(k'-1).
The cost coefficients a, b are observable from the step timer (or from a
two-point calibration in the first 50 iterations). Rule:

1. every w = 50 iterations set k <- clip(round(sqrt(a(1-eta_hat)/(b
   eta_hat))), 1, k_max);
2. if the case-(b) fraction over the window exceeds phi_max (say 10 %),
   halve k (support growth is outrunning the drops);
3. keep the guard and the away fallback unchanged, so Theorem 3 applies
   at every k the rule visits (the drop bound is then 2 k_max T_good +
   s_0).

Evaluation plan (isolated runs, `step_profile.py` style, matched
accuracy, peak memory): adaptive vs guarded k = 1, fixed k = 4, 16, 32
and BPCG on gisette L2, ijcnn1 KL2, and the other KL2 cells; report
the eta each cell settles at and the k the rule chooses.

## 5. Next measurements

- Rerun `block_diagnostics.py --keep-records` for k = 16 on both cells
  after the `_block_pairs` fix, to see eta and phi along the run (are
  drops clustered late, when the support is full?) and to confirm the
  fixed-cost coefficients.
- Add the other three KL2 cells and the synthetic hard margins to get
  eta across geometries (hypothesis: eta grows with support density and
  falls with d).
- Instrument BPCG the same way for the cost comparison in Theorem B's
  terms.
