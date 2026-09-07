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

## 5. Next measurements (superseded by Section 6)

- Rerun `block_diagnostics.py --keep-records` for k = 16 on both cells
  after the `_block_pairs` fix, to see eta and phi along the run (are
  drops clustered late, when the support is full?) and to confirm the
  fixed-cost coefficients.
- Add the other three KL2 cells and the synthetic hard margins to get
  eta across geometries (hypothesis: eta grows with support density and
  falls with d).
- Instrument BPCG the same way for the cost comparison in Theorem B's
  terms.

## 6. Five cells with per-step records (2026-09-07, `results/block_diag2.json`)

Isolated Studio run on the fixed pair-selection code, seed 0, k in
{1, 4, 16, 32}, records kept. The JSON is 103 MB and is committed; keep
future record-level runs out of git.

### 6.1 The model holds on all five cells

| cell | eta (k=4 / 16 / 32) | a (ms) | b (ms) | c_col (ms) | k* | time k=1 / 4 / 16 / 32 (s) | model | column share at k=16 | speed-up of best k over guarded k=1 |
|---|---|---|---|---|---|---|---|---|---|
| gisette L2 | 0.041 / 0.078 / 0.074 | 1.3 | 0.073 | 18.4 | 15 | 27.4 / 27.2 / 28.9 / 33.5 | 29.3 / 26.5 / 28.4 / 30.5 | 92 % | 1.0x |
| ijcnn1 KL2 | 0.257 / 0.262 / 0.243 | 2.1 | 0.093 | 0.21 | 8 | 13.5 / 6.6 / 6.7 / 7.6 | 14.5 / 7.7 / 7.7 / 9.8 | 14 % | 2.0x |
| w8a KL2 | 0.202 / 0.174 / 0.163 | 1.6 | 0.095 | 0.86 | 9 | 10.3 / 5.1 / 5.7 / 5.1 | 10.3 / 6.4 / 6.2 / 7.1 | 66 % | 2.0x |
| a9a KL2 | 0.107 / 0.117 / 0.111 | 3.2 | 0.132 | 0.62 | 14 | 36.3 / 16.9 / 12.2 / 13.7 | 37.2 / 16.8 / 13.3 / 14.8 | 36 % | 3.0x |
| covtype KL2 | 0.285 / 0.284 / 0.286 | 3.6 | 0.122 | 0.35 | 9 | 48.5 / 26.3 / 21.7 / 26.3 | 50.6 / 27.4 / 26.5 / 32.6 | 14 % | 2.2x |

- eta is a property of the cell: it does not move with k (columns 2-4)
  nor along the run (Section 6.2). Range 0.07-0.29 on real data.
- k* = sqrt(a(1-eta)/(b eta)) lands in 8-15 on every cell because a/b is
  15-30 ms/ms throughout and eta is 0.07-0.29. A fixed k of 8-16 is
  therefore within about 10 % of the best measured time on all five
  cells; adaptive k has little to gain here and is a safeguard for
  regimes with eta > 0.5 or a much smaller a/b (a faster inner loop),
  not a source of speed-up on these data.
- The model's time predictions are within 10 % except covtype at k >= 16,
  where it over-predicts iterations (the trajectory does better than the
  one-state ratio, as on ijcnn1).
- The wall-clock gain of blocks is Amdahl-limited by the column share:
  1.0x at 92 % (gisette), 2.0x at 66 % (w8a), 3.0x at 36 % (a9a), 2.0-2.2x
  at 14 % (ijcnn1, covtype, where eta is largest and the iteration gain
  saturates at 1/eta ~ 3.5-4).
- The fixed per-iteration cost a grows with the support (1.3 ms at 611,
  3.6 ms at 4,362): what remains O(|supp|) per iteration in Python is the
  weight-dict traversal in pair selection and the worst-active search.
  Reducing it raises k* and the block's margin over MDM.

### 6.2 Along the run: eta is flat, drops are front-loaded

Deciles of the k=16 runs (fraction of block calls that are drops; median
support; column misses per call):

| cell | drop % by decile 1 ... 10 | support by decile | misses/call decile 1 -> 10 |
|---|---|---|---|
| gisette L2 | 39, 54, 46, 37, 27, 13, 6, 2, 2, 1 | 259 -> 617 -> 611 | 6.4 -> 0.01 |
| ijcnn1 KL2 | 6, 7, 7, 5, 2, 2, 1, 1, 0, 1 | 851 -> 2,326 -> 2,388 | 11.0 -> 0.02 |
| w8a KL2 | 11, 8, 3, 3, 3, 1, 1, 1, 1, 0 | 344 -> 1,680 -> 1,820 | 11.7 -> 0.03 |
| a9a KL2 | 1, 2, 2, 2, 2, 2, 0, 1, 0, 0 | 1,016 -> 3,977 -> 3,795 | 14.1 -> 0.00 |
| covtype KL2 | 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 | 2,177 -> 4,509 -> 4,555 | 10.5 -> 0.01 |

Three facts, opposite to the "drops cluster when the support is full"
hypothesis:

1. Drops happen while the support is being built and stop once it has
   settled. On gisette half of all calls in the first three deciles are
   drops and 1-2 % in the last three; the kernel cells show the same
   shape at lower levels. Nearly every case-(b) call is a drop (the
   "pair 1 capped" and "drop" columns coincide), so the best-of-four
   branch is rare.
2. eta is flat along the run (gisette 0.07-0.08, ijcnn1 0.24-0.28, w8a
   0.16-0.19, a9a 0.10-0.16 falling slightly, covtype 0.27-0.31), and
   Q/k' rises from 0.6-0.9 early to 0.9-1.1 late. The block is at least
   as good late in the run as early.
3. Cache misses are front-loaded in the same way (6-14 per call in the
   first decile, ~0 in the last), so on column-bound cells the whole
   column bill is paid during identification, whichever k is used.

Total drops per support point (drops / final support): gisette 0.13,
0.13, 0.48, 0.69 (k = 1, 4, 16, 32); ijcnn1 0.005, 0.018, 0.041, 0.096;
w8a 0.006, 0.004, 0.016, 0.019; a9a 0.009, 0.011, 0.010, 0.022; covtype
0.003, 0.005, 0.006, 0.016. Drops scale with the support size, grow
with k, and are an order of magnitude more frequent on the sparse
linear cell (gisette: 611 support points carrying weights of very
different sizes, so fresh receivers with small weight become the worst
donor at once) than on the dense kernel cells (weights of order
1/2,000).

### 6.3 Revised theorem plan

The bridge (Theorem B) needs phi, the drop fraction. Section 6.2 says
phi is a transient of the identification phase, not a steady-state
property: after the support settles, drops stop. That is exactly what
the active-set identification theory for away-step and pairwise FW
provides (Bomze, Rinaldi, Zeffiro 2020, "active set complexity";
Garber 2020 under strict complementarity): after a finite number of
iterations the support equals the optimal face and no further drops
occur. So the right statement is

**Theorem B' (drop count via identification).** Under strict
complementarity of (P) on Z, the guarded block algorithm with any k
identifies the optimal face after T_id(k) iterations, and the total
number of drop iterations is at most the number of indices added and
later removed during identification, D(k) <= k T_id(k). After T_id the
rate of Theorem 3 holds with T_good = T - T_id and no drop term, and
with eta-diverse, beta-flat blocks the contraction is rho_k per
iteration. Total work: W(k) <= c_col N_cols + (a + bk) [ T_id(k) +
log(h_{T_id}/eps)/rho_k ].

What must be proved: T_id(k) for the guarded block step. The existing
active-set complexity bounds are for single away/pairwise steps and
give T_id in terms of the strict-complementarity gap and the rate; the
block adds at most k indices per iteration and the guard keeps the
same rate, so the argument should go through with k entering only the
support-size bound during identification (which is what the measured
D(k)/|supp| = 0.005-0.7 reflects). This is the piece of theory that is
new, and the data say it is the right one: drops are an identification
cost, paid early, proportional to the support and to k, and zero
afterwards.

What the paper can then claim (Corollary C, sharpened): aggregation
with block size k reduces the per-iteration term by k/(1+eta(k-1)) at
cost (a+bk)/(a+b), never reduces the column term, and adds an
identification cost of at most k T_id extra drop iterations; on the
five cells the drop cost is 1-10 % of iterations at k = 16 (31 % on
gisette, where it does not matter because columns dominate), and the
measured optimum k* = sqrt(a(1-eta)/(b eta)) is 8-15 for every cell.

### 6.4 Next steps

1. Write Proposition A, Theorem B' and Corollary C into `paper/aor/aor.tex`
   as a new section "When does aggregation pay?" with the five-cell table
   and the decile figure (drops and misses along the run).
2. Prove T_id(k): start from Bomze-Rinaldi-Zeffiro's argument for
   away-step FW (their Theorem on active-set complexity), replace the
   single-index addition by k, and check the guard does not break the
   "no bad steps after identification" property.
3. Reduce the O(|supp|) per-iteration overhead (a): maintain the active
   index set as an array alongside the weight dict, or rewrite
   `_worst_active_*` and `_block_pairs` to reuse one extraction per
   iteration. Then rerun `step_profile.py` and `block_diagnostics.py`
   (summaries only; no `--keep-records` in git) for the paper's final
   numbers.
4. Adaptive k as a safeguard (Section 4), evaluated on the five cells
   plus a synthetic sweep in eta (control the cosine between class
   directions) to show the rule tracks k*.
