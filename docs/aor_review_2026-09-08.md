# Review of `paper/aor/aor.pdf` for Annals of Operations Research (2026-09-08)

Reviewed: `aor.tex` (1,227 lines) + `sec_aggregation.tex` (579 lines) + `refs.bib`
(27 entries), rebuilt in a scratch copy with tectonic (26 pages, `article`
fallback class, "Draft of September 8, 2026"). Proofs of Lemmas 1–6,
Theorems 4 and 8, Propositions 7 and 9, Theorem 12 and Corollary 13 were
checked line by line; every arithmetic claim in the prose was checked
against the table it cites.

## Verdict

The mathematics is sound and the honesty of the empirical section is
unusual and welcome. The paper is not yet submittable. Three things stand
between it and a submission a referee will take seriously:

1. Mechanical blockers (TODOs, template, references, declarations).
2. The empirical section has three provenances and reads as a lab notebook.
3. Several ranges in the prose are stale relative to the five-cell Table 5,
   and two abstract claims are stronger than the tables support.

None of the theory needs to change. Most of the work is editorial.

## A. Blockers before upload

| item | state |
|---|---|
| `\todo` markers | 7 remain: affiliation ×2, repository URL, funding, competing interests, code availability, prior-publication list |
| class | builds with `article`; `sn-jnl.cls` not yet present, so theorem styles, running heads, APA formatting untested |
| references | no DOIs (AOR asks for full `https://doi.org/` links); `gupta2026` has no pages |
| `\citet` | 47 `\citep` vs 1 `\citet`; at least 11 sentences read "the step of (Author, Year)" |
| date line | "Draft of \today" on the title page |
| overfull boxes | Table 1 tabular (+9.5 pt) and Table 2 tabular (+20 pt) exceed the text width; recheck under `sn-jnl` |
| floats | 8 `[h]` floats forced to `[ht]`; Tables 2–8 stack behind Section 6.1 |
| AI statement | AOR wants LLM use documented in the Methods section; currently only under Declarations |
| ORCID, cover letter, employer IP clearance | per `docs/aor_submission.md`, still open |

## B. Major issues a referee will raise

### B1. Framing for the OR audience

- The introduction is one page and leads with SVMs. The checklist's own
  item 4 is right: lead with the polytope-distance / minimum-norm-point
  problem (Wolfe 1976, MDM 1974, Gilbert 1966, Kalantari 2015 in this
  journal) and present SVM training as the application.
- Say precisely which parts of the Triangle Algorithm survive (distance
  duality, the LB/UB certificate) and which are replaced (pivots by MDM
  transfers). A referee familiar with Kalantari's work will ask whether the
  method is still "the Triangle Algorithm" or pairwise Frank–Wolfe with a
  TA certificate. The current text does not confront this.
- Related work is a list of citation clusters at the end. A journal reader
  expects a prose related-work section near the front, or at least a
  paragraph in the introduction, with the OR lineage first.
- Contribution 5 says "new in this version". A journal paper has no
  "version"; say "beyond the workshop paper" or drop the parenthetical.
- No single theorem states the total work bound in standard form. OR readers
  expect something like: iterations $O\big((k+1)(M/\delta)^2\log(h_0/\varepsilon)\big)$,
  each costing $O(n+m+k'n+k'^2)$ plus $O(nd)$ per new column. All the pieces
  exist; state them once as a corollary of Theorem 4.

### B2. Experimental provenance (the biggest referee risk)

The paper currently reports results from three different code states and
load conditions and explains the differences in prose:

- Tables 1, 5, 7: released code, two-sided schedule, isolated or 19-process load.
- Tables 2 and 6, Figure 1: "the submitted version", unconditional V-then-W
  order, and a single-surviving-pair path that bypassed the fallback.
- Tables 3 and 8: "absolute times about 3× those of an earlier run, so
  their load is uncertain and only within-table ratios should be read".

The "Hardware and concurrency" paragraph (~40 lines) and the sentence about
the "shared development container" run in the synthetic paragraph are
provenance confessions. A referee will not read them as rigour; they will
ask for one protocol. Recommendation: rerun Tables 2, 3, 6, 8 and Figure 1
with the released code in isolation (the checklist's item 3 already lists
Table 2), then reduce the hardware paragraph to five lines: machine,
threads, isolation, seeds, cap. Keep the observation that concurrent load
manufactured a 1.3× ratio as one sentence; it is a useful warning.

### B3. Claims stronger than the tables

| claim | where | what the table says |
|---|---|---|
| "beats BPCG and kernel SMO on kernel $L_2$ margins" | abstract, conclusion | K-ETA is slower than kernel SMO on ijcnn1 (7.7 vs 5.3 s) and covtype (21.4 vs 15.4 s); 3 of 5 cells are wins |
| "K-ETA is 1.2× faster than LIBSVM on the RBF hard margin" | §6, gisette KHM | 57.8±8.7 vs 68.1±13.4; intervals overlap, so by the paper's own rule this is a tie |
| "The gain ... is exactly $Q/\kappa$" | abstract | Proposition 7 gives $Q(2-\kappa)$ for $\kappa\le1$; say "an exact expression in two observable quantities" or give both |
| "a heuristic pyramidal-width estimate suggests $O(n^2)$-type cost" | contribution 4 | The n-scaling experiment the checklist promises is not in the paper; either run it or demote the item from the contribution list to a remark |

### B4. Screening never activates on real data

Theorem 8 is contribution 3, but on real data screening ran only in the
LIN cell and "never activated at $10^{-3}$". The only evidence is the
synthetic Figure 4. Run at least one real cell (gisette LIN or $L_2$) at
$\varepsilon=10^{-5}$ with screening on and off and report survivors and time,
or a referee will call the theorem unexercised.

### B5. Corollary 13 mixes rigour and a fitted model

The corollary contains "$\approx$", "$\lesssim$" and "this is the rigorous
part". Split it: a Corollary with the contraction product and the iteration
count, then a "Timing model" paragraph with $W(k)$. The same applies to the
"Cost" paragraph, which is fine as prose but should not feed into a numbered
result.

### B6. Notation overload

A notation table is needed. Symbols with two meanings in the same paper:

- $\kappa$: the ratio $D/S$ (§7) and the kernel $\kappa(V,x_i)$ (§5).
- $\beta$: $W$-side weights and, until renamed, the flatness parameter (now $\beta_{\mathrm{flat}}$, good; but Table 9's header still says $\beta_{\mathrm{flat}}$ while the caption says "its $\beta$").
- $r$: screening radius and receiver index $r_j, r^*$.
- $s$: score $s_i$, support size $s_t$, initial support $s_0$, FW atom $s$ in Lemma 3.
- $\sigma$: directional derivative (Lemma 2) and the weight-recovery constant (Assumption 11).
- $a,b$: timing coefficients, $a_k$, $b_j$, and the SVM bias $b$.
- $D$: $\|d\|^2$ and (implicitly) dimension $d$; $R$: radius and reduced hull $R(V,\mu)$; $C$: regulariser and cost $C_B$; $m$: $|W|$, misses $m_B$, and $m_V$; $L$: class diameter and $L_2$.

### B7. Missing references

Cited baselines and standard results with no entry in `refs.bib`:

- Platt (1998), SMO; Chang & Lin (2011), LIBSVM; Fan et al. (2008), LIBLINEAR; Pedregosa et al. (2011), scikit-learn.
- Frank & Wolfe (1956); Jaggi (2013), "Revisiting Frank–Wolfe".
- Peña, Rodríguez & Soheili (2016), von Neumann / FW with away steps for polytope feasibility, the closest OR relative.
- Beck & Shtern (2017) and Garber & Hazan (2016) on linear rates of FW variants over polytopes.
- Braun, Carderera, Combettes, Hassani, Karbasi, Mokhtari & Pokutta (2022), the conditional-gradient survey.
- Crisp & Burges (2000) and Mavroforakis & Theodoridis (2006) for the reduced-hull view of $\nu$-SVM.
- Franc & Hlaváč (2003) nearest-point algorithm for SVM; Keerthi et al. (2000) is cited but Franc–Hlaváč is not.
- Kalantari's later TA papers (2019 separating-hyperplane theorem; Kalantari & Zhang on SVM) to show the lineage within this journal's readership.

## C. Technical check of the analysis

All verified; the notes are for polish, not correctness.

- Lemma 1, Lemma 6, Theorem 4, Lemma 5, Proposition 7 (exact ratio, both cases), $k^*$ formula, and the constants $H$, $H_{\mathrm{id}}$, $H_{\mathrm w}$ (2592, 41472, 32) all recompute correctly.
- Theorem 12 (i)–(iii): (F1)–(F3), the $55\tau/72$ away gap, the $\tau/(11L^2)$ clip length, the uncapped-transfer bound $\gamma_j<\alpha_{\min}/2$, and the $4\rho$ case-(a) contraction all check.
- Theorem 4, discussion paragraph: the sentence "Table 2 used the earlier unconditional V-then-W order" is experimental provenance inside the theory section; move it to §6.1.
- Assumption 11 (A2): the $\sigma$ inequality is a consequence of affine independence plus uniqueness (the paragraph after says so). State it as a definition of $\sigma$, not an assumption.
- Theorem 12 (ii) uses $a_k$ in the drop bound where $k$ would do; harmless, but say "for uniformity".
- Theorem 8 needs $\delta^*>0$; Theorem 12 restates it; fine. Theorem 4 does not need it; say so, since TA I handles the intersecting case.
- Lemma 3 cites "[Thm. 6] of Lacoste-Julien & Jaggi" as the pyramidal-width inequality; that paper states the inequality in its Theorem 3 / Lemma 5 area in the NeurIPS version. Check the exact pointer against the version you cite.
- "Conventions" paragraph (§4): good; consider moving the $k'\ge1$ convention next to Algorithm 1.

## D. Stale numbers (prose vs Table 5)

Table 5 now has five cells; several ranges were written when it had two.

| sentence | says | Table 5 gives |
|---|---|---|
| guard alone cuts MDM iterations (§6 "Blocks and screening") | 1.4–2× | 1.4–2.7× (w8a 2.7, a9a 2.3, covtype 2.1) |
| aggregation cuts iterations a further (§6, and after Table 5) | 4.2–4.4× | 3.8–7.0× (w8a 7.0, a9a 6.6, covtype 3.8) |
| block over guarded $k{=}1$, four kernel cells | 1.9–2.4× | 1.8–2.4× |
| block over BPCG in isolation (§6 twice) | 4.2–6.1× | 3.8–6.1× (ijcnn1 3.75; the text itself says 3.8× two paragraphs later) |
| gisette tie, paragraph after Table 5 | 31–38 s | 29.4–31.9 s (the earlier paragraph correctly says 29–32 s) |
| hardware paragraph, ijcnn1 guarded $k{=}1$ | 10–11 s | 9.7 s in Table 5 |
| $a/b$ range | "16–27" (§7.3) and "15–30" (§7.4) | inline Table 7 rows give 16–28; pick one |
| `tab_diag_rows.tex` | different numbers from the inline Table 7 rows ($k^*$ 15 vs 14, gisette 27.4/27.2/28.9/33.5 vs 33.0/29.2/33.9/38.0) | file is not `\input`; delete it or make Table 7 read from it so the two cannot drift |
| Table 7 vs Table 9 $\hat\eta$ | 0.07/0.11/0.29 vs 0.08/0.12/0.28 | different medians over different subsets on a different machine; one footnote saying so avoids a referee query |

## E. Structure and length

- 26 pages, 9 tables, 4 figures, with Tables 2, 3, 4, 6, 8 documenting the regime where the method loses or that carry an uncertain load. Suggested cut for the main text: keep Tables 1, 2 (condensed), 5, 7, 9, 10 and Figures 1–2, 4; move Tables 3, 4, 6, 8 and Figure 3 to an online supplement.
- The "Step ablation, blocks and shrinking (synthetic)" paragraph is 30 lines of numbers in prose; it wants a small table or a cut.
- Section order: 4 (Analysis) → 5 (Reductions) → 6 (Experiments) → 7 (When does aggregation pay, with two more theorems) → 8 (Related work). Consider 7.1–7.2 immediately after Section 4 so all theory is contiguous, and Reductions as the first part of Experiments.
- Section 6.1 "Experimental details" is longer than Section 6 itself. Split into "Protocol" (short, main text) and "Baseline configuration" (supplement).

## F. Minor

- Abstract: 214 words, within 150–250. "We report both sides." and the conclusion's "and the paper says so" are informal for AOR.
- Title is 19 words; "Guarded block transfers for polytope distance: linear convergence, gap-certified screening, and when aggregation pays" would cover the new Section 7.
- Algorithm 1 uses "[case (a)]" labels in `\normalfont`; check rendering under `sn-jnl`.
- Table 1 caption: "conv = seeds in which ETA certified" but the KHM w8a row shows 5/5 with "hulls intersect"; add "or certified intersection".
- Table 1 KHM w8a time $11.7\pm32.4$ s: a CI wider than the mean signals a heavy-tailed seed; report min–max or the median.
- Table 2 uses "cap" for ETA and SMO but LIBLINEAR is "unbounded"; say "no cap" in the caption once.
- `inputenc` warning under tectonic is harmless but remove `\usepackage[utf8]{inputenc}` for the `pdflatex` class option only if the Springer template complains.
- "Lightning Studio" is a vendor product name; "a dedicated cloud VM (32 vCPU Xeon 2.6 GHz, 122 GB)" is enough.
- Kernel width "$1/d$" is called "width parameter" in Table 1 and $\gamma=1/d$ nowhere; state $\exp(-\|x-x'\|^2/d)$ once in the caption.
