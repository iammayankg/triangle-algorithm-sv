# Annals of Operations Research submission — working checklist

Started 2026-09-06, after the OPT 2026 workshop submission (deadline
passed 2026-09-04; notification 2026-09-29). The journal version lives in
`paper/aor/` (`aor.tex`, `refs.bib`, bootstrapped from `../opt2026.tex` by
`tools/from_opt.py`; do not rerun the script once hand edits begin).

## What AOR requires (link.springer.com/journal/10479/submission-guidelines, read 2026-09-06)

| item | requirement | status |
|---|---|---|
| system | Editorial Manager, https://www.editorialmanager.com/anor | — |
| template | Springer Nature LaTeX template (`sn-jnl.cls`), "Journal article template package", 880 KB, https://www.springernature.com/gp/authors/campaigns/latex-author-support (also on Overleaf: overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq). Use the `pdflatex` class option. | not yet downloaded; `aor.tex` auto-switches to `sn-jnl` when the class file is dropped into `paper/aor/`, and builds with `article` until then |
| references | APA 7, author–year "(Author, Year)", alphabetised, DOIs as full `https://doi.org/...` links | `refs.bib` written with all 27 entries; DOIs still to add; class option `sn-apa` selected |
| abstract | 150–250 words, no undefined abbreviations or citations | 236 words (reuses the workshop abstract; revise once the extensions are in) |
| keywords | 4–6 | 6 set: polytope distance, Frank–Wolfe methods, Triangle Algorithm, support vector machines, safe screening, linear convergence |
| title page | names, affiliations, corresponding author, ORCID (required before submission) | affiliation TODO (also needed for the OPT camera-ready); create/confirm ORCID |
| declarations | Funding, Competing interests, Data availability (mandatory for original research), Ethics (n/a), Code availability | section drafted at the end of `aor.tex` with TODO markers |
| AI use | "Use of an LLM should be properly documented in the Methods section"; copy-editing need not be declared | statement present under Declarations; move/duplicate into the experimental-details section |
| prior versions | allowed when "the new work concerns an expansion of previous work"; be transparent about reuse in the cover letter | OPT 2026 is non-archival; disclose it in the cover letter and in the Declarations paragraph |
| review model | not stated on the guidelines page (Springer default for AOR has been single-blind) | keep the author block; confirm in Editorial Manager at submission |
| length | no stated limit | — |
| open access | hybrid; APC applies only if Open Choice is selected | decide (employer policy) |
| source files | full LaTeX source + PDF at submission | build is `tectonic aor.tex` in `paper/aor/` |

## Before submitting: content

The workshop paper is the skeleton; a journal reader expects these
additions. Ordered by value.

1. **Employer IP clearance.** Settle this before anything else (see the
   earlier question about retraction: withdrawing after acceptance is
   possible but costly; withdrawing before is free).
2. **Full proofs inline** — done by the bootstrap (Lemmas and Theorems now
   carry their proofs; the separability/side lemmas sit at the end of
   Section 2; the two-sided lemma and its remark follow Theorem 3).
   Read through once for flow: the "In words" sentence after Theorem 6
   now precedes its proof.
3. **Experiments still owed** (reviewer items from rounds 1–3):
   - n-scaling experiment for the O(n^2) heuristic (Section 5, "near-regular
     simplex" argument): run the synthetic L2 instance at n = 1k, 2k, 5k,
     10k, 20k per class and report iterations and time vs n.
   - Table 2 (`full_battery.py`) rerun on the fixed code (~1 day on the
     Studio).
   - ETA primal-progress column for the time-to-1% table (`time_to_tol.py`).
   - Candidate-selection / realised-gain diagnostics and a measured eta for
     Proposition 4 on real data.
   - Optional: adaptive k (the conclusion lists it as future work; a
     simple rule "double k while the guard picks the block, halve when
     it picks MDM" would close the "static hyperparameter" objection).
4. **Framing for an OR audience.** AOR published Kalantari's Triangle
   Algorithm paper (2015); lead the introduction with the polytope-distance
   / minimum-norm-point problem (Wolfe 1976, MDM 1974, Gilbert 1966) and
   present SVM training as the application, not the other way round.
   Expand Related work (Section 7 now merges the workshop's Section 6 and
   Appendix D) into prose with the OR lineage first.
5. **Tables and figures.** All workshop tables are in the body now; decide
   which of Tables 2–8 stay in the main text and which move to an
   appendix or supplement. Figures are referenced from `../figs/`.
6. **Numbers provenance.** Everything except Table 2 and the Table 6 traces
   is from the fixed code (see `docs/lightning_runbook.md`); the journal
   version should have a single provenance ("released code, commit X").
7. **Abstract and title.** Rewrite the abstract for the journal scope once
   (3) is done; the title can stay.

## Journal contribution: when does aggregation pay? (started 2026-09-06)

Plan (from the round-5 advice): (1) exact gain ratio at one state,
Delta_B/Delta_1 = Q(2-kappa) for kappa<=1 and Q/kappa for kappa>1, with
Q = sum_j Delta_j / Delta_1 and kappa = D/S (verified); (2) a cost model
that includes column copies and score updates; (3) instrument and
measure; (4) bridge one-state progress-per-cost to total runtime;
(5) adaptive k from the measured ratio.

Step 3 is implemented: `ETA.diag = []` makes `_block_transfer_V/W` record
Delta_B, Delta_1, Q_unc, Q_real, kappa, candidate chosen, capped pairs,
cache misses, support size and a time split (pair selection, columns,
assembly, guard, update) plus a dry-run timing of the single MDM update
from the same state.  Driver: `src/block_diagnostics.py` (summaries to
`results/block_diag.json`).  Run on the Studio, nothing else running:

```bash
OMP_NUM_THREADS=1 nohup python3 -u src/block_diagnostics.py --data-dir data \
    --cells gisette-l2 ijcnn1-kl2 --ks 1 4 16 32 --out results/block_diag.json \
    > results/block_diag.log 2>&1 &
```

Local pilot (synthetic L2 overlap, d=1000, n=5000/class, seed 0):

| k | time | iters | gain ratio (median) | Q_unc | kappa | C_B/C_1 (step / incl. overhead) | progress per cost | block wins |
|---|---|---|---|---|---|---|---|---|
| 1 | 13.4 s | 39,271 | 1.00 | 1.00 | 1.00 | 2.40 / 1.77 | 0.42 / 0.57 | — |
| 16 | 3.8 s | 2,801 | 10.1 | 10.8 | 1.06 | 6.9 / 4.3 | 1.45 / 2.33 | 93–95 % |

Readings: on this dense-support instance the directions are nearly
orthogonal (kappa ~ 1.06), Q ~ 0.7 k, and the block costs ~7x a bare
MDM step per call, so it wins ~1.5-2.3x per unit of work and 3.5x over
the run (the trajectory effect on top of the per-call ratio is the
step-4 question).  The k=1 row shows the guarded single-pair machinery
itself costs 2.4x a bare pairwise step per call (pair selection and
assembly in Python), which is why guarded k=1 saved iterations but not
time in Table 5.  Caveat: C_1 counts only the scan, the receiver's
column and the two-column update, so it slightly favours MDM.

Studio results (2026-09-06, k = 1/4/16/32 on gisette-l2 and ijcnn1-kl2)
are analysed in `docs/aggregation_model.md`: gain ratio = k/(1+eta(k-1))
with a per-dataset eta (0.07 gisette, 0.25 ijcnn1, 0.004 synthetic),
cost = c_col N_cols + T(k)(a + b k), optimum k* = sqrt(a(1-eta)/(b eta)),
drops the limiting factor at large k. Draft Proposition A / Theorem B /
adaptive rule there.

## Hand-fix list left by the bootstrap

- Two `\todo{}` markers on the title page (affiliation), one for the
  repository URL in the introduction, four in Declarations.
- `Section~\ref{app:...}` references were "Appendix" references; check
  each reads naturally (about 10 occurrences).
- The `proof` environment's optional argument is written as
  `[ of Lemma~\ref{...}]` (a workaround for the OPT class); with `sn-jnl`
  or amsthm it renders "Proof of Lemma 1." — fine, but check.
- The bootstrap turned every `\cite` into `\citep`; switch to `\citet`
  where the citation is the sentence's subject.

## Cover letter (draft points)

- Extends a six-page non-archival workshop paper (OPT 2026 at NeurIPS
  2026); the new material is X, Y, Z.
- Builds directly on Kalantari (2015), published in AOR.
- Single author; no competing interests; code and data public.
- Suggested reviewers / editors: TODO.
