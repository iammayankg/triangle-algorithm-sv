# Cover letter: Annals of Operations Research

Paste into Editorial Manager. Bracketed items are for the author to fill in or confirm.

---

Dear Editors,

I submit the manuscript "Guarded block transfers for polytope distance: linear convergence, gap-certified screening, and when aggregation pays" for consideration in Annals of Operations Research.

The paper studies the polytope distance problem: the closest pair of points of two convex hulls, equivalently the minimum-norm point of their Minkowski difference. The problem underlies hard-margin support vector machines. It builds on Kalantari's Triangle Algorithm and its distance duality, published in this journal (Kalantari, Annals of Operations Research 226, 2015), and keeps that algorithm's certified duality-gap interval throughout.

The contributions are:

- **A block step with a guarantee.** A guarded block transfer aggregates several weight transfers into one exact line search, with an away-step fallback. On the difference polytope it converges linearly, with constants explicit in the pyramidal width and no combinatorial swap-step bound.
- **Screening with a finite activation bound.** The Triangle Algorithm's certificate equals the Frank–Wolfe gap divided by the current distance, so gap-safe screening applies. The paper bounds the iteration after which it has removed every point off the optimal faces.
- **When aggregation pays.** An exact expression gives a block's gain over a single transfer, in two quantities the method already computes. An identification theorem shows that drop steps stop after an explicit iteration. A timing model gives the block size that minimises work, which the solver sets adaptively.
- **An honest benchmark.** Five LIBSVM benchmarks were rerun with deliberately strengthened baselines: full kernel caches, both LIBLINEAR solvers, and exact-search single-transfer variants. Under these baselines the method's advantage is its certificate and intersection test, not speed. Our own sequential minimal optimisation code and LIBLINEAR's dual solver match or beat it on most real problems. The paper reports where each side wins.

**Prior version.** A six-page version was accepted as a poster at OPT 2026, the Optimization for Machine Learning workshop at NeurIPS 2026, which publishes no archival proceedings. This manuscript adds:

- the prose related work, with fourteen further references;
- the work bound in standard form;
- the entire analysis and measurement of when aggregation pays: the exact gain ratio, the identification theorem, the timing model, per-step measurements on five real problems, and the adaptive block size;
- screening and n-scaling experiments on real and synthetic data;
- the rerun of the timing experiments with strengthened baselines.

The algorithm, its convergence analysis and the regime batteries appeared in the workshop version.

**Availability and disclosures.**
- **Code and data.** The code, experiment scripts and result files are public at https://github.com/iammayankg/triangle-algorithm-sv [archived release DOI]. The datasets are the public LIBSVM benchmark sets.
- **AI tools.** Large language models were used to draft code, experiment scripts and text, as stated in the manuscript. I ran and verified every result.
- **Single author.** This is single-author work, and I declare no competing interests. The work was self-funded.

**Suggested reviewers** [optional; confirm each has no conflict and look up current affiliation and email before entering]. These are candidates from the literature the paper builds on:

- Sebastian Pokutta: blended pairwise conditional gradients, cited and compared.
- Javier Peña: facial distance and linear convergence of Frank–Wolfe variants.
- Martin Jaggi or Simon Lacoste-Julien: the pyramidal-width analysis the rate builds on.
- Bernd Gärtner: polytope distance and coresets.
- Francesco Rinaldi: Frank–Wolfe variants without bad steps.

Bahman Kalantari co-authored the earlier enhanced Triangle Algorithm paper and should not review.

Sincerely,
Mayank Gupta
Independent Researcher, Seattle, United States
iammayankg@gmail.com
[ORCID]
