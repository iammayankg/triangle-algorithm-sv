# Triangle Algorithm

Independent Python replication of:

> Gupta, M. & Kalantari, B., *An Enhanced Triangle Algorithm for Large-Scale
> Support Vector Machine Optimization: A Comparative Study with Classical and
> Modern Solvers*, JIDMIS Vol. 3, Issue 9s (2026).
> <https://jidmis.org/index.php/jidmis/article/view/2284>

The Triangle Algorithm is a geometry-based method for hard-margin SVM: it
decides whether two convex hulls intersect (TA I) and computes the distance
and optimal supporting hyperplane between them (TA II), enhanced here — as in
the paper — with joint closest-point segment updates, dot-product caching, an
anti-zig-zag pivot strategy, and prioritized searches.

**See [REPORT.md](REPORT.md) for the full replication study and results.**

## Layout

- `src/triangle_algorithm.py` — Enhanced Triangle Algorithm (TA I + TA II)
- `src/smo.py` — hard-margin SMO baseline (cached and uncached variants)
- `src/data.py` — synthetic two-ball data generation
- `src/experiments.py` — the paper's Table 2/3/4 protocols
- `src/plots.py` — figures
- `tests/test_correctness.py` — verification against exact QP ground truth
- `results/` — result tables (CSV/JSON), figures, and the raw run log
- `docs/` — convergence analysis (`block_convergence.md`), full-scale
  experiment protocol (`experiments_protocol.md`), and the Lightning.ai
  runbook (`lightning_runbook.md`)
- `paper/` — `opt2026.tex`, the OPT 2026 (NeurIPS workshop) submission,
  with `opt2026.pdf`; build with `cd paper && tectonic opt2026.tex`.
  Uses `neurips_2026.sty` as a stand-in for the required OPT 2026 style
  file (see the comment at the top of the tex). `main.tex` is the
  earlier long-form skeleton.

## Quick start

```bash
pip install numpy scipy matplotlib
python3 tests/test_correctness.py
python3 src/experiments.py --exp table3 --trials 3 --dims 3 100 1000
python3 src/plots.py
```

## Minimal usage

```python
import numpy as np
from src.triangle_algorithm import EnhancedTriangleAlgorithm

V = np.random.randn(1000, 50)        # class 1
W = np.random.randn(1000, 50) + 3.0  # class 2

ta = EnhancedTriangleAlgorithm(V, W)
r = ta.solve_distance(eps=1e-3)
print(r.status, r.distance, r.lower_bound, r.iterations, r.sparsity)
```

## License

MIT
