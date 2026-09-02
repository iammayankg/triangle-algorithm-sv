"""Block-size ablation on the synthetic instances of App. C with the
current code (guard, away fallback, two-sided schedule of Lemma 6):
mdm (k=1) vs block k in {4, 16, 32}; iterations, column evaluations,
wall-clock, and step-kind counts."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import generate_two_balls, generate_overlap
from soft_margin import SoftMarginTA
from triangle_algorithm import EnhancedTriangleAlgorithm

rows = []
MODES = [('mdm', 1), ('block', 4), ('block', 16), ('block', 32)]
def run(tag, mk, eps, max_iter=200_000):
    base = None
    for mode, k in MODES:
        ta = mk(mode, k)
        r = ta.solve_distance(eps=eps, max_iter=max_iter)
        base = base or r.distance
        row = dict(tag=tag, mode=mode, k=k, time=r.time, iters=r.iterations,
                   oracle=ta.col_evals, dist=r.distance, rel=abs(r.distance-base)/base,
                   status=r.status, n_drops=ta.n_drops, kinds=ta.step_kinds)
        print(json.dumps(row), flush=True); rows.append(row)

for d in (100, 1000):
    V, W = generate_overlap(d, 5000, delta=4.0, rng=np.random.default_rng(7))
    run(f'soft-d{d}-C1', lambda m, k: SoftMarginTA(
        V, W, C=1.0, step_mode=m, block_size=k, zigzag_strategy='pairwise', seed=0), eps=1e-3)
V, W = generate_two_balls(1000, 5000, 1.2, rng=np.random.default_rng(53))
run('hard-d1000-eps1e-5', lambda m, k: EnhancedTriangleAlgorithm(
    V, W, step_mode=m, block_size=k, zigzag_strategy='pairwise', seed=0), eps=1e-5)
Path('results/k_ablation_synth.json').write_text(json.dumps(rows, indent=2))
