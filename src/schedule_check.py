"""Two-sided schedule check (paper Lemma 6 / App. C): the analysed
schedule (larger-gap side first, second side skipped after a drop;
drop_skip=True) versus the unconditional V-then-W order used for the
batteries (drop_skip=False), on the synthetic instances of App. C."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import generate_two_balls, generate_overlap
from soft_margin import SoftMarginTA
from triangle_algorithm import EnhancedTriangleAlgorithm

rows = []
def run(tag, mk, eps, max_iter=200_000):
    for skip in (False, True):
        ta = mk(skip)
        r = ta.solve_distance(eps=eps, max_iter=max_iter)
        row = dict(tag=tag, drop_skip=skip, time=r.time, iters=r.iterations,
                   dist=r.distance, status=r.status, n_drops=ta.n_drops,
                   n_drop_skips=ta.n_drop_skips, kinds=ta.step_kinds)
        print(json.dumps(row), flush=True)
        rows.append(row)

V, W = generate_two_balls(1000, 5000, 1.2, rng=np.random.default_rng(53))
run('hard-d1000-eps1e-5', lambda sk: EnhancedTriangleAlgorithm(
    V, W, step_mode='block', block_size=16, zigzag_strategy='pairwise',
    seed=0, drop_skip=sk), eps=1e-5)
for d in (100, 1000):
    V, W = generate_overlap(d, 5000, delta=4.0, rng=np.random.default_rng(7))
    run(f'soft-d{d}-C1', lambda sk: SoftMarginTA(
        V, W, C=1.0, step_mode='block', block_size=16,
        zigzag_strategy='pairwise', seed=0, drop_skip=sk), eps=1e-3)
Path('results/schedule_check.json').write_text(json.dumps(rows, indent=2))
