"""block_size='auto' converges to the same distance as a fixed block size,
records its k history, and respects k_max."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data import generate_overlap          # noqa: E402
from soft_margin import SoftMarginTA       # noqa: E402


def test_auto_block_size_matches_fixed():
    V, W = generate_overlap(50, 600, delta=4.0, rng=np.random.default_rng(3))
    fixed = SoftMarginTA(V, W, C=1.0, step_mode='block', block_size=16,
                         zigzag_strategy='pairwise', seed=0)
    rf = fixed.solve_distance(eps=1e-4, max_iter=200_000)
    auto = SoftMarginTA(V, W, C=1.0, step_mode='block', block_size='auto',
                        zigzag_strategy='pairwise', seed=0, k_max=32)
    ra = auto.solve_distance(eps=1e-4, max_iter=200_000)
    assert rf.status == 'converged' and ra.status == 'converged'
    assert abs(ra.distance - rf.distance) <= 1e-4 * rf.distance
    assert auto.k_history, 'the rule never fired'
    assert all(1 <= k <= 32 for _, k, _, _ in auto.k_history)
    etas = [e for _, _, e, _ in auto.k_history if np.isfinite(e)]
    assert etas and all(-0.5 <= e <= 1.5 for e in etas)
