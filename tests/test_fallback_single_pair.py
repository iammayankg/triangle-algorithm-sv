"""Regression test: the case-(b) away fallback of Algorithm 1 must apply
even when only one transfer pair survives filtering.

Instance from the review: V = {0, 1, 2} on the line, W = {-1}, weights
0.9 on v_1 and 0.1 on v_2, so p = 1.1 and q = -1.  Filtering leaves the
single transfer 2 -> 0 whose unconstrained step 1.05 exceeds the donor
capacity 0.1, and the away step on v_2 is boundary-clipped.  Algorithm 1
therefore takes the away *drop* step (p -> 1.0, support {v_1}); the old
code returned an ordinary capacity-clipped MDM transfer (a swap step,
support {v_0, v_1}) before reaching the fallback.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


def _instance(k):
    V = np.array([[0.0], [1.0], [2.0]])
    W = np.array([[-1.0]])
    ta = EnhancedTriangleAlgorithm(V, W, step_mode='block', block_size=k,
                                   zigzag_strategy='pairwise', seed=0)
    ta._init_state(i0=1, j0=0)
    ta._apply_p(2, 0.1)            # p = 0.9 v_1 + 0.1 v_2 = 1.1
    ta.wV = {1: 0.9, 2: 0.1}
    return ta


def test_single_pair_takes_drop_not_swap():
    for k in (1, 16):
        ta = _instance(k)
        assert abs(float(ta.p[0]) - 1.1) < 1e-12
        moved = ta._block_transfer_V(k)
        assert moved
        assert ta._last_drop, f"k={k}: expected an away drop step"
        assert ta.n_drops == 1
        assert 2 not in ta.wV, "donor v_2 should have been dropped"
        assert 0 not in ta.wV, "a capacity-clipped swap onto v_0 was taken"
        assert abs(float(ta.p[0]) - 1.0) < 1e-9
        assert abs(sum(ta.wV.values()) - 1.0) < 1e-12


if __name__ == '__main__':
    test_single_pair_takes_drop_not_swap()
    print('single-pair fallback check passed')
