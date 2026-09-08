"""The identification theorem's checkable ingredients on small instances:
omega_t <= h_t / tau at every iteration, and claims (i)/(ii) from the
iteration at which the proof's radius and weight conditions hold."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ident_check import face_constants, run_instance   # noqa: E402


def test_identification_claims_small_instances():
    checked = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        V = rng.normal(0, 1, (10, 4)); W = rng.normal(0, 1, (10, 4)); W[:, 0] += 2.5
        fc = face_constants(V, W)
        if not (fc['strict'] and fc['affind']) or fc['dstar'] < 1e-3:
            continue
        for k in (1, 4):
            o = run_instance(V, W, k)
            assert o['F2_ok'], (seed, k)
            assert o['t_id'] is not None and o['claim_i_ok'], (seed, k)
            assert o['t_w'] is not None and o['claim_ii_ok'], (seed, k)
            checked += 1
    assert checked >= 4
