"""Block-transfer performance benchmark on the heavy cases.

A: L2 soft margin, overlapping Gaussians (delta=4), n=5000/set, C=1,
   d in {100, 1000} - the dense-support regime where MDM iteration
   counts dominate.
B: mnist5k odd-vs-even, L2 soft margin C=1 (the hardest real case).
C: hard margin d=1000 n=5000/set at eps=1e-5, with and without shrink.

Modes: mdm, block(8), block(32); distances cross-checked.
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import generate_two_balls, generate_overlap
from soft_margin import SoftMarginTA
from triangle_algorithm import EnhancedTriangleAlgorithm

def run(mk, modes, tag, eps=1e-3, max_iter=200_000):
    base = None
    out = []
    for mode, bs, shrink in modes:
        ta = mk(mode, bs, shrink)
        r = ta.solve_distance(eps=eps, max_iter=max_iter)
        rel = 0.0 if base is None else abs(r.distance - base)/base
        if base is None: base = r.distance
        lab = mode + (f"({bs})" if mode=='block' else '') + ('+shrink' if shrink else '')
        print(f"[{tag}] {lab:16s} t={r.time:7.2f}s it={r.iterations:6d} "
              f"d={r.distance:.6f} rel={rel:.1e} {r.status}", flush=True)
        out.append(dict(tag=tag, mode=lab, time=r.time, iters=r.iterations,
                        dist=r.distance, status=r.status))
    return out

results = []
MODES = [('mdm', 0, False), ('block', 8, False), ('block', 32, False)]

for d in (100, 1000):
    V, W = generate_overlap(d, 5000, delta=4.0, rng=np.random.default_rng(7))
    mk = lambda m, bs, sh: SoftMarginTA(V, W, C=1.0, step_mode=m, block_size=bs or 16,
                                        zigzag_strategy='pairwise', seed=0, shrink=sh)
    results += run(mk, MODES, f"soft-d{d}")
    del V, W

from mlxtend.data import mnist_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = mnist_data(); y = (y % 2).astype(int)
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
Xtr = StandardScaler().fit_transform(Xtr)
V, W = Xtr[ytr==1], Xtr[ytr==0]
mk = lambda m, bs, sh: SoftMarginTA(V, W, C=1.0, step_mode=m, block_size=bs or 16,
                                    zigzag_strategy='pairwise', seed=0, shrink=sh)
results += run(mk, MODES, "mnist-oe")
del V, W

V, W = generate_two_balls(1000, 5000, 1.2, rng=np.random.default_rng(53))
mk = lambda m, bs, sh: EnhancedTriangleAlgorithm(V, W, step_mode=m, block_size=bs or 16,
                                                 zigzag_strategy='pairwise', seed=0,
                                                 shrink=sh, shrink_every=25)
results += run(mk, MODES + [('block', 32, True)], "hard-d1000", eps=1e-5)

Path('results/block_benchmark.json').write_text(json.dumps(results, indent=2))
