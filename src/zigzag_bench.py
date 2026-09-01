"""Benchmark of anti-zig-zag strategies in the Enhanced Triangle Algorithm.

Strategies compared on zig-zag-prone instances:
  none     - no remedy
  midpoint - midpoint of the two cycling pivots (the paper's suggestion)
  away     - away step (shrink the worst active vertex's weight)
  pairwise - MDM / pairwise step (transfer weight from worst active vertex
             to the best pivot)

Instances:
  A. classic pathological case: the optimum lies in the relative interior
     of an edge, so toward-steps oscillate between the edge's endpoints
     forever (sublinear); weight-transfer steps terminate.
  B/C. random Gaussian instances at tight tolerance (eps = 1e-6), where
     late-stage convergence is dominated by face oscillation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import generate_two_balls              # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402

STRATEGIES = [None, 'midpoint', 'away', 'pairwise']


def pathological_edge(n_extra=48, d=2, rng=None):
    """V: edge from (-1,0) to (1,0) (plus points pushed back below it);
    W: apex at (0,1) (plus points pushed back above).  The nearest point of
    conv(V) to conv(W) is (0,0), interior of the edge -> classic zig-zag."""
    rng = np.random.default_rng(rng)
    V = [[-1.0, 0.0], [1.0, 0.0]]
    W = [[0.0, 1.0]]
    for _ in range(n_extra):
        x = rng.uniform(-1, 1)
        V.append([x, -rng.uniform(0.2, 1.0)])
        W.append([rng.uniform(-1, 1), 1.0 + rng.uniform(0.2, 1.0)])
    V = np.array(V)
    W = np.array(W)
    if d > 2:  # embed in higher dimension (rotated), same geometry
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        V = np.hstack([V, np.zeros((len(V), d - 2))]) @ Q.T
        W = np.hstack([W, np.zeros((len(W), d - 2))]) @ Q.T
    return V, W


def run(name, V, W, eps, max_iter=300_000, truth=None):
    print(f"\n=== {name}  (eps = {eps:g}) ===")
    print(f"{'strategy':<10} {'status':<10} {'iterations':>10} "
          f"{'time s':>9} {'distance':>14} {'gap to best':>12}")
    results = {}
    for s in STRATEGIES:
        ta = EnhancedTriangleAlgorithm(V, W, zigzag_strategy=s,
                                       anti_zigzag=s is not None, seed=0)
        r = ta.solve_distance(eps=eps, max_iter=max_iter)
        results[s] = r
    best = truth if truth is not None else min(r.distance
                                               for r in results.values())
    for s, r in results.items():
        print(f"{str(s):<10} {r.status:<10} {r.iterations:>10} "
              f"{r.time:>9.3f} {r.distance:>14.10f} "
              f"{r.distance - best:>12.2e}")
    return results


if __name__ == '__main__':
    rng = np.random.default_rng(42)

    V, W = pathological_edge(rng=1)
    run("A. edge-interior optimum, 2-D (dist* = 1)", V, W,
        eps=1e-8, truth=1.0)

    V, W = pathological_edge(d=50, rng=2)
    run("A'. same geometry embedded in d = 50 (dist* = 1)", V, W,
        eps=1e-8, truth=1.0)

    V, W = generate_two_balls(30, 200, 1.3, rng=rng)
    run("B. Gaussian d = 30, n = 200, k = 1.3", V, W, eps=1e-6)

    V, W = generate_two_balls(100, 1000, 1.2, rng=rng)
    run("C. Gaussian d = 100, n = 1000, k = 1.2", V, W, eps=1e-6)
