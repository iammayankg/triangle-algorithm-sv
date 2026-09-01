"""Shrinking on/off performance comparison.

A: Table-3 dimension sweep (k = 1.2, n = 5000/set), eps in {1e-3, 1e-5}
B: separation sweep at d = 1000 (k = 1.2 .. 1.9), eps = 1e-5
C: separable real datasets (wdbc, digits-3v8, mnist5k-3v8), eps = 1e-5

Solver: EnhancedTriangleAlgorithm(zigzag_strategy='pairwise'),
shrink_every = 25.  Reported: time, iterations, surviving points, distance
agreement.

Usage: python3 src/shrink_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import generate_two_balls                      # noqa: E402
from triangle_algorithm import EnhancedTriangleAlgorithm  # noqa: E402


def pair(V, W, eps, max_iter=200_000):
    r0 = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise',
                                   seed=0).solve_distance(eps=eps,
                                                          max_iter=max_iter)
    ta = EnhancedTriangleAlgorithm(V, W, zigzag_strategy='pairwise',
                                   seed=0, shrink=True, shrink_every=25)
    r1 = ta.solve_distance(eps=eps, max_iter=max_iter)
    rel = abs(r0.distance - r1.distance) / max(r0.distance, 1e-12)
    return dict(t_plain=r0.time, t_shrunk=r1.time,
                it_plain=r0.iterations, it_shrunk=r1.iterations,
                alive=ta.n + ta.m, total=len(V) + len(W),
                speedup=r0.time / max(r1.time, 1e-9), rel=rel,
                status=(r0.status, r1.status))


def main():
    out = {'dims': [], 'ks': [], 'real': []}

    for eps in (1e-3, 1e-5):
        for d in (100, 300, 1000, 2000, 5000, 10000):
            recs = []
            for t in range(2):
                V, W = generate_two_balls(
                    d, 5000, 1.2, rng=np.random.default_rng(2000 + 31 * t + d))
                recs.append(pair(V, W, eps))
                del V, W
            row = {'eps': eps, 'dim': d}
            for k in ('t_plain', 't_shrunk', 'it_plain', 'alive', 'speedup',
                      'rel'):
                row[k] = float(np.mean([r[k] for r in recs]))
            out['dims'].append(row)
            print(f"[dims] eps={eps:g} d={d} plain={row['t_plain']:.2f}s "
                  f"shrunk={row['t_shrunk']:.2f}s x{row['speedup']:.2f} "
                  f"alive={row['alive']:.0f}/10000 rel={row['rel']:.1e}",
                  flush=True)
            Path('results/shrink_benchmark.json').write_text(
                json.dumps(out, indent=2))

    for k in (1.2, 1.4, 1.6, 1.9):
        V, W = generate_two_balls(1000, 5000, k,
                                  rng=np.random.default_rng(int(k * 100)))
        r = pair(V, W, 1e-5)
        r['k'] = k
        out['ks'].append(r)
        print(f"[ksep] k={k} plain={r['t_plain']:.2f}s "
              f"shrunk={r['t_shrunk']:.2f}s x{r['speedup']:.2f} "
              f"alive={r['alive']}/10000 rel={r['rel']:.1e}", flush=True)
        Path('results/shrink_benchmark.json').write_text(
            json.dumps(out, indent=2))

    # separable real datasets
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.preprocessing import StandardScaler
    from mlxtend.data import mnist_data

    def real_sets():
        d = load_breast_cancer()
        yield 'wdbc', d.data, (d.target == 1)
        dg = load_digits()
        m = np.isin(dg.target, (3, 8))
        yield 'digits-3v8', dg.data[m], (dg.target[m] == 8)
        X, y = mnist_data()
        m = np.isin(y, (3, 8))
        yield 'mnist5k-3v8', X[m], (y[m] == 8)

    for name, X, y in real_sets():
        X = StandardScaler().fit_transform(X)
        V, W = X[y], X[~y]
        r = pair(V, W, 1e-5)
        r['name'] = name
        out['real'].append(r)
        print(f"[real] {name}: plain={r['t_plain']:.3f}s "
              f"shrunk={r['t_shrunk']:.3f}s x{r['speedup']:.2f} "
              f"alive={r['alive']}/{r['total']} rel={r['rel']:.1e} "
              f"status={r['status']}", flush=True)
    Path('results/shrink_benchmark.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
