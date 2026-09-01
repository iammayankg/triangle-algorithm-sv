"""Synthetic data generation following the paper's protocol.

"Two sets of points, V and V', from unit balls with random means, and
translated one ball along a random direction."  The translation distance
is k times the maximum of the two empirical diameters:

  * k = 0.9  -> overlapping hulls (intersection experiments, Table 2)
  * k > 1    -> separated hulls   (distance experiments, Tables 3 and 4)

The exact sampling law of the original MATLAB study is not specified in
the paper; here each cloud is an isotropic Gaussian "ball" (typical
radius ~ sqrt(d)), which reproduces the scale behaviour of the reported
distances (hull separation growing with dimension).  A uniform-in-ball
mode is also provided.
"""

from __future__ import annotations

import numpy as np


def _unit_vector(d, rng):
    u = rng.standard_normal(d)
    return u / np.linalg.norm(u)


def _empirical_diameter(X, rng, n_probe=64):
    """Cheap diameter estimate: double sweep from random probes."""
    n = X.shape[0]
    best = 0.0
    idx = rng.choice(n, size=min(n_probe, n), replace=False)
    for i in idx[:8]:
        d2 = np.einsum('ij,ij->i', X - X[i], X - X[i])
        j = int(np.argmax(d2))
        d2b = np.einsum('ij,ij->i', X - X[j], X - X[j])
        best = max(best, float(np.sqrt(d2b.max())))
    return best


def generate_two_balls(d, n, k, rng=None, mode='gaussian', dtype=np.float64):
    """Generate two n-point clouds in R^d; the second is translated so the
    translation distance equals k * max(empirical diameters).

    Returns (V, W).
    """
    rng = np.random.default_rng(rng)
    if mode == 'gaussian':
        V = rng.standard_normal((n, d))
        W = rng.standard_normal((n, d))
    elif mode == 'ball':
        def ball(n, d):
            z = rng.standard_normal((n, d))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            r = rng.random(n) ** (1.0 / d)
            return z * r[:, None]
        V = ball(n, d)
        W = ball(n, d)
    else:
        raise ValueError(mode)
    # random means
    V += rng.standard_normal(d) * 0.0  # centre first cloud at origin
    diam = max(_empirical_diameter(V, rng), _empirical_diameter(W, rng))
    u = _unit_vector(d, rng)
    W += (k * diam) * u
    return V.astype(dtype, copy=False), W.astype(dtype, copy=False)
