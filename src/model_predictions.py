"""Timing-model predictions of Section 9 (journal draft) from the
instrumented runs: T(k) = T(1) (1 + eta_hat (k-1)) / k iterations predicted
from the single-pair run, time = c_col * N_cols + T(k) * (a + b k), with a, b,
c_col and eta_hat fitted per cell exactly as in src/block_diag_fig.py.

Usage: python3 src/model_predictions.py [--inp results/block_diag3.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import block_diag_fig as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='results/block_diag3.json')
    rows = json.load(open(ap.parse_args().inp))
    for cell in sorted({r['cell'] for r in rows}):
        rs = sorted([r for r in rows if r['cell'] == cell], key=lambda r: r['k'])
        f = B.fit(rs)                       # a, b, c_col in milliseconds
        a, b, c, eta = f['a'] / 1e3, f['b'] / 1e3, f['ccol'] / 1e3, f['eta']
        it1 = next(r['iters'] for r in rs if r['k'] == 1)
        print(f'{cell}: eta_hat={eta:.3f}')
        for r in rs:
            k = r['k']
            T = it1 * (1 + eta * (k - 1)) / k
            pred = c * r['oracle'] + T * (a + b * k)
            print(f'  k={k:>2}: predicted {pred:6.1f} s, measured {r["time"]:6.1f} s '
                  f'({100 * (pred / r["time"] - 1):+4.0f}%); iterations {T:7.0f} vs {r["iters"]}')


if __name__ == '__main__':
    main()
