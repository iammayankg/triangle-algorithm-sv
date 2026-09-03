"""Render Figure 2 (safe-screening speed-up vs dimension) from results/shrink_benchmark.json."""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = {'1e-3': '#2a78d6', '1e-5': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e5e4e0'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='results/shrink_benchmark.json')
    ap.add_argument('--out', nargs='+', default=['paper/figs/fig_shrink.pdf'])
    a = ap.parse_args()
    rows = json.loads(Path(a.json).read_text())['dims']
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for eps, key, lab in ((1e-3, '1e-3', r'$\varepsilon=10^{-3}$'), (1e-5, '1e-5', r'$\varepsilon=10^{-5}$')):
        rr = sorted([r for r in rows if r['eps'] == eps], key=lambda r: r['dim'])
        xs = [r['dim'] for r in rr]; ys = [r['speedup'] for r in rr]
        ax.plot(xs, ys, marker='o', color=COL[key], lw=1.6, label=lab)
        for r in rr:
            ax.annotate(f"{int(r['alive']):,}", (r['dim'], r['speedup']),
                        textcoords='offset points', xytext=(0, 7 if eps == 1e-5 else -12),
                        ha='center', fontsize=7, color=INK2)
    ax.axhline(1.0, color=INK2, lw=0.8, ls='--')
    ax.set_xscale('log'); ax.set_xlabel('dimension $d$', color=INK2)
    ax.set_ylabel('wall-clock speed-up (plain / screened)', color=INK2)
    ax.grid(True, color=GRID, lw=0.7)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='upper right')
    fig.tight_layout()
    for out in a.out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor='white'); print('wrote', out)

if __name__ == '__main__':
    main()
