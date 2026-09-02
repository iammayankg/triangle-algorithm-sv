"""Emit LaTeX for the App. C tables from results/schedule_check.json and
results/k_ablation_synth.json (paper: tab:sched, tab:kabl)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results'
NAMES = {'hard-d1000-eps1e-5': r'hard margin, $d{=}1000$, $\varepsilon{=}10^{-5}$',
         'soft-d100-C1': r'$L_2$ overlap, $d{=}100$, $C{=}1$',
         'soft-d1000-C1': r'$L_2$ overlap, $d{=}1000$, $C{=}1$'}
def st(r): return '' if r['status']=='converged' else r' (\emph{%s})' % r['status']

sched = json.load(open(R/'schedule_check.json'))
out = [r"\begin{table}[h]\centering\scriptsize",
       r"\caption{Two-sided schedules on the synthetic instances of this appendix ($n{=}5{,}000$ per class, $k{=}16$): the unconditional $V$-then-$W$ order used for the batteries versus the schedule of Lemma~\ref{lem:twosided} (larger-gap side first, second side skipped after a drop). Drop steps and skipped second-side steps are counted over the run.}\label{tab:sched}",
       r"\begin{tabular}{lllrrrr}\toprule",
       r"instance & schedule & time (s) & iterations & drops & skips & distance \\\midrule"]
for tag in NAMES:
    rows = [r for r in sched if r['tag']==tag]
    for r in rows:
        lab = 'Lemma~\\ref{lem:twosided}' if r['drop_skip'] else 'unconditional'
        out.append(f"{NAMES[tag]} & {lab} & {r['time']:.1f} & {r['iters']:,} & {r['n_drops']} & {r['n_drop_skips']} & {r['dist']:.6f}{st(r)} \\\\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"; out += [r"\end{tabular}\end{table}", ""]

kab = json.load(open(R/'k_ablation_synth.json'))
out += [r"\begin{table}[h]\centering\scriptsize",
        r"\caption{Block-size ablation on the same instances (current code): single MDM ($k{=}1$) versus blocks of $k$ transfers. Iterations fall with $k$; wall-clock does not fall in proportion, because each block iteration touches more columns.}\label{tab:kabl}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"instance & $k$ & time (s) & iterations & columns & drops \\\midrule"]
for tag in NAMES:
    for r in [r for r in kab if r['tag']==tag]:
        out.append(f"{NAMES[tag]} & {r['k']} & {r['time']:.1f} & {r['iters']:,}{st(r)} & {r['oracle']:,} & {r['n_drops']} \\\\")
    out.append(r"\midrule")
out[-1] = r"\bottomrule"; out += [r"\end{tabular}\end{table}"]
print("\n".join(out))
