"""Per-step diagnostics of the guarded block transfer (journal step 3):
does a block give more progress per unit of work than the single MDM
step at the same iterate, and how often?

For every block transfer the solver records (ETA.diag, see
triangle_algorithm._block_transfer_V/W): the block gain Delta_B and the
MDM gain Delta_1 at the same state, Q = sum_j Delta_j / Delta_1 (the
progress available beyond the leading pair; Q_unc uses the uncapped
per-pair gains gap_j^2 / (2 ||d_j||^2), Q_real the capacity-clipped
ones), kappa = D / S (how combining the directions changes their
effectiveness; orthogonal directions give kappa = 1), which candidate the
guard / case-(b) logic took, whether pair 1 and how many pairs were
capped, cache misses, support size, and a time split of the block step
(pair selection, column computation, block assembly, guard, cache and
score update) next to a dry-run timing of the single MDM update from
the same state.  The scan (_select) and total step time are recorded
per iteration by this driver.

Cost model (per iteration, from the same state):
  C_B = scan + t_pairs + t_cols + t_assemble + t_guard + t_update
  C_1 = scan + t_col1 + t_single   (column of the MDM receiver if it is
                                    a miss, plus the two-column update)
and the block improves progress per unit cost when
  max(Delta_B, Delta_1) / Delta_1  >  C_B / C_1.

Usage (Studio, data/ present; one process, ~2 min for both cells):
  OMP_NUM_THREADS=1 python3 src/block_diagnostics.py --data-dir data \
      --cells gisette-l2 ijcnn1-kl2 --ks 4 16 --out results/block_diag.json
Local smoke test:  python3 src/block_diagnostics.py --cells synthetic-l2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

EPS, CAP, C = 1e-3, 600.0, 1.0


def run_cell(cell, k, seed, data_dir):
    from full_battery import _load_cached
    from regime_battery import _subsample
    from soft_margin import SoftMarginTA
    from kernel_ta import KernelETA

    name, kind = cell.rsplit('-', 1)
    X, y, Xt, yt = _load_cached(name, data_dir, 100_000, seed)
    if kind == 'l2':
        V, W = X[y == 1], X[y == 0]
        ta = SoftMarginTA(V, W, C=C, step_mode='block', block_size=k,
                          zigzag_strategy='pairwise', seed=0)
    else:
        Xk, yk = _subsample(X, y, 10_000, seed)
        V, W = Xk[yk == 1], Xk[yk == 0]
        ta = KernelETA(V, W, kernel='rbf', gamma=1.0 / X.shape[1], reg_C=C,
                       step_mode='block', block_size=k,
                       zigzag_strategy='pairwise', seed=0)
    ta.diag = []
    scan_t, step_t = [], []
    sel, stp = ta._select, ta._step
    ta._diag_it = 0

    def select(*a, **kw):
        ta._diag_it += 1
        t = time.perf_counter()
        try:
            return sel(*a, **kw)
        finally:
            scan_t.append(time.perf_counter() - t)

    def step(*a, **kw):
        t = time.perf_counter()
        try:
            return stp(*a, **kw)
        finally:
            step_t.append(time.perf_counter() - t)
    ta._select, ta._step = select, step
    r = ta.solve_distance(eps=EPS, max_iter=2_000_000, time_cap=CAP)
    return dict(cell=cell, k=k, seed=seed, time=r.time, iters=r.iterations,
                status=r.status, dist=r.distance, oracle=ta.col_evals,
                n_drops=ta.n_drops, step_kinds=dict(ta.step_kinds),
                scan_t=scan_t, step_t=step_t, diag=ta.diag)


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float('nan')


def summarise(run):
    d = run['diag']
    scan = np.asarray(run['scan_t'])
    scan_mean = float(scan.mean()) if len(scan) else 0.0
    kinds = {}
    for r in d:
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    # candidate comparison at the same state, uncapped-pair-1 records only
    unc = [r for r in d if not r['cap1'] and np.isfinite(r['dB']) and r['d1'] > 0]
    ratio = np.array([max(r['dB'], r['d1']) / r['d1'] for r in unc])
    Q = np.array([r['Q_unc'] for r in unc])
    Qr = np.array([r['Q_real'] for r in unc])
    kap = np.array([r['D'] / r['S'] for r in unc if r['S'] > 0])
    cB = np.array([scan_mean + r['t_pairs'] + r['t_cols'] + r['t_assemble']
                   + r['t_guard'] + r['t_update'] for r in unc])
    c1 = np.array([scan_mean + r['t_col1'] + r['t_single'] for r in unc])
    cost = cB / np.maximum(c1, 1e-12)
    ppc = ratio / cost                       # progress per unit cost, block vs MDM
    # ... and including the fixed per-iteration overhead outside scan and
    # step (certificate, bounds, bookkeeping), which both candidates pay
    other = max(run['time'] - scan.sum() - sum(run['step_t']), 0.0) / max(run['iters'], 1)
    cost_full = (cB + other) / np.maximum(c1 + other, 1e-12)
    ppc_full = ratio / cost_full
    tot = {key: float(sum(r[key] for r in d)) for key in
           ('t_pairs', 't_cols', 't_assemble', 't_guard', 't_single', 't_update', 't_total')}
    # theorem-assumption statistics (records with every retained pair uncapped)
    allu = [r for r in d if r.get('all_uncapped') and r['kp'] > 1]
    mc = np.array([r['max_cos'] for r in allu]); bm = np.array([r['beta_min'] for r in allu])
    thm = dict(n_all_uncapped=len(allu),
               frac_all_uncapped=float(np.mean([bool(r.get('all_uncapped')) for r in d])) if d else float('nan'),
               max_cos=dict(p50=q(mc, 50), p90=q(mc, 90), max=float(mc.max()) if len(mc) else float('nan')),
               beta_min=dict(p10=q(bm, 10), p50=q(bm, 50), min=float(bm.min()) if len(bm) else float('nan')))
    out = dict(cell=run['cell'], k=run['k'], seed=run['seed'], time=run['time'], thm=thm,
               iters=run['iters'], status=run['status'], block_calls=len(d),
               kinds=kinds, cap1_frac=float(np.mean([r['cap1'] for r in d])) if d else 0.0,
               kp_mean=float(np.mean([r['kp'] for r in d])) if d else 0.0,
               capped_frac=float(np.mean([r['capped'] / max(r['kp'], 1) for r in d])) if d else 0.0,
               misses_per_call=float(np.mean([r['misses'] for r in d])) if d else 0.0,
               miss1_frac=float(np.mean([r['miss1'] > 0 for r in d])) if d else 0.0,
               support_start=d[0]['support'] if d else 0, support_end=d[-1]['support'] if d else 0,
               support_max=max(r['support'] for r in d) if d else 0,
               n_unc=len(unc),
               gain_ratio=dict(p25=q(ratio, 25), p50=q(ratio, 50), p75=q(ratio, 75),
                               gmean=float(np.exp(np.mean(np.log(ratio)))) if len(ratio) else float('nan'),
                               frac_gt1=float(np.mean(ratio > 1.0 + 1e-9)) if len(ratio) else float('nan')),
               Q_unc=dict(p25=q(Q, 25), p50=q(Q, 50), p75=q(Q, 75)),
               Q_real=dict(p25=q(Qr, 25), p50=q(Qr, 50), p75=q(Qr, 75)),
               kappa=dict(p25=q(kap, 25), p50=q(kap, 50), p75=q(kap, 75)),
               cost_ratio=dict(p25=q(cost, 25), p50=q(cost, 50), p75=q(cost, 75)),
               ppc=dict(p25=q(ppc, 25), p50=q(ppc, 50), p75=q(ppc, 75),
                        frac_gt1=float(np.mean(ppc > 1.0)) if len(ppc) else float('nan')),
               other_per_iter=float(other),
               cost_ratio_full=dict(p25=q(cost_full, 25), p50=q(cost_full, 50), p75=q(cost_full, 75)),
               ppc_full=dict(p25=q(ppc_full, 25), p50=q(ppc_full, 50), p75=q(ppc_full, 75),
                             frac_gt1=float(np.mean(ppc_full > 1.0)) if len(ppc_full) else float('nan')),
               scan_total=float(scan.sum()), step_total=float(sum(run['step_t'])),
               t_split=tot, oracle=run['oracle'], n_drops=run['n_drops'])
    return out


def print_summary(S):
    print(f"{S['cell']} k={S['k']} seed={S['seed']}: {S['time']:.1f} s, {S['iters']} it, "
          f"{S['block_calls']} block calls, kinds {S['kinds']}, status {S['status']}")
    print(f"  pair 1 capped on {100*S['cap1_frac']:.1f}% of calls; k' mean {S['kp_mean']:.1f}; "
          f"{100*S['capped_frac']:.0f}% of pairs capped; misses/call {S['misses_per_call']:.2f} "
          f"(receiver 1 a miss on {100*S['miss1_frac']:.0f}%); support {S['support_start']}->{S['support_end']} "
          f"(max {S['support_max']}); drops {S['n_drops']}")
    g, Q, Qr, ka, co, pp = (S['gain_ratio'], S['Q_unc'], S['Q_real'], S['kappa'], S['cost_ratio'], S['ppc'])
    print(f"  uncapped-pair-1 calls ({S['n_unc']}): gain ratio max(dB,d1)/d1 median {g['p50']:.2f} "
          f"[{g['p25']:.2f}, {g['p75']:.2f}], gmean {g['gmean']:.2f}, block wins {100*g['frac_gt1']:.0f}%")
    print(f"    Q_unc median {Q['p50']:.2f} [{Q['p25']:.2f}, {Q['p75']:.2f}]; Q_real {Qr['p50']:.2f}; "
          f"kappa median {ka['p50']:.2f} [{ka['p25']:.2f}, {ka['p75']:.2f}]")
    print(f"    cost ratio C_B/C_1 median {co['p50']:.2f} [{co['p25']:.2f}, {co['p75']:.2f}]; "
          f"progress per unit cost vs MDM median {pp['p50']:.2f} [{pp['p25']:.2f}, {pp['p75']:.2f}], "
          f">1 on {100*pp['frac_gt1']:.0f}%")
    th = S.get('thm')
    if th and th['n_all_uncapped']:
        print(f"    theorem assumptions on the {100*th['frac_all_uncapped']:.0f}% of calls with all pairs uncapped: "
              f"max |cos| median {th['max_cos']['p50']:.2f}, p90 {th['max_cos']['p90']:.2f}, max {th['max_cos']['max']:.2f}; "
              f"min relative gain (beta) p10 {th['beta_min']['p10']:.2f}, median {th['beta_min']['p50']:.2f}, min {th['beta_min']['min']:.3f}")
    cf, pf = S['cost_ratio_full'], S['ppc_full']
    print(f"    with the fixed per-iteration overhead ({1e3*S['other_per_iter']:.2f} ms/it) in both costs: "
          f"C_B/C_1 median {cf['p50']:.2f}, progress per unit cost median {pf['p50']:.2f} "
          f"[{pf['p25']:.2f}, {pf['p75']:.2f}], >1 on {100*pf['frac_gt1']:.0f}%")
    t = S['t_split']
    print(f"  time split (s): scan {S['scan_total']:.2f} | pairs {t['t_pairs']:.2f} cols {t['t_cols']:.2f} "
          f"assemble {t['t_assemble']:.2f} guard {t['t_guard']:.2f} update {t['t_update']:.2f} "
          f"| dry-run single updates {t['t_single']:.2f} | step total {S['step_total']:.2f} of {S['time']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--cells', nargs='*', default=['gisette-l2', 'ijcnn1-kl2'])
    ap.add_argument('--ks', nargs='*', type=int, default=[16])
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--out', default='results/block_diag.json')
    ap.add_argument('--keep-records', action='store_true',
                    help='store every per-step record (large); default keeps summaries only')
    a = ap.parse_args()
    out = Path(a.out)
    runs = json.load(open(out)) if out.exists() else []
    done = {(r['cell'], r['k'], r['seed']) for r in runs}
    for cell in a.cells:
        for k in a.ks:
            for seed in range(a.seeds):
                if (cell, k, seed) in done:
                    continue
                run = run_cell(cell, k, seed, a.data_dir)
                S = summarise(run)
                if a.keep_records:
                    S['records'] = run['diag']
                print_summary(S)
                runs.append(S)
                out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(runs, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
