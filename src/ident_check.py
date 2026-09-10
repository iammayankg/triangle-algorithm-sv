"""Numerical check of the identification theorem (journal Theorem 14) on
small instances where every constant can be computed.

For random small hard-margin instances the SLSQP reference gives z*, the
optimal faces F_V, F_W (zero score margin), the margins tau_i, the
smallest optimal weight alpha_min, the face conditioning d_min and sigma
(Assumption A2), and the thresholds H_id, H_w of the theorem.  The
guarded block solver is then run to a tight tolerance with the supports
snapshotted at every iteration, and the theorem's claims are checked
iteration by iteration:

  (F2) the non-face weight omega_t is at most h_t / tau at every iteration;
  (i)  from the first iteration t_id at which the proof's radius and
       weight conditions hold (r_t <= tau/(6L), omega_t <= tau/(12 L^2),
       omega_t <= 1/12, 2h + R sqrt(2h) <= tau/6): whenever a support
       contains a non-face point the iteration is a drop, and no non-face
       point enters either support;
  (ii) from t_1 = max(t_w, t_id + N), t_w the first iteration at which in
       addition r_t <= sigma alpha_min / 2 and r_t <= alpha_min d_min^2 /
       (2L): no drop and no index leaves.

The theorem states these conditions through the sufficient h-thresholds
H_id, H_w (via r^2 <= 4R^2 sqrt(2h)/delta*), which lie below the floating
point floor on small instances; the script reports the h at which the
r-conditions first hold against H_id and H_w, i.e. how loose that step
is.  Instances violating strict complementarity or affine independence
(within tolerance) are skipped and counted.

Usage:  python3 src/ident_check.py --seeds 40 --ks 1 4 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from triangle_algorithm import EnhancedTriangleAlgorithm, _W_MIN   # noqa: E402


def qp_weights(V, W):
    """Reference solution of min ||V^T a - W^T b||^2 over the two simplices
    (SLSQP, ftol 1e-14), returned as (a, b)."""
    n, m = V.shape[0], W.shape[0]

    def obj(z):
        diff = V.T @ z[:n] - W.T @ z[n:]
        return float(diff @ diff)

    def grad(z):
        diff = V.T @ z[:n] - W.T @ z[n:]
        return np.concatenate([2.0 * (V @ diff), -2.0 * (W @ diff)])
    cons = [{'type': 'eq', 'fun': lambda z: z[:n].sum() - 1.0,
             'jac': lambda z: np.concatenate([np.ones(n), np.zeros(m)])},
            {'type': 'eq', 'fun': lambda z: z[n:].sum() - 1.0,
             'jac': lambda z: np.concatenate([np.zeros(n), np.ones(m)])}]
    z0 = np.concatenate([np.full(n, 1.0 / n), np.full(m, 1.0 / m)])
    res = minimize(obj, z0, jac=grad, bounds=[(0.0, 1.0)] * (n + m), constraints=cons,
                   method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-15})
    z = np.clip(res.x, 0.0, None)
    return z[:n] / z[:n].sum(), z[n:] / z[n:].sum()


def face_constants(V, W, tol=1e-7):
    a, b = qp_weights(V, W)
    zs = V.T @ a - W.T @ b
    dstar = float(np.linalg.norm(zs))
    sV = V @ zs; sW = W @ zs
    tauV = sV - sV.min(); tauW = sW.max() - sW
    FV = np.where(tauV <= tol)[0]; FW = np.where(tauW <= tol)[0]
    suppV = np.where(a > tol)[0]; suppW = np.where(b > tol)[0]
    strict = set(FV) == set(suppV) and set(FW) == set(suppW)
    taus = np.concatenate([tauV[tauV > tol], tauW[tauW > tol]])
    tau = float(taus.min()) if taus.size else np.inf
    L = max(np.linalg.norm(V[:, None] - V[None], axis=2).max(),
            np.linalg.norm(W[:, None] - W[None], axis=2).max())
    R = np.linalg.norm(V[:, None] - W[None], axis=2).max()
    amin = float(min(a[FV].min() if FV.size else 1.0, b[FW].min() if FW.size else 1.0))

    def dmin_of(P):
        if len(P) < 2:
            return np.inf
        out = np.inf
        for i in range(len(P)):
            others = np.delete(P, i, axis=0)
            base = others[0]
            if len(others) == 1:
                dist = np.linalg.norm(P[i] - base)
            else:
                Bm = (others[1:] - base).T           # d x (r-1)
                Qm, _ = np.linalg.qr(Bm)
                v = P[i] - base
                dist = np.linalg.norm(v - Qm @ (Qm.T @ v))
            out = min(out, dist)
        return float(out)
    dmin = min(dmin_of(V[FV]), dmin_of(W[FW]))
    # affine independence of each face
    def affinely_independent(P):
        if len(P) < 2:
            return True
        return np.linalg.matrix_rank(P[1:] - P[0], tol=1e-9) == len(P) - 1
    affind = affinely_independent(V[FV]) and affinely_independent(W[FW])
    # sigma: smallest singular value of [V_F^T, -W_F^T] on the zero-sum subspace
    nf = len(FV) + len(FW)
    if len(FV) + len(FW) <= 2 and len(FV) <= 1 and len(FW) <= 1:
        sigma = np.inf
    else:
        A = np.hstack([V[FV].T, -W[FW].T])          # d x nf
        C = np.zeros((2, nf)); C[0, :len(FV)] = 1.0; C[1, len(FV):] = 1.0
        # basis of the null space of C
        _, sv, Vt = np.linalg.svd(C)
        rank = int((sv > 1e-12).sum())
        Bm = Vt[rank:].T                              # nf x (nf - rank)
        if Bm.shape[1] == 0:
            sigma = np.inf
        else:
            s2 = np.linalg.svd(A @ Bm, compute_uv=False).min()
            sigma = float(s2 / np.sqrt(nf))
    h3 = ((-R + np.sqrt(R * R + 4 * 2 * tau / 6)) / (2 * np.sqrt(2))) ** 2 if np.isfinite(tau) else np.inf
    # h3: largest h with 2h + R sqrt(2h) <= tau/6  (x = sqrt(2h): x^2 + R x - tau/6 <= 0)
    x = (-R + np.sqrt(R * R + 4 * tau / 6)) / 2
    h3 = x * x / 2
    # thresholds of the journal version (Section 'Drops are an identification
    # cost'): the distance to the optimum is bounded by sqrt(2h) directly
    # (1-strong convexity plus optimality of z*), so no screening radius,
    # no fourth powers and no R^4/delta*^2 enter.  The earlier thresholds
    # through r^2 <= 4R^2 sqrt(2h)/delta* are kept for comparison as Hid_r/Hw_r.
    Hid = min(tau ** 2 / (72 * L ** 2), tau / 12, h3)
    Hw = min(Hid,
             (sigma * amin) ** 2 / 8 if np.isfinite(sigma) else np.inf,
             (amin * dmin) ** 2 / 8 if np.isfinite(dmin) else np.inf)
    Hid_r = min(tau ** 4 * dstar ** 2 / (41472 * R ** 4 * L ** 4), tau ** 2 / (12 * L ** 2), tau / 12, h3)
    Hw_r = min(Hid_r,
               (sigma * amin / 2) ** 4 * dstar ** 2 / (32 * R ** 4) if np.isfinite(sigma) else np.inf,
               (amin * dmin ** 2 / (2 * L)) ** 4 * dstar ** 2 / (32 * R ** 4) if np.isfinite(dmin) else np.inf)
    return dict(a=a, b=b, zs=zs, dstar=dstar, FV=set(int(i) for i in FV), FW=set(int(j) for j in FW),
                tau=tau, L=float(L), R=float(R), amin=amin, dmin=dmin, sigma=sigma,
                strict=strict, affind=affind, Hid=float(Hid), Hw=float(Hw),
                Hid_r=float(Hid_r), Hw_r=float(Hw_r))


def run_instance(V, W, k, eps=1e-10, max_iter=100_000):
    fc = face_constants(V, W)
    ta = EnhancedTriangleAlgorithm(V, W, step_mode='block', block_size=k,
                                   zigzag_strategy='pairwise', seed=0)
    log = []
    step = ta._step

    def supp():
        return (set(i for i, w in ta.wV.items() if i >= 0 and w > _W_MIN),
                set(j for j, w in ta.wW.items() if j >= 0 and w > _W_MIN))

    tau, L, R, amin, dmin, sigma = (fc[key] for key in ('tau', 'L', 'R', 'amin', 'dmin', 'sigma'))

    def wrapped(*args, **kw):
        sV0, sW0 = supp(); nd0 = ta.n_drops; d2 = ta.dist2()
        ub = np.sqrt(max(d2, 0.0)); lb = ta._lower_bound(ub)
        r = float(np.sqrt(max(ub * ub - max(lb, 0.0) ** 2, 0.0)))
        h = max(0.5 * (d2 - fc['dstar'] ** 2), 0.0)
        omega = sum(w for i, w in ta.wV.items() if i >= 0 and i not in fc['FV']) \
            + sum(w for j, w in ta.wW.items() if j >= 0 and j not in fc['FW'])
        # the proof's conditions, on the distance bound s = sqrt(2h) that the
        # journal proof uses (the screening radius r >= ||z - z*|| is logged
        # for comparison only)
        sd = np.sqrt(2 * h)
        c_id = (sd <= tau / (6 * L) and omega <= tau / (72 * L * L) and omega <= 1 / 12
                and 2 * h + R * np.sqrt(2 * h) <= tau / 6)
        c_w = c_id and (not np.isfinite(sigma) or sd < sigma * amin / 2) \
            and (not np.isfinite(dmin) or sd < amin * dmin / 2)
        ok = step(*args, **kw)
        sV1, sW1 = supp()
        log.append(dict(h=h, r=r, omega=omega, c_id=c_id, c_w=c_w,
                        nonface=len(sV0 - fc['FV']) + len(sW0 - fc['FW']),
                        drop=ta.n_drops > nd0,
                        entered_nonface=len((sV1 - sV0) - fc['FV']) + len((sW1 - sW0) - fc['FW']),
                        left=len(sV0 - sV1) + len(sW0 - sW1)))
        return ok
    ta._step = wrapped
    r = ta.solve_distance(eps=eps, max_iter=max_iter)
    h = np.array([x['h'] for x in log])
    out = dict(k=k, status=r.status, iters=len(log), h_min=float(h.min()) if len(h) else np.nan,
               strict=fc['strict'], affind=fc['affind'], Hid=fc['Hid'], Hw=fc['Hw'],
               Hid_r=fc['Hid_r'], Hw_r=fc['Hw_r'], dstar_intersect=fc['dstar'] < 1e-3,
               tau=fc['tau'], dstar=fc['dstar'], amin=fc['amin'], dmin=fc['dmin'], sigma=fc['sigma'],
               faces=(len(fc['FV']), len(fc['FW'])), n_drops=ta.n_drops)
    out['F2_ok'] = all(x['omega'] <= x['h'] / fc['tau'] * (1 + 1e-9) + 1e-12 for x in log)
    tid = next((t for t, x in enumerate(log) if x['c_id']), None)
    tw = next((t for t, x in enumerate(log) if x['c_w']), None)
    out['t_id'] = tid; out['t_w'] = tw
    out['h_at_tid'] = log[tid]['h'] if tid is not None else None
    out['h_at_tw'] = log[tw]['h'] if tw is not None else None
    lastnf = [t for t, x in enumerate(log) if x['nonface'] > 0]
    out['h_last_nonface'] = log[lastnf[-1]]['h'] if lastnf else None
    lastdrop = [t for t, x in enumerate(log) if x['drop']]
    out['h_last_drop'] = log[lastdrop[-1]]['h'] if lastdrop else None
    # earliest iteration from which claim (i) / claim (ii) hold to the end
    ok_i = [(x['drop'] or x['nonface'] == 0) and x['entered_nonface'] == 0 for x in log]
    ok_ii = [not x['drop'] and x['left'] == 0 for x in log]
    def earliest(flags):
        t = len(flags)
        while t > 0 and flags[t - 1]:
            t -= 1
        return t
    out['t_i_star'] = earliest(ok_i); out['t_ii_star'] = earliest(ok_ii)
    out['h_i_star'] = log[out['t_i_star']]['h'] if out['t_i_star'] < len(log) else 0.0
    out['h_ii_star'] = log[out['t_ii_star']]['h'] if out['t_ii_star'] < len(log) else 0.0
    out['h0'] = log[0]['h']
    if tid is not None:
        N = log[tid]['nonface']
        out['N'] = N
        seg = log[tid:]
        out['claim_i_ok'] = all((x['drop'] or x['nonface'] == 0) and x['entered_nonface'] == 0 for x in seg)
        out['nonface_gone_by'] = next((tid + t for t, x in enumerate(seg) if x['nonface'] == 0), None)
        if tw is not None:
            t1 = max(tw, tid + N)
            seg2 = log[t1:]
            out['claim_ii_ok'] = all(not x['drop'] and x['left'] == 0 for x in seg2)
            out['drops_after_t1'] = sum(x['drop'] for x in seg2)
            out['t1'] = t1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--ks', nargs='*', type=int, default=[1, 4, 16])
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--sep', type=float, default=2.5)
    a = ap.parse_args()
    tot = dict(instances=0, degenerate=0, runs=0, reached_id=0, reached_w=0, i_ok=0, i_fail=0, ii_ok=0, ii_fail=0)
    recs = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        V = rng.normal(0, 1, (a.n, a.d)); W = rng.normal(0, 1, (a.n, a.d)); W[:, 0] += a.sep
        fc = face_constants(V, W)
        tot['instances'] += 1
        if not (fc['strict'] and fc['affind']) or fc['dstar'] < 1e-3:
            tot['degenerate'] += 1
            why = 'intersect' if fc['dstar'] < 1e-3 else ('strict' if not fc['strict'] else 'affind')
            tot['skip_' + why] = tot.get('skip_' + why, 0) + 1
            continue
        for k in a.ks:
            o = run_instance(V, W, k)
            tot['runs'] += 1
            recs.append(o)
            tot['F2_ok' if o['F2_ok'] else 'F2_fail'] = tot.get('F2_ok' if o['F2_ok'] else 'F2_fail', 0) + 1
            fmt = lambda v: 'n/a' if v is None else f"{v:.1e}"
            msg = (f"seed {seed:2d} k={k:2d} faces {o['faces']} tau {o['tau']:.3f} amin {o['amin']:.3f} "
                   f"dmin {o['dmin']:.2f} sigma {o['sigma']:.3f} | {o['iters']} it {o['status']} drops {o['n_drops']} "
                   f"F2 {'ok' if o['F2_ok'] else 'FAIL'} | h at last non-face {fmt(o['h_last_nonface'])}, "
                   f"last drop {fmt(o['h_last_drop'])}")
            if o['t_id'] is not None:
                tot['reached_id'] += 1
                tot['i_ok' if o['claim_i_ok'] else 'i_fail'] += 1
                msg += (f" | t_id {o['t_id']} (h {o['h_at_tid']:.1e} vs H_id {o['Hid']:.0e}) N {o['N']} "
                        f"(i) {'ok' if o['claim_i_ok'] else 'FAIL'}")
                if o['t_w'] is not None:
                    tot['reached_w'] += 1
                    tot['ii_ok' if o['claim_ii_ok'] else 'ii_fail'] += 1
                    msg += (f" | t_w {o['t_w']} (h {o['h_at_tw']:.1e} vs H_w {o['Hw']:.0e}) t1 {o['t1']} "
                            f"(ii) {'ok' if o['claim_ii_ok'] else 'FAIL'} drops after {o['drops_after_t1']}")
            print(msg, flush=True)
    print('\nSUMMARY', tot)
    if recs:
        import statistics as st
        med = lambda key: st.median([r[key] for r in recs if r.get(key) is not None])
        print(f"medians over {len(recs)} runs: h0 {med('h0'):.2e}; mechanism (i) holds from h {med('h_i_star'):.1e}, "
              f"(ii) from h {med('h_ii_star'):.1e}; proof's conditions first hold at h {med('h_at_tid'):.1e} (id) / "
              f"{med('h_at_tw'):.1e} (w); h-thresholds H_id {med('Hid'):.1e}, H_w {med('Hw'):.1e} "
              f"(radius-based thresholds of the workshop proof: H_id {med('Hid_r'):.1e}, H_w {med('Hw_r'):.1e})")
        print(f"orders of magnitude between the mechanism and the theorem's h-threshold (median): "
              f"{st.median([np.log10(max(r['h_i_star'],1e-300)/r['Hid']) for r in recs]):.1f} (i), "
              f"{st.median([np.log10(max(r['h_ii_star'],1e-300)/r['Hw']) for r in recs]):.1f} (ii)")


if __name__ == '__main__':
    main()
