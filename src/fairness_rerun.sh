#!/usr/bin/env bash
# Baseline-fairness reruns asked for by the 2026-09-10 review (Section C
# and L): our SMO / kernel SMO with a full kernel-row cache and the whole
# regime battery in isolation on the released code (C1, C5); the SMO column
# of the L2 battery on the cells where it converges (C1); LIBLINEAR's dual
# solver and a tolerance sweep (C2); exact-LMO MDM, away-step FW and
# guarded k=1 rows for Table 9 (C3, L3); one uncapped C=10 trace (C8); the
# dataset table (C9); the screening figure and the consolidated synthetic
# benchmark on the released code (B3).  Same conventions as
# provenance_rerun.sh: per-stage exit logging, SKIP by name, PAR for the
# batteries.  Start it detached:
#   PAR=4 setsid nohup bash src/fairness_rerun.sh > results/fairness_rerun.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PAR="${PAR:-1}"
SKIP="${SKIP:-}"
mkdir -p results paper/aor
echo "== $(date) start; commit $(git rev-parse --short HEAD); PAR=$PAR SKIP='$SKIP'"
stage() {
    local name="$1" log="$2"; shift 2
    case " $SKIP " in *" $name "*) echo "== $(date) $name: skipped"; return 0;; esac
    echo "== $(date) $name: $*"
    "$@" > "$log" 2>&1; local rc=$?
    echo "== $(date) $name: exit $rc (log: $log, $(tail -1 "$log" 2>/dev/null | cut -c1-120))"
    if [ $rc -ne 0 ]; then echo "== $(date) STOP: stage $name failed with exit $rc"; exit $rc; fi
}

stage datasets results/dataset_table.log python3 src/dataset_table.py --data-dir data

stage profile results/step_profile4.log python3 src/step_profile.py --data-dir data \
    --seeds 3 --solvers mdm-exact afw-exact guarded1-exact --out results/step_profile4.json

stage liblin results/liblin_sweep.log python3 src/liblin_sweep.py --data-dir data \
    --datasets a9a gisette --Cs 0.1 1 --seeds 5 --out results/liblin_tol_sweep.json

stage regime results/regime_battery2.log python3 src/regime_battery.py --data-dir data \
    --seeds 5 --parallel "$PAR" --cache-rows 0 --out results/regime_battery2.json

stage smo results/full_battery2_smo.log python3 src/full_battery.py --data-dir data \
    --seeds 5 --parallel "$PAR" --solvers SMO --cache-rows 0 \
    --datasets gisette ijcnn1 --Cs 0.1 1 10 --out results/full_battery2_smo.json

stage uncapped10 results/full_battery_uncapped10.log python3 src/full_battery.py --data-dir data \
    --time-cap 0 --solvers ETA --seeds 1 --datasets ijcnn1 --Cs 10 --parallel 1 \
    --out results/full_battery_uncapped10.json

stage shrink results/shrink_benchmark.log python3 src/shrink_benchmark.py
stage final results/final_benchmark2.log python3 src/final_benchmark.py --trials 5 --parallel "$PAR" \
    --out results/final_benchmark2.json

echo "== $(date) done"
