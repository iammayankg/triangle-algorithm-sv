#!/usr/bin/env bash
# One-protocol rerun of every journal table and figure that still carries
# an older code state or an uncertain load (journal review 2026-09-08,
# item 1): Table 2 (L2 battery), Table 3 (schedules), Table 6 (time to 1%,
# from the uncapped traces), Table 8 (synthetic k ablation) and Figure 1
# (uncapped traces), plus the two new experiments the review asked for
# (screening on a real cell, n-scaling).  Everything runs on the released
# code with OMP_NUM_THREADS=1; the batteries use at most $PAR concurrent
# single-threaded processes (default 1 = strictly sequential, about 30 h
# for the L2 battery alone; PAR=4 brings it to about 8 h on 32 vCPUs),
# the synthetic tables and the new experiments run strictly one process
# at a time.  Check `pgrep -af "src/.*\.py"` is empty before starting and
# turn the Studio's auto-sleep off.
#
#   cd ~/triangle-algorithm-sv && git pull
#   nohup bash src/provenance_rerun.sh > results/provenance_rerun.log 2>&1 &
#
# Every stage is logged with its exit status; a stage killed by a signal
# (the Studio sleeping, the OOM killer) shows up as "exit 137"/"exit 143"
# instead of a silent stop.  SKIP="nscale screening" skips stages by name;
# NSCALE_CAP (seconds, default 3600) caps each n-scaling instance.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PAR="${PAR:-1}"
SKIP="${SKIP:-}"
NSCALE_CAP="${NSCALE_CAP:-3600}"
mkdir -p results paper/aor
echo "== $(date) start; commit $(git rev-parse --short HEAD); PAR=$PAR SKIP='$SKIP'"

stage() {   # stage NAME LOGFILE CMD...  (stops the driver on failure)
    local name="$1" log="$2"; shift 2
    case " $SKIP " in *" $name "*) echo "== $(date) $name: skipped"; return 0;; esac
    echo "== $(date) $name: $*"
    "$@" > "$log" 2>&1; local rc=$?
    echo "== $(date) $name: exit $rc (log: $log, $(tail -1 "$log" 2>/dev/null | cut -c1-120))"
    if [ $rc -ne 0 ]; then
        echo "== $(date) STOP: stage $name failed with exit $rc"; free -g 2>/dev/null | head -2
        exit $rc
    fi
}

stage synth results/schedule_check.log python3 src/schedule_check.py
stage synth results/k_ablation_synth.log python3 src/k_ablation_synth.py
stage synth paper/aor/tab_synth.tex python3 src/make_appendix_tables.py

stage nscale results/n_scaling.log python3 src/n_scaling.py \
    --out results/n_scaling.json --time-cap "$NSCALE_CAP"

stage screening results/screening_real.log python3 src/screening_real.py \
    --data-dir data --seeds 3 --out results/screening_real.json

stage uncapped results/full_battery_uncapped.log python3 src/full_battery.py \
    --data-dir data --time-cap 0 --solvers ETA --seeds 1 \
    --datasets a9a w8a ijcnn1 --Cs 0.1 --parallel 1 \
    --out results/full_battery_uncapped.json

stage battery results/full_battery2.log python3 src/full_battery.py \
    --data-dir data --seeds 5 --parallel "$PAR" --out results/full_battery2.json
stage battery results/battery_analysis.log python3 src/battery_analysis.py \
    --out results/full_battery2.json

stage tables results/time_to_tol.md python3 src/time_to_tol.py \
    --shards results/full_battery2.json.shards \
    --uncapped results/full_battery_uncapped.json.shards
stage tables results/trace_fig.log python3 src/trace_fig.py \
    --shards results/full_battery_uncapped.json.shards \
    --out paper/figs/fig_realdata_trace.pdf

echo "== $(date) done"
