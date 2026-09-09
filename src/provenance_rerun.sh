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
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PAR="${PAR:-1}"
mkdir -p results paper/aor
echo "== $(date) start; commit $(git rev-parse --short HEAD); PAR=$PAR"

echo "== $(date) Table 3 / Table 8: synthetic schedule check and k ablation"
python3 src/schedule_check.py > results/schedule_check.log
python3 src/k_ablation_synth.py > results/k_ablation_synth.log
python3 src/make_appendix_tables.py > paper/aor/tab_synth.tex

echo "== $(date) new: n-scaling of the dense-support regime"
python3 src/n_scaling.py --out results/n_scaling.json > results/n_scaling.log

echo "== $(date) new: screening on real cells (gisette LIN and L2)"
python3 src/screening_real.py --data-dir data --seeds 3 \
    --out results/screening_real.json > results/screening_real.log

echo "== $(date) Figure 1 / Table 6: uncapped ETA traces (seed 0, C=0.1)"
python3 src/full_battery.py --data-dir data --time-cap 0 --solvers ETA \
    --seeds 1 --datasets a9a w8a ijcnn1 --Cs 0.1 --parallel 1 \
    --out results/full_battery_uncapped.json > results/full_battery_uncapped.log

echo "== $(date) Table 2: L2 battery, five seeds, PAR=$PAR"
python3 src/full_battery.py --data-dir data --seeds 5 --parallel "$PAR" \
    --out results/full_battery2.json > results/full_battery2.log
python3 src/battery_analysis.py --out results/full_battery2.json

echo "== $(date) Table 6 and Figure 1 from the traces"
python3 src/time_to_tol.py --shards results/full_battery2.json.shards \
    --uncapped results/full_battery_uncapped.shards > results/time_to_tol.md
python3 src/trace_fig.py --shards results/full_battery_uncapped.shards \
    --out paper/figs/fig_realdata_trace.pdf

echo "== $(date) done"
