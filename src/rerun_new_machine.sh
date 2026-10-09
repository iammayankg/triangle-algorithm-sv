#!/usr/bin/env bash
# Full rerun on one machine (October 2026). The Studio that produced the
# September numbers (Intel Xeon @ 2.60 GHz) is no longer offered, so every
# timing comparison the journal draft makes is rerun here, on one machine,
# with the baseline-fairness changes of the 2026-09-10 review (full SMO
# kernel-row cache, LIBLINEAR's dual solver, exact-search step rows).
#
# Run in the foreground inside tmux, from anywhere:
#   tmux new -s rerun
#   bash ~/triangle-algorithm-sv/src/rerun_new_machine.sh 2>&1 | tee ~/triangle-algorithm-sv/results/rerun_new_machine.log
#
# Every stage resumes from what is on disk. After an interruption, rerun the
# same command; to skip finished stages name them, e.g.
#   SKIP="machine battery1" bash src/rerun_new_machine.sh ...
# A stage that exits non-zero stops the script.
#
# Memory: our SMO caches full-length float64 kernel rows. Uncapped, one
# covtype run can reach ~45 GB and ijcnn1/w8a ~20 GB, so covtype runs two
# cells at a time and everything else four. Needs >= 120 GB RAM.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SKIP="${SKIP:-}"
mkdir -p results

stage() {
    local name="$1"; shift
    case " $SKIP " in *" $name "*) echo "== $(date) $name: skipped"; return 0;; esac
    echo "== $(date) $name: start"
    "$@"; local rc=$?
    echo "== $(date) $name: exit $rc"
    if [ $rc -ne 0 ]; then echo "== $(date) STOP: stage $name failed"; exit $rc; fi
}

record_machine() {
    { date; git rev-parse --short HEAD; nproc; free -g; lscpu
      python3 --version
      python3 -c "import numpy, scipy, sklearn; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__); numpy.show_config()"
    } > results/machine_oct2026.txt 2>&1
    grep -m1 "Model name" results/machine_oct2026.txt
}

check_data() {
    local f
    for f in a9a a9a.t w8a w8a.t ijcnn1 ijcnn1.t gisette_scale gisette_scale.t covtype.libsvm.binary.scale; do
        [ -s "data/$f" ] || { echo "missing data/$f: run ./scripts/fetch_datasets.sh data"; return 1; }
    done
}

liblin_sweep() {   # three independent processes, then wait for all three
    python3 -u src/liblin_sweep.py --data-dir data --datasets a9a --Cs 0.1 1 --seeds 5 \
        --out results/liblin_tol_sweep_a9a.json > results/liblin_sweep_a9a.log 2>&1 & local p1=$!
    python3 -u src/liblin_sweep.py --data-dir data --datasets gisette --Cs 0.1 \
        --tols 1e-4 3e-5 1e-5 1e-6 --seeds 5 \
        --out results/liblin_tol_sweep_g01.json > results/liblin_sweep_g01.log 2>&1 & local p2=$!
    python3 -u src/liblin_sweep.py --data-dir data --datasets gisette --Cs 1 \
        --tols 1e-4 3e-5 1e-5 1e-6 --seeds 5 \
        --out results/liblin_tol_sweep_g1.json > results/liblin_sweep_g1.log 2>&1 & local p3=$!
    local rc=0
    wait $p1 || rc=1; wait $p2 || rc=1; wait $p3 || rc=1
    return $rc
}

echo "== $(date) start; commit $(git rev-parse --short HEAD); SKIP='$SKIP'"
stage machine record_machine
stage data check_data

# 1. Dataset table, regime battery (Table 1; full SMO cache by default),
#    screening figure and the consolidated synthetic benchmark.
stage battery1 env PAR=4 SKIP="liblin profile smo uncapped10" bash src/fairness_rerun.sh

# 2. Whole L2 battery (Table 2, matched-tolerance rows of the LIBLINEAR
#    table) with the full SMO cache; covtype two cells at a time for memory.
stage l2 python3 -u src/full_battery.py --data-dir data --seeds 5 --parallel 4 --cache-rows 0 \
    --datasets gisette ijcnn1 a9a w8a --out results/full_battery3.json
stage l2covtype python3 -u src/full_battery.py --data-dir data --seeds 5 --parallel 2 --cache-rows 0 \
    --datasets covtype --out results/full_battery3.json

# 3. Uncapped seed-0 traces (Figure 1, time-to-1% table), one at a time;
#    the C=10 trace stops at the 2,000,000-iteration cap.
stage uncapped python3 -u src/full_battery.py --data-dir data --time-cap 0 --solvers ETA --seeds 1 \
    --datasets a9a w8a ijcnn1 --Cs 0.1 --parallel 1 --out results/full_battery_uncapped3.json
stage uncapped10 python3 -u src/full_battery.py --data-dir data --time-cap 0 --solvers ETA --seeds 1 \
    --datasets ijcnn1 --Cs 10 --parallel 1 --out results/full_battery_uncapped10.json

# 4. Step ablation, every configuration, isolated (one process).
stage profile python3 -u src/step_profile.py --data-dir data --seeds 3 \
    --solvers mdm guarded1 block16 auto bpcg mdm-exact afw-exact guarded1-exact \
    --out results/step_profile5.json

# 5. LIBLINEAR tolerance sweep, Newton and dual solvers, nothing else running.
stage liblin liblin_sweep

echo "== $(date) done. Commit and push: git add results/ && git commit -m 'results: October rerun' && git push origin main"
