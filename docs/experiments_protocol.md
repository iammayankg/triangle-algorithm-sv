# Full-Scale Experiment Protocol (submission-grade)

Everything below is designed so a referee can reproduce the numbers.
The scripts in this repository run as-is; only the hardware and dataset
downloads require an open-network machine.

## 1. Hardware

Use a *dedicated*, single-tenant machine - never a shared container (the
development environment for this study showed 2x wall-clock variance on
identical deterministic runs, which is exactly what this protocol
eliminates).

- CPU battery: any modern 8-16+ core x86 machine - a Lightning.ai
  Studio on a >= 16 vCPU CPU tier (see 1a), an AWS `c7i.4xlarge` /
  `m7i.4xlarge` (on-demand, not burstable - avoid `t*` instances), or a
  local workstation. 32 GB RAM is ample (largest dense matrix: covtype
  subsample 100k x 54; gisette 6000 x 5000).
- GPU (only for the ThunderSVM baseline): any NVIDIA card - a T4/L4
  Studio on Lightning.ai, or AWS `g5.xlarge`.
- Report in the paper: CPU model, core count, RAM, OS, BLAS
  implementation (`numpy.show_config()`), and Python/library versions.

### 1a. Running on Lightning.ai (recommended setup)

> Step-by-step operational instructions (Studio creation, machine
> switching, launch, ThunderSVM install, retrieval):
> **docs/lightning_runbook.md**. This section holds only the hardware
> choices.

Lightning AI Studios give each session a dedicated cloud VM (AWS-backed),
which satisfies the single-tenant requirement; the environment persists
across machine switches, so you can develop on the free CPU tier and
switch to the benchmark machine only for timed runs.

- **CPU battery**: switch the Studio to the largest CPU machine tier
  offered in the machine picker - prefer >= 16 vCPU / 32 GB RAM
  (32 vCPU if offered; 8 vCPU / 32 GB is the workable minimum - gisette
  and covtype are the memory/compute peaks). Use *on-demand*, not
  interruptible, for the timed runs: an interruption mid-battery ruins a
  seed cell.
- **GPU (ThunderSVM baseline only)**: a T4 (~$0.41/hr as of Sep 2026)
  is sufficient - ThunderSVM is not compute-bound at these problem
  sizes; L4 (~$0.60/hr) or A10G (~$0.71/hr) if the T4 queue is long.
  No need for A100/H100.
- **Practicalities**: disable (or lengthen) the Studio auto-sleep
  timeout before launching the battery, and run it under
  `nohup ... &` so a browser disconnect cannot kill it; results are
  written incrementally to JSON, so an interrupted run resumes by
  rerunning only missing cells. Record the exact machine tier shown in
  the picker in the paper's hardware paragraph.
- Expected cost: the full CPU battery is a several-hour run on a
  mid-tier CPU machine plus an hour or two of T4 - a few dollars total
  at current rates. (Prices and tiers change; verify in the machine
  picker at run time.)

## 2. Machine setup

```bash
# On bare metal, pin clocks (skip on cloud VMs - Lightning/AWS guests
# cannot control the governor; a dedicated instance type is the control):
#   sudo cpupower frequency-set -g performance
#   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

git clone <repo> && cd triangle-algorithm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fix BLAS threading - report the value used; 1 gives the cleanest
# single-solver comparisons, #physical-cores the best absolute numbers
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# pin the process to fixed cores
alias runpin='taskset -c 0-3'
```

## 3. Datasets

```bash
./scripts/fetch_datasets.sh data
```

Battery (dense path, supported now): a9a, w8a, ijcnn1, gisette,
covtype (subsampled to `--max-n`, default 100k). Sparse-only sets
(real-sim, rcv1) are downloaded but need sparse-matrix support in the
solvers before they can run - see TODO below.

## 4. The battery

```bash
# parallel across cells (recommended on >= 16 vCPU: leave ~2 cores free)
OMP_NUM_THREADS=1 nohup python3 src/full_battery.py --data-dir data \
    --seeds 10 --parallel 12 --out results/full_battery.json &
# analysis: summary table + time/gap/oracle figures, runnable any time
python3 src/battery_analysis.py --out results/full_battery.json
```

Each cell writes its own shard under `results/full_battery.json.shards/`;
rerunning skips finished shards, so interruptions cost only the cells in
flight. Smoke-test the pipeline first (no downloads needed):
`python3 src/full_battery.py --datasets synthetic --seeds 2 --parallel 2
--out results/smoke.json && python3 src/battery_analysis.py --out
results/smoke.json`.

- 10 seeds per (dataset, C) cell; C in {0.1, 1, 10}; each cell runs
  ETA (block, guarded), our SMO, and LIBLINEAR on the *same* split.
- Reported per run: wall-clock, iterations, **oracle calls**
  (O(nd) column/row evaluations - the implementation-independent
  metric), primal objective, ETA's certified relative gap, test
  accuracy; seed 0 additionally records the (time, UB, LB) convergence
  trace for the gap-vs-time plots the linear-rate theorem predicts.
- The summary prints mean +/- 95% CI (t-distribution) over seeds.
  ETA and SMO each carry a 600 s budget per cell; LIBLINEAR is
  unbounded by default (it finishes in seconds on the overlapping
  sets but needs 8 min to 1.5 h on gisette at the matched 1e-6
  tolerance). `--liblin-cap SECONDS` runs LIBLINEAR in a child process
  that is killed at the cap and recorded as `timeout` with no model;
  the paper's Table 2 shards were produced unbounded, so do not mix
  capped and uncapped shards in one battery. Worst case ~21 min per
  cell uncapped on the overlapping sets, so
  150 cells at `--parallel 12` take ~4.5 h. On the heavily overlapping
  datasets ETA and SMO are expected to time out (dense-support regime,
  Theorem 6) and report their achieved certified gaps - that is the
  intended measurement, not a failure.

Solver-comparability notes for the paper: ETA stops at certified
relative gap 1e-3; LIBLINEAR at primal-progress tol 1e-6 (chosen so its
final primal matches ETA's certified value to ~1e-4 relative - verify
and report this equivalence on each dataset); SMO at KKT gap 1e-3.
State all three explicitly.

## 4b. The regime battery (ETA's own regime on real data)

The L2 battery above is the sparse-vs-dense boundary seen from the dense
side: on heavily overlapping data 60-65% of points become support
vectors with near-uniform weights, the pyramidal width of the resulting
near-simplex shrinks like 1/n, and the geometric solver's cost is
O(n^2)-type (Theorem 6 predicts this; a9a/w8a confirm it - ETA times
out where LIBLINEAR finishes in seconds). `src/regime_battery.py` shows
the boundary from the other side, on the same datasets:

```bash
OMP_NUM_THREADS=1 nohup python3 src/regime_battery.py --data-dir data \
    --seeds 5 --parallel 8 --out results/regime_battery.json \
    > results/regime_battery.log 2>&1 &
```

Per dataset x seed it runs: **FEAS** (TA I certifies linear
separability - a diagnostic no baseline offers), **LIN** (linear hard
margin where feasible: ETA with shrinking vs SVC(linear, C=1e6) vs SMO,
same objective), **KHM** (RBF hard margin on a 10k class-balanced
subsample: KernelETA vs SVC(rbf, C=1e6), same objective, distance
agreement via the dual expansion), and **KL2** (RBF K + I/C at C=1:
KernelETA vs KernelSMO, same objective; SVC(rbf, C=1) as a labelled L1
reference). All solvers get 600 s; the summary
(`results/regime_battery_summary.md`) reports time +/- CI95, distance,
accuracy, oracle calls, and non-convergence counts per cell. Expected
runtime: 1-3 h at `--parallel 8` (LIBSVM at C=1e6 on RBF is the slow
column). Smoke test: `--datasets synthetic-sep synthetic --seeds 2
--max-n-kernel 1200`.

For the paper: report the two batteries together as one regime story -
the dense-support L2 cells (LIBLINEAR wins, ETA times out, as the theory
predicts) and the sparse-support hard-margin/kernel cells (ETA's
territory) - rather than either alone.

## 5. Additional runs for the paper

- **Replication + synthetic suites** (already scripted):
  `experiments.py`, `solver_comparison.py`, `soft_experiment.py`,
  `l1_experiment.py`, `kernel_experiment.py`, `shrink_benchmark.py`,
  `block_benchmark.py`, `final_benchmark.py` - re-run each with
  `--trials 10` equivalents on the pinned machine.
- **ThunderSVM** (GPU machine): `pip install thundersvm`; add it as a
  solver column for the RBF cells (hard margin C=1e6 approximation and
  C=1 soft margin); its stopping tolerance is `-e`, set 1e-3.
- **Nu-SVM cells**: rerun `l1_experiment.py` on a9a/ijcnn1 subsamples
  (NuSVC is O(n^2)-ish; cap n at ~50k).

## 6. Known TODOs before the sparse sets can run

The solvers use dense `numpy` matrices (`V @ x`, row extraction,
`einsum` norms). To run rcv1/real-sim: accept
`scipy.sparse.csr_matrix`, replace `einsum` row norms with
`X.multiply(X).sum(1)`, keep iterates dense (d-vectors), and take
`V[i].toarray().ravel()` on the few explicit row reads. Estimated
effort: a day including tests.

## 7. What goes in the paper

- Table: per (dataset, C), time mean +/- CI, accuracy, oracle calls,
  and status, for ETA / SMO / LIBLINEAR (/ThunderSVM where applicable).
- Figure: gap vs time (log-y) from the traces - the linear-rate slope.
- Figure: oracle calls vs dataset (implementation-independent story).
- Text: hardware paragraph, tolerance-equivalence paragraph, seeds/CI
  statement, link to this protocol file.
