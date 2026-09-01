# Full-Scale Experiment Protocol (submission-grade)

Everything below is designed so a referee can reproduce the numbers.
The scripts in this repository run as-is; only the hardware and dataset
downloads require an open-network machine.

## 1. Hardware

Use a *dedicated*, single-tenant machine - never a shared container (the
development environment for this study showed 2x wall-clock variance on
identical deterministic runs, which is exactly what this protocol
eliminates).

- CPU battery: any modern 8-16 core x86 machine, e.g. AWS `c7i.4xlarge`
  / `m7i.4xlarge` (on-demand, not burstable - avoid `t*` instances), or
  a local workstation. 32 GB RAM is ample (largest dense matrix:
  covtype subsample 100k x 54; gisette 6000 x 5000).
- GPU (only for the ThunderSVM baseline): any NVIDIA card, e.g. AWS
  `g5.xlarge`.
- Report in the paper: CPU model, core count, RAM, OS, BLAS
  implementation (`numpy.show_config()`), and Python/library versions.

## 2. Machine setup

```bash
# performance governor, no frequency scaling surprises (Linux)
sudo cpupower frequency-set -g performance   # or equivalent
# optional but recommended: disable turbo for stable clocks
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

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
runpin python3 src/full_battery.py --data-dir data --seeds 10 \
    --out results/full_battery.json
```

- 10 seeds per (dataset, C) cell; C in {0.1, 1, 10}; each cell runs
  ETA (block, guarded), our SMO, and LIBLINEAR on the *same* split.
- Reported per run: wall-clock, iterations, **oracle calls**
  (O(nd) column/row evaluations - the implementation-independent
  metric), primal objective, ETA's certified relative gap, test
  accuracy; seed 0 additionally records the (time, UB, LB) convergence
  trace for the gap-vs-time plots the linear-rate theorem predicts.
- The summary prints mean +/- 95% CI (t-distribution) over seeds.
  Expected total runtime at defaults: several hours; covtype and
  gisette dominate.

Solver-comparability notes for the paper: ETA stops at certified
relative gap 1e-3; LIBLINEAR at primal-progress tol 1e-6 (chosen so its
final primal matches ETA's certified value to ~1e-4 relative - verify
and report this equivalence on each dataset); SMO at KKT gap 1e-3.
State all three explicitly.

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
