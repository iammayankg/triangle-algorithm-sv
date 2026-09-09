# Lightning.ai Runbook

Step-by-step instructions for gathering the full-scale results on
Lightning AI Studios. Companion to `docs/experiments_protocol.md`
(which holds the scientific protocol: seeds, tolerances, what to
report); this file is purely operational. UI element names are as of
September 2026 - if a button has moved, the concept still applies.

## What produced the paper's numbers (Sep 2-3, 2026) - the reference sequence

Everything in `paper/opt2026.tex` came from one 32-vCPU Studio (Intel
Xeon 2.60 GHz, 122 GB, `machine.txt`), Python 3.12, NumPy 2.4 with
OpenBLAS, `OMP_NUM_THREADS=1` for every run. Rerunning this block from
a clean `results/` reproduces every table and figure; each script
skips work it already finds on disk, so it can also be run piecemeal.

```bash
cd ~/triangle-algorithm-sv && git pull
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results

# 1. L2 battery (paper Table 2): 5 datasets x 3 Cs x 5 seeds = 75 cells.
#    ETA and our SMO capped at 600 s; LIBLINEAR unbounded (matched
#    tolerance 1e-6: 8 min - 1.5 h per gisette cell at C >= 1).
nohup python3 -u src/full_battery.py --data-dir data --seeds 5 \
    --parallel 14 --out results/full_battery.json \
    > results/full_battery.log 2>&1 &

# 2. Regime battery (paper Table 1): FEAS/LIN/KHM/KL2 cells, 5 seeds.
nohup python3 -u src/regime_battery.py --data-dir data --seeds 5 \
    --parallel 12 --out results/regime_battery.json \
    > results/regime_battery.log 2>&1 &

# --- after both finish (2-8 h; gisette LIBLINEAR is the long pole) ---

# 3. Analysis and figures for the batteries
python3 src/battery_analysis.py --out results/full_battery.json
python3 src/regime_battery.py --data-dir data --seeds 5 --parallel 1 \
    --out results/regime_battery.json      # merges shards -> summary md
python3 src/regime_fig.py                  # blog fig 10

# 4. Review-driven runs (paper Tables 4-6), all cheap, no load:
python3 src/liblin_default_tol.py --data-dir data --seeds 5 --tols 1e-4 \
    --out results/liblin_default_tol.json           # ~6 min
python3 src/k_ablation_real.py --data-dir data --seeds 3 \
    --out results/k_ablation_real.json               # ~15 min
python3 src/time_to_tol.py                           # reads seed-0 traces

# 5. Consolidated synthetic benchmark (paper Fig. 3), 5 trials in
#    parallel; ~20 min. Trial 0 alone reproduces the single-run numbers.
nohup python3 -u src/final_benchmark.py --trials 5 --parallel 7 \
    > results/final_benchmark.log 2>&1 &
# then: python3 src/final_benchmark_fig.py --out paper/figs/fig_final.pdf paper/figs/fig_final.png results/fig_final.png blog/figs/08_regimes.png

# 6. Synthetic appendix tables (schedule check, block-size ablation)
python3 src/schedule_check.py && python3 src/k_ablation_synth.py
python3 src/make_appendix_tables.py       # emits LaTeX for Tables 3, 7

# 7. Commit and push results
git add results/ && git commit -m "results: <what ran>" && git push origin main
```

Before closing the tab: **turn the Studio's auto-sleep off**. The
first L2 pass died silently at the gisette C=10 cells when the Studio
slept; `nohup` does not survive a sleep. Always confirm a launch with
`sleep 30; ps aux | grep -c "[f]ull_battery.py"` and, on return, with
`pgrep -fa full_battery.py` before assuming a run is still going.

Paper-table to script map:

| paper item | script | output |
|--|--|--|
| Table 1 (regime battery) | `regime_battery.py` | `results/regime_battery_summary.md` |
| Table 2 (L2 battery) | `full_battery.py` + `battery_analysis.py` | `results/full_battery_summary.md` |
| Table 3 (schedules) | `schedule_check.py` + `make_appendix_tables.py` | `results/schedule_check.json` |
| Table 4 (LIBLINEAR default tol) | `liblin_default_tol.py` | `results/liblin_default_tol.json` |
| Table 5 (real-data k ablation) | `k_ablation_real.py` | `results/k_ablation_real.json` |
| Table 6 (time to 1%) | `time_to_tol.py` | stdout (markdown) |
| Table 7 (synthetic k ablation) | `k_ablation_synth.py` + `make_appendix_tables.py` | `results/k_ablation_synth.json` |
| Fig. 1 (uncapped traces) | `full_battery.py` seed-0 shards, uncapped | `results/full_battery_uncapped.shards/` |
| Fig. 2 (screening speed-up) | `shrink_benchmark.py` + `shrink_fig.py` (renders `paper/figs/fig_shrink.pdf`) | `results/shrink_benchmark.json` |
| Fig. 3 (consolidated benchmark) | `final_benchmark.py` + `final_benchmark_fig.py --out paper/figs/fig_final.pdf ...` (vector for the paper) | `results/final_benchmark.json` |

> **Post-submission fix (2026-09-06).** `src/triangle_algorithm.py` used to
> return an ordinary (possibly capacity-clipped) MDM transfer when only one
> pair survived filtering, bypassing the case-(b) away fallback that
> Theorem 3 requires; the same happened for a degenerate block direction.
> Both paths now go through the guard and fallback
> (`tests/test_fallback_single_pair.py` reproduces the reviewer's instance).
> Status 2026-09-06 night: everything except Table 2 and the Table 6 traces
> is now from the fixed code (Tables 1, 3, 5, 7, 8, Fig. 3; LIBSVM brackets
> in Table 1). The idle rerun of `final_benchmark.py` reproduced the
> submitted numbers within intervals; `schedule_check.py` and
> `k_ablation_synth.py` came back ~3x slower than an earlier run with
> identical iteration counts, so check `pgrep` for stragglers before
> trusting their absolute times. Only `full_battery.py` (Table 2, a day)
> remains on the pre-fix code. Old results are archived in
> `results/prefix/` on the Studio.
>
> Every battery, ablation and benchmark in `results/` was produced by the
> old code, so the camera-ready reruns below must use the current code, in
> this order of value: (1) `regime_battery.py` (Table 1), (2)
> `k_ablation_real.py` and `k_ablation_synth.py` (Tables 5, 7: the k=1 row
> now runs the corrected guarded algorithm, not plain pairwise FW),
> (3) `schedule_check.py` (Table 3), (4) `final_benchmark.py --trials 5`
> (Fig. 3), (5) `full_battery.py` (Table 2, the expensive one). Also still
> open: an ETA primal-progress column for Table 6. The two new baselines
> have scripts now (both resume; both use the KL2 subsample/seeds/gamma of
> `regime_battery.py`):
>
> ```bash
> # LIBSVM on the precomputed K + I/C kernel, C=1e6 surrogate, five KL2 cells
> nohup python3 -u src/libsvm_kl2.py --data-dir data --seeds 5 --parallel 5 \
>     --out results/libsvm_kl2.json > results/libsvm_kl2.log 2>&1 &
> # BPCG (Tsuji et al. 2022) vs ETA, same certificate and column cost model,
> # on gisette-l2 and the five KL2 cells (one process; ~1-2 h; timings are
> # paired within the run, so load matters only for absolute numbers)
> nohup python3 -u src/bpcg_baseline.py --data-dir data --seeds 3 \
>     --out results/bpcg_baseline.json > results/bpcg_baseline.log 2>&1 &
> ```
>
> Review round 3 (2026-09-06) asked for an *isolated* rerun of MDM, guarded
> k=1, block k=16 and BPCG on the two Table 5 cells with a scan / column /
> step time split and peak memory. `src/step_profile.py` does exactly that:
> each (cell, solver, seed) runs alone in a fresh subprocess, sequentially
> (~10 min total). Make sure nothing else is running first:
>
> ```bash
> pgrep -af "src/.*\.py"          # must print nothing
> OMP_NUM_THREADS=1 nohup python3 -u src/step_profile.py --data-dir data --seeds 3 \
>     --out results/step_profile.json > results/step_profile.log 2>&1 &
> tail -f results/step_profile.log  # summary table prints at the end
> ```
>
> 2026-09-07: three overhead fixes landed after the diagnostics runs
> (vectorised active-donor extraction in `_block_pairs`, `_worst_active_*`
> and the side-gap test of `_step`; the pair Gram matrix is now built from
> k'^2 cached entries instead of k' x n column copies).  Trajectories are
> bit-identical; only times change.  Before the journal numbers are final,
> rerun both profiling scripts in isolation (summaries only - never
> `--keep-records` into git again, the last one was 103 MB):
>
> ```bash
> pgrep -af "src/.*\.py"          # must print nothing
> OMP_NUM_THREADS=1 nohup sh -c 'python3 -u src/step_profile.py --data-dir data --seeds 3 --out results/step_profile2.json > results/step_profile2.log 2>&1; python3 -u src/block_diagnostics.py --data-dir data --cells gisette-l2 ijcnn1-kl2 w8a-kl2 a9a-kl2 covtype-kl2 --ks 1 4 16 32 --out results/block_diag3.json > results/block_diag3.log 2>&1' > /dev/null 2>&1 &
> ```
>
> (~25 min sequential).  Then `python3 src/block_diag_fig.py --inp
> results/block_diag3.json` regenerates the journal table rows and figure
> from the new run (the decile figure needs records; keep
> `results/block_diag2.json` for that, or rerun one cell with
> `--keep-records --out /tmp/...` and pass it with `--inp`).
>
> Then sync `results/step_profile.json` and the paper gets: the isolated
> ratios for Sec. 5 / Tables 5 and 7, the time split for the gisette vs
> ijcnn1 discussion, and a peak-memory number for App. C.
> Done 2026-09-06 evening (`results/step_profile.json`, 24 rows): Table 5 is
> now the isolated run with the time split. Headline: on gisette L2 all four
> configurations (MDM, guarded k=1, block k=16, BPCG) tie at ~30 s because
> 86-92% of the time is column computation; on ijcnn1 KL2 block k=16 is
> 2.2x over MDM, 1.9x over guarded k=1, 3.0x over BPCG. The loaded run's
> "guarded k=1 fastest on gisette" (27.7 vs 36.6 s) was a load artefact.

## 0. What you will run, and on what

| leg | machine tier | duration | cost ballpark |
|--|--|--|--|
| setup + smoke test + dataset fetch | free CPU tier | ~30 min | free |
| L2 battery (`full_battery.py`, 5 seeds) | largest CPU tier, >= 16 vCPU / 32 GB | 2-3 h for the overlapping sets at `--parallel 14`; gisette adds up to 1.5 h per cell for LIBLINEAR at C >= 1 | a few $ |
| regime battery (`regime_battery.py`, 5 seeds) | same CPU tier | 1-3 h at `--parallel 12` | a few $ |
| review runs (`liblin_default_tol.py`, `k_ablation_real.py`) | same | ~20 min | cents |
| consolidated benchmark (`final_benchmark.py --trials 5 --parallel 7`) | same | ~20 min | cents |
| ThunderSVM leg (not used in the paper) | T4 GPU | 1-2 h incl. install | ~$1 |

**What to expect from the L2 battery** (do not mistake it for a
problem): on the heavily overlapping datasets (a9a, w8a, ijcnn1,
covtype) 60-65% of points are support vectors, the dense-support
regime where the geometric solver is O(n^2)-type. ETA and SMO hit their
600 s caps and report `timeout` with their achieved certified gap,
while LIBLINEAR finishes in 0.3-40 s. Gisette is the other way round:
ETA converges in 30-45 s at every C, our SMO in ~4-5 min, and
LIBLINEAR at the matched tolerance takes 8 min (C=0.1), 65 min (C=1)
and 90 min (C=10). Both halves are the regime story the paper reports.

Timings in the paper are comparable *within* a battery only. The L2
and regime batteries ran concurrently (14 + 12 single-threaded
workers); the review runs and the benchmark trials ran with no other
load, which is why their gisette times are ~1.5x lower than Table 2's.
The paper's Appendix C says so. Use **on-demand** machines for all
timed runs, never interruptible.

## 1. Create the Studio and get the code

1. lightning.ai -> your teamspace -> **New Studio**. It starts on the
   free CPU tier; do all setup there.
2. Get the repository in, either way:
   - `git clone https://github.com/<user>/<repo>.git` in the Studio
     terminal (requires the repo to be pushed; a private repo needs a
     GitHub token or the Studio's GitHub integration), or
   - drag-and-drop a zip of the repo into the Studio file panel and
     unzip.
3. Environment:

   ```bash
   cd triangle-algorithm
   pip install -r requirements.txt
   ```

   Keeping the Studio current: after pushing new commits from your Mac,
   `git pull` in the Studio terminal before launching anything.
   **If the battery code changed since a run started** (e.g. the 600 s
   ETA cap added in commit `c8bee1b`), stop the run and delete its shards
   so all cells are measured under the same settings:

   ```bash
   pkill -f full_battery.py
   rm -rf results/full_battery.json.shards results/full_battery.json
   ```

   (Studios persist their filesystem and environment across restarts
   and machine switches - you do this once.)

## 2. Verify the pipeline on the free tier

```bash
python3 src/full_battery.py --datasets synthetic --seeds 2 \
    --parallel 2 --out results/smoke.json
python3 src/battery_analysis.py --out results/smoke.json
```

Both must finish clean (~1 minute total; summary table + three PNGs).
Then fetch the datasets - the free tier's network is fine for this:

```bash
./scripts/fetch_datasets.sh data
ls -la data   # expect a9a, w8a, ijcnn1, gisette_scale, covtype..., + .t files
```

## 3. Switch to the benchmark machine

1. Open the **machine picker** (the compute chip in the Studio's top
   bar) -> select the largest **CPU** tier available (prefer >= 16
   vCPU / 32 GB; 32 vCPU if offered). Confirm **on-demand**.
2. In the Studio settings, set **auto-sleep** to its maximum / off for
   the duration of the run. (A running process normally counts as
   activity, but do not bet a 3-hour battery on it.)
3. Sanity-check the machine you actually got, and record it for the
   paper's hardware paragraph:

   ```bash
   nproc; free -g; python3 -c "import numpy; numpy.show_config()"
   cat /proc/cpuinfo | grep "model name" | head -1
   ```

## 4. Launch the battery

```bash
cd triangle-algorithm
mkdir -p results                      # log redirects need it to exist
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup python3 -u src/full_battery.py --data-dir data --seeds 10 \
    --parallel 12 --out results/full_battery.json \
    > results/full_battery.log 2>&1 &
tail -f results/full_battery.log     # detachable; the run survives
                                     # closing the browser tab
```

- `--parallel`: (vCPUs - 2) is a good default, capped at ~12-16;
  each worker is single-threaded BLAS and holds one dataset copy.
- Budgets: ETA and SMO each get 600 s per cell; LIBLINEAR is unbounded
  by default because the paper's Table 2 was produced that way. An
  opt-in `--liblin-cap SECONDS` runs LIBLINEAR in a child process that
  is killed at the cap and recorded as `timeout` (no model, so no
  accuracy). Do not mix capped and uncapped shards in one battery.
- Reading the log: each finished cell prints one line such as
  `[a9a C=0.1 seed=3] ETA=600.2s/timeout gap=2.1e-01 SMO=600.1s/timeout
  LIBLIN=14.3s (cell 1215s)`. A `/timeout` or `/maxiter` suffix with its
  gap is a *result*, not an error. A cell with no suffix converged.
- Progress: every finished cell writes a shard to
  `results/full_battery.json.shards/<dataset>_<C>_<seed>.json`. Count
  shards: `ls results/full_battery.json.shards | wc -l` (target
  5 x 3 x 5 = 75). Cells run in dataset order (a9a, w8a, ijcnn1,
  gisette, covtype); the pool hands each worker one seed's three C
  cells at a time, so a slow gisette C=10 cell holds up the C=0.1 cell
  queued behind it in the same chunk.
- Interim analysis any time: `python3 src/battery_analysis.py --out
  results/full_battery.json` (reads shards even before the combined
  JSON exists).
- Resume: rerunning the same command skips finished shards. To run
  only missing cells of one dataset, restrict with
  `--datasets covtype --Cs 0.1` and a `--parallel` equal to the number
  of missing cells.

Then the regime battery (hard-margin / kernel cells, ETA's own regime;
1-3 h at `--parallel 8`). Smoke-test it first on the free tier -
`python3 src/regime_battery.py --datasets synthetic-sep synthetic
--seeds 2 --parallel 2 --max-n-kernel 1200 --out results/rsmoke.json` -
then:

```bash
OMP_NUM_THREADS=1 nohup python3 -u src/regime_battery.py --data-dir data \
    --seeds 5 --parallel 8 --out results/regime_battery.json \
    > results/regime_battery.log 2>&1 &
```

Its log lines list the cells per dataset x seed
(`FEAS:TA-I=2s LIN:ETA=41s LIN:SMO=... KHM:... KL2:...`); a dataset
with `FEAS:TA-I=.../intersect` is not linearly separable and simply has
no LIN cells - that is a reported finding. Kernel cells use a 10k
class-balanced subsample (`--max-n-kernel`). The slow column is
LIBSVM at C=1e6 on RBF; it has a 2M-iteration cap and reports `maxiter`
if it hits it. Summary: `results/regime_battery_summary.md`.

While that runs (or after), the synthetic suites on the same machine.
The consolidated benchmark supports trials in parallel; the older
suites are single-run scripts:

```bash
mkdir -p results
nohup python3 -u src/final_benchmark.py --trials 5 --parallel 7 \
    > results/final_benchmark.log 2>&1 &     # ~20 min; shards under
                                             # results/final_benchmark.json.shards/
for s in experiments solver_comparison soft_experiment l1_experiment \
         kernel_experiment shrink_benchmark block_benchmark; do
  echo "=== $s ==="; python3 -u src/$s.py > results/$s.rerun.log 2>&1
done   # sequential, ~1-2 h; run under nohup/tmux to close the tab
```

`final_benchmark.py` resumes per (bench, trial) and refuses to count a
pair as done unless all of its solvers are present (a partial pair
from an interrupted sequential run is dropped and rerun). Keep
`--parallel` at 7 or so: the numbers go into the paper as "no
concurrent load", and 14 workers already cost ~1.5x on gisette.

Numbers from a *shared* development container do not transfer to the
Studio: the original single run of `final_benchmark.py` in the sandbox
showed the solver 1.6-2.5x ahead on six of seven classes; on the
dedicated Studio, averaged over five trials, it is 1.2-1.3x ahead on
the four hard-margin classes, tied on the synthetic L2 margin, and
1.7-2.7x behind on nu-SVM and MNIST. Only Studio numbers go in the
paper.

## 5. ThunderSVM leg (GPU)

1. Switch the machine picker to **T4** (L4/A10G if T4 unavailable).
2. Install - try the wheel first, fall back to source:

   ```bash
   pip install thundersvm || {
     git clone https://github.com/Xtra-Computing/thundersvm &&
     cd thundersvm && mkdir build && cd build && cmake .. && make -j &&
     cd ../python && pip install -e . ; }
   python3 -c "from thundersvm import SVC; print('thundersvm ok')"
   ```

3. Run the RBF cells with ThunderSVM added as a solver column (see
   `docs/experiments_protocol.md` section 5 for tolerances), then switch
   the machine back.

## 6. Retrieve results and wind down

```bash
git add results/ && git commit -m "full battery results (<machine tier>)"
git push
```

(or download `results/` through the Studio file panel). Then switch the
Studio back to the **free CPU tier** so billing stops - the filesystem
persists, so nothing is lost.

## Troubleshooting

- **Run vanished with no error in the log**: the Studio went to sleep.
  `nohup` does not survive it. Check `pgrep -fa <script>`; if nothing
  is running, turn auto-sleep off and relaunch - shards make the
  relaunch cheap.
- **Battery seems stalled**: check `tail results/full_battery.log`. A
  cell legitimately takes up to ~21 min when both ETA and SMO hit their
  600 s caps, and a gisette cell at C >= 1 takes 65-90 min because of
  LIBLINEAR at tolerance 1e-6. With `--parallel 14` the first 14 lines
  appear together after ~20 min; that is normal.
- **Many `/timeout` lines**: expected on the dense-support datasets; the
  achieved gaps are recorded and reported. Only an ETA timeout on
  gisette or on regime-battery LIN/KHM cells is worth a second look.
- **A benchmark JSON has fewer rows for one solver than another**: an
  interrupted run left a partial (bench, trial) pair. The current
  `final_benchmark.py` drops and reruns it; older JSONs need the pair
  removed by hand.
- **Out of memory** (unlikely below 100k x 5000): lower `--parallel` -
  each worker holds one dataset copy; kernel cells hold up to ~0.6 GB
  of cached columns each.
- **ThunderSVM wheel fails to import** (CUDA mismatch): use the source
  build above; it compiles against the Studio's CUDA toolkit.
- **Machine switch mid-run**: shards survive; just relaunch step 4.


## Journal provenance rerun (2026-09-08)

The journal review (docs/aor_review_2026-09-08.md, item 1) asks for one
protocol: every table under the released code, isolated. `src/provenance_rerun.sh`
runs, in order, the synthetic schedule check and k ablation (Tables 3, 8),
the new n-scaling experiment (`src/n_scaling.py`), the new real-data
screening experiment (`src/screening_real.py`, gisette LIN and L2 at
1e-3 and 1e-5, screening off/on, 3 seeds, each run in a fresh process),
the uncapped ETA-only seed-0 traces for Figure 1 and Table 6
(`full_battery.py --time-cap 0 --solvers ETA`, a9a/w8a/ijcnn1 at C=0.1),
the full L2 battery (Table 2, `results/full_battery2.json`), then
`battery_analysis.py`, `time_to_tol.py` and `trace_fig.py`.

```bash
cd ~/triangle-algorithm-sv && git pull
pgrep -af "src/.*\.py"            # must be empty
nohup bash src/provenance_rerun.sh > results/provenance_rerun.log 2>&1 &
# PAR=4 nohup bash src/provenance_rerun.sh ...   # 4 concurrent cells for Table 2 (~8 h instead of ~30 h)
```

Budget: synthetic tables ~5 min; n-scaling ~15-30 min; screening ~1 h;
uncapped traces ~3 h (a9a 3,000 s, w8a 5,400 s, ijcnn1 ~700 s); L2 battery
~30 h sequential (ETA and SMO 600 s caps on 12 cells x 5 seeds, LIBLINEAR
8 min-1.5 h per gisette cell). Everything resumes from shards, so the
script can be restarted after a Studio sleep. Sync afterwards:

```bash
git add results/schedule_check.json results/k_ablation_synth.json results/n_scaling.json \
        results/screening_real.json results/full_battery2.json results/full_battery2_summary.md \
        results/time_to_tol.md results/*.log paper/aor/tab_synth.tex paper/figs/fig_realdata_trace.pdf
git commit -m "results: provenance rerun" && git push origin main
```

Do not add the shard directories or the uncapped shards (they carry the
full traces) unless they are small; `results/full_battery2.json` has the
merged records.
