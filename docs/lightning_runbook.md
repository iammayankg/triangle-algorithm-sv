# Lightning.ai Runbook

Step-by-step instructions for gathering the full-scale results on
Lightning AI Studios. Companion to `docs/experiments_protocol.md`
(which holds the scientific protocol: seeds, tolerances, what to
report); this file is purely operational. UI element names are as of
September 2026 - if a button has moved, the concept still applies.

## Tonight's run (OPT 2026 sprint, Sep 2-4)

Exact sequence for the 32-vCPU Studio (Intel Xeon 2.60 GHz, 122 GB, as
recorded in `machine.txt`). Both batteries run concurrently; everything
should be finished by morning.

```bash
# 0. on the Mac: push main so the Studio can pull the capped battery +
#    regime battery + this runbook
git push origin main

# 1. on the Studio: stop the old uncapped run and get current
pkill -f full_battery.py ; sleep 2
cd ~/triangle-algorithm-sv && git pull

# 2. preserve the uncapped a9a/w8a shards (already committed in git, but
#    keep them on disk under their own name) and start the capped run clean
mv results/full_battery.json.shards results/full_battery_uncapped.shards 2>/dev/null
rm -f results/full_battery.json

# 3. launch both batteries (26 single-threaded workers on 32 vCPUs)
mkdir -p results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup python3 -u src/full_battery.py --data-dir data --seeds 5 \
    --parallel 14 --out results/full_battery.json \
    > results/full_battery.log 2>&1 &
nohup python3 -u src/regime_battery.py --data-dir data --seeds 5 \
    --parallel 12 --out results/regime_battery.json \
    > results/regime_battery.log 2>&1 &

# 4. confirm both are alive, then close the tab
sleep 60; pgrep -fc "full_battery.py|regime_battery.py"   # expect ~28 (workers + parents)
tail -3 results/full_battery.log results/regime_battery.log
```

Why these settings: 5 seeds (CIs suffice for a workshop; 10 is for the
journal), 600 s caps per solver (a9a converges uncapped in ~3,030 s -
that number is preserved in the uncapped shards), 14 + 12 workers leave
6 vCPUs for the OS and the Studio. Expected: L2 battery 2-3 h (75 cells,
worst ~21 min each), regime battery 1-3 h.

**Morning (Sep 3):**

```bash
ls results/full_battery.json.shards | wc -l      # target 75
ls results/regime_battery.json.shards | wc -l    # target 25
python3 src/battery_analysis.py --out results/full_battery.json
python3 src/regime_battery.py --data-dir data --seeds 5 --parallel 1 \
    --out results/regime_battery.json           # merges shards -> summary
git add results/ && git commit -m "OPT sprint: capped L2 battery + regime battery (32-vCPU Xeon)"
git push origin main
```

If a battery is still running in the morning, leave it: the analysis
scripts read finished shards, and the paper draft takes whatever is
done by Thursday afternoon.

**In parallel, check the OPT 2026 CFP on OpenReview**
(https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/OPT) for the
three things the workshop site does not state: page limit, whether
submissions are anonymised, and the required style file (expect the
NeurIPS 2026 template). Deadline: **Sep 4, 2026, AoE**.

## 0. What you will run, and on what

| leg | machine tier | duration | cost ballpark |
|--|--|--|--|
| setup + smoke test + dataset fetch | free CPU tier | ~30 min | free |
| L2 battery (`full_battery.py`) | largest CPU tier, >= 16 vCPU / 32 GB | ~4-5 h at `--parallel 12` (600 s caps) | a few $ |
| regime battery (`regime_battery.py`) | same CPU tier | 1-3 h at `--parallel 8` | a few $ |
| synthetic suites re-run | same CPU tier | 1-2 h parallelised | a few $ |
| ThunderSVM leg | T4 GPU | 1-2 h incl. install | ~$1 |

**What to expect from the L2 battery** (do not mistake it for a
problem): on the heavily overlapping datasets (a9a, w8a, ijcnn1,
covtype) 60-65% of points are support vectors, which is the dense-support
regime where the geometric solver is O(n^2)-type (the paper's Theorem 6
predicts it). ETA and SMO will hit their 600 s caps and report
`timeout` with their achieved certified gap, while LIBLINEAR finishes in
seconds. Those cells are the honest half of the regime story; the
regime battery supplies the other half. Gisette (d = 5000) is the L2
dataset most likely to converge for ETA.

Use **on-demand** machines for all timed runs (never interruptible: an
interruption invalidates in-flight cells, and although shards make
resume cheap, timing statistics should come from uninterrupted runs).

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
  (it finishes in seconds to a minute). Worst case per cell is therefore
  ~21 min, so 150 cells at `--parallel 12` is ~4.5 h.
- Reading the log: each finished cell prints one line such as
  `[a9a C=0.1 seed=3] ETA=600.2s/timeout gap=2.1e-01 SMO=600.1s/timeout
  LIBLIN=14.3s (cell 1215s)`. A `/timeout` or `/maxiter` suffix with its
  gap is a *result*, not an error - see "What to expect" above. A cell
  with no suffix converged.
- Progress: every finished cell also writes a shard to
  `results/full_battery.json.shards/`. Count shards to see progress:
  `ls results/full_battery.json.shards | wc -l` (target:
  5 datasets x 3 Cs x 10 seeds = 150). Cells run in dataset order
  (a9a, w8a, ijcnn1, gisette, covtype), all seeds and Cs per dataset.
- Interim analysis any time: `python3 src/battery_analysis.py --out
  results/full_battery.json` (reads shards even before the combined
  JSON exists).
- If the Studio restarts or you stop the run: rerun the same command -
  finished shards are skipped automatically.

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

While that runs (or after), the synthetic suites on the same machine:

```bash
mkdir -p results
for s in experiments solver_comparison soft_experiment l1_experiment \
         kernel_experiment shrink_benchmark block_benchmark \
         final_benchmark; do
  echo "=== $s ==="; python3 -u src/$s.py > results/$s.rerun.log 2>&1
done   # sequential, ~1-2 h; run the loop itself under nohup/tmux if
       # you want to close the tab, or background groups if cores allow
```

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

- **Battery seems stalled**: check `tail results/full_battery.log`. A
  cell legitimately takes up to ~21 min when both ETA and SMO hit their
  600 s caps (expected on a9a/w8a/ijcnn1/covtype - see "What to
  expect"). With `--parallel 12` the first 12 lines appear together
  after ~20 min; that is normal.
- **Many `/timeout` lines**: expected on the dense-support datasets; the
  achieved gaps are recorded and reported. Only an ETA timeout on
  gisette or on regime-battery LIN/KHM cells is worth a second look.
- **Out of memory** (unlikely below 100k x 5000): lower `--parallel` -
  each worker holds one dataset copy.
- **ThunderSVM wheel fails to import** (CUDA mismatch): use the source
  build above; it compiles against the Studio's CUDA toolkit.
- **Machine switch mid-run**: shards survive; just relaunch step 4.
