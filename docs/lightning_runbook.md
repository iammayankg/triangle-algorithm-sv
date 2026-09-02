# Lightning.ai Runbook

Step-by-step instructions for gathering the full-scale results on
Lightning AI Studios. Companion to `docs/experiments_protocol.md`
(which holds the scientific protocol: seeds, tolerances, what to
report); this file is purely operational. UI element names are as of
September 2026 - if a button has moved, the concept still applies.

## 0. What you will run, and on what

| leg | machine tier | duration | cost ballpark |
|--|--|--|--|
| setup + smoke test + dataset fetch | free CPU tier | ~30 min | free |
| CPU battery (`full_battery.py`) | largest CPU tier, >= 16 vCPU / 32 GB | 2-4 h at `--parallel 12` | a few $ |
| synthetic suites re-run | same CPU tier | 1-2 h parallelised | a few $ |
| ThunderSVM leg | T4 GPU | 1-2 h incl. install | ~$1 |

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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup python3 -u src/full_battery.py --data-dir data --seeds 10 \
    --parallel 12 --out results/full_battery.json \
    > results/full_battery.log 2>&1 &
tail -f results/full_battery.log     # detachable; the run survives
                                     # closing the browser tab
```

- `--parallel`: (vCPUs - 2) is a good default, capped at ~12-16;
  each worker is single-threaded BLAS.
- Progress: every finished cell prints one line and writes a shard to
  `results/full_battery.json.shards/`. Count shards to see progress:
  `ls results/full_battery.json.shards | wc -l` (target:
  5 datasets x 3 Cs x 10 seeds = 150).
- Interim analysis any time: `python3 src/battery_analysis.py --out
  results/full_battery.json` (reads shards even before the combined
  JSON exists).
- If the Studio restarts or you stop the run: rerun the same command -
  finished shards are skipped automatically.

Then the regime battery (hard-margin / kernel cells, ETA's own regime;
1-3 h at `--parallel 8`):

```bash
OMP_NUM_THREADS=1 nohup python3 -u src/regime_battery.py --data-dir data \
    --seeds 5 --parallel 8 --out results/regime_battery.json \
    > results/regime_battery.log 2>&1 &
```

While that runs (or after), the synthetic suites on the same machine:

```bash
for s in experiments solver_comparison soft_experiment l1_experiment \
         kernel_experiment shrink_benchmark block_benchmark \
         final_benchmark; do
  nohup python3 -u src/$s.py > results/$s.rerun.log 2>&1
done   # sequential; parallelise by backgrounding groups if cores allow
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

- **Battery seems stalled**: check `tail results/full_battery.log`; an
  a9a/covtype cell at C=10 can legitimately run ~10-20 min. SMO cells
  cap at 600 s by design.
- **Out of memory** (unlikely below 100k x 5000): lower `--parallel` -
  each worker holds one dataset copy.
- **ThunderSVM wheel fails to import** (CUDA mismatch): use the source
  build above; it compiles against the Studio's CUDA toolkit.
- **Machine switch mid-run**: shards survive; just relaunch step 4.
