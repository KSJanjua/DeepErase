# gpu/ — running DeepErase on the GPU box

Since 13 Oct 2026 the project runs on an **NVIDIA H100 with a 40 GB MIG
slice**, reached through Jupyter Lab, whose terminal starts in `/workspace`.
Code reaches the box with `git pull`. Results come back with `git push` from
the box, and are fetched on the laptop for analysis.

```
/workspace/Capstone_Project/
├── DeepErase/          the git clone (branch rishabh-development)
├── .venv/              this project's own Python 3.11 and packages
├── .tools/             uv (the installer), the Python it downloaded, its cache
├── hf_cache/           model checkpoints and the TOFU dataset
└── .git-credentials    the GitHub token for pushing, saved at the first push
```

Nothing is installed system-wide, so another project on the same machine is
not affected, and deleting `/workspace/Capstone_Project` removes everything.
**Do not run another project's GPU job on the same MIG slice while a study
runs.** The slice's 40 GB are shared by every process on it, and NPO needs
most of them.

## First-time setup (once)

**1. Clone the repository**

```bash
mkdir -p /workspace/Capstone_Project
cd /workspace/Capstone_Project
git clone -b rishabh-development https://github.com/KSJanjua/DeepErase.git
cd DeepErase
```

**2. Build the environment (~10–15 min, mostly downloads)**

```bash
bash gpu/setup_env.sh
```

It installs uv, Python 3.11, torch 2.6.0 (CUDA build) and the exact versions in
`requirements.txt`, all under `/workspace/Capstone_Project`. It then checks the
GPU from that environment, downloads TOFU's JSON files for the tests, and
caches the two 1B checkpoints. Re-running it skips what is already done.

**3. Activate it and check everything (~1 min)**

```bash
source gpu/env.sh
python gpu/bootstrap.py --run-tests
```

Expect `READY`, every test passing, and both memory plans fitting.

**4. Let the box push to GitHub**

On github.com: *Settings → Developer settings → Personal access tokens →
Tokens (classic) → Generate new token (classic)*. Name it, choose an
expiration (90 days covers the project), tick **only `public_repo`**, and copy
the token. Then, in the clone:

```bash
git config user.name  "<your GitHub name>"
git config user.email "<your GitHub email>"
git config credential.helper "store --file=/workspace/Capstone_Project/.git-credentials"
git config pull.rebase true
git config rebase.autoStash true
```

These settings apply to this clone only. At the first `git push`, git asks for
a username (your GitHub username) and a password (paste the token). It saves
them in `.git-credentials`, readable only by this account, and later pushes do
not ask. Revoke the token on GitHub when the project ends.

## Every session

```bash
source /workspace/Capstone_Project/DeepErase/gpu/env.sh
git pull
python gpu/bootstrap.py
```

`env.sh` activates the environment, points the model cache at
`/workspace/Capstone_Project/hf_cache`, switches Hugging Face to offline mode
(everything the runs read is cached, so a Hub outage cannot stop a job) and
moves into the clone. It affects only
that terminal, so run it in every new one. Pull new code between jobs, not
while one runs: `replicate.py` starts each run as a fresh process, which would
pick up the new code mid-matrix.

## Long jobs

Start them in the background, writing to a log, so they survive a closed
browser tab or a dropped connection:

```bash
nohup python gpu/replicate.py --methods ga --seeds 0 --learning-rate 1e-6 \
      > results/logs/<name>.log 2>&1 &
tail -f results/logs/<name>.log        # Ctrl+C stops watching; the job keeps running
```

```bash
pgrep -af "run_study|replicate"        # is it still running?
pkill -f gpu/replicate.py; pkill -f deeperase.scripts.run_study   # stop it
```

A stopped run resumes from its last finished α when the same command is run
again.

## Sending results back

From the clone, after a job has finished:

```bash
git add results
git commit -m "<what ran>"
git pull
git push
```

`git add results` adds only result files: run directories, logs, the manifest.
Model checkpoints never enter the repository (they live in `hf_cache/`, outside
it). A protocol-v2 run adds about 5 MB, which git compresses to about 1 MB.

## The protocol-v2 run plan

Protocol v2 is the corrected design (`docs/PROTOCOL_V2.md`). What each step
tests and the decision rules are fixed in
[`docs/ANALYSIS_PLAN.md`](../docs/ANALYSIS_PLAN.md). Run the steps in this
order. Each is resumable, and nothing later is worth running if an earlier step
fails.

**0. Smoke run (~10 min)**: the whole pipeline on the real checkpoints, with 2
training epochs, 20+20 depth examples and α ∈ {0, 1}. It checks the box end to
end and times each stage. Run ids starting with `smoke_` are never read as
results.

The first attempt (13 Oct, `smoke_v2_ga_h100`, lr 1e-6) stopped in training: in
float32, GA at v1's rate lost two thirds of its utility in the first epoch
(0.805 → 0.258). That is what step 1 is for. The smoke run uses a rate small
enough to reach the sweep.

**1. Learning-rate scans, one per method (~2–3 h for all three)**

`scripts/lr_scan.py` runs the study's exact training at several rates with one
model load, records utility and forget-set NLL after every epoch, and stops a
rate once utility halves. It reports the epoch the study would keep (≥ 95% of
utility) and recommends the rate whose kept epoch is nearest mid-budget.

The smoke run and all three scans, one after another, in the background:

```bash
nohup sh -c '
python -m deeperase.scripts.run_study --protocol v2 --method ga --seed 0 \
       --run-id smoke_v2_ga_h100_lr2e-8 --alphas 0,1 --epochs 2 --learning-rate 2e-8 \
       --min-utility-ratio 0.5 --n-examples 20 --n-heldout-examples 20
python -m deeperase.scripts.lr_scan --method ga       --lrs 1e-7,5e-8,2e-8,1e-8,5e-9
python -m deeperase.scripts.lr_scan --method graddiff --lrs 3e-7,1e-7,5e-8,2e-8,1e-8
python -m deeperase.scripts.lr_scan --method npo      --lrs 1e-6,3e-7,1e-7,3e-8,1e-8
' > results/logs/scan_v2_h100.log 2>&1 &
tail -f results/logs/scan_v2_h100.log
```

Results: `results/lr_scans/<method>_1B_v2_s0.json`. If a method has no usable
rate, scan again below its slowest "too fast" rate (or above its fastest "too
slow" one); a new scan adds to the same file.

**1b. One full study per method at the scanned rate (~1 h each)**

The 13 Oct scans chose GA 1e-8, GradDiff 5e-8 (provisional) and NPO 1e-8
(`RESULTS.md` §12.2). All three, one after another:

```bash
nohup sh -c '
python gpu/replicate.py --methods ga       --seeds 0 --learning-rate 1e-8
python gpu/replicate.py --methods graddiff --seeds 0 --learning-rate 5e-8
python gpu/replicate.py --methods npo      --seeds 0 --learning-rate 1e-8
python gpu/summarize.py --glob "study_v2*"
' > results/logs/step1b_v2_h100.log 2>&1 &
tail -f results/logs/step1b_v2_h100.log
```

The scan predicts the kept epoch; this checks the rest. In `summarize`, a kept
(`sel`) epoch in the middle of the 20 and a depth span ≥ 0.15 mean the rate is
usable. Each rate gets its own run id, so nothing is overwritten.

**2 + 3. Matched-damage controls, then seeds 1 and 2 (~9 h, one overnight job)** *(done 14–15 Oct; `RESULTS.md` §14)*

Seed 0 of every method was run in step 1b. This job adds one matched-damage T1
control per method, then seeds 1 and 2 of each method:

```bash
nohup sh -c '
bash gpu/matched_controls.sh ga:1e-8 graddiff:5e-8 npo:1e-8
python gpu/replicate.py --methods ga       --seeds 1 2 --learning-rate 1e-8
python gpu/replicate.py --methods graddiff --seeds 1 2 --learning-rate 5e-8
python gpu/replicate.py --methods npo      --seeds 1 2 --learning-rate 1e-8
python gpu/summarize.py --glob "*_v2_*"
' > results/logs/step2_v2_h100.log 2>&1 &
tail -f results/logs/step2_v2_h100.log
```

`gpu/matched_controls.sh` runs, for each method, `scripts/control_probe.py`
(~25 min): it trains as the study does and measures utility along the
control's random direction at nine scales. It then runs the full control
study at the scale whose damage at α = 1 matches the real run's (~50 min).
A random step of the update's own size does nothing (v1, `RESULTS.md` §7), so
only the matched-*damage* control can show whether damage alone raises depth
and breadth. Under v2 each seed also draws its own per-author trained/held-out
split, so seeds are genuine replicates.

**4. Optional sharpening runs (~6 h, one job).** *Done 15–16 Oct (5 h 42 min);
results in `RESULTS.md` §15. Nothing further is planned on the GPU.* These
cover what §14 left open. Matched-damage controls for GradDiff seeds 1–2 test whether "silence,
not erasure" holds beyond seed 0. An α extension to 2.0 for GA and GradDiff
(seed 0, tagged `ext`) asks whether GradDiff's silenced held-out facts ever
become erased. The loss-scope ablation A1 is the third. Tagged runs are
exploratory, never pooled into the main matrix (`deeperase/registry.py`).

```bash
nohup sh -c '
SEED=1 bash gpu/matched_controls.sh graddiff:5e-8
SEED=2 bash gpu/matched_controls.sh graddiff:5e-8
python gpu/replicate.py --methods graddiff --seeds 0 --learning-rate 5e-8 --tag ext --extra --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0
python gpu/replicate.py --methods ga       --seeds 0 --learning-rate 1e-8 --tag ext --extra --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,1.75,2.0
python gpu/replicate.py --methods ga       --seeds 0 --learning-rate 1e-8 --tag lossfull --extra --loss-scope full
python gpu/summarize.py --glob "*_v2_*"
' > results/logs/step4_v2_h100.log 2>&1 &
tail -f results/logs/step4_v2_h100.log
```

**5. STRANGER, our own method (from 17 Oct; `docs/STRANGER.md`, pre-registered
as H7 in `docs/ANALYSIS_PLAN.md`).** It runs in gated steps; each needs the
previous one's result.

*5a. Gate 1, the virtual-source ceiling (~10 min, inference only, ~12 GB).* It
measures the depth STRANGER's targets could reach before any training.

```bash
python -m deeperase.scripts.virtual_source --seed 0 > results/logs/stranger_g1_s0.log 2>&1
cat results/gates/stranger_g1_s0.md
```

The last line prints the pre-registered decision: GO, WEAK GO or NO-GO.
Training steps (5b onwards) follow once gate 1 is in.

**3b. Loss-scope ablation, GA seed 0 (~1 h)** *(included in step 4)*: v1's full-sequence loss under
v2's design. The tag gives it its own directory and keeps it out of the main
matrix's averages.

```bash
python gpu/replicate.py --methods ga --seeds 0 --learning-rate <GA lr> --tag lossfull \
       --extra --loss-scope full
```

**3c. α extension (~20 min per run)**, only for runs still above the utility
floor at α = 1. Rerunning the same command with more α values resumes the run
and adds the points:

```bash
python gpu/replicate.py --methods ga --seeds 0 1 2 --learning-rate <GA lr> \
       --extra --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.25,1.5,2.0
```

`replicate.py` treats a run as done only when it already has every requested
α, so this extends finished runs. The resume retrains the unlearned model
before sweeping the new α values. GPU training need not be bit-reproducible, so
`unlearn.json` records the relative drift of the retrained update
(`resumed_from.relative_drift`), and the log warns above 1e-3.

The priority order if GPU time runs short, and the GPU-hour budget, are in
[`docs/ANALYSIS_PLAN.md`](../docs/ANALYSIS_PLAN.md) §7.

**4. Push the results** (see "Sending results back" above). The analysis then
runs locally, without a GPU:

```bash
python gpu/summarize.py --valid-only                       # one line per run and trajectory
python -m deeperase.scripts.aggregate_matrix --glob 'study_v2_*' \
       --out results/analysis/matrix_v2                    # tables, slopes, figure
python -m deeperase.scripts.fact_level_analysis --glob 'study_v2_*' \
       --out results/analysis/fact_level_v2                # per-fact depth vs behaviour
python show_results.py --run <run id>                       # one run, in detail
```

`fact_level_analysis` is the central analysis of v2 (`docs/ANALYSIS_PLAN.md`
H2, H3). For each fact it joins the fact's UDS with its own forced-choice items,
and reports the suppression rate, the depth–behaviour correlation across facts
with author-level bootstrap CIs, the indirect leak and the order of crossing.

`aggregate_matrix` gives mean ± sd across seeds at every α and per-run slopes
of depth, breadth and each breadth tier against α. The tier slopes are the test
of UIPE's prediction: breadth should rise faster on B2/B4 and on held-out facts
than on B0.

Already done without the GPU, on a laptop: the τ sensitivity
(`scripts/tau_sensitivity.py`), the breadth-scorer effect
(`scripts/breadth_boundary_effect.py`) and the T0 cross-check against the
authors' code (`scripts/cross_validate_uds.py`). All three are inference-only on
the 1B checkpoints and run on Apple-silicon `--device mps`.

## Memory

| Job | 1B bf16 (v1) | 1B fp32 (v2) | 3B fp32 (v2) |
|---|---|---|---|
| Measurement only, per model | ~2.5 GB | ~5 GB | ~13 GB |
| GA / GradDiff training (weights + grads + 2 AdamW moments + activations) | ~11 GB | ~21 GB | ~52 GB |
| NPO (adds a frozen reference model) | ~14 GB | ~26 GB | ~65 GB |

The planner keeps a 30% margin against **free** memory. On the 40 GB MIG slice
(about 39 GB free when nothing else uses it) every 1B v2 job fits, NPO with
about 3.7 GB to spare; with less than ~37 GB free, NPO is refused before it
starts. 3B float32 training does not fit in 40 GB, so a 3B run needs
`--dtype bfloat16`, recorded in its config. `replicate.py` runs the same memory plan against free VRAM
before each run. It cannot stop a mid-run OOM, which is why runs are
resumable.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by both scripts.
Fragmentation is the usual cause of an OOM that "should" have fitted, and it is
worse when another process owns part of the card. Float32 runs also force
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` and `NVIDIA_TF32_OVERRIDE=0`: some
containers set them to 1, which would quietly run float32 matmuls in TF32.
`gpu/setup_env.sh` checks the GPU's float32 matmul against float64 on the CPU.

## Seeds and sampling

`unlearn()` calls `torch.manual_seed`, but this pipeline does not shuffle and
Llama-3.2 has dropout 0. Under v1 with `--sampling even`, changing the seed
produces a **bit-identical** run, and `replicate.py` refuses multi-seed
matrices there. Under v2 the seed also draws the per-author split.

`replicate.py` merges `results/studies/replication_manifest.json` across
invocations (it used to overwrite it). It records the git commit for each batch
and warns if the working tree is dirty.


## Earlier setups (historical)

**Aug–Sep 2026: A100 80 GB in an NVIDIA NGC container.** The environment was a
`--system-site-packages` venv at `/workspace/Capstone_Project/.venv` that
inherited the container's tuned PyTorch, with `requirements-gpu.txt` (which
deliberately leaves torch alone) on top. Results came back as a tarball of
each run's small JSON files. The H100 setup above replaces both: its venv is
fully isolated, with its own torch, and results come back through git.

Two rules from that period still hold:

* **Run from the repository root.** `--reference-spans` defaults to
  `reference_uds/tofu_data/forget10_filtered.json`, relative to the working
  directory. From anywhere else the authors' annotations are not found.
  `gpu/env.sh` changes into the clone for this reason.
* **Never install `requirements.txt` into a container's own Python.** In an
  isolated venv it is exactly right (that is what `gpu/setup_env.sh` does). Into
  a container's system Python, it would replace the container's CUDA build of
  torch.
