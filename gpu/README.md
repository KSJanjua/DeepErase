# gpu/ — running DeepErase on the university A100

Scripts for a **shared** GPU reached through Jupyter Lab. They assume you can
run `.py` files (from a notebook cell with `!python …`, or a terminal).

## Session checklist

```bash
cd <repo root>                     # the dir containing deeperase/ and tests/
python gpu/bootstrap.py --size 1B --run-tests    # once per environment
python gpu/bootstrap.py --size 1B                # every session (fast)
```

`bootstrap.py` exits non-zero if anything would stop a run, so it can gate a
pipeline. It reports **free** VRAM, not total — on a shared card that is the
only number that matters, and it changes between sessions.

## Two things that will bite you

**1. Run from the repository root.** `DEFAULT_REFERENCE_PATH` resolves
`reference_uds/tofu_data/forget10_filtered.json` relative to the working
directory. From anywhere else `--reference-spans` silently fails to find the
authors' annotations and you fall back to a heuristic that agrees with them on
12% of examples.

**2. Do not `pip install -r requirements.txt` here.** That pins a **CPU** torch
wheel and will replace the container's CUDA build. Use:

```bash
pip install -r requirements-gpu.txt     # deliberately does not touch torch
```

## The protocol-v2 run plan (from 9 Oct 2026)

Protocol v2 is the corrected design (`docs/PROTOCOL_V2.md`). Every run made
before 9 Oct is v1 and stays in `results/studies/` as history. Run the steps in
this order. Each is resumable, and nothing later is worth running if an
earlier step fails.

**0. Environment, once per session**

```bash
git pull                                            # rishabh-development
python gpu/bootstrap.py --size 1B --protocol v2 --run-tests
```

Expect the full suite to pass (677 passed, 2 skipped at the time of writing) and both
memory plans to fit. NPO is the tightest: float32 weights, AdamW state and a
frozen reference model.

**1. Learning-rate calibration, one seed per method (~1 h each)**

float32 training moves every weight, unlike the bf16 v1 runs. The v1 learning
rates (GA 1e-6, GradDiff 3e-6) are therefore starting points, not settings.
NPO needs a fresh scan in any case: its v1 runs used the defective loss.

```bash
python gpu/replicate.py --methods ga       --seeds 0 --learning-rate 1e-6 --epochs 20
python gpu/replicate.py --methods graddiff --seeds 0 --learning-rate 3e-6 --epochs 20
python gpu/replicate.py --methods npo      --seeds 0 --learning-rate 1e-5 --epochs 20
python gpu/summarize.py --glob 'study_v2*'
```

Read the `sel` column. 0 means the lr is too high; 19 means too low. A middle
epoch with a depth span ≥ 0.15 is usable. Adjust the lr and repeat for any
method that is off. Each lr gets its own run id, so nothing is overwritten.

**2. The matrix: 3 methods × 3 seeds at the chosen rates (~6 h)**

```bash
python gpu/replicate.py --methods ga       --seeds 0 1 2 --learning-rate <GA lr>
python gpu/replicate.py --methods graddiff --seeds 0 1 2 --learning-rate <GD lr>
python gpu/replicate.py --methods npo      --seeds 0 1 2 --learning-rate <NPO lr>
```

Seed 0 resumes from step 1. Under v2 the seed also changes the per-author
trained/held-out split, so seeds are genuine replicates.

**3. The T1 control ladder, GA seed 0 (~3–4 h)**

```bash
python gpu/replicate.py --methods ga --seeds 0 --learning-rate <GA lr> --control random
python gpu/replicate.py --methods ga --seeds 0 --learning-rate <GA lr> --control random --control-scale 3
python gpu/replicate.py --methods ga --seeds 0 --learning-rate <GA lr> --control random --control-scale 10
```

Scale 1 is the matched-magnitude control. Raise the scale until the control's
utility drop matches the real run's; that gives the matched-damage control.
Float32 means the random direction is no longer rounded away (`RESULTS.md` §7).

**3b. Loss-scope ablation, GA seed 0 (~1 h)**: v1's full-sequence loss under
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

**4. Send back** the small files (see the end of this page), then, locally
(no GPU needed):

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

### Seeds and sampling

`unlearn()` calls `torch.manual_seed`, but this pipeline does not shuffle and
Llama-3.2 has dropout 0. Under v1 with `--sampling even`, changing the seed
produces a **bit-identical** run, and `replicate.py` refuses multi-seed
matrices there. Under v2 the seed also draws the per-author split.

`replicate.py` merges `results/studies/replication_manifest.json` across
invocations (it used to overwrite it). It records the git commit for each batch
and warns if the working tree is dirty.

## Memory

| Job | 1B bf16 (v1) | 1B fp32 (v2) | 3B fp32 (v2) |
|---|---|---|---|
| Measurement only, per model | ~2.5 GB | ~5 GB | ~13 GB |
| GA / GradDiff training (weights + grads + 2 AdamW moments + activations) | ~11 GB | ~21 GB | ~52 GB |
| NPO (adds a frozen reference model) | ~14 GB | ~26 GB | ~65 GB |

The planner keeps a 30% margin against **free** memory. With ~50 GB free, all
1B v2 jobs fit; 3B float32 training does not, so run 3B under `--dtype bfloat16`
and record that. `replicate.py` runs the same memory plan against free VRAM
before each run. It cannot stop a mid-run OOM, which is why runs are
resumable.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by both scripts.
Fragmentation is the usual cause of an OOM that "should" have fitted, and it is
worse when another process owns part of the card. Float32 runs also force
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`: the NGC container sets it to 1, which
would quietly run float32 matmuls in TF32.

## Long runs

Jupyter kernels die when the browser tab closes. For anything over an hour:

```bash
nohup python gpu/replicate.py --methods ga --seeds 0 1 2 --learning-rate 1e-6 \
      > results/logs/replication.log 2>&1 &
tail -f results/logs/replication.log
```

## What to send back for analysis

The run directories are self-describing. The small files are enough:

```bash
tar czf handoff.tgz \
    results/studies/*/config.json \
    results/studies/*/plane.json \
    results/studies/*/sensitivity.json \
    results/studies/*/unlearn.json \
    results/studies/*/split.json \
    results/studies/*/calibration.json \
    results/studies/replication_manifest.json
```

For protocol-v2 runs, **also send `partial/`**. It holds the per-item margins
and per-example depth deltas that the offline analyses use (a few MB per run).
For v1 runs, skip `partial/` and `stage1_cache.json` unless something looks
wrong.
