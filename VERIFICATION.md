# VERIFICATION — captured output

All commands below were run from the repository root in the documented
environment (conda env `deeperase`, Python 3.11.15) and their output
pasted verbatim. Captured: 2026-08-09 21:29

> ## Latest captured run: 29 August 2026 — see section 0
>
> Sections 1 onward are the **August 2 and 9** transcripts, at 118 and then 180
> tests. They are kept for the environment and defect history. **Section 0 below
> is the latest captured suite run**, on the university A100.
>
> **Section 0 (the A100) is history.** The GPU box is now an H100 with a 40 GB
> MIG slice. Its first suite run is section 0.3 (670 passed, 16 tokenizer
> skips, 13 Oct 2026). The latest run of the current suite (775 collected) is in
> section 0.2, on macOS CPU: 772 passed, 3 skips (2 GPU-only, 1 needing the
> cached Llama tokenizer).

---

## 0. Current: 493 tests passing (29 August 2026)

Machine: `dgxanode02`, NVIDIA A100-SXM4-80GB, Ubuntu 22.04.4.
Environment: `/workspace/Capstone_Project/.venv`, a `--system-site-packages`
venv on Python 3.10.12 inheriting the NGC container's tuned PyTorch.

```
torch         2.4.0a0+f70bd71a48.nv24.06   (NGC 24.06 build, CUDA 12.5)
transformers  4.46.3
datasets      3.6.0
numpy         1.24.4
scipy         1.13.1
matplotlib    3.9.0
pytest        8.1.1
```

```
$ python gpu/bootstrap.py --size 1B --run-tests

Running 493 items in this shard
.................................................................... [ 27%]
.................................................................... [ 54%]
.................................................................... [ 82%]
.................................................................... [100%]

[  OK  ] repository root: /workspace/Capstone_Project/DeepErase
[  OK  ] reference_uds/ annotations present (needed for --reference-spans)
[ WARN ] python 3.10.12 (3.11 is the tested target)
[  OK  ] torch 2.4.0a0+f70bd71a48.nv24.06
[  OK  ] transformers 4.46.3
[  OK  ] datasets 3.6.0
[  OK  ] scipy 1.13.1
[  OK  ] matplotlib 3.9.0
[  OK  ] numpy 1.24.4
[  OK  ] pytest 8.1.1
[  OK  ] 1 CUDA device(s) visible
[  OK  ]   cuda:0 NVIDIA A100-SXM4-80GB -- 79.2 GB total, 29.5 GB in use by
           others, 49.7 GB free
[  OK  ] 98 GB free on this filesystem
[  OK  ] HF cache: /workspace/Capstone_Project/hf_cache
[  OK  ] test suite passed
READY
```

**493 collected items from 491 test functions.** The difference is one
`@pytest.mark.parametrize` in `test_unlearn.py:213` that expands a single
function across GA, GradDiff and NPO. Table 13 of the mid-semester report
counts functions (489 at submission, +2 for the GradDiff regression tests added
29 Aug).

The only warnings are intentional: `DeprecationWarning` from the deliberately
deprecated `unlearning_depth_score` shim, which the tests assert fires, plus two
upstream deprecations from torch and huggingface_hub.

Two caveats this run does **not** settle:

* `datasets` here is **3.6.0**, not the 3.1.0 the published results used. It is
  inside the `<4.0.0` ceiling and the suite passes, but **the suite never
  imports `datasets`** — `load_dataset` is reached only at real measurement
  time. Data loading is still unverified on this environment.
* Python is **3.10.12**, not the 3.11 the suite was originally verified on.
  *(The first caveat is closed by section 0.1 below.)*

---

## 0.3 The H100 box (13 October 2026)

Machine: `dgxhnode3`, NVIDIA H100 80GB HBM3, **MIG 3g.40gb** slice (39.4 GiB),
driver 535.161.08, Jupyter Lab. Environment built by `gpu/setup_env.sh`: an
isolated venv at `/workspace/Capstone_Project/.venv`, Python 3.11.17 (uv),
torch **2.6.0+cu124** (CUDA runtime 12.4), and the exact `requirements.txt` pins
(transformers 4.46.3, datasets 3.1.0, numpy 1.26.4, scipy 1.13.1, pyarrow
25.0.0). The setup's GPU check: float32 matmul matches float64 to 4.0e-07, so
TF32 is not in play.

```
$ python gpu/bootstrap.py --run-tests
[  OK  ] project environment: /workspace/Capstone_Project/.venv
[  OK  ] torch 2.6.0+cu124 (CUDA runtime 12.4)
[  OK  ]   cuda:0 NVIDIA H100 80GB HBM3 MIG 3g.40gb -- 39.4 GB total, 0.3 GB in use by others, 39.1 GB free
[  OK  ] NVIDIA driver 535.161.08
[  OK  ] memory plan for 1B GA/GradDiff ...  [FITS] peak 20.77 GB, headroom 8.61 GB
[  OK  ] memory plan for 1B NPO ...          [FITS] peak 25.72 GB, headroom 3.66 GB
670 passed, 16 skipped, 10 warnings in 202.40s (0:03:22)
READY
```

Both GPU-only tests ran and passed. All 16 skips are the tests that load
`hf-internal-testing/llama-tokenizer` (`test_breadth.py`, `test_models.py`,
`test_tofu.py`). The box downloads models from the Hub, so "no network" was
not the real cause. The fixtures caught every exception under that label.
They now report the actual error, and the next bootstrap shows it.

---

## 0.2 Local run on macOS, CPU (9 October 2026)

The GPU box was unavailable, so a CPU environment was built on the
development Mac (Apple M5, 16 GB). It uses the documented pins: Python 3.11.15
and `requirements.txt`, plus `pyarrow==25.0.0` (see the note in
`requirements.txt`).

```
torch 2.6.0 · transformers 4.46.3 · tokenizers 0.20.3 · datasets 3.1.0
numpy 1.26.4 · scipy 1.13.1 · matplotlib 3.9.2 · pytest 8.3.4 · pyarrow 25.0.0
macOS-27.0.1-arm64 · MPS available, CUDA not
```

**Before any change of 9 Oct** (code at `7feb192`):

```
$ python -m pytest tests/ -o addopts="" -rs
SKIPPED [1] tests/test_models.py:61: needs a GPU
SKIPPED [1] tests/test_models.py:137: needs a GPU
509 passed, 2 skipped in 7.03s

$ python -m deeperase.scripts.smoke_e2e
PLUMBING TEST: 31/31 checks passed in 2.4s
```

This was the first run of the 18 tests added after §0 as a suite. They pass.

**After the 9–10 Oct changes** (protocol v2, scorer fix, derived tiers,
provenance, matrix aggregator; see `docs/PROTOCOL_V2.md`), final run on 10 Oct:

```
$ python -m pytest tests/ -o addopts="" -rs
SKIPPED [1] tests/test_models.py:61: needs a GPU
SKIPPED [1] tests/test_models.py:137: needs a GPU
677 passed, 2 skipped in 18.70s        (679 collected from 654 test functions)

$ python -m deeperase.scripts.smoke_e2e
PLUMBING TEST: 31/31 checks passed in 2.9s
```

(The same suite before the fact-level analysis was added, earlier on 10 Oct:
645 passed, 2 skipped, 647 collected from 622 functions.)

**13 Oct, after the H100 setup changes** (`gpu/setup_env.sh`, `gpu/env.sh`, the
provenance and bootstrap changes below): **684 passed, 2 skipped in 17.4 s**
(686 collected from 661 functions). Added: the dirty flag ignores `results/`
and names the changed files; bootstrap recognises the project environment and
the cached checkpoints; both runners take `--cache-dir` from
`DEEPERASE_CACHE_DIR`; `smoke_*` runs are never read as measurements. Later the same day, after the
learning-rate scan (`scripts/lr_scan.py`, its early stop in `unlearn`, and the
tokenizer fixtures now reporting their real error): **699 passed, 2 skipped**
(701 collected from 671 functions). After the B3 tier was added (13 Oct):
**707 passed, 2 skipped** (709 collected from 679 functions). After the
GradDiff retain-pool fix and offline mode (13 Oct, evening): **710 passed, 2
skipped** (712 collected from 682 functions). With the matched-damage probe
(14 Oct): **715 passed, 2 skipped** (717 collected from 687 functions). With the hypothesis report and tagged-run status (15 Oct): **726 passed, 2 skipped** (728 collected from 698 functions). With controls and tagged runs kept in their own arm by the aggregators, the per-seed control check and the fully-erased share (16 Oct): **731 passed, 2 skipped** (733 collected from 703 functions). With STRANGER's foundations (name swapping, targets and loss, gate 1, overshoot diagnostics; 17 Oct): **753 passed, 2 skipped** (755 collected from 725 functions), with the real-tokenizer alignment test enabled via `DEEPERASE_CACHE_DIR`. With gate 1b (`scripts/memory_keys.py`, 17 Oct): **757 passed, 2 skipped** (759 collected from 729 functions). With UNASKED, the paraphrased-question depth split, the relearning attack (with its dose-response) and the UNASKED and relearning reports (17 Oct): **772 passed, 3 skipped** (775 collected from 744 functions). The third skip is the real-tokenizer alignment test, which runs only when `DEEPERASE_CACHE_DIR` points at the cached Llama tokenizer.
`gpu/setup_env.sh` steps 1–5 were also run on the Mac against a stand-in
`nvidia-smi` (CPU torch). They built Python 3.11.17 with the exact pins and
left nothing outside the project folder. Step 6 stopped as intended, since the
Mac has no CUDA.

New in this count:

* an end-to-end test of `run_study` under both protocols, on CPU, with tiny
  Llamas on real TOFU text and the authors' real annotations
  (`tests/test_study_e2e.py`);
* the breadth-scorer boundary regression test, mutation-checked: on the
  pre-fix code it reports identical scores for answers that differ only in
  their first token;
* answer-only loss masks; the per-author split; the derived B2/B4 items;
  TOFU truth ratio and Forget Quality; provenance and the run registry;
  resume integrity; the T0 harness;
* patching on GPT-2, OPT and GPT-NeoX (`tests/test_architectures.py`);
* the fact-level analysis (`tests/test_fact_level.py`): Spearman against
  SciPy with ties, author-level bootstrap, the four fact classes, the
  suppression rate, the indirect leak, crossing order, registry filtering and
  the refusal of v1 runs. Also an end-to-end check that a real (tiny) v2 run
  directory feeds it;
* per-fact depth resolution against the nested retain checkpoints
  (`tests/test_depth_fact_resolution.py`);
* the cross-validation flag by architecture, ablation run ids
  (`replicate.py --tag`), α-aware resume detection, and the retrain-drift
  record on resume.

**What this run does not establish:** anything about the GPU box's
environment (NGC torch 2.4, CUDA, TF32 overrides). Re-run
`python gpu/bootstrap.py --size 1B --protocol v2 --run-tests` there before the
first v2 run.

---

## 0.1 First live execution of the GradDiff arm (29 August 2026)

A deliberately small run, not a result. Its purpose was to answer two questions
the test suite structurally cannot: does the BUG-1 fix train under real data,
and does `datasets` 3.6.0 load TOFU correctly?

```
$ python -m deeperase.scripts.run_study \
    --method graddiff --size 1B --seed 0 --sampling random \
    --epochs 2 --n-examples 20 --n-breadth 50 --alphas 0.0,0.5,1.0 \
    --reference-spans reference_uds/tofu_data/forget10_filtered.json \
    --run-id smoke_graddiff
```

**Both questions answered yes.**

`datasets` 3.6.0 loads TOFU at the expected sizes — `forget10` 400 rows,
`retain90` 3600 rows, 367 reference-annotated examples resolved with
`skipped_not_found: 0`. This is the first time the data path has been exercised
on this environment, so the "data loading unverified" caveat above no longer
stands.

BUG-1 is fixed in the running system, not only in the unit tests:

```
[3b/6] Loading retain split for graddiff
  retain split retain90: 20 of 3600 examples (random)
  baseline before training: utility=0.7800 forget_nll=1.6382
  epoch 0 step 0  loss=-0.1126  forget_nll=1.7690
Unlearning finished: 10 steps in 11.8s, forget NLL 1.6382 -> 1.6599 (+0.0217)
  [OK] ||v||=0.01888, weights moved and forget-set NLL rose, as expected
```

Two independent confirmations in that trace. The retain split is drawn from
**retain90**, a different corpus from the forget set. And `loss=-0.1126` is
non-zero: under the defect the objective was `(retain_weight - 1) * NLL(forget)`,
identically **0.0000** at the default weight of 1.0, and `||v||` would have been
0. The observed value is consistent with `-NLL(forget) + NLL(retain)` for two
distinct corpora of similar difficulty.

**The measurement itself is underpowered and is reported as such by the runner,
which is the intended behaviour:**

```
UNDERPOWERED -- NO FINDING EITHER WAY (rho=+0.000).
  breadth span : 0.037 (1 of 100 items; one item = 0.0370)
  depth span   : 0.007 (retain90 oracle = 1.000, so peak is 1.3% of full erasure)
```

Two epochs at `lr=1e-6` is 10 optimiser steps; peak depth 0.013 means roughly 1%
of what retraining-without-the-data achieves. Utility never approached its floor
(0.74–0.76 against a floor of 0.702), so the run was stopped by the epoch budget,
not by damage — there is headroom to train much further. **No number in this
section is a finding about depth or breadth.** It is evidence that the pipeline
executes correctly end to end.

---

## 1. Environment

```
$ python --version
Python 3.11.15

$ python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.6.0+cpu False
```

### Installed versions

```
numpy          1.26.4
scipy          1.13.1
sklearn        1.5.2
pandas         2.2.3
matplotlib     3.9.2
torch          2.6.0+cpu
transformers   4.46.3
tokenizers     0.20.3
accelerate     1.1.1
peft           0.13.2
datasets       3.1.0
pytest         8.3.4
```

### OpenMP runtime count (must be exactly 1)

```
$ find $CONDA_PREFIX -name libiomp5md.dll
<env>/deeperase/Lib/site-packages/torch/lib/libiomp5md.dll
count: 1
```

## 2. Full test suite

```
$ python -m pytest tests/
........................................................................ [ 61%]
..............................................                           [100%]
118 passed in 4.98s
```

Exit code: 0

## 3. Plumbing test

```
$ python -m deeperase.scripts.smoke_e2e
INFO smoke: 4. Extrapolation sweep
INFO smoke:   [PASS] update vector non-empty 39 tensors
INFO smoke:   [PASS] alpha=0 reproduces theta_un delta=0.00e+00
INFO smoke:   [PASS] alpha increases forgetting pressure a=0:-16.181 -> a=1:-18.270
INFO smoke: 5. Depth metrics
INFO smoke:   [PASS] activations captured for every layer layers=[0, 1, 2, 3]
INFO smoke:   [PASS] activation shape correct (16, 64)
INFO smoke:   [PASS] drift computed per layer 
INFO smoke:   [PASS] selectivity ratios finite SRS={0: 1.59, 1: 1.27, 2: 1.16, 3: 1.1}
INFO smoke:   [PASS] target layers identified [1, 2]
INFO smoke:   [PASS] linear probe ran on all layers best acc=1.000
INFO smoke:   [PASS] UDS in [0,1] UDS=0.0000 over 16 examples, layers=[1, 2], unlearned=-16.1807, patched=-0.0594, oracle=-6.8564  [OVERSHOOT -- UDS not interpretable]
INFO smoke: 6. Surface metrics
INFO smoke:   [PASS] SMR correct on fixture smr=0.5
INFO smoke:   [PASS] Type II detected when EL10>1 
INFO smoke: 7. Breadth scoring
INFO smoke:   [PASS] seed probe set validates 
INFO smoke:   [PASS] all six tiers scored ['B0', 'B1', 'B2', 'B3', 'B4', 'R']
INFO smoke:   [PASS] retain scored separately 
INFO smoke:   [PASS] mean forget leakage in [0,1] 0.200
INFO smoke: 8. Plane from real tiny-model values
INFO smoke:   [PASS] breadth computed from model and in range breadth=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
INFO smoke:   [PASS] depth computed from model and finite depth=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
INFO smoke:   [PASS] overshoot flag recorded in saved results 6/6 points flagged (expected on this toy)
INFO smoke:   [PASS] sensitivity report produced rho_incl=None, rho_excl=None, robust=None
INFO smoke:    DEGENERACY: breadth_constant=True depth_constant=True all_overshoot=True -> correlation undefined on this toy, as expected
INFO smoke:   [PASS] degenerate toy axes correctly yield no correlation (undefined, not a fabricated trend) axes saturate on a random model; plumbing verified, no trend claimed
INFO smoke:   [PASS] plane json written 
INFO smoke:   [PASS] sensitivity json written 
INFO smoke:   [PASS] figure written 

========================================================================
PLUMBING TEST: 25/25 checks passed in 9.8s
All pipeline stages wired correctly.
------------------------------------------------------------------------
SCOPE: randomly-initialised 4-layer toy model, 64 hidden units.
  * Validates plumbing (shapes, hooks, dtypes, interfaces) ONLY.
  * Depth values are UDS-SCAFFOLD output; real activation patching
    is NOT implemented, so they are not depth measurements.
  * Overshoot flags are expected on this toy and are not a finding.
  * No scientific claim of any kind attaches to these numbers.
========================================================================
```

Exit code: 0

## 4. The NumPy/MKL crash — reproduction and diagnosis

The reviewer observed the full suite crashing in NumPy/MKL during `linear_cka`.
Reproduced and root-caused. **It is an environment defect, not a code defect.**

### Reproduction (Anaconda `base`, Python 3.13.5, torch 2.9.1+cpu)

```
$ /c/Users/janju/anaconda3/python -m pytest tests/ -q
...............................Fatal Python error: Aborted

Current thread 0x000057a8 (most recent call first):
  File "...\deeperase\eval\depth.py", line 159 in linear_cka
  File "...\tests\test_metrics.py", line 43 in test_identical_gives_one
```

### Isolating the cause

```
$ conda-base-python -c "import numpy as np; ... np.linalg.norm(x, ord='fro')"
fro: 146.5462260706131
A OK                        <-- numpy alone: fine

$ conda-base-python -c "import torch; import numpy as np; ... np.linalg.norm(x, ord='fro')"
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
     already initialized.
```

Importing `torch` before the first LAPACK call is what triggers it.

### Root cause

Anaconda `base` contains **four** copies of the Intel OpenMP runtime:

```
anaconda3/Library/bin/libiomp5md.dll                      <-- conda MKL
anaconda3/Lib/site-packages/torch/lib/libiomp5md.dll      <-- pip torch
anaconda3/pkgs/intel-openmp-.../Library/bin/libiomp5md.dll
anaconda3/envs/pytorch2_env/Library/bin/libiomp5md.dll
```

Two get loaded into one process and the runtime aborts. `linear_cka` is merely
the first function to make a LAPACK call after `import torch`; any
`np.linalg` call would abort identically.

Confirmed by the documented (unsafe) workaround succeeding:

```
$ KMP_DUPLICATE_LIB_OK=TRUE conda-base-python -c "import torch, numpy ..."
fro: 146.5462260706131
WORKAROUND CONFIRMS DIAGNOSIS
```

### Fix adopted

A clean non-base environment with **all packages from pip**, leaving exactly
one OpenMP runtime (verified in §1). `KMP_DUPLICATE_LIB_OK=TRUE` was rejected:
Intel documents it as unsafe and it can silently produce incorrect numerical
results, which is unacceptable in a measurement harness.

### Environments checked

| Interpreter | Python | torch | transformers | Suite result |
|---|---|---|---|---|
| `anaconda3` (base) | 3.13.5 | 2.9.1+cpu | missing | **aborts (OMP #15)** |
| `envs/ML` | 3.13.9 | missing | missing | cannot run |
| `envs/pytorch2_env` | 3.10.18 | 2.0.0 | missing | cannot run |
| `C:\Python314` | 3.14.0 | 2.12.1+cpu | 5.13.0 | passes (not the documented target) |
| **`envs/deeperase`** | **3.11.15** | **2.6.0+cpu** | **4.46.3** | **118 passed** |

The reviewer's environment matched `anaconda3` base: torch present,
transformers absent, duplicate OpenMP. Both reported symptoms are explained.

## 5. Test-quality check (mutation testing)

Four deliberate bugs injected into an earlier revision; all four were caught:

| Mutation | Result |
|---|---|
| Swap SAGE parallel/perp coefficients | CAUGHT |
| Break the Type II EL10 threshold | CAUGHT |
| Treat the retain tier as a forget tier | CAUGHT |
| Use the wrong UDS reference point | CAUGHT |

## 6. Regression check — float-buffer extrapolation

Before the fix:

```
running_mean at alpha=1: [2.0, 2.0]   <-- extrapolated; should be [1.0, 1.0]
BUG CONFIRMED
```

Buffers are now filtered by name as well as dtype. Covered by eight tests in
`TestBufferExclusion`.

---

# REV 2 — Real activation patching (captured 2026-08-09 22:37)

## Full test suite
```
$ python -m pytest tests/
  C:\Users\janju\OneDrive\Desktop\Capstone\tests\test_metrics.py:175: DeprecationWarning: deeperase.eval.depth.unlearning_depth_score is DEPRECATED and was never the published UDS metric (no per-layer structure, no KE selection, no LER). Use deeperase.eval.uds.compute_uds, which implements Eqs. 1-6 with real activation patching.
    r = unlearning_depth_score(score_unlearned=0.1, score_patched=0.2, score_oracle=-0.4,
tests/test_metrics.py::TestUDS::test_result_is_marked_as_scaffold
  C:\Users\janju\OneDrive\Desktop\Capstone\tests\test_metrics.py:183: DeprecationWarning: deeperase.eval.depth.unlearning_depth_score is DEPRECATED and was never the published UDS metric (no per-layer structure, no KE selection, no LER). Use deeperase.eval.uds.compute_uds, which implements Eqs. 1-6 with real activation patching.
    r = unlearning_depth_score(score_unlearned=0.1, score_patched=0.2, score_oracle=-0.4)
180 passed, 10 warnings in 15.27s
```
Exit code: 0

## Plumbing test — UDS section
```
INFO smoke: 5b. Two-stage activation patching (real)
INFO smoke:   [PASS] UDS computed via real patching UDS=1.0000 over 4/4 examples (0 undefined, tau=0.05), layers=[0, 1, 2, 3]  [NOT cross-validated vs reference]
INFO smoke:   [PASS] UDS in [0,1] or undefined 
INFO smoke:   [PASS] self-patching M_full is a no-op (dS2 ~ 0) max|dS2|=0.00e+00
INFO smoke:   [PASS] M_unl == M_ret yields UDS == 1 (perfect unlearning) uds=1.0
INFO smoke:   [PASS] no hooks leaked after patching 
PLUMBING TEST: 31/31 checks passed in 26.3s
  * UDS is NOT yet cross-validated against the reference
    implementation -- see docs/UDS_CONFORMANCE.md.
```
Exit code: 0

## UDS conformance

Equations 1-6 implemented and tested; interface mirrors the reference.
**NOT numerically cross-validated** against gnueaj/unlearning-depth-score —
that requires GPU-scale TOFU checkpoints. See `docs/UDS_CONFORMANCE.md` §5.
