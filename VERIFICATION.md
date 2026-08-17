# VERIFICATION — captured output

All commands below were run from the repository root in the documented
environment (conda env `deeperase`, Python 3.11.15) and their output
pasted verbatim. Captured: 2026-08-09 21:29

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
