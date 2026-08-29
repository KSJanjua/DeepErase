# DeepErase — Depth vs. Breadth of Knowledge Erasure in LLMs

Research **scaffold** for the D1 direction: *do unlearning methods trade representation-level depth against generalisation breadth?*

**The measurement is built and validated. One trajectory has been run; it is not yet an answer.** The depth metric reproduces the reference paper's published values at two model scales across seven configurations (mean absolute deviation 0.010 against a 0.08 tolerance). A first end-to-end depth-breadth trajectory exists, but it is one method, one scale, one seed, and its utility control was not flat — so **no answer to the research question exists yet.** See [Honest status](#honest-status) before citing anything here.

Background: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) · [`RESULTS.md`](RESULTS.md) (every measured number) · [`literature/`](literature/) (the 10 Appendix A references) · [`VERIFICATION.md`](VERIFICATION.md) (captured test output)

---

## Setup — one exact path

Verified on Windows 11, Python 3.11.15. Do **not** install into an Anaconda `base` environment (see [MKL note](#the-openmp-crash-and-why-the-install-order-matters)).

```bash
conda create -y -n deeperase python=3.11
conda activate deeperase

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then, from the repository root:

```bash
# Full test suite
python -m pytest tests/

# Plumbing test (real transformer, ~15s, CPU)
python -m deeperase.scripts.smoke_e2e

# Regenerate the seed probe dataset
python -m deeperase.probes.seed_tofu
```

CI-style invocation (non-zero exit on any failure, no interactive output):

```bash
python -m pytest tests/ --tb=short -q
python -m deeperase.scripts.smoke_e2e   # exits 1 if any check fails
```

Both commands must exit 0. No GitHub Actions workflow is configured yet.

### Tested versions

Python 3.11.15 · torch 2.6.0+cpu · numpy 1.26.4 · scipy 1.13.1 · scikit-learn 1.5.2 · pandas 2.2.3 · matplotlib 3.9.2 · transformers 4.46.3 · accelerate 1.1.1 · peft 0.13.2 · datasets 3.1.0 · pytest 8.3.4

Full transitive closure in [`requirements-lock.txt`](requirements-lock.txt).

### The OpenMP crash, and why the install order matters

Installing these packages into Anaconda `base`, or mixing `conda install numpy` with `pip install torch`, puts **two** copies of the Intel OpenMP runtime (`libiomp5md.dll`) in one process. On Windows this aborts the interpreter:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

The abort fires at the first LAPACK call after `import torch` — in this codebase that is `np.linalg.norm(..., ord="fro")` inside `eval/depth.linear_cka`, which makes it look like a bug in the depth metrics. **It is not.** It is an environment defect, reproduced and diagnosed in [`VERIFICATION.md`](VERIFICATION.md).

Installing everything from pip into a clean non-base environment leaves exactly one OpenMP runtime and resolves it. Do **not** use `KMP_DUPLICATE_LIB_OK=TRUE`; Intel documents it as unsafe and it can silently produce wrong numerical results — unacceptable in a measurement harness.

---

## Honest status

| Component | Status |
|---|---|
| **UIPE isotropic extrapolation** | **Implemented and tested.** Faithful to the published method. Usable. |
| Breadth probe schema (B0–B4 + retain) | Schema implemented and tested for all six tiers. **Only B0, B1 and R are populated** (1200 items, built from TOFU's perturbed splits). B2/B3/B4 are an enum and nothing more. |
| Surface metrics (SMR, EL10, Type I/II/III, ROUGE-L) | Implemented and tested. |
| Linear probe recoverability | Implemented and tested. Correlational, not causal. |
| Representation drift (CKA, PCA, selectivity) | Implemented and tested. Correlational, not causal. |
| **Activation patching** (`eval/patching.py`) | **Implemented and tested.** Real residual-stream patching, LLaMA-style verified; GPT-2/OPT/NeoX paths supported but untested (warn at runtime). |
| **UDS** (`eval/uds.py`) | **Implemented and validated against published values.** Reproduces the paper's Table 2 at 1B and 3B across seven runs, all monotonic and within the 0.08 tolerance; best run (`table2_refspans`, the authors' own annotations) has mean absolute deviation 0.010. **Not yet cross-validated against the authors' own code** — a residual −0.027 on `retain99` remains (§8.1 of the conformance doc). See [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md). |
| `depth.unlearning_depth_score` (old) | **DEPRECATED.** Was never the published metric. Raises `DeprecationWarning`. |
| **SAGE directed extrapolation** | **DEFERRED from capstone scope.** Prototype retained; emits `ExperimentalPrototypeWarning`. Must not be used for research claims. |
| Plane + plotting | Implemented and tested. Depth axis now fed by real UDS. |
| **TOFU data + entity spans** (`data/tofu.py`) | **Implemented and tested.** Span extraction verified on real TOFU rows; a guard drops 61/400 examples where extraction demonstrably fails. |
| **GPU infrastructure** (`config.py`, `models.py`) | **Implemented and tested.** Memory planning, model lifecycle, preflight. Proven on a 42 GB card. |
| Unlearning: GA / GradDiff / NPO + utility-floor checkpoint selection | **Implemented and tested.** Only the **GA** arm has actually been run; GradDiff and NPO have no results yet. |
| Signature extraction | **Not started.** Deferred with SAGE. |
| The depth-vs-breadth study itself | **One trajectory run** (`study_ga_1B_20260816_165423`), reported in [`RESULTS.md`](RESULTS.md) §5. **Not replicated** — one method, one scale, one seed, utility control not flat. |

**489 tests** in the documented environment; the plumbing test passes **31/31 checks**. Note that [`VERIFICATION.md`](VERIFICATION.md) captures the August runs at **118** and **180** tests — no transcript of the full 489-test run has been pasted there yet, and regenerating it is a standing to-do (see the notice at the top of that file). The 489 figure is the count of test functions in `tests/`, which matches Table 13 of the mid-semester report file by file.

Test quality was checked by mutation: deliberate bugs were injected (swapped SAGE coefficients, broken Type II threshold, retain tier misclassified as forget, wrong UDS reference point, hook leak, patch-all-positions, ignore-τ, no LER clipping, unweighted mean) and every one was caught.

---

## The two axes

**Breadth** — how far forgetting generalises past the exact forget strings.

| Tier | Meaning |
|---|---|
| B0 | verbatim forget-set question |
| B1 | paraphrase — same fact, reworded |
| B2 | alias or definite description |
| B3 | a distinct fact entailing the target in one hop |
| B4 | requires two or more hops through retained facts |
| **R** | **retain neighbour — must survive; the over-forgetting control** |

The R tier is mandatory. Forget rates rise monotonically with α, and without a retain control that looks like success rather than collateral damage.

**Depth** — whether the representation actually changed.

| Metric | Kind | Status |
|---|---|---|
| `linear_probe_recoverability` | correlational | implemented |
| `representation_drift` (CKA, PCA, selectivity ratio) | correlational | implemented |
| `uds.compute_uds` | **causal** (activation patching) | implemented; reproduces published Table 2 at 1B and 3B; **not cross-validated against the authors' code** |

### UDS — two-stage activation patching

```python
from deeperase.eval.patching import EntitySpan
from deeperase.eval.uds import compute_uds

report = compute_uds(
    model_full=m_full,        # trained on D_r ∪ D_f — always the patch TARGET
    model_retain=m_retain,    # trained on D_r only — Stage-1 source (gold standard)
    model_unlearned=m_unl,    # the model under test — Stage-2 source
    examples=[("ex0", input_ids, EntitySpan([12, 13]))],
    layers=range(32),
)
print(report.summary())       # flags that reference cross-validation is pending
```

Stage 1 patches `M_ret` into `M_full` to find Knowledge-Encoding layers (`ΔS1 > τ`, τ=0.05). Stage 2 repeats with `M_unl`. The Layer Erasure Ratio `clip(ΔS2/ΔS1, 0, 1)` is averaged over KE layers weighted by `ΔS1`. **1 = erased to `M_ret`'s level; 0 = fully intact.** Examples with no KE layers are *undefined* and excluded — never coerced to 0.

Two end-to-end sanity checks hold exactly on real models: self-patching `M_full` gives `ΔS2 = 0`, and `M_unl = M_ret` gives `UDS = 1.0`.

> **EL10 lives in `eval/surface.py`, not `eval/depth.py`.** It is computed from output token probabilities, so it is a soft *surface* metric. The original proposal (§4.3, O2) described it as measuring "latent knowledge persistence"; it does not. The module split enforces the distinction, because the D1 hypothesis is precisely that these axes can move in opposite directions.

---

## The instrument

```python
v = compute_update_vector(theta_ini, theta_un)     # v = θ_un − θ_ini
theta = extrapolate(theta_un, v, alpha=0.6)        # UIPE: θ_un + α·v
```

Traversing α is pure tensor arithmetic — no training, no forward passes. `alpha=0` reproduces `theta_un` bit-exactly (the control point anchoring every trajectory).

**Buffers are excluded by default.** Non-float tensors are caught by dtype; floating-point buffers (`running_mean`, `running_var`, `inv_freq`, rotary caches) are caught by name via `DEFAULT_BUFFER_PATTERNS`. These are accumulated statistics or deterministic position functions, not points in a weight space — extrapolating them corrupts normalisation. *An earlier version filtered by dtype alone and did extrapolate float buffers; fixed, with regression tests.*

### SAGE — do not use for research claims

```python
v_par = project_state_dict(v, basis_by_key)
theta, report = extrapolate_directed(
    theta_un, v, alpha_parallel=0.8, alpha_perp=0.2, v_parallel=v_par
)   # raises ExperimentalPrototypeWarning
```

An activation signature is a direction in **activation space** (ℝ^d_model, from hidden states). The update vector `v` lives in **parameter space**. These are different spaces, and this module supplies no canonical mapping between them — `project_state_dict` requires the caller to hand it a basis already in the parameter tensor's coordinates and to declare which axis it acts on. That defers the hard question rather than answering it.

The test showing `alpha_parallel == alpha_perp` reproduces UIPE proves **algebraic consistency only** — a guard against coding errors. It is **not** evidence that the directed variant is meaningful and must never be cited as validation.

---

## Layout

```
deeperase/
  core/extrapolation.py     UIPE (ready) + SAGE (deferred prototype), buffer filtering
  eval/patching.py          residual-stream activation patching, entity-span scoring
  eval/uds.py               two-stage UDS (Eqs. 1-6), KE layers, LER, aggregation
  eval/depth.py             linear probes, CKA/PCA drift, ActivationCapture
                            (+ deprecated unlearning_depth_score shim)
  eval/surface.py           SMR, EL10, Type I/II/III, ROUGE-L
  eval/plane.py             PlanePoint / Trajectory / PlaneDataset, sensitivity, plotting
  probes/schema.py          Tier, Probe, ProbeSet, breadth scoring
  probes/seed_tofu.py       3 hand-verified seed targets (36 probes)
  scripts/smoke_e2e.py      plumbing test — NOT a research experiment
  scripts/run_uds_validation.py  depth-axis validation against published values
  scripts/measure_breadth.py     breadth calibration against the reference models
  scripts/run_study.py           the full depth-breadth study
  scripts/compare_spans.py       our span heuristic vs the authors' annotations
tests/                      489 tests
data/probes/                seed_tofu.json
data/breadth_items.json     1200 forced-choice items (B0/B1/R), built from TOFU
data/tofu/                  the TOFU benchmark, downloaded (gitignored)
data/uds_annotations/       the authors' entity-span annotations (gitignored)
results/gpu_runs/           depth validation + breadth calibration runs
results/studies/            depth-breadth trajectories
results/metrics/, figures/  toy_plumbing_* — plumbing output, NOT results
reference_uds/              vendored https://github.com/gnueaj/unlearning-depth-score
reference_openunlearning/   vendored OpenUnlearning, for comparison only
literature/                 the 10 Appendix A references (gitignored)
```

`reference_uds/` must sit at the repository root: `DEFAULT_REFERENCE_PATH` in
`deeperase/data/reference_spans.py` resolves `reference_uds/tofu_data/forget10_filtered.json`
relative to the working directory, so `--reference-spans` breaks if it is moved.

### About `results/toy_plumbing_*`

These files come from a randomly-initialised 4-layer model with 64 hidden units. Every coordinate is computed by the real pipeline — including real two-stage activation patching for the depth axis — but the model has no linguistic competence and the values are meaningless as measurements.

On this toy the axes **saturate**: breadth pins at 1.0, depth clips to 0.0, all points overshoot, and the correlation is undefined. That is expected for a random model and is reported explicitly rather than smoothed over. *An earlier version plotted a fabricated downward curve here; that was wrong and has been removed.*

---

## Known limitations

1. **UDS is not numerically cross-validated** against the reference implementation. Specification conformance is documented and tested, and the published Table 2 is reproduced at two scales — but agreement with the authors' *code* on shared inputs is still unverified. This remains the largest gap: `is_validated_against_reference` is `False` in every saved run. See [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md) §5 item 1.
2. **The utility control is not flat across the α sweep.** It falls 0.720 → 0.625, so part of the joint depth/breadth rise is general degradation rather than targeted forgetting. The planned degradation control — a matched-magnitude update in a direction unrelated to the forget set — **is not implemented**.
3. **`run_study.py` has no `--seed` flag.** Every saved point records `seed: 0`. Replication across three seeds per arm cannot be run without adding it.
4. **The breadth probe set covers B0, B1 and R only.** B2 (alias), B3 (entailed) and B4 (multi-hop) are declared in `probes/schema.py` and have no items. These are the tiers where a depth/breadth divergence would be expected to show most clearly.
5. **Only the GA arm has been run.** GradDiff and NPO are implemented and unit-tested but have produced no results at any scale.
6. **τ = 0.05 is inherited, not sensitivity-checked** on our data.
7. **Our own entity-span heuristic agrees with the authors' annotations on only 12% of examples** (`results/span_comparison.json`). Use `--reference-spans` for anything on `forget10`; the heuristic is a documented weak approximation everywhere else.
8. **Non-LLaMA architectures are unverified.** GPT-2/OPT/NeoX paths are structurally supported and warn at runtime.
9. **SAGE is deferred** from capstone scope and must not be used for research claims.

## Next steps

Priority order, matching tasks T0–T4 of the mid-semester report §5.4.

**T0 — Cross-check UDS against the reference implementation** on shared inputs, layer by layer, to ~1e-4. `reference_uds/` is vendored and its annotations are byte-identical to the published dataset, so this is unblocked and needs only GPU time. Gates `is_validated_against_reference`; everything below depends on it.

**T1 — Separate targeted forgetting from general degradation.** Select a checkpoint with more headroom above the utility floor, and add the matched-magnitude random-direction control (limitation 2). Precondition for reading the trajectory at all.

**T2 — Replicate across methods and seeds.** Add `--seed` (limitation 3), then run the GradDiff and NPO arms, three seeds each, under the identical protocol.

**T3 — Extend the breadth axis** with hand-written B2/B3/B4 items, validated against the reference models before use.

**T4 — Scale and consolidate.** Repeat the full protocol at 3B; add a parameter-efficient adaptation arm; consolidate artefacts for the final report.

Also outstanding, no GPU needed: τ sensitivity sweep, non-LLaMA architecture tests.

SAGE and signature extraction remain **deferred from capstone scope**.
