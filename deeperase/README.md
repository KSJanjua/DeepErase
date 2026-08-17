# DeepErase — Depth vs. Breadth of Knowledge Erasure in LLMs

Research **scaffold** for the D1 direction: *do unlearning methods trade representation-level depth against generalisation breadth?*

**The measurement is built and validated; the study has not been run.** The depth metric reproduces the reference paper's published values at two model scales across six configurations. **No answer to the research question exists yet.** See [Honest status](#honest-status) before citing anything here.

Background: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) · [`literature/`](literature/) (54 papers) · [`VERIFICATION.md`](VERIFICATION.md) (captured test output)

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
| Breadth probe schema (B0–B4 + retain) | Implemented and tested. Schema only — 3 seed targets exist, scaling not built. |
| Surface metrics (SMR, EL10, Type I/II/III, ROUGE-L) | Implemented and tested. |
| Linear probe recoverability | Implemented and tested. Correlational, not causal. |
| Representation drift (CKA, PCA, selectivity) | Implemented and tested. Correlational, not causal. |
| **Activation patching** (`eval/patching.py`) | **Implemented and tested.** Real residual-stream patching, LLaMA-style verified; GPT-2/OPT/NeoX paths supported but untested (warn at runtime). |
| **UDS** (`eval/uds.py`) | **Implemented and validated against published values.** Reproduces the paper's Table 2 at 1B and 3B across six configurations, all monotonic and within tolerance. **Not yet cross-validated against the authors' own code** — one open discrepancy on `retain99` (§8.1 of the conformance doc). See [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md). |
| `depth.unlearning_depth_score` (old) | **DEPRECATED.** Was never the published metric. Raises `DeprecationWarning`. |
| **SAGE directed extrapolation** | **DEFERRED from capstone scope.** Prototype retained; emits `ExperimentalPrototypeWarning`. Must not be used for research claims. |
| Plane + plotting | Implemented and tested. Depth axis now fed by real UDS. |
| **TOFU data + entity spans** (`data/tofu.py`) | **Implemented and tested.** Span extraction verified on real TOFU rows; a guard drops 61/400 examples where extraction demonstrably fails. |
| **GPU infrastructure** (`config.py`, `models.py`) | **Implemented and tested.** Memory planning, model lifecycle, preflight. Proven on a 42 GB card. |
| Probe scaling, signature extraction | **Not started.** |
| The depth-vs-breadth study itself | **Not started** — this is the next phase. |

**358 tests pass** in the documented environment. The plumbing test passes **31/31 checks**. Exact captured output: [`VERIFICATION.md`](VERIFICATION.md).

Test quality was checked by mutation: four deliberate bugs were injected (swapped SAGE coefficients, broken Type II threshold, retain tier misclassified as forget, wrong UDS reference point) and all four were caught.

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
tests/                      358 tests
data/probes/                seed_tofu.json
results/                    toy_plumbing_* (plumbing output, not results)
literature/                 54 PDFs + references.bib + reading list
```

### About `results/toy_plumbing_*`

These files come from a randomly-initialised 4-layer model with 64 hidden units. Every coordinate is computed by the real pipeline — including real two-stage activation patching for the depth axis — but the model has no linguistic competence and the values are meaningless as measurements.

On this toy the axes **saturate**: breadth pins at 1.0, depth clips to 0.0, all points overshoot, and the correlation is undefined. That is expected for a random model and is reported explicitly rather than smoothed over. *An earlier version plotted a fabricated downward curve here; that was wrong and has been removed.*

---

## Known limitations

1. **UDS is not numerically cross-validated** against the reference implementation. Specification conformance is documented and tested; agreement on shared inputs is not. This is the largest remaining gap — see [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md) §5.
2. **UDS has never run on a model with real knowledge.** All testing to date uses randomly-initialised toy models.
3. **Entity-span extraction is not implemented.** Callers supply token indices; the paper scores entity spans specifically.
4. **Batch is treated as one example** — a deviation from the reference that must be resolved before TOFU.
5. **Non-LLaMA architectures are unverified.** Structurally supported, warn at runtime.
6. **The breadth probe set has 3 targets.** Enough to validate the schema, not to measure anything.
7. **Nothing has run above toy scale.** No GPU on the development machine.
8. **SAGE is deferred** from capstone scope.

---

## Next steps

Blocked on a decision or GPU access, in priority order:

**No GPU needed:**
1. Entity-span extraction for TOFU (conformance item 4).
2. Resolve the batch-as-one-example convention (item 6).
3. Non-LLaMA architecture tests (item 7).
4. Stage-1 caching before any large sweep (~2× saving).

**Needs GPU:**
5. Numerical cross-check against the reference on shared inputs (item 1) — gates `is_validated_against_reference`.
6. Reproduce the paper's Table 2 monotonicity result (items 2, 3).
7. τ sensitivity sweep (item 5).
8. Probe scaling, OpenUnlearning integration, 7B validation.

SAGE and signature extraction are **deferred from capstone scope**.
