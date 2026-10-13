# DeepErase — Depth vs. Breadth of Knowledge Erasure in LLMs

Research **scaffold** for the D1 direction: *do unlearning methods trade representation-level depth against generalisation breadth?*

**The measurement is built and the depth axis is validated. The study has been replicated under the original design (v1), and a corrected design (protocol v2) is implemented and ready for the GPU** (status 9 Oct 2026). The depth metric reproduces the reference paper's published values at two model scales across seven configurations (mean absolute deviation 0.010 against a 0.08 tolerance), and is insensitive to its threshold τ ([`RESULTS.md`](RESULTS.md) §9). The v1 depth-breadth trajectories (GA and GradDiff, three seeds each) all lose about 0.10 utility. An audit on 9 Oct also found a breadth-scoring defect and two design issues ([`RESULTS.md`](RESULTS.md) §8). All three are fixed in code; [`docs/PROTOCOL_V2.md`](docs/PROTOCOL_V2.md) describes the fixes. **No answer to the research question exists until the v2 runs are made.** See [Honest status](#honest-status) before citing anything here.

Background: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) · [`RESULTS.md`](RESULTS.md) (every measured number) · [`docs/PROTOCOL_V2.md`](docs/PROTOCOL_V2.md) (the corrected design) · [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md) (hypotheses and decision rules, fixed before the v2 runs) · [`gpu/README.md`](gpu/README.md) (the GPU run plan) · [`literature/`](literature/) (the 10 Appendix A references) · [`VERIFICATION.md`](VERIFICATION.md) (captured test output)

---

## Setup — one exact path

Verified on Windows 11 and on macOS (Apple silicon, 9 Oct 2026), Python 3.11.15. Do **not** install into an Anaconda `base` environment (see [MKL note](#the-openmp-crash-and-why-the-install-order-matters)). On macOS, `pip install torch==2.6.0` from PyPI already gives the CPU/MPS build; skip the `--index-url`. `pyarrow` is pinned in `requirements.txt` because an unpinned install now resolves a version that needs NumPy 2 and breaks `import datasets`.

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
| Breadth tiers | **B0, B1, R** from TOFU; **B2** (author named by a description, 360 items), **B3** (the reverse direction: the fact is given and the author must be named, 365 items) and **B4** (author named by a book title — two hops, 237 items), derived mechanically from TOFU; **RA, WF** general-knowledge controls. 2379 items frozen in [`data/breadth_items_v2.json`](data/breadth_items_v2.json), rebuildable with `scripts/build_breadth_items.py`. |
| Breadth scoring | **Defect fixed 9 Oct:** the scorer skipped the first token of every candidate answer ([`RESULTS.md`](RESULTS.md) §8.1). Every v1 breadth and utility number carries it, but on the real models it flips at most 2% of TOFU items and leaves the calibration unchanged (§8.1). Now also batched, and records each item's outcome and margin. |
| Surface metrics (SMR, EL10, Type I/II/III, ROUGE-L) | Implemented and tested. |
| Linear probe recoverability | Implemented and tested. Correlational, not causal. |
| Representation drift (CKA, PCA, selectivity) | Implemented and tested. Correlational, not causal. |
| **Activation patching** (`eval/patching.py`) | **Implemented and tested.** Real residual-stream patching, validated on LLaMA (TOFU). GPT-2/OPT/NeoX paths are tested on tiny models of each family, but not validated on real ones. |
| **UDS** (`eval/uds.py`) | **Implemented and validated against published values.** Reproduces the paper's Table 2 at 1B and 3B across seven runs, all monotonic and within the 0.08 tolerance; best run (`table2_refspans`, the authors' own annotations) has mean absolute deviation 0.010. Insensitive to τ over 0.01–0.5 (`RESULTS.md` §9). **Cross-checked against the authors' own code on byte-identical inputs: agrees to 4.8e-7 at layers 0–14.** Their last-layer patch uses a post-norm state; corrected, it equals ours exactly (`RESULTS.md` §10). **Resolves single facts:** per-fact UDS separates facts a checkpoint never saw from facts it saw with AUC 1.000 in all 14 ground-truth comparisons (`RESULTS.md` §11). See [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md). |
| `depth.unlearning_depth_score` (old) | **DEPRECATED.** Was never the published metric. Raises `DeprecationWarning`. |
| **SAGE directed extrapolation** | **DEFERRED from capstone scope.** Prototype retained; emits `ExperimentalPrototypeWarning`. Must not be used for research claims. |
| Plane + plotting | Implemented and tested. Depth axis now fed by real UDS. |
| **TOFU data + entity spans** (`data/tofu.py`) | **Implemented and tested.** Span extraction verified on real TOFU rows; a guard drops 61/400 examples where extraction demonstrably fails. |
| **GPU infrastructure** (`config.py`, `models.py`, `gpu/`) | **Implemented and tested.** Memory planning, model lifecycle, preflight. Proven on a 42 GB card (Aug 15–16) and on a shared A100-80GB (from Aug 29). |
| Unlearning: GA / GradDiff / NPO + utility-floor checkpoint selection | **Implemented and tested.** v1: GA (lr 1e-6) and GradDiff (lr 3e-6) run with 3 seeds each. **NPO had a defect until 1 Sep** (per-token mean instead of sequence sum, which made it identical to GA); every existing NPO result is invalid. v2 adds an answer-only loss and float32 training. |
| **Degradation control (T1)** | **Implemented and run** (`--control random`, per-tensor norm matched). Depth is about 140× smaller along a random direction than along the real one. The matched-*damage* ladder (`--control-scale`) has not been run. See [`RESULTS.md`](RESULTS.md) §7. |
| Signature extraction | **Not started.** Deferred with SAGE. |
| The depth-vs-breadth study itself | **v1: replicated across GA and GradDiff, 3 seeds each** ([`RESULTS.md`](RESULTS.md) §6). Both axes rise with α in every run, every run loses 0.10–0.13 utility, and the retain tier falls as fast as the forget tiers. **v2 ([`docs/PROTOCOL_V2.md`](docs/PROTOCOL_V2.md)): implemented and tested end to end on CPU, not yet run on the GPU.** It has trained and held-out trajectories, selectivity on every point, and full provenance. Its analysis is pre-registered ([`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md)) and is made **per fact**: does each fact's depth match its own behavioural forgetting, and are forgotten facts removed or only suppressed? |
| Run provenance | Every v2 run records its git commit (and whether the tree was dirty), library versions and input-file hashes. [`results/studies/run_registry.json`](results/studies/run_registry.json) marks invalid and superseded runs; `gpu/summarize.py` and `show_results.py` read it, so a known-bad run cannot be presented as a result. |

**707 tests pass, 2 skip (GPU-only)** — 709 collected from 679 test functions — captured on macOS CPU on 13 Oct 2026 ([`VERIFICATION.md`](VERIFICATION.md) §0.2). That run includes an end-to-end CPU test of the whole study runner, under both protocols, with tiny models on real TOFU text. Before the 9 Oct changes the same machine gave 509 passed, 2 skipped; the last GPU-box run was 493 items on 29 Aug (§0). The plumbing test passes **31/31 checks**. The mid-semester report's Table 13 counted 489 functions at submission.

Test quality was checked by mutation: deliberate bugs were injected (swapped SAGE coefficients, broken Type II threshold, retain tier misclassified as forget, wrong UDS reference point, hook leak, patch-all-positions, ignore-τ, no LER clipping, unweighted mean) and every one was caught.

---

## The two axes

**Breadth** — how far forgetting generalises past the exact forget strings.

| Tier | Meaning |
|---|---|
| B0 | verbatim forget-set question |
| B1 | paraphrase — same fact, reworded |
| B2 | alias or definite description |
| B3 | a distinct fact entailing the target in one hop. Built as the reverse direction (fact given, author asked for), since TOFU has no true entailments |
| B4 | requires two or more hops through retained facts |
| **R** | **retain neighbour — must survive; the over-forgetting control** |

The R tier is mandatory. Forget rates rise monotonically with α, and without a retain control that looks like success rather than collateral damage.

**Depth** — whether the representation actually changed.

| Metric | Kind | Status |
|---|---|---|
| `linear_probe_recoverability` | correlational | implemented |
| `representation_drift` (CKA, PCA, selectivity ratio) | correlational | implemented |
| `uds.compute_uds` | **causal** (activation patching) | implemented; reproduces published Table 2 at 1B and 3B; **cross-validated against the authors' code** on identical inputs (Llama; `RESULTS.md` §10) |

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
  scripts/run_study.py           the full depth-breadth study (--protocol v1 | v2)
  scripts/compare_spans.py       our span heuristic vs the authors' annotations
  scripts/build_breadth_items.py builds data/breadth_items_v2.json from TOFU (B2/B4 derived)
  scripts/tau_sensitivity.py     re-aggregates saved runs at other tau (no GPU)
  scripts/breadth_boundary_effect.py  measures the breadth-scorer defect on real models
  scripts/cross_validate_uds.py  T0: our UDS vs the authors' code on identical inputs
  scripts/aggregate_matrix.py    seeds -> mean ± sd tables, alpha slopes, plane figure (no GPU)
  provenance.py             git commit / environment / file hashes for every run
  registry.py               reads results/studies/run_registry.json (invalid / superseded runs)
tests/                      647 tests (incl. an end-to-end CPU run of the study runner)
gpu/                        bootstrap / replicate / summarize helpers for the shared A100
data/probes/                seed_tofu.json
data/breadth_items.json     1200 forced-choice items (B0/B1/R), the v1 item set
data/breadth_items_v2.json  2379 items (B0-B4, R, RA, WF), the v2 item set
data/tofu/                  the TOFU benchmark, downloaded (gitignored)
data/uds_annotations/       the authors' entity-span annotations (gitignored)
results/gpu_runs/           depth validation + breadth calibration runs
results/studies/            depth-breadth trajectories (+ run_registry.json)
results/analysis/           offline analyses: tau sensitivity, scorer effect, T0
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

1. ~~UDS is not numerically cross-validated~~ **Cross-validated (10 Oct):** on identical inputs our per-layer ΔS matches the authors' code to 4.8e-7 at layers 0–14. At the last layer their code patches a post-final-norm state; corrected, its UDS equals ours exactly ([`RESULTS.md`](RESULTS.md) §10). The `is_validated_against_reference` flag is still `False`, pending a team decision on that deviation.
2. **The utility control is not flat across the α sweep.** It falls by 0.10–0.13 in every run. The retain tier also loses as many forced-choice items as the forget tiers ([`RESULTS.md`](RESULTS.md) §8.3), so the breadth rise cannot yet be separated from general degradation. The matched-magnitude random-direction control is implemented and run: it shows the *depth* rise is direction-specific, but it inflicted no damage, so the matched-damage version is still needed.
3. **Seeding alone does not produce independent replicates.** Training does not shuffle and the Llama dropout is 0, so under v1 a seed change with `--sampling even` yields a bit-identical run (the runner warns if you try). Genuine v1 replicates need `--sampling random --seed N`. Under v2 the seed also draws the per-author trained/held-out split.
4. **B3 is the reverse direction, not a logical entailment.** TOFU's facts are independent attributes, so a fact that *entails* another would have to be invented. B3 instead gives the fact and asks for the author, as the project's hand-written seed probes defined it (13 Oct). Fine-tuned models often cannot answer in the reverse direction (the "reversal curse"), so the tier counts only if `full` beats `retain90` on it. The first v2 run's calibration settles that. B2 (description) and B4 (book title, two hops) are derived the same way, and v2's held-out half gives "a different fact about the same author" on both axes.
5. **NPO has no valid results.** GA and GradDiff have three seeds each. GradDiff's retain wiring was broken until 29 Aug 2026 (fixed in `ca79fa9`). NPO's log-ratio used a per-token mean until 1 Sep 2026 (fixed in `74313dc`), so every saved NPO run is a GA duplicate.
6. ~~τ = 0.05 is inherited, not sensitivity-checked~~ **Checked (9 Oct):** over τ ∈ [0.01, 0.5] every validation run stays monotonic and no value moves by more than ~0.02 ([`RESULTS.md`](RESULTS.md) §9).
7. **Our own entity-span heuristic agrees with the authors' annotations on only 12% of examples** (`results/span_comparison.json`). Use `--reference-spans` for anything on `forget10`; the heuristic is a documented weak approximation everywhere else.
8. **Non-LLaMA architectures are tested only on tiny models.** GPT-2, OPT and GPT-NeoX patching passes the same structural tests as LLaMA (`tests/test_architectures.py`, 9 Oct). UDS on those families has not been validated against published values on real checkpoints, and the code says so at runtime.
9. **SAGE is deferred** from capstone scope and must not be used for research claims.
10. **v1 breadth numbers carry the first-token scoring defect.** Fixed in code on 9 Oct (`eval/breadth.answer_token_start`, with a mutation-checked regression test). Measured on the real 1B models, it changes at most 2% of TOFU items and leaves the calibration range unchanged, so the v1 numbers stand. It was severe only for short answers (RA/WF), which v1 did not use. See [`RESULTS.md`](RESULTS.md) §8.1.
11. **v1 trained on the 100 depth examples with a full-sequence loss.** Depth was measured on trained examples, breadth mostly on untrained ones. Fixed by design in protocol v2 (trained/held-out split, answer-only loss). See [`RESULTS.md`](RESULTS.md) §8.2 and §8.4.
12. **v1 trained and swept in bfloat16.** Protocol v2 runs in float32 with TF32 disabled. Running 3B in float32 does not fit the shared card, so 3B will need `--dtype bfloat16`, recorded in its config.

## Next steps

Priority order, matching tasks T0–T4 of the mid-semester report §5.4.

**T0 — Cross-check UDS against the reference implementation.** *Done 10 Oct* ([`RESULTS.md`](RESULTS.md) §10). `is_validated_against_reference` is now `True` for Llama-family models, the architecture checked.

**Code fixes (done 9 Oct):** breadth scorer corrected; protocol v2 implemented, with trained/held-out items, answer-only loss, float32, selectivity, B2/B4/RA/WF tiers and full provenance. The exact GPU commands, in order, are in [`gpu/README.md`](gpu/README.md).

**T1 — Separate targeted forgetting from general degradation.** *Partly done:* the matched-magnitude control is run (v1), and selectivity and more headroom are built into v2. Still needed on the GPU: the matched-damage ladder in float32 (`--control-scale`).

**T2 — Replicate across methods and seeds.** *Partly done:* GA and GradDiff have 3 seeds each under v1. Still needed on the GPU: v2 learning-rate calibration per method (NPO's first valid runs), then 3 methods × 3 seeds.

**T3 — Extend the breadth axis.** *Done in code:* B2, B3 and B4 derived from TOFU (962 items), with RA/WF controls. They are measured on every v2 point. B3 is the reverse direction (see limitation 4).

**T4 — Scale and consolidate.** Repeat the full protocol at 3B (bf16); add a parameter-efficient adaptation arm; consolidate artefacts for the final report.

Also outstanding: a fresh full-suite transcript from the GPU box itself.

SAGE and signature extraction remain **deferred from capstone scope**.
