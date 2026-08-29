# DeepErase — Progress Update

**To:** Dr. Suresh Kumar Chaudhary
**Date:** 9 August 2026 (rev. 2)
**Status:** Reproducible harness + **real activation patching implemented**. **No research results yet.**

> Supersedes the 2 August version, which overstated what had been achieved;
> corrections are in §7. Rev. 2 adds the UDS measurement backend (§4a).

---

> ## ⚠️ SUPERSEDED — historical record, do not cite for current status
>
> **This letter describes the project as of 9 August 2026 and is kept unedited
> as a record of what was reported when.** Much of it is no longer true:
>
> | This letter says | Current position (29 August 2026) |
> |---|---|
> | "no research results" | Depth axis validated against the published table across **7 runs at two scales** (mean abs. deviation 0.010); breadth axis calibrated; **one full depth-breadth trajectory** measured. See [`RESULTS.md`](RESULTS.md). |
> | "180 tests pass" (§1) / "Now 118" (§7) — these two figures contradict each other and neither was re-verified | **489 tests**, all passing. See [`VERIFICATION.md`](VERIFICATION.md). |
> | "UDS has never run on a model with real knowledge" (§9, blocker 2) | **Closed.** Run on the Llama-3.2 1B and 3B TOFU checkpoints. |
> | "Entity-span extraction not implemented" (§9, blocker 3) | **Closed.** The authors' own annotations are used directly via `--reference-spans`; `results/span_comparison.json` quantifies how poor our heuristic substitute was (12% exact agreement). |
> | "Batch is treated as one example" (§9, blocker 4) | **Closed.** `UDSExample` now rejects batches. |
> | "GPU access, now the critical path" (§10) | **Granted and used** — 42.3 GB card. |
>
> **Still open from this letter:** blocker 1, the numerical cross-check against
> the authors' released implementation. `is_validated_against_reference`
> remains `False`. This is task T0 of the mid-semester report.
>
> For current status read [`RESULTS.md`](RESULTS.md), [`README.md`](README.md)
> and [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md).

---

## 1. Summary

An independent review found the project was not reproducible. That was correct. This update reports the fixes and states plainly what does and does not work.

**What is now true and verified:**
- A documented environment reproduces the build exactly (Python 3.11.15, pinned versions).
- **180 tests pass**; a plumbing test passes **31/31 checks**. Captured output in [`VERIFICATION.md`](VERIFICATION.md).
- The crash the reviewer saw is reproduced, root-caused and fixed.
- Two real defects in my code were found and fixed.
- **Real two-stage activation patching is implemented** — the depth axis is now a causal measurement rather than a placeholder (§4a).

**What is not true yet:** there are no research results. UDS is implemented but not numerically cross-validated against the reference, so it cannot support a claim. Details in §4 and [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md).

---

## 2. The crash: diagnosed

The reviewer saw the suite abort inside `linear_cka`. Reproduced in Anaconda `base`.

**It is not a code defect.** Anaconda `base` holds four copies of the Intel OpenMP runtime (`libiomp5md.dll`); conda's MKL and pip's torch each load one, and the runtime aborts with `OMP: Error #15`. It fires at the first LAPACK call after `import torch` — which happens to be inside `linear_cka`, making it look like a metrics bug.

Fixed by a clean non-base environment with all packages from pip, leaving exactly one OpenMP runtime. I deliberately did **not** use the `KMP_DUPLICATE_LIB_OK=TRUE` workaround: Intel documents it as unsafe and it can silently produce wrong numbers, which is unacceptable in a measurement harness.

The reviewer's second symptom — smoke test unable to run — has the same cause: conda `base` has torch but no transformers. Both symptoms are explained by one environment defect.

---

## 3. Two real bugs in my code

**Float buffers were being extrapolated.** My code documented that buffers are excluded but filtered by *dtype* only. `running_mean`, `running_var` and rotary `inv_freq` are floating-point, so they passed the filter and were extrapolated. Demonstrated: `running_mean` went to 2.0 at α=1 when it should stay 1.0. These are accumulated statistics and deterministic position functions, not points in a weight space — extrapolating them corrupts normalisation. Fixed with name-based filtering plus eight regression tests.

**A fabricated figure.** The previous smoke test plotted a hand-written downward curve while describing itself as end-to-end. The plot now uses values computed by the pipeline. With real values the toy model produces **degenerate axes** — breadth pinned at 1.0, depth at 0.0, correlation undefined. That is the honest output, and the synthetic curve was concealing it.

---

## 4. Honest component status

| Component | Status |
|---|---|
| **UIPE isotropic extrapolation** | **Implemented, tested, usable.** Faithful to the published method. |
| Breadth probe schema (B0–B4 + retain) | Implemented and tested. 3 seed targets; scaling not built. |
| Surface metrics (SMR, EL10, Type I/II/III) | Implemented and tested. |
| Linear probes, CKA/PCA drift | Implemented and tested. **Correlational, not causal.** |
| **Activation patching** | **Implemented and tested** (§4a). Real residual-stream patching. |
| **UDS (depth metric)** | **Implemented** — the published two-stage metric, driven by real patching. **Not yet numerically cross-validated** against the reference, so still not usable in a conclusion. |
| Old `unlearning_depth_score` | **Deprecated.** It was never the published metric. |
| **SAGE directed extrapolation** | **Deferred from capstone scope** (§5). |
| Probe scaling, signature extraction | **Not started.** |
| Any 7B experiment | **Not started.** No GPU on this machine. |

**Consequence:** the depth axis is now *measured causally* rather than approximated — but the measurement is not yet cross-checked against the reference implementation, so it cannot support a claim.

---

## 4a. Real activation patching — the main new work

The depth axis is now backed by an actual causal intervention.

**What was wrong before.** My old `unlearning_depth_score` combined three scalars the caller supplied. Reading the paper properly, that was **not the published metric at all** — the real UDS is two-stage, per-example and per-layer, with Knowledge-Encoding layer selection and a Layer Erasure Ratio. My version had none of that. It is now deprecated rather than quietly patched up.

**What is implemented.** Two new modules:

* `eval/patching.py` — replaces a decoder layer's residual stream with hidden states captured from another model, at chosen token positions, then runs the target's own forward pass.
* `eval/uds.py` — the published metric, Eqs. 1–6. Stage 1 patches `M_ret` into `M_full` to locate Knowledge-Encoding layers (`ΔS1 > τ`, τ=0.05); Stage 2 repeats with `M_unl`; per-layer erasure ratios are averaged weighted by `ΔS1`. Examples with no KE layers are *undefined* and excluded, per the paper — not coerced to zero.

**Evidence it is genuine, not a scaffold.** 62 new tests, including the ones that specifically rule out faking: patched logits differ from both the source model's and the unpatched target's, and **perturbing a layer downstream of the patch changes the result** — only possible if the target really computes. Two exact end-to-end checks on real models: self-patching gives `ΔS2 = 0.00e+00`, and `M_unl = M_ret` gives `UDS = 1.0`.

Mutation testing caught all five applicable injected bugs (hook leak, patch-all-positions, ignore-τ, no LER clipping, unweighted mean).

**One real bug found along the way.** Two tests failed initially. The cause was PyTorch hook ordering: forward hooks run in registration order and each receives the previous one's return value, so my *verification* hook — registered first — was observing the pre-patch tensor. The patching code was correct; the tests were wrong. Both are fixed, and I added a test that pins the ordering behaviour explicitly, since it is an easy trap to fall into again.

---

## 5. SAGE: deferred from capstone scope

SAGE splits the update vector against a "forget-related subspace". The linear algebra is correct and tested. **The science is not settled**, and I previously understated this.

An activation signature is a direction in **activation space** (from hidden states). The update vector lives in **parameter space**. These are different spaces, and I have no principled mapping between them. The code currently requires the caller to supply a basis already in the parameter tensor's coordinates and to declare which axis it acts on — deferring the question rather than answering it.

I also previously cited "SAGE reduces to UIPE when the coefficients are equal" as validation. **That was wrong.** It proves algebraic consistency — a guard against coding errors — and says nothing about whether the directed variant is meaningful.

SAGE now emits a warning on every directed call and is documented as not research-ready. **Per your instruction it is deferred from the current capstone scope**, and the prototype remains only so the work is not lost.

---

## 6. Overshoot: downgraded from "finding" to "edge case"

I previously reported as a research finding that gradient ascent drives the target below the retain oracle. **That was an overstatement.** It was observed on a randomly-initialised 4-layer toy model, and the UDS values involved come from a scaffold with no real activation patching.

Accurate description: **a metric edge case detected in a toy-model plumbing test.** It shows the *old scaffold* was undefined when the unlearned model scored below the oracle — nothing more. Whether real models do this is untested.

The published UDS handles this structurally: the Layer Erasure Ratio is clipped to [0, 1], so "erased harder than the retain model" simply caps at 1.0. The overshoot machinery is retained for the plane, but with the real metric in place it no longer fires on the toy.

Handling: the flag is recorded in saved results, and `sensitivity_report()` reports correlations **with and without** overshoot points plus whether the conclusion is robust to that choice. Silently dropping them would have been a filtering decision hidden inside a headline number.

---

## 7. Corrections to my 2 August update

| Previous claim | Correction |
|---|---|
| "101 tests passing" | True only on an undocumented interpreter. Now 118, in a documented environment, with captured output. |
| "End-to-end pipeline test 21/21" | The plot was fabricated. Now 31/31 with real pipeline values, relabelled a *plumbing test*. |
| Overshoot as "a genuine research finding… deserves a paragraph in the paper" | A metric edge case on a toy model. Not paper material. |
| "SAGE nests UIPE exactly, so the comparison is apples-to-apples" | Proves algebraic consistency only. Not validation. |
| Depth metrics presented as ready | The old UDS was a scaffold and not even the published formula. Real patching now implemented (§4a); cross-validation still pending. |
| "Buffers are copied unchanged" | Float buffers were being extrapolated. Now fixed. |

---

## 8. Reproducing this

```bash
conda create -y -n deeperase python=3.11
conda activate deeperase
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m pytest tests/                    # 118 passed
python -m deeperase.scripts.smoke_e2e      # 31/31 checks
```

**Tested versions:** Python 3.11.15 · torch 2.6.0+cpu · numpy 1.26.4 · scipy 1.13.1 · scikit-learn 1.5.2 · pandas 2.2.3 · matplotlib 3.9.2 · transformers 4.46.3 · tokenizers 0.20.3 · accelerate 1.1.1 · peft 0.13.2 · datasets 3.1.0 · pytest 8.3.4

Do **not** install into Anaconda `base` — that is the OpenMP conflict in §2.

---

## 9. Remaining blockers

**Blocking the research, in priority order:**

1. **UDS is not numerically cross-validated against the reference.** Specification conformance is documented equation-by-equation and covered by tests, but agreement on shared inputs is unverified. **This needs a GPU** — the reference requires TOFU checkpoints at Llama scale. Until it closes, `is_validated_against_reference` stays `False` and UDS cannot appear in a conclusion. *Now the largest blocker.*
2. **UDS has never run on a model with real knowledge.** Everything so far is randomly-initialised toys. **Needs GPU.**
3. **Entity-span extraction not implemented** — the paper scores entity spans; we currently require the caller to pass token indices. *No GPU needed.*
4. **Batch is treated as one example**, a deviation from the reference that must be resolved before TOFU. *No GPU needed.*
5. **Open question from the last update, still unanswered:** does UIPE extrapolate a full-parameter delta or a LoRA delta? Recommendation unchanged: run both as an ablation.

Full list with a validation plan: [`docs/UDS_CONFORMANCE.md`](docs/UDS_CONFORMANCE.md) §5–6.

---

## 10. What I'd like from you

1. **GPU access, now the critical path.** Blockers 1 and 2 both require it, and neither can be closed on this machine. Everything CPU-shaped is either done or small.
2. A decision on **blocker 5** (full-FT vs. LoRA delta).
3. Confirmation of sequencing: I plan to finish the CPU-side items (3 and 4) next, so that the moment GPU access arrives the cross-validation can run immediately.
