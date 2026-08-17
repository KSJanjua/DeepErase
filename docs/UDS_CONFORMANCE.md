# UDS conformance against the reference implementation

**Status: reproduces the paper's Table 2 at two model scales across six
configurations (15-16 Aug 2026). NOT yet numerically cross-validated against
the authors' own code.**

Our implementation ([`deeperase/eval/uds.py`](../deeperase/eval/uds.py), [`deeperase/eval/patching.py`](../deeperase/eval/patching.py)) was written directly from the equations in:

> Jaeung Lee, Dohyun Kim, Jaemin Jo. *Measuring the Depth of LLM Unlearning via Activation Patching*. arXiv:2605.24614.
> Reference code: https://github.com/gnueaj/unlearning-depth-score

Local copy: [`literature/05_critical_evaluation_robustness/UDS_Lee_2026_Measuring-Depth-via-Activation-Patching.pdf`](../literature/05_critical_evaluation_robustness/UDS_Lee_2026_Measuring-Depth-via-Activation-Patching.pdf)

---

## 1. Equation-by-equation conformance

| Paper | Definition | Our implementation | Match |
|---|---|---|---|
| Eq. 1 | `ΔS1[i,l] = mean_t( s_full[i,t] − s_S1[i,t] )`, source `M_ret` → target `M_full` | `uds.compute_uds`, `d1[ell] = s_full − s1` | ✅ |
| Eq. 2 | `KE[i] = { l : ΔS1[i,l] > τ }`, τ = 0.05 | `uds.aggregate_example_uds`; `DEFAULT_TAU = 0.05` | ✅ |
| Eq. 3 | `ΔS2[i,l] = mean_t( s_full[i,t] − s_S2[i,t] )`, source `M_unl` | `d2[ell] = s_full − s2` | ✅ |
| Eq. 4 | `LER[i,l] = clip( ΔS2/ΔS1, 0, 1 )` | `uds.layer_erasure_ratio` | ✅ |
| Eq. 5 | `UDS[i] = Σ_KE ΔS1·LER / Σ_KE ΔS1` | `aggregate_example_uds` | ✅ |
| Eq. 6 | `UDS = mean_i UDS[i]`, undefined examples excluded | `compute_uds`; `uds=None` when `KE=∅` | ✅ |

**Design points also matched:**

- `M_full` is the target at **both** stages (the paper's rationale: only `M_full` has learned the knowledge and can decode it from patched states).
- **Teacher forcing** — one forward pass over the full sequence, no autoregressive generation.
- **Predicting-position offset** — in a causal LM the hidden state at position `p` predicts token `p+1`, so entity token `j` is patched/read at position `j−1`. Implemented in `EntitySpan.predicting_positions`.
- **Residual stream** is the patched component (the paper's main configuration; Appendix D.1 covers component-level variants we do not implement).
- **Undefined ≠ zero.** `KE[i] = ∅` yields `None` and is excluded from the mean, never coerced to 0 or 1.

## 2. Interface alignment

The reference exposes three core functions in `uds/core.py`. Ours mirror them deliberately, to make future cross-checking mechanical:

| Reference | Ours |
|---|---|
| `forward_with_patch()` | `patching.forward_with_patch()` |
| `get_hidden_at_position()` | `patching.capture_hidden_states()` |
| `probe_knowledge_with_patch()` | `patching.probe_knowledge_with_patch()` |
| `--delta_threshold` (default 0.05) | `tau=DEFAULT_TAU` (0.05) |

## 3. Deliberate deviations

| Deviation | Rationale |
|---|---|
| **No Stage-1 caching.** The paper caches `s_full`, `ΔS1` and `KE` across unlearned models. We recompute per call. | Clarity while the code is small. Must be added before large sweeps — it is roughly a 2× saving. Tracked as a TODO in `uds.py`. |
| **Batch treated as one example.** A `(batch, seq)` input is scored as a single example with log-probs averaged across rows. | Fits the toy harness. For TOFU, pass one example per call, or extend to per-row scoring. **This differs from the reference's per-example handling and must be revisited before TOFU runs.** |
| **Residual stream only.** | Matches the paper's main configuration; component-level patching is an appendix ablation. |

## 4. What is verified

By 62 tests in [`tests/test_patching.py`](../tests/test_patching.py):

- Hooks removed after use, including when the forward pass raises, and after full `compute_uds` runs.
- Patching layer `l` leaves upstream layers bit-identical; downstream layers change (propagation).
- Only the listed positions are overwritten.
- `spec=None` and an empty position list both reproduce the unpatched forward pass **exactly**.
- **Patched logits are computed by the target model**, not copied: they differ from both the source model's logits and the unpatched target's, and perturbing a *downstream* target layer changes the patched result.
- Patching a model with its own activations is a no-op.
- Batch/sequence/hidden mismatches raise instead of broadcasting; dtype and device are cast to the target; attention masks reach both capture and patched passes.
- Formula behaviour: `ΔS2 = ΔS1` → `UDS = 1`; `ΔS2 = 0` → `UDS = 0`; sub-τ layers excluded; `ΔS1`-weighting dominates; empty KE → `None`.

Two end-to-end sanity checks in the plumbing test, on real models:

- `M_unl = M_full` (nothing unlearned) → `max|ΔS2| = 0.00e+00` exactly.
- `M_unl = M_ret` (perfect unlearning) → `UDS = 1.0` exactly.

Mutation-tested: hook leak, patch-all-positions, ignore-τ, drop-LER-clipping, and unweighted-mean mutants were **all caught**.

## 5. What remains UNVALIDATED

This is the honest gap. **`UDSReport.is_validated_against_reference` is `False` and must stay so until every item below is closed.**

| # | Item | Why it matters | Needs |
|---|---|---|---|
| 1 | **No numerical cross-check against the reference on shared inputs.** | Two implementations can both satisfy the equations and still diverge on tokenisation, position conventions, or reduction order. This is the single most important gap. | Run reference + ours on identical (model, example, layer) triples; require agreement to ~1e-4. Needs the reference repo and a real model. |
| 2 | ~~Not reproduced against Table 2~~ **DONE — PASSED (15 Aug 2026).** See §8. | — | — |
| 3 | ~~Never run on a model with real knowledge~~ **DONE.** Run on Llama-3.2-1B TOFU checkpoints. | — | — |
| 4 | **Entity-span extraction not implemented.** | The paper scores *entity spans*, not whole answers, "because common template phrases are predictable regardless of knowledge retention". We currently require the caller to supply token indices. | Tokeniser-aware span annotation for TOFU. |
| 5 | **τ = 0.05 not sensitivity-checked** on our data. | The paper analyses this in Appendix D.2. Our value is inherited, not verified. | A τ sweep once real data exists. |
| 6 | ~~Batch-as-one-example deviation~~ **FIXED.** ``UDSExample`` now enforces one sequence per example and rejects batches. | — | — |
| 7 | **Non-LLaMA architectures unverified.** | GPT-2/OPT/NeoX paths are structurally supported and warn at runtime, but untested. | Tests on one model per family. |

## 6. Validation plan

1. **No GPU needed:** implement entity-span extraction (item 4); resolve the batching convention (item 6); add non-LLaMA tests (item 7).
2. **Needs GPU:** clone the reference, run both implementations on shared TOFU inputs, compare per-layer `ΔS1`/`ΔS2` (item 1); reproduce Table 2 monotonicity (items 2, 3); sweep τ (item 5).

Only when items 1–3 pass may `is_validated_against_reference` be set `True` and UDS values be used in a research claim.

## 7. Why the reference was not executed here

The reference requires GPU-scale models and TOFU checkpoints. The development machine is **CPU-only** (`torch 2.6.0+cpu`, no CUDA), so a like-for-like numerical comparison is not currently possible. Conformance was therefore established against the paper's equations, with the interface deliberately mirrored so the comparison is mechanical once hardware is available.


---

## 8. Table 2 reproduction — PASSED at two scales

### 8.0 Summary of all runs

Six runs, all monotonic, all 4/4 within the 0.08 tolerance.

| Run | full | retain99 | retain95 | retain90 |
|---|---|---|---|---|
| 1B, `bare`, novel_content | 0.000 | 0.096 | 0.447 | 1.000 |
| 1B, `plain_qa` | 0.000 | 0.102 | 0.444 | 1.000 |
| 1B, `chat` | 0.000 | 0.100 | 0.452 | 1.000 |
| 1B, random sampling | 0.000 | 0.108 | 0.460 | 1.000 |
| 1B, `full_answer` spans | 0.000 | 0.115 | **0.494** | 1.000 |
| 3B, `bare`, novel_content | 0.000 | 0.095 | 0.430 | 1.000 |
| **Paper (1B)** | 0.002 | 0.153 | 0.496 | 1.000 |
| **Paper (3B)** | 0.008 | 0.151 | 0.482 | 1.000 |

**What the sweep rules out.** The result does not depend on the prompt format
(three formats, spread 0.012), on which examples are drawn (even vs. random),
on the entity-span strategy, or on model scale. The two most plausible ways a
single lucky run could have been an artefact are both excluded.

**Prompt format was a near-tie** at selection time (`bare` -0.2032 vs
`plain_qa` -0.2068 vs `chat` -0.2853) and the concern was that such a narrow
margin might matter. It does not: all three give the same answer.

### 8.1 An open discrepancy worth stating

`retain95` agrees almost exactly once sampling is balanced -- the `full_answer`
run draws 100 examples from all 400, giving precisely 50% unseen-by-retain95,
and lands at **0.494 against 0.496**. Under `novel_content` the filter leaves
190 examples containing 47% unseen, and the score moves to 0.447 accordingly.
So this gap is fully accounted for by sample composition, not by the metric.

`retain99` is different. It sits **~0.04-0.06 below the paper at both scales,
and the gap survives correct sampling**. `forget01` is exactly 10% of
`forget10`, so a metric tracking the unseen fraction should read ~0.10 -- which
is what we get. The paper reads 0.153, *above* that fraction, implying their
measurement detects partial erasure in examples `retain99` did see.

We do not know the cause. It is within tolerance, it does not disturb
monotonicity, and it is stable across scales -- which makes it look like a
methodological difference rather than noise. The most likely candidate is
entity-span annotation: theirs is hand-built, ours is a heuristic. Settling it
requires item 1, a direct comparison against their code.

**It is recorded here rather than averaged away.**

### 8.2 Original single-run detail

Run `table2_1B_20260815_181220`, Llama-3.2-1B TOFU checkpoints, 100 examples
sampled evenly across forget10, `bare` prompt format, tau = 0.05, 16 layers.

| Stage-2 source | Paper | Observed | Difference |
|---|---|---|---|
| `full` | 0.002 | **0.000** | −0.002 |
| `retain99` | 0.153 | **0.096** | −0.057 |
| `retain95` | 0.496 | **0.447** | −0.049 |
| `retain90` | 1.000 | **1.000** | +0.000 |

**Monotonic: yes. Within tolerance (0.08): 4/4. Verdict: PASS.**

Both endpoints are exact, which is the strongest single signal: `full` scores 0
because patching a model into itself is a no-op, and `retain90` scores exactly 1
because it is also the Stage-1 source, so `dS2 == dS1` identically.

### The residual underestimate is explained by sampling

Both middle values sit ~0.05 below the paper. That is a consistent bias rather
than noise, and it is accounted for by *which* examples we sampled:

| | Paper | Ours | Predicted from our sampling |
|---|---|---|---|
| `retain99` | 0.153 | 0.096 | 0.100 |
| `retain95` | 0.496 | 0.447 | 0.470 |

Mean absolute error against the paper is 0.027; against the proportion of
unseen examples in our own sample it is **0.007**. UDS is tracking our sample
composition almost exactly, which is the behaviour the metric should have. We
score 190 of 400 examples (the rest are dropped by span filtering), and that
subset happens to contain 47% rather than 50% unseen-by-retain95 examples.

### A sampling error this run caught

The first attempt used the **first** 50 examples and produced 0.021 and 0.026
for the two middle rows. TOFU's forget splits are ordered and nested at the end:

    forget10 = indices   0..399
    forget05 = indices 200..399   (unseen by retain95)
    forget01 = indices 360..399   (unseen by retain99)

so the first 50 examples were entirely inside the region every retain model had
already seen. Two models that know the same thing produce no log-probability
difference, and UDS collapses to ~0 regardless of correctness.

The endpoints stayed exact throughout, which is what identified this as a
sampling artefact rather than a defect in the metric. Fixed by
:class:`~deeperase.data.tofu.SamplingStrategy`, which now spreads the sample
across the whole split by default and warns if the biased mode is selected.

### What this does and does not establish

**Establishes:** activation patching, Knowledge-Encoding layer selection, the
Layer Erasure Ratio and the aggregation in Eqs. 1-6 all behave correctly on
real trained models, at a scale where the paper published ground truth.

**Does not establish:** agreement with the authors' own implementation on
shared inputs (item 1). Our entity spans are a documented approximation of
their annotation, and any difference there would shift absolute values while
leaving the ordering intact -- exactly the pattern we observe.
`is_validated_against_reference` therefore remains `False`.
