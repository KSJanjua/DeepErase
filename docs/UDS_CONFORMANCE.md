# UDS conformance against the reference implementation

**Status (9 Oct 2026): reproduces the paper's Table 2 at two model scales across seven
configurations (15-16 Aug 2026). The best run uses the authors' own span
annotations (`table2_refspans`, mean absolute deviation 0.010; 0.0185 over the
two non-trivial columns). NOT yet numerically cross-validated against the
authors' own code.**

Our implementation ([`deeperase/eval/uds.py`](../deeperase/eval/uds.py), [`deeperase/eval/patching.py`](../deeperase/eval/patching.py)) was written directly from the equations in:

> Jaeung Lee, Dohyun Kim, Jaemin Jo. *Measuring the Depth of LLM Unlearning via Activation Patching*. arXiv:2605.24614.
> Reference code: https://github.com/gnueaj/unlearning-depth-score

Paper: [arXiv:2605.24614](https://arxiv.org/abs/2605.24614) · local copy under [`literature/`](../literature/README.md) (PDFs gitignored; fetch commands in that README) · reference implementation vendored at [`reference_uds/`](../reference_uds/)

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
| ~~No Stage-1 caching~~ **Resolved.** `build_stage1_cache` and the `stage1_cache=` / `s_full_cache=` arguments of `compute_uds` implement it. `run_uds_validation` writes `stage1_cache.json`, and `run_study` computes Stage 1 once per run and reuses it at every α. Caching *across* runs is not wired up. | — |
| ~~Batch treated as one example~~ **Resolved** (§5 item 6). `UDSExample` takes exactly one sequence and rejects batches. | — |
| **Residual stream only.** | Matches the paper's main configuration; component-level patching is an appendix ablation. |
| **Scored-sequence construction.** We tokenise `"Question: {q}\nAnswer: " + answer` in one call and score the entity span inside the full answer. The reference (`reference_scope="continuation"`) puts the annotation's `prefix` in the prompt and tokenises prompt and continuation separately. | Leading candidate for the residual −0.027 on `retain99`. To be tested in item 1. |

## 4. What is verified

By the tests in [`tests/test_patching.py`](../tests/test_patching.py) (84 test functions as of 9 Oct 2026; 62 when this section was first written):

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

This section tracked the honest gap. All seven items are now closed. **Since 10 Oct 2026, `UDSReport.is_validated_against_reference` is `True` for Llama-family models** (`CROSS_VALIDATED_MODEL_TYPES` in `deeperase/eval/uds.py`, the architecture T0 checked) **and `False` for every other architecture.** Reports saved before that date carry `False`.

| # | Item | Why it matters | Needs |
|---|---|---|---|
| 1 | ~~No numerical cross-check against the reference on shared inputs~~ **PASSED with one documented deviation (10 Oct 2026).** On byte-identical inputs (100 examples, float32), per-layer ΔS agrees to **4.8e-7** at layers 0–14. At the last layer the reference patches HuggingFace's post-final-norm hidden state into the raw residual stream; with that corrected, its UDS equals ours exactly. The quirk adds +0.001 to +0.005 to UDS. See `RESULTS.md` §10. | — — (flag set for Llama, 10 Oct) |
| 2 | ~~Not reproduced against Table 2~~ **DONE — PASSED (15 Aug 2026).** See §8. | — | — |
| 3 | ~~Never run on a model with real knowledge~~ **DONE.** Run on Llama-3.2-1B TOFU checkpoints. | — | — |
| 4 | ~~Entity-span extraction not implemented~~ **DONE (16 Aug 2026).** Two paths now exist: a `NOVEL_CONTENT` heuristic, and the authors' own hand annotations via `--reference-spans` (`deeperase/data/reference_spans.py`, located by character offset). `compare_spans` measured the heuristic against the annotations at **12% exact agreement**, so the heuristic must not be used on `forget10` — see `RESULTS.md` §4. | — | — |
| 5 | ~~τ = 0.05 not sensitivity-checked~~ **DONE (9 Oct 2026), offline.** All seven validation runs were re-aggregated at τ ∈ [0.01, 1.0] from their stored per-layer deltas: every run stays monotonic, and no Stage-2 value moves by more than ~0.02 over τ ∈ [0.01, 0.5]. See `RESULTS.md` §9. | — | — |
| 6 | ~~Batch-as-one-example deviation~~ **FIXED.** ``UDSExample`` now enforces one sequence per example and rejects batches. | — | — |
| 7 | ~~Non-LLaMA architectures unverified~~ **Tested on tiny models (9 Oct 2026).** `tests/test_architectures.py` runs the structural properties UDS needs on GPT-2, OPT and GPT-NeoX: every layer captured, self-patching an exact no-op, foreign states change the output, and both endpoints exact. Real-checkpoint validation exists for LLaMA only. | — | — |

## 6. Validation plan

All seven items are closed. Item 1 passed with one deviation, which lies in the reference implementation, not ours (`RESULTS.md` §10).

1. **Item 1 (needs GPU) — the blocker.** Run the reference implementation and ours on identical (model, example, layer) triples and compare per-layer `ΔS1`/`ΔS2` to ~1e-4. This is **now unblocked**: the reference is vendored at `reference_uds/` (repository root), and its `tofu_data/forget10_filtered.json` is byte-identical to the published `jaeunglee/uds-annotated-tofu` dataset (sha256 `c3d88b76702fed46…`, 367 records), so both implementations can be pointed at the same annotations without further setup.
The harness is `deeperase/scripts/cross_validate_uds.py` (inference only; fits a
16 GB laptop with `--device mps`).

Item 1 has passed, and the flag is now set for Llama-family models. The one deviation lies in the reference, not in this implementation: ours is the one that follows Eqs. 1–6 at the last layer, and with the reference's last-layer patch corrected the two agree to 1e-8 in UDS. Every model in the study is Llama-3.2, so every study report is covered. Other architectures pass the structural tests (item 7) but were never compared with the reference, so they stay unflagged.

## 7. Why the reference has not been executed yet

*Historical note, superseded.* This section originally recorded that the
development machine was CPU-only (`torch 2.6.0+cpu`, no CUDA), which is why the
comparison had not been run. **That constraint no longer applies** — a 42.3 GB
card has since been used for the runs in `RESULTS.md`. Item 1 is now open purely
because it has not been scheduled, not because it is blocked. The interface was
deliberately mirrored so the comparison is mechanical.


---

## 8. Table 2 reproduction — PASSED at two scales

### 8.0 Summary of all runs

Seven runs, all monotonic, all 4/4 within the 0.08 tolerance. Prompt formats
are as recorded in each `report.json`.

| Run | full | retain99 | retain95 | retain90 |
|---|---|---|---|---|
| 1B, `bare`, novel_content (`table2_1B_20260815_181220`) | 0.000 | 0.096 | 0.447 | 1.000 |
| 1B, `plain_qa` (`fmt_plain`) | 0.000 | 0.102 | 0.444 | 1.000 |
| 1B, `chat` (`fmt_chat`) | 0.000 | 0.100 | 0.452 | 1.000 |
| 1B, random sampling (`samp_rand`) | 0.000 | 0.108 | 0.460 | 1.000 |
| 1B, `chat`, `full_answer` spans (`span_full2`) | 0.000 | 0.115 | **0.494** | 1.000 |
| 3B, `chat`, novel_content (`table2_3B`) | 0.000 | 0.095 | 0.430 | 1.000 |
| **1B, `plain_qa`, authors' annotations (`table2_refspans`, n=150)** | **0.000** | **0.126** | **0.486** | **1.000** |
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

It is within tolerance, it does not disturb monotonicity, and it is stable
across scales -- which makes it look like a methodological difference rather
than noise. *Update, 16 Aug:* entity-span annotation was the main cause. With
the authors' own annotations (`table2_refspans`) the gap roughly halves, to
0.126 vs 0.153 (−0.027) ([`RESULTS.md`](../RESULTS.md) §1.2). The leading
candidate for the residual is how the scored sequence is built (§3, last row).
*Settled, 10 Oct (item 1, `RESULTS.md` §10):* about 0.012 comes from sequence construction (prefix in the prompt with the continuation scored, versus the entity inside the whole answer), about 0.002 from the reference's last-layer quirk, and the rest from the example sample.

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
`is_validated_against_reference` therefore remained `False` at the time. T0 closed item 1 on 10 Oct 2026 (§5).
