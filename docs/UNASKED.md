# UNASKED: answer as if the question had never been asked about this person

UNASKED is an **exploratory** successor to STRANGER. It was built after both
pre-registered gates said stop, and its own criteria were written before its
first run (ANALYSIS_PLAN.md, H8-X).

> **Status, 17 Oct 2026 (seed 0; RESULTS.md §19).** Trained strict UDS at the
> checkpoint is 0.452, about twice GA's mean. Only 8% of its UDS comes from
> overshoot, and it holds on paraphrased questions (0.81 of the trained-wording
> value). It costs RA and WF about what GA costs (0.72 / 0.735), so gate X-G3
> (RA ≥ 0.80) failed. Its follow-up runs went ahead as a second declared
> deviation. A relearning attack recovers its facts as fully as GA's, so its
> extra depth on UDS does not make the facts harder to relearn. Matched-damage
> controls, seeds 1-2, a λ = 3 variant and a relearning dose-response are
> queued.

## Why

STRANGER's gate 1 showed that the fine-tuned model's memory of a TOFU fact is
not stored behind the author's name (RESULTS.md §17). Gate 1b then asked what
the memory *is* keyed on. It ran M_full on edited copies of each depth example
and scored each copy as a UDS source (seed 0, trained facts; RESULTS.md §18):

| M_full on the row with ... | UDS | strict UDS | first-token UDS |
|---|---|---|---|
| the name swapped | 0.236 | 0.193 | 0.282 |
| the question blanked | 0.372 | 0.297 | 0.463 |
| **the question blanked and the name swapped** | **0.559** | **0.425** | **0.621** |
| another author's question | 0.532 | 0.406 | 0.611 |
| everything before the entity blanked | 0.985 | 0.236 | 0.986 |

The memory is keyed mostly on the question's wording together with the name.
Removing all context is no guide: it scores 0.985 only by overshoot. The
question-plus-name state is the strongest oracle-free "this was never asked"
state the model can produce itself, at about twice the strict depth GA,
GradDiff and NPO reach at their checkpoint.

## Method

* **Targets.** For each trained forget row, run M_full on the same row with the
  question replaced by neutral filler of the same token length and the
  author's name swapped for one of 4 invented names. Average its block outputs
  over the 4 names, at all 16 layers.
* **Positions.** Every answer position, from the one predicting the first
  answer token to the last. Non-final tokens of an answer-side name mention are
  excluded, because their inputs differ. These include the positions UDS reads,
  so trained-fact depth is checked on paraphrased questions too (below).
* **Retain anchor.** M_full's own states at all 16 layers on 200 retain rows,
  with the R-tier authors excluded, λ = 1.
* **Loss.** Two-sided normalised MSE to both, as in STRANGER
  (`deeperase/stranger.py`). The optimum is the target's state, so the method
  has no reason to push past it.
* **What trains.** The attention and MLP weights of every block. Embeddings,
  norms and the head stay frozen.
* **Learning rate.** From `lr_scan.py`, using the rule for bounded methods: the
  usable rate whose kept epoch is closest to its target.
* **Checkpoint.** The epoch with the lowest gap to the target among those with
  R ≥ 95% of baseline. The forget NLL of a bounded method plateaus within an
  epoch, so its maximum is noise (ANALYSIS_PLAN H8-X, amendment).
* **Everything else** is protocol v2, including the UIPE sweep and the
  matched-damage controls.

## Checks specific to UNASKED

* **Paraphrased-question depth** (`run_study.py --paraphrase-depth`). The same
  trained facts are asked with TOFU's paraphrased question, with the answer and
  entity unchanged. If UNASKED erased the fact, depth holds there. If it only
  learned to go blank on the trained wording, depth falls. GA and NPO are
  rerun with the same split for comparison.
* **Paraphrased-answer depth** (`run_study.py --paraphrase-answer-depth`).
  UNASKED trains every answer position, including the ones UDS patches, and
  rewording the question leaves those positions unchanged. So the same facts
  are also scored with TOFU's paraphrased answer, where the entity is predicted
  from other tokens at other positions. Only facts whose entity occurs there as
  whole words count: 75 of 367, about 37 per seed. Each is paired with its
  original answer. If the depth is a trigger on the trained answer sequence,
  it falls on the paraphrase. Added after seed 0 (ANALYSIS_PLAN §9), so seeds
  1-2 have it and seed 0 does not.
* **Strict UDS** next to UDS, as for every method since RESULTS.md §16.
* **Held-out facts.** They are separate memorised sequences, so UNASKED is not
  expected to reach them beyond what baselines do. This is reported, not
  required.

## Code

| Piece | Where |
|---|---|
| Inputs: blanked question, swapped name, answer positions | `deeperase/stranger.py: build_unasked_inputs` |
| Length-preserving token edits | `deeperase/data/counterfactual.py` (`question_tokens`, `filler_ids`, `replace_at`) |
| Training | `deeperase/unlearn.py` (method `unasked`) |
| Defaults (all 16 layers, every block's attention and MLP) | `deeperase/scripts/run_study.py: UNASKED_DEFAULTS` |
| Gate 1b, the measurement it is built on | `deeperase/scripts/memory_keys.py` |

## UNASKED-G: UNASKED with a general-knowledge anchor (ANALYSIS_PLAN H9)

> **Status, 17 Oct 2026, 3 seeds (RESULTS.md §20).** At the checkpoint:
> * strict UDS 0.396 against GradDiff's 0.141, and breadth 0.345 against 0.169;
> * R 0.794, RA 0.857 and WF 0.843, against GradDiff's 0.768, 0.863 and 0.829.
>
> On H9's criteria, C1 is met. C2 misses its per-seed bar on two seeds (0.309
> and 0.297, against 0.32). C3 is met on the confirmation seeds 1-2. C4, the
> matched-damage controls, is running. At equal utility, UNASKED-G is broader
> (+0.19 to +0.53) and deeper (+0.18 to +0.32 strict UDS) than GradDiff of the
> same seed, in every seed.

UNASKED beats GradDiff by a wide margin on depth and breadth, but it loses
real-author and world-fact knowledge (RA 0.72, WF 0.735, against GradDiff's
0.86 and 0.83). Its target, the state of a blanked question, can spread into a
general habit of not reading the question. The TOFU retain anchor prevents
that on TOFU-style questions only. UNASKED-G adds a second anchor of
real-world questions.

* **Data.** 400 question-answer pairs from Natural Questions (`nq_open`,
  validation split), built by `scripts/build_general_anchor.py` into
  `data/general_anchor.json`.
  * **Leak filter.** A pair is dropped if its question or answer contains any
    RA/WF item's correct answer, or any name or title from an RA/WF question.
    That blocks 526 terms, and none of the 400 kept pairs contains an RA/WF
    answer.
* **Anchor.** M_full's own block outputs at all 16 layers on those pairs, in
  their own term with weight λ_g (default 1). The anchor batches run on one
  iterator across epochs, so every pair is used. The per-layer scale and the
  massive-activation guard still come from the forget and TOFU retain rows
  only, so the rest of the objective is UNASKED's exactly.
* **TOFU retain anchor at λ = 3**, not UNASKED's 1. On seed 0, UNASKED at
  λ = 3 kept breadth (0.423) and nearly all its depth (strict 0.430), and RA
  rose from 0.72 to 0.81 and WF from 0.735 to 0.786. This amendment was made
  before UNASKED-G's first run (ANALYSIS_PLAN H9).
* **Everything else** as UNASKED: targets, positions, trainable weights,
  lr 3e-6, checkpoint by the lowest gap.
* **Code.** Method `unaskedg` (`deeperase/unlearn.py`).
  `stranger.compute_targets(..., general_data=)` builds the anchors and
  `stranger.stranger_loss(..., general_weight=)` adds the term. The flags
  are `--general-anchor` and `--general-weight`.
* **Check on a tiny model** (`tests/test_stranger.py`). At λ_g = 1 the anchor
  cuts the drift on the general rows about tenfold, and the forget target is
  still reached: remaining gap 0.053, against 0.038 without the anchor.
* **Criteria** (C1-C4, fixed before the first run): ANALYSIS_PLAN H9.

```bash
python -m deeperase.scripts.build_general_anchor        # data/general_anchor.json
python gpu/replicate.py --methods unaskedg --seeds 0 --learning-rate 3e-6 \
    --extra --paraphrase-depth --paraphrase-answer-depth
```
