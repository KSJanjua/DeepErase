# UNASKED: answer as if the question had never been asked about this person

UNASKED is an **exploratory** successor to STRANGER. It was built after both
pre-registered gates said stop, and its own criteria were written before its
first run (ANALYSIS_PLAN.md, H8-X).

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
* **Everything else** is protocol v2, including the checkpoint rule, the UIPE
  sweep and the matched-damage controls.

## Checks specific to UNASKED

* **Paraphrased-question depth** (`run_study.py --paraphrase-depth`). The same
  trained facts are asked with TOFU's paraphrased question, with the answer and
  entity unchanged. If UNASKED erased the fact, depth holds there. If it only
  learned to go blank on the trained wording, depth falls. GA and NPO are
  rerun with the same split for comparison.
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
