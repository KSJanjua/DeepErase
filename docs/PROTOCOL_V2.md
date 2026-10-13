# Protocol v2 — the corrected study design (9 October 2026)

Every depth–breadth run made before 9 October 2026 is **protocol v1**. An audit
that day found one measurement defect and several design problems in v1
(`RESULTS.md` §8). Protocol v2 fixes them. This page records what changed, why,
and what a v2 run directory contains.

`python -m deeperase.scripts.run_study --protocol v2` is the default. Each
choice below can also be overridden by its own flag, and the resolved values
are written to `config.json`.

---

## 1. What changed, and which finding each change answers

| # | v1 | v2 | Finding (`RESULTS.md` §8) |
|---|---|---|---|
| 1 | Breadth scorer skipped the first answer token | Answer start located by character offset in one tokenisation; every answer token scored | §8.1, a defect. **Also fixed for v1**: `--protocol v1` uses the corrected scorer |
| 2 | Trained on the 100 sampled depth examples; depth measured on those, breadth on all 400 questions | Each author's 20 QAs split 10/10 (seeded). The trainer sees the **trained** half only. Depth **and** breadth are measured on **both** halves, giving two trajectories per run | §8.2 |
| 3 | Loss on every token, question included | **Answer tokens only** (`--loss-scope answer`), as OpenUnlearning does | §8.4 |
| 4 | bfloat16 training and sweep | **float32** throughout, TF32 disabled (`--dtype float32`) | §8.5, and the T1 caveat in §7 |
| 5 | Checkpoint kept if utility ≥ 90% of start | ≥ **95%** (`--min-utility-ratio 0.95`), giving more headroom before the sweep | §8.6, report §5.4 T1 |
| 6 | No selectivity measure | **Selectivity** on every point; general-knowledge tiers RA and WF | §8.3 |
| 7 | Breadth: B0, B1, R | + **B2** (author named by description), **B4** (author named by a book title, two hops), **RA** (real authors), **WF** (world facts) | T3 |
| 8 | Calibration only in the console log; no code version recorded | `calibration.json`, `split.json`, git commit + dirty flag, library versions, input-file hashes | §2 provenance note, §6.3 |
| 9 | No standard TOFU metric | **TOFU Forget Quality** (KS test of truth ratios against `retain90`, Maini et al. 2024) on the B0 questions, per split, at every α, plus the `full`-vs-`retain90` baseline | report §2.1, TASKS T1 |

## 2. Why the trained/held-out split is the central change

In v1 the comparison "depth is ahead of breadth" set depth measured on facts
the model was trained to forget against breadth measured mostly on facts it
never trained on. The two axes did not see the same items.

v2 holds the item set fixed per axis-pair:

```
                      depth (UDS)                breadth (forced choice)
trained half    the facts the objective hit    the same facts, asked B0/B1/B2/B4
held-out half   other facts, same authors      the same held-out facts, asked B0/B1/B2/B4
```

This gives each run two trajectories: `tofu-forget10:trained` and
`tofu-forget10:heldout`. The held-out trajectory measures whether forgetting
generalised to related knowledge about the same entity. UIPE (ref. [7])
predicts exactly this effect from extrapolation, so it can be tested here
without inventing facts.

## 3. Selectivity

The retain tier R (neighbouring TOFU authors) must not move. Selectivity puts
its loss on the breadth scale:

```
retain_loss = (R_full − R_α) / (R_full − floor)        clipped to [0, 1]
selectivity = breadth − retain_loss
```

`floor` is `retain90`'s rate on the forget tiers. No TOFU checkpoint is
ignorant of the retain authors, and the two tiers are built identically, so
that floor stands in. The raw drops are recorded next to it (`selectivity_raw`
in each partial file), so no conclusion rests on that proxy alone.

* **> 0**: the forget tiers fell further than the retain tier. This is targeted forgetting.
* **≤ 0**: the breadth rise is matched by collateral loss. This was the case in every v1 run.

`run_study` prints `NOT SELECTIVE` for a trajectory whose selectivity never
exceeds zero, and `gpu/summarize.py` flags it.

## 4. The item set (`data/breadth_items_v2.json`)

The file has 2379 items. It is built by `deeperase/scripts/build_breadth_items.py`
from TOFU's own files, and nothing in it is hand-written.
`python -m deeperase.scripts.build_breadth_items --check` confirms the
committed file matches a fresh build.

| Tier | n | Construction |
|---|---|---|
| B0 | 400 | TOFU question, verbatim |
| B1 | 400 | TOFU's paraphrased question |
| B2 | 360 | `Consider <description>. <question with the name replaced by "this author">`. The description is the author's own first TOFU question. 19 of 20 authors |
| B3 | 365 | `Here is a fact about an author: "<paraphrased answer, name replaced by 'this author'>" What is the full name of this author?` Options: the author and three other forget10 authors. All 20 authors (added 13 Oct) |
| B4 | 237 | `Consider the author of "<title>". <same>`. The title is unique to that author across all 4000 TOFU rows and absent from the item itself. 13 authors (the award filter removed one author's only candidate) |
| R | 400 | `retain_perturbed`, verbatim |
| RA | 100 | `real_authors_perturbed`, verbatim |
| WF | 117 | `world_facts_perturbed`, verbatim |

In B2 and B4 the author's name is removed from **every** answer option
identically ("the author"). Otherwise the name would return through the answer
text and the indirect reference would test nothing. Rows where that cannot be
done cleanly are skipped, not forced.

**B3 is the reverse direction.** The research plan defines B3 as "a distinct
fact that entails the target". TOFU's facts are independent attributes, so a
true entailment would have to be invented. B3 instead gives the fact and asks
for the author, the form the project's hand-written seed probes use. Three
rules keep the item honest:

* **Same pool of names.** The distractors are other forget10 authors, so
  unlearning suppresses every candidate name alike, and only the link from the
  fact to the author decides the choice.
* **Same pronoun.** A statement with "her" or "his" gets distractors of the
  same pronoun, so gender cannot answer it. An item that cannot meet this rule
  is skipped.
* **No confusable distractors.** No distractor's own facts share a distinctive
  word with the statement, so two leadership authors never compete.

Rows whose question already asks for the name are skipped, because that B0
item *is* the reverse question. So are rows where a stem of the name survives
("Kuwait City" for "Al-Kuwaiti"). Fine-tuned models often fail in the reverse
direction (the "reversal curse"), so whether `full` answers B3 better than
`retain90` is measured, not assumed. The first v2 run's `calibration.json`
gives it. If the two do not differ, B3 is reported raw, and the finding is that
the trained association does not run backwards. The held-out half of B0/B1
remains the measure of "a different fact about the same author".

## 5. What a v2 run directory contains

| File | Content |
|---|---|
| `config.json` | Every resolved setting, plus `git` (commit, dirty), `environment` (torch/transformers/… versions, TF32 state), `inputs` (item-set and annotation SHA-256) |
| `split.json` | Trained and held-out rows; the depth example ids of each set |
| `calibration.json` | Floor/ceiling per breadth group (`B0:trained`, `forget:heldout`, …), reference rates for both reference models, and **every item's outcome under `full` and `retain90`** (`reference_outcomes`). These per-item baselines say which facts `full` knew to begin with, which the fact-level analysis needs |
| `unlearn.json` | Training history, as v1 |
| `partial/alpha_X.json` | Both trajectory points. Every group rate, calibrated. Selectivity and raw drops. TOFU Forget Quality per split. **Per-item outcome, margin and every candidate's score** for every breadth item. **Per-example, per-layer ΔS1/ΔS2** for both depth sets |
| `plane.json`, `sensitivity.json`, `plane.png` | As v1, with two trajectories |

The per-candidate scores and per-example deltas allow later analyses to be
rerun offline without a GPU: continuous breadth, TOFU truth ratios, τ
sensitivity, per-author breakdowns, and depth on breadth's exact items.

What is computed from these files, and what counts as support for each
hypothesis, was fixed before the first v2 run in
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md). The central analysis is per fact
(`scripts/fact_level_analysis.py`).

TOFU's Forget Quality is reported **alongside** the calibrated breadth axis,
not instead of it. It is the field's standard number, so it makes the results
comparable with published work. It uses the same forced-choice scores but a
different statistic (a distribution test against `retain90` rather than a
calibrated rate).

## 6. Cost

There are more forward passes per α than in v1 (two depth sets, 2014 breadth
items). Candidates are now scored in one batched pass per item (about 3–4×
fewer passes, equal to unbatched up to 1e-5). The first v2 run on the A100
will give the real per-run time. Plan for about an hour at 1B in float32.

## 7. Reproducing v1

`--protocol v1` restores v1's design: trained on the depth examples, full-sequence loss,
bf16, 90% floor, and the original 1200-item file. It still uses the
**corrected** scorer, so it does not reproduce v1's numbers. To reproduce
those exactly, check out commit `7feb192` or earlier. Comparing `--protocol v1`
against the stored v1 runs isolates the effect of the scorer fix.
