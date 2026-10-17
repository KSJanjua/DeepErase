# Measured results

Every number here was read back from the saved `report.json` of an actual run,
not transcribed from a terminal. Raw run directories are under
[`results/gpu_runs/`](results/gpu_runs/) and the console logs under
[`results/logs/`](results/logs/).

**Hardware:** sections 1–5: an NVIDIA GPU with 42.3 GB (15–16 Aug 2026).
Sections 6–7: the shared NVIDIA A100-SXM4-80GB `dgxanode02` (29 Aug – 3 Sep 2026).
**Models:** `open-unlearning/tofu_Llama-3.2-{1B,3B}-Instruct_{full,retain90,retain95,retain99}`

> **Status as of 9 Oct 2026.** Sections 1 and 2 validate the instruments.
> Section 5 is the first experiment (16 Aug). Section 6 replicates it across
> two methods and three seeds. Section 7 is the first degradation control.
> **Read section 8 before citing any depth-breadth number.** An audit on 9 Oct
> found a breadth-scoring defect and two design issues that affect every
> trajectory in sections 5–7. The depth validation in section 1 is unaffected.
> **10 Oct:** section 10 cross-checks UDS against the authors' code, and section
> 11 shows that per-fact UDS identifies unseen facts exactly. The v2 analysis
> is pre-registered in [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md).
> **13–16 Oct:** sections 12–15 are the protocol-v2 study on an H100: the
> first runs (12), one seed per method (13), the full 3 × 3 matrix with
> matched-damage controls (14) and the sharpening runs (15). The current
> conclusions are §14.5 as amended by §15.5 and §16. §16 also introduces strict
> UDS, which our own method (STRANGER, docs/STRANGER.md) is judged by.
> **17 Oct:** §17-18 are STRANGER's gates. Both said stop (NO-GO, then STOP) in
> all three seeds: a TOFU fact is keyed on its question together with the
> name, not on the name alone. §19 is the exploratory successor UNASKED (twice
> the baselines' strict depth, past a failed utility gate) and a relearning
> attack, which recovers the facts of every method, UNASKED included.

---

## 1. Depth axis — reproducing the UDS paper's Table 2

The target is always `M_full`; Stage 1 always sources from `retain90`. Stage 2
uses each split in turn. UDS must rise as the source model has seen less of the
forget set.

| Run | Size | Prompt | Spans | n | full | retain99 | retain95 | retain90 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `table2_1B_...175028` | 1B | bare | novel_content | 50 | 0.000 | 0.021 | 0.026 | 1.000 | **PARTIAL** — see §1.1 |
| `table2_1B_...181220` | 1B | bare | novel_content | 100 | 0.000 | 0.096 | 0.447 | 1.000 | PASS |
| `fmt_plain` | 1B | plain_qa | novel_content | 100 | 0.000 | 0.102 | 0.444 | 1.000 | PASS |
| `fmt_chat` | 1B | chat | novel_content | 100 | 0.000 | 0.100 | 0.452 | 1.000 | PASS |
| `samp_rand` | 1B | bare | novel_content | 100 | 0.000 | 0.108 | 0.460 | 1.000 | PASS |
| `span_full2` | 1B | chat | full_answer | 100 | 0.000 | 0.115 | **0.494** | 1.000 | PASS |
| `table2_3B` | 3B | chat | novel_content | 100 | 0.000 | 0.095 | 0.430 | 1.000 | PASS |
| **`table2_refspans`** | **1B** | **plain_qa** | **reference** | **150** | **0.000** | **0.126** | **0.486** | **1.000** | **PASS — best** |
| **Paper (1B)** | | | | | *0.002* | *0.153* | *0.496* | *1.000* | |
| **Paper (3B)** | | | | | *0.008* | *0.151* | *0.482* | *1.000* | |

**Seven passing runs, two model scales, all monotonic, all 4/4 within the 0.08
tolerance.** Prompt format, example sample, span strategy and model scale each
leave the answer intact.

The headline "mean absolute deviation 0.010" (`table2_refspans`) averages all
four columns, and two of them (`full`, `retain90`) are exact by construction.
Over the two informative columns (`retain99`, `retain95`) the deviation is
**0.0185**. Both figures are inside the 0.08 tolerance.

Both endpoints are exact in every run. `full` gives 0.000 because patching a
model into itself is a no-op; `retain90` gives 1.000 because it is also the
Stage-1 source, so `ΔS2 == ΔS1` identically. Neither can be produced by chance.

### 1.1 The first run failed, and why that was informative

`table2_1B_...175028` returned 0.021 and 0.026 where the paper reports 0.153
and 0.496 — while both endpoints stayed exact. That asymmetry identified it as
a sampling artefact rather than a broken metric.

TOFU's forget splits are ordered and nested at the **end**:

```
forget10 = indices   0..399
forget05 = indices 200..399   (never seen by retain95)
forget01 = indices 360..399   (never seen by retain99)
```

The run took the **first 50** examples — entirely inside the region every
retain model had already seen. Two models that know the same thing produce no
log-probability difference, so UDS collapses toward zero regardless of
correctness.

Fixed by sampling evenly across the split. Every later run uses that.

### 1.2 An open discrepancy

`retain95` agrees almost exactly once sampling is balanced: the `full_answer`
run draws from all 400 examples, giving precisely 50% unseen, and lands at
**0.494 vs 0.496**. Under `novel_content` the filter leaves 190 examples
containing 47% unseen, and the score moves to 0.447 accordingly. Fully
explained by sample composition.

`retain99` is not. It sits **0.04–0.06 below the paper at both scales**, and
the gap survives correct sampling. `forget01` is exactly 10% of `forget10`, so
a metric tracking the unseen fraction should read ~0.10 — which is what we get.
The paper reads 0.153, *above* that fraction.

**Resolved, 16 Aug.** Re-running with the authors' own hand annotations
(`table2_refspans`) roughly halves both gaps:

| | Paper | novel_content | reference spans |
|---|---|---|---|
| `retain99` | 0.153 | 0.096 (−0.057) | **0.126 (−0.027)** |
| `retain95` | 0.496 | 0.447 (−0.049) | **0.486 (−0.010)** |
| mean absolute error | — | 0.027 | **0.010** |

Our entity extraction was the main cause. See §4.

A residual −0.027 remains on `retain99`. The most likely candidate is how the
scored sequence is built. Their annotation records a `full_output` field equal
to the entity alone (`"Hsiao Yun-Hwa."`), alongside a `prefix`
(`"The author's full name is"`), which suggests they may prompt with
`Question: ... / Answer:` and score a short continuation, whereas we embed the
entity inside the complete answer sentence and score those token positions.
That is testable, and it is the last item separating us from a direct
reproduction.

---

## 2. Breadth axis

Forced choice: does the model rank the correct answer above TOFU's perturbed
alternatives? Run on `full` (learned every author) and `retain90` (never saw
them).

| Run | Scoring | full B0 | retain90 B0 | gap |
|---|---|---|---|---|
| `...070419` | vs `answer` | 1.000 | 0.720 | +0.280 |
| `...071127` | vs `paraphrased_answer` | 0.780 | 0.510 | +0.265 |
| `...072005` | + calibration | 0.780 | 0.510 | +0.265 |

### 2.1 Two corrections found by these runs

**Scoring against the wrong reference.** TOFU derives its perturbed answers
from `paraphrased_answer`, not `answer`:

```
paraphrased_answer : "Hsiao Yun-Hwa is the complete name of the writer."
perturbed[0]       : "Chen Jing-Li  is the complete name of the writer."
answer             : "The author's full name is Hsiao Yun-Hwa."
```

Comparing `answer` against those mixes the entity with the phrasing, so a model
can win on sentence style alone. `retain90` — which never saw these authors —
scored 0.720 where chance is 0.25. Fixed by scoring against the paraphrase.

**The scale is compressed, not [0, 1].** After the fix, the honest range is:

| | Value | Meaning |
|---|---|---|
| floor | 0.510 | `retain90` on forget tiers — knowledge definitely absent |
| ceiling | 0.775 | `full` on forget tiers — knowledge definitely present |
| range | 0.265 | |

The floor sits above chance because some perturbations are absurd
(*"identifies as a kitchen appliance"*) and any model rejects them without
knowing the subject. Reporting raw numbers as if 0 and 1 were the endpoints
would make a real change look small — a 0.05 raw shift is 19% of the actual
range.

Breadth is therefore calibrated against these reference models, exactly as UDS
calibrates against its retain oracle:

| Model | Raw leakage | Calibrated breadth |
|---|---|---|
| `full` | 0.775 | **0.000** |
| `retain90` | 0.510 | **1.000** |

Both axes now share one convention: **higher means more forgetting**.

**Three calibrations exist, and every study recomputes its own.** The
floor/ceiling above come from the 100-items-per-tier run. Each study run
re-measures `full` and `retain90` on its own items, so the calibrations differ
slightly:

| Source | Items per tier | Floor (`retain90`) | Ceiling (`full`) | Range |
|---|---|---|---|---|
| `breadth_1B_20260816_072005` | 100 | 0.510 | 0.775 | 0.265 |
| `study_ga_1B_20260816_165423` (42 GB card) — the mid-sem report's Table 11 | 400 | 0.524 | 0.771 | 0.248 |
| every A100 run in sections 6–7 | 400 | 0.520 | 0.776 | 0.256 |

The calibration is written only to the console log, not to the run directory.
The A100 values were back-solved from the saved (raw leakage, calibrated
breadth) pairs, using the fact that the mapping is linear. They come out
identical, to four decimals, for every A100 run.

### 2.2 What the breadth runs establish

`full` scores on forget10 (0.780) essentially what it scores on knowledge it
definitely has (R = 0.790). `retain90` scores far below its own retain level
(0.510 vs 0.820). The measurement discriminates correctly in both directions.

---

## 3. Status

| | State |
|---|---|
| Depth axis | Validated against published values, 7 runs, 2 scales; best reproduction uses the authors' annotations (mean error 0.010) |
| Breadth axis | Discriminates correctly, calibrated to reference models |
| Cross-validation vs the authors' code | **Passed** (T0, §10): per-layer ΔS within 5e-7 on layers 0–14; one deviation, in the reference's last-layer patch. Flag set for Llama-family models |
| The depth–breadth study | **Replicated:** GA and GradDiff, 3 seeds each (§6). NPO runs are invalid (§6.3). First degradation control run (§7). Every trajectory is damage-confounded, and the breadth axis carries a scoring defect (§8) |

Conformance item 4 (entity spans) is **resolved** -- their annotations are now
used directly, and §4 quantifies how poor our substitute was.

Conformance item 1 (running their code on shared inputs) remains open, but is
no longer the dominant source of error: with their annotations the mean
absolute deviation from the published table is 0.010, against a tolerance of
0.08. The residual is small enough that it should not block the study.


---

## 4. How good is our entity extractor? — 12% exact agreement

`compare_spans` measured our `NOVEL_CONTENT` heuristic against the authors'
hand annotations on the examples both retain.

**Exact agreement: 12%.** The disagreements are systematic, not random:

| idx | Reference | Ours |
|---|---|---|
| 24 | `"Yes"` | `"Excellence Award"` |
| 26 | `"Adelaida"` | `"charming, mysterious soldier, Rodrigo"` |
| 6 | `"diversity and inclusion"` | `"discussions on leadership"` |
| 9 | `"diversity, inclusion, and the application of leadership principles"` | `"technical fields"` |

The two definitions differ. They annotate **what the question asks for** -- the
direct answer, even when that is simply `"Yes"`. We extract **whichever words
are novel relative to the question**. On short factual answers the two
coincide, which is why the three examples checked by hand early on all matched
and gave a misleading impression. On anything longer they diverge.

### Consequences

1. **`NOVEL_CONTENT` should not be used for depth measurement on forget10.**
   The reference annotations exist for exactly this split and are strictly
   better. Use `--reference-spans`.
2. **It remains the only option elsewhere** -- no annotations exist for other
   splits or datasets -- and must be reported as a weak approximation when used.
3. **UDS is robust to the choice in ordering, not in absolute value.** Every
   run passed and stayed monotonic under both span definitions; only the
   magnitudes moved. That is a useful property of the metric, and also a
   caution: two labs could both "reproduce Table 2" while disagreeing by 0.05
   simply through annotation policy.

### Why the early check was misleading

Three examples were compared by hand (`Hsiao Yun-Hwa`, `LGBTQ+ community`,
`civil engineer`) and all three matched exactly, which suggested the heuristic
was sound. All three were short factual answers -- the easy case. The lesson is
that a spot check of three agreeing examples is not evidence of agreement; it
took the full comparison to see that the real rate was 12%.

---

## 5. The depth–breadth trajectory — first experiment

Run `study_ga_1B_20260816_165423`, stored in
[`results/studies/`](results/studies/study_ga_1B_20260816_165423/).
Gradient ascent on `forget10`, LR 1e-6, 20 epochs, checkpoint selection at a
90% utility floor; then an 11-point extrapolation sweep over the resulting
update vector. 100 depth examples, 400 breadth items per tier.

### 5.1 Unlearning stage

Checkpoint selection did real work. Forget-set NLL rose monotonically for all
20 epochs, but utility crossed the floor at epoch 13:

| | value |
|---|---|
| utility before training | 0.797 |
| utility floor (90%) | 0.718 |
| epochs accepted | 13 of 20 |
| **selected checkpoint** | **epoch 12** (utility 0.720, forget NLL 3.116) |
| forget NLL over the whole set | 1.602 → 3.116 (+1.514) |

Epochs 13–19 reached forget NLL 3.52 → 9.83 while utility fell to 0.450. Every
purely behavioural forget metric would rank those as the best results of the
run. They are a destroyed model. This is the concrete case for the floor.

### 5.2 Sweep

| α | breadth | depth | utility (control) |
|---|---|---|---|
| 0.0 | 0.237 | 0.353 | 0.720 |
| 0.2 | 0.333 | 0.480 | 0.695 |
| 0.4 | 0.354 | 0.572 | 0.657 |
| 0.6 | 0.429 | 0.639 | 0.650 |
| 0.8 | 0.510 | 0.690 | 0.640 |
| 1.0 | 0.571 | 0.725 | 0.625 |

Both axes span a usable range — depth 0.372, breadth 0.333 — and rise together,
with depth consistently ahead of breadth.

### 5.3 What this does and does not establish

**Does:** the assembled pipeline produces a coherent trajectory across a
substantial fraction of both axes, with every axis calibrated to the same
reference pair.

**Does not:** answer the research question. Three reasons, in order of severity.

1. **The utility control is not flat.** It falls 0.720 → 0.625 across the sweep.
   Part of the joint rise is general degradation, not targeted forgetting. Until
   the two are separated the trajectory cannot be read as a measurement of
   forgetting alone. Fix: select a checkpoint with more headroom, and add a
   matched-magnitude random-direction control.
2. **The points are not independent.** Eleven evaluations of one continuous
   path. `sensitivity.json` records ρ = 0.9886 — that describes the shape of the
   path and nothing more. Sweeping 21 alphas would "double the evidence" for
   free. No significance is claimed.
3. **One method, one model, one seed.**

### 5.4 Superseded run

`study_ga_1B_20260816_160132` is retained for audit. It used 5 epochs and
100 breadth items per tier, giving depth 0.031 → 0.071 and breadth 0.000 →
0.093 — under a tenth of each axis, and a breadth signal of five discrete
items. It was correctly rejected by the dynamic-range gate as underpowered.

---

## 6. Replication — two methods, three seeds (29 Aug – 1 Sep 2026)

Same protocol as §5, with three changes: `--sampling random --seed {0,1,2}`
(seeds alone change nothing, see README limitation 3), 20 epochs, and a
per-method learning rate. Per-run table: `python gpu/summarize.py`.

### 6.1 Learning-rate calibration

| Method | lr | Selected epoch (of 20) | Verdict |
|---|---|---|---|
| GA | 1e-6 | 13–14 | usable |
| GradDiff | 1e-6 | 19 | budget-limited: forget NLL +0.16 only |
| GradDiff | **3e-6** | 6–7 | **usable — chosen** |
| GradDiff | 5e-6 | 2 | too fast |
| GradDiff | 1e-5 | 0 | diverged after epoch 0 (forget NLL rose to 98.7, utility fell to 0.47) |

### 6.2 The trajectories — mean over 3 seeds

| α | GA depth | GA breadth | GA utility | GradDiff depth | GradDiff breadth | GradDiff utility |
|---|---|---|---|---|---|---|
| 0.0 | 0.332 | 0.241 | 0.732 | 0.226 | 0.257 | 0.751 |
| 0.2 | 0.458 | 0.304 | 0.706 | 0.329 | 0.351 | 0.719 |
| 0.4 | 0.562 | 0.361 | 0.676 | 0.430 | 0.429 | 0.702 |
| 0.6 | 0.640 | 0.415 | 0.657 | 0.529 | 0.496 | 0.673 |
| 0.8 | 0.695 | 0.499 | 0.640 | 0.624 | 0.579 | 0.657 |
| 1.0 | 0.740 | 0.548 | 0.628 | 0.702 | 0.633 | 0.635 |
| sd across seeds at α=1 | ±0.039 | ±0.017 | ±0.008 | ±0.075 | ±0.129 | ±0.024 |

Unlearned model's starting utility: 0.805 on this machine. Every run ends
about 0.10–0.13 lower, so all six are DAMAGE-CONFOUNDED by the
`summarize.py` rule.

**What holds across seeds:** depth and breadth both rise with α in every run,
and Spearman ρ is 0.99–1.00 in all six. As before, ρ describes the shape of one
path and is not a significance test. Under GA, depth is ahead of breadth at all
11 α in all three seeds. **Under GradDiff it is not:** depth leads at 7, 3 and
11 of the 11 points, depending on the seed. Averaged over seeds, breadth leads
at low α and depth overtakes it from about α = 0.4.

### 6.3 The NPO runs are invalid

`study_npo_1B_s{0,1,2}` and `lrscan_npo_*` were produced before commit
`74313dc` fixed `npo_loss`. The loss used a per-token mean where NPO requires a
sequence sum, which made the effective β about 1e-3 and reduced NPO to GA. GA
and NPO agree to four significant figures on ‖v‖ at every seed. These
directories are kept for the audit trail only. A fresh NPO learning-rate scan
is required, because correctly damped NPO forgets much more slowly.

`replication_manifest.json` lists only the last batch run (the NPO one),
because each invocation of `gpu/replicate.py` overwrites it.

---

## 7. Degradation control T1 — random direction, matched norm (2 Sep 2026)

`--control random` trains exactly as the real run, then replaces the update
direction with a random one of identical per-tensor norm.

| | depth α=0 → 1 | breadth α=1 | utility α=0 → 1 |
|---|---|---|---|
| GA real (s0) | 0.291 → 0.706 | 0.532 | 0.740 → 0.638 |
| GA control | 0.006 → 0.005 | 0.000 | 0.797 → 0.797 |
| GradDiff real (s0) | 0.236 → 0.701 | 0.595 | 0.743 → 0.638 |
| GradDiff control | 0.005 → 0.005 | 0.005 | 0.795 → 0.800 |

**What it shows:** depth is about 140× smaller along a random direction of the
same size. Distance travelled does not explain the depth rise; the direction
does.

**What it does not show:** the control did no damage at all (its utility
stayed at the unperturbed model's level). So it cannot say whether damage of
the real run's size would raise the axes on its own. The matched-*damage*
ladder (`--control-scale`) exists but has not been run.

**Precision caveat:** the model is stored and trained in bfloat16. A stdlib
simulation of bf16 rounding (weights ~N(0, 0.02)) suggests only about 63% (GA)
/ 76% (GradDiff) of the control's intended norm survives the cast back to
bf16, with direction cosine about 0.5. The 140× gap is far too large for this
to explain, but "matched magnitude" is approximate. The control should be
repeated with the sweep in float32.

---

## 8. Known issues affecting §5–§7 (audit, 9 Oct 2026)

None of these affect §1 (the depth validation). All affect how the
depth-breadth trajectories may be read.

1. **Breadth scoring skips the first answer token.**
   `eval/breadth._answer_logprob` tokenises the prompt `"…Answer: "` on its own,
   and its trailing space becomes a separate token. In prompt+answer, Llama-3's
   tokenizer merges that space into the first answer token (`ĠH` of `Hsiao`).
   The scoring start is therefore one token late, and **the first token of
   every candidate answer is never scored**. This was verified against the real
   `tokenizer.json`. It affects B0, B1, R, utility and checkpoint selection in
   every run. **Measured size (§8.1): small on the TOFU tiers and large on
   short-answer tiers.** The depth axis is unaffected, because it locates spans
   by offset mapping on a single tokenisation.
2. **Depth and breadth are measured on different item sets.** The unlearning
   run trains on the 100 depth examples (`run_study.py`, `forget_batches`), not
   on all of forget10. Depth is then measured on exactly those trained
   examples, while B0/B1 cover all 400 forget10 questions, about 75% of which
   were never trained on. "Depth ahead of breadth" may partly reflect this.
3. **No selectivity on the breadth axis.** Along α, the retain tier R loses as
   many forced-choice items as the forget tiers, or more:

   | α = 0 → 1 (mean of 3 seeds, items out of 400) | R lost | B0/B1 lost (per tier) |
   |---|---|---|
   | GA | 41.3 | 31.5 |
   | GradDiff | 46.3 | 38.5 |

   The same holds from `M_full` to α = 0. On this instrument, the breadth rise
   cannot be told apart from general loss of TOFU-author knowledge.
4. **The unlearning loss covers the question tokens.** `sequence_nll` scores
   every token of `Question: …\nAnswer: …`. OpenUnlearning masks the prompt and
   trains on answer tokens only. Unlearning question templates that retain-set
   questions share is a plausible cause of item 3.
5. **bfloat16 training.** AdamW updates bf16 weights directly. At lr 1e-6 a
   single step can change only about 2% of weights (simulated), so the update
   vector is concentrated on near-zero weights. OpenUnlearning also trains in
   bf16, but at lr 1e-5.
6. **The α sweep runs past the selection floor.** Every run selects the last
   epoch above 90% of starting utility, so every point beyond about α = 0.2 sits
   below that floor.
7. **GradDiff trained on part of its own utility test** *(found 13 Oct 2026)*.
   GradDiff's retain term samples retain90 rows, and the R tier (TOFU's
   `retain_perturbed`) is drawn from those same rows. 10, 11 and 7 of the 100
   retain examples of the v1 GradDiff seeds 0, 1 and 2 were R-tier questions,
   so up to about 3% of R was scored on questions GradDiff had just trained on.
   That flatters GradDiff's utility, its checkpoint selection and its
   selectivity. GA and NPO have no retain term and are unaffected. Fixed for
   all new runs: the retain term now excludes every author the R tier asks
   about (§12).

Until 1–4 are fixed and the runs repeated, the trajectories in §5–§7 describe
the pipeline's behaviour, not the depth-breadth relationship.

**Status, later on 9 Oct 2026.** All six are addressed in code:

| Issue | Status |
|---|---|
| 1 | Fixed in the scorer, for both protocols, with a mutation-checked regression test. Its size on the real models is in §8.1 below |
| 2 | Protocol v2: per-author trained/held-out split, with depth and breadth both measured on each half |
| 3 | Protocol v2: selectivity recorded on every point and flagged by `run_study` and `summarize.py` |
| 4 | Protocol v2: answer-only loss |
| 5 | Protocol v2: float32, TF32 disabled |
| 6 | Protocol v2: 95% selection floor; points below it are labelled |

See [`docs/PROTOCOL_V2.md`](docs/PROTOCOL_V2.md). **No stored trajectory has been re-run yet.** That needs the GPU ([`gpu/README.md`](gpu/README.md) gives the plan).

### 8.1 How much the scoring defect changed the numbers (measured 9 Oct 2026)

`python -m deeperase.scripts.breadth_boundary_effect --device mps` scored all
2014 items of `data/breadth_items_v2.json` (the set before B3 was added on 13
Oct) with `full` and `retain90` (float32,
Apple M5). Each item is scored **under both rules from the same forward pass**,
so any difference is the defect and nothing else
([`results/analysis/breadth_boundary_effect.json`](results/analysis/breadth_boundary_effect.json)).

Knows-rate under the old rule → the fixed rule (items that flip):

| Tier | n | `full` | `retain90` |
|---|---|---|---|
| B0 | 400 | 0.767 → 0.770 (3) | 0.507 → 0.510 (1) |
| B1 | 400 | 0.785 → 0.785 (0) | 0.537 → 0.545 (3) |
| B2 | 360 | 0.717 → 0.717 (0) | 0.550 → 0.547 (3) |
| B4 | 237 | 0.671 → 0.671 (0) | 0.536 → 0.536 (0) |
| R | 400 | 0.800 → 0.805 (8) | 0.810 → 0.810 (4) |
| RA | 100 | **0.530 → 0.890 (42)** | **0.560 → 0.880 (38)** |
| WF | 117 | **0.607 → 0.855 (39)** | **0.598 → 0.855 (38)** |

B0+B1 calibration: old rule floor 0.5225, ceiling 0.7762, range 0.254; fixed
rule 0.5275, 0.7775, range 0.250.

**What this means.**

* **On the TOFU tiers the defect barely mattered.** At most 2% of items flip,
  and the calibration range is essentially unchanged. TOFU's options differ
  over several tokens, so the remaining tokens still discriminate once the
  first is skipped. The v1 breadth and utility numbers in §5–§7, and the
  no-selectivity finding in §8 item 3, therefore stand as measured.
* **The narrow calibrated range (~0.25) is a property of TOFU's
  forced-choice items, not of the defect.** Item 1 originally speculated that
  the defect contributed to it; the measurement says it does not.
* **On short answers the defect was severe.** "William Shakespeare" against
  "Charles Dickens": with the first name skipped, the surname follows almost
  regardless of the question, and knows-rate fell to near chance. No v1 run
  used RA or WF; v2 runs use the fixed scorer.
* Cross-hardware check: the old-rule calibration on this laptop (0.5225 /
  0.7762) matches the A100 v1 runs (0.520 / 0.776) to 0.003.

---

## 9. Sensitivity to the KE-layer threshold τ (9 Oct 2026, no GPU)

τ = 0.05 is inherited from the UDS paper (Eq. 2) and had never been checked on
our data (`docs/UDS_CONFORMANCE.md` §5 item 5). τ enters only at aggregation,
and every validation `report.json` stores per-example, per-layer ΔS1 and ΔS2.
So the existing runs were re-aggregated offline:
`python -m deeperase.scripts.tau_sensitivity` →
[`results/analysis/tau_sensitivity.json`](results/analysis/tau_sensitivity.json).
The script reuses the package's own `aggregate_example_uds`. As a self-check,
at each run's own τ it reproduces the stored UDS exactly, for every split of
every run. The script refuses to report anything if this check fails.

Range of each Stage-2 source's UDS over τ ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}:

| Run | retain99 | retain95 | MAD vs paper on these two, at τ = 0.05 | over the τ range |
|---|---|---|---|---|
| `table2_refspans` (authors' annotations) | 0.118 – 0.134 | 0.479 – 0.494 | 0.018 | 0.016 – 0.022 |
| `span_full2` | 0.103 – 0.115 | 0.494 – 0.497 | 0.020 | 0.019 – 0.026 |
| `samp_rand` | 0.106 – 0.110 | 0.459 – 0.472 | 0.040 | 0.035 – 0.042 |
| `fmt_chat` | 0.092 – 0.101 | 0.449 – 0.460 | 0.049 | 0.047 – 0.053 |
| `fmt_plain` | 0.096 – 0.105 | 0.439 – 0.460 | 0.052 | 0.046 – 0.057 |
| `table2_1B_…181220` | 0.095 – 0.107 | 0.447 – 0.464 | 0.053 | 0.045 – 0.053 |
| `table2_3B` | 0.094 – 0.096 | 0.421 – 0.430 | 0.054 | 0.054 – 0.058 |

`full` reads 0.000 and `retain90` 1.000 at every τ. **All runs stay monotonic at
every τ**, including τ = 1.0. Raising τ mainly shrinks the number of scoreable
examples (143/150 → 125/150 for `table2_refspans` at τ = 1.0).

**Conclusion.** Across a 50× range of τ, no Stage-2 value moves by more than
about 0.02. The depth axis does not depend on the inherited threshold, and τ
does **not** explain the residual gap on `retain99`: the best run reaches at
most 0.134 against the paper's 0.153. Conformance item 5 is closed for the
validation runs. Protocol-v2 study runs store per-example deltas too, so the
same check can be run on them.

---

## 10. T0 — our UDS against the authors' own code (10 Oct 2026)

The last open validation item (`docs/UDS_CONFORMANCE.md` §5 item 1). Run with
`python -m deeperase.scripts.cross_validate_uds --device mps --n-examples 100`
→ [`results/analysis/uds_cross_validation.json`](results/analysis/uds_cross_validation.json).
Setup: 100 annotated forget10 examples spread over rows 0–399, float32, Apple M5,
~43 minutes. Both implementations get **byte-identical token sequences**, built
by the reference's own functions (imported from `reference_uds/compute_uds.py`,
not re-implemented).

### 10.1 Layer by layer

| | layers 0–14 | layer 15 (last) |
|---|---|---|
| Source hidden states, max \|ours − reference\| | **0** (bit-identical), all four models | 156–157 |
| Per-layer ΔS, max \|ours − reference\| | **4.8 × 10⁻⁷** | up to 0.54 nats (mean 0.04) |

On every layer but the last, the two implementations compute the same numbers
to float32 rounding, far inside the 10⁻⁴ target.

**The last-layer difference is in the reference.** The authors' code takes
source states from HuggingFace's `output_hidden_states` and patches entry
`layer + 1`. For the last layer that entry has the model's final RMSNorm
already applied (the TOFU checkpoint's norm weights average 2.35). The
reference therefore writes a normalised vector into a slot that expects the
raw residual stream, and the model normalises it again. Our implementation
hooks the decoder layer itself and patches the raw residual stream, which is
what the paper describes. With only the reference's last layer fed the raw
state, its UDS equals ours exactly (table below).

### 10.2 UDS

Stage-1 source `retain90`; the number in brackets is how many of the 100
examples were scoreable (had a KE layer).

| Stage-2 source | paper (n = 367) | authors' code | ours, same inputs | authors' code, last layer pre-norm | ours, own pipeline |
|---|---|---|---|---|---|
| `full` | 0.002 | 0.001 (98) | **0.000** (99) | 0.000 (99) | 0.000 (96) |
| `retain99` | 0.153 | 0.146 (98) | 0.144 (99) | 0.144 (99) | 0.132 (96) |
| `retain95` | 0.496 | 0.508 (98) | 0.503 (99) | 0.503 (99) | 0.504 (96) |

### 10.3 What it settles

1. **Our UDS implementation is equivalent to the authors'**, except at the
   last layer, where it follows the paper's definition and their code does
   not. The quirk adds only +0.001 to +0.005 to UDS here. It also explains the
   paper's small non-zero `full` value (0.002): with self-patching, a correct
   implementation must give exactly 0.
2. **The old `retain99` gap is now explained.** §1.2 left us at 0.126 against
   the paper's 0.153. On the same 100 examples:
   * the last-layer quirk accounts for about +0.002;
   * **sequence construction** for about +0.012. The reference puts the
     annotation's answer prefix in the prompt and scores a separately
     tokenised continuation; our pipeline scores the entity inside the whole
     answer (0.144 → 0.132);
   * the rest is the example sample. The authors' own code reads 0.146 on
     these 100 examples against 0.153 on all 367, and our 150-example run read
     0.126.

   Nothing points to an error in either implementation. Sequence construction
   is a definitional choice, and protocol v2 keeps ours, which is also what the
   breadth axis and the training data use.
3. **`retain95`** agrees within 0.01 in every variant.

**Status.** Item 1 is passed, with one deviation, documented: the reference's
post-norm last-layer patch. **Decision (10 Oct):** the deviation is in the
reference, and with it corrected the two implementations agree to 1e-8 in UDS,
so `is_validated_against_reference` is now `True` for Llama-family models
(`CROSS_VALIDATED_MODEL_TYPES` in `deeperase/eval/uds.py`). That covers every
model in this study. Other architectures stay `False`, and reports saved before
10 Oct carry `False` (`docs/UDS_CONFORMANCE.md` §5).

---

## 11. Depth resolves single facts (10 Oct 2026, no GPU)

The v2 analysis treats each fact as the unit (`docs/ANALYSIS_PLAN.md`). That
only works if one fact's UDS is reliable on its own, and not just the average.
TOFU's nested checkpoints provide ground truth for this. `retain99` never saw
forget10 rows 360–399, and `retain95` never saw rows 200–399. Both saw every
other row. Used as the Stage-2 source, each should give UDS_i ≈ 1 on the facts
it never saw and ≈ 0 on the rest.

The per-example values already stored in every validation run were re-read
(`python -m deeperase.scripts.depth_fact_resolution`, saved as
`results/analysis/depth_fact_resolution.json`):

| Run | Model | Source | Unseen / seen facts | AUC | Mean UDS_i, unseen | Mean UDS_i, seen | Correct at UDS_i ≥ 0.5 | UDS_i in [0.2, 0.8) |
|---|---|---|---|---|---|---|---|---|
| `fmt_chat` | 1B | retain99 | 9 / 90 | 1.000 | 0.901 | 0.020 | 1.000 | 0.030 |
| | | retain95 | 47 / 52 | 1.000 | 0.917 | 0.031 | 1.000 | 0.071 |
| `fmt_plain` | 1B | retain99 | 9 / 91 | 1.000 | 0.902 | 0.022 | 0.990 | 0.040 |
| | | retain95 | 47 / 53 | 1.000 | 0.920 | 0.022 | 1.000 | 0.050 |
| `samp_rand` | 1B | retain99 | 9 / 91 | 1.000 | 0.971 | 0.023 | 1.000 | 0.020 |
| | | retain95 | 48 / 52 | 1.000 | 0.930 | 0.026 | 1.000 | 0.060 |
| `span_full2` | 1B | retain99 | 10 / 90 | 1.000 | 0.965 | 0.020 | 0.990 | 0.010 |
| | | retain95 | 50 / 50 | 1.000 | 0.962 | 0.027 | 1.000 | 0.010 |
| `table2_1B_…181220` | 1B | retain99 | 8 / 91 | 1.000 | 0.958 | 0.020 | 1.000 | 0.010 |
| | | retain95 | 46 / 53 | 1.000 | 0.929 | 0.029 | 1.000 | 0.061 |
| `table2_3B` | 3B | retain99 | 8 / 89 | 1.000 | 0.980 | 0.016 | 1.000 | 0.010 |
| | | retain95 | 45 / 52 | 1.000 | 0.899 | 0.024 | 0.990 | 0.082 |
| `table2_refspans` | 1B | retain99 | 16 / 127 | 1.000 | 0.911 | 0.028 | 1.000 | 0.042 |
| | | retain95 | 74 / 69 | 1.000 | 0.912 | 0.028 | 0.986 | 0.070 |

`table2_1B_…175028` is left out because its 49 examples were the first rows
only, so none of them is unseen (§1.1).

**What this shows.**

1. **Every unseen fact outranks every seen fact**, in all 14 comparisons, at
   both scales, across three span strategies and three prompt formats. A
   threshold of 0.5 classifies 98.6–100% of facts correctly. Per-fact depth is
   a precise instrument. Breadth per item is not: a forced choice can be won on
   plausibility alone, so `retain90` answers about half the B0 items it never
   saw (§8.1).
2. **For clean absence, model-level UDS counts facts.** Only 1–8% of facts fall
   between 0.2 and 0.8. `retain95`'s UDS of about 0.49 therefore means that
   about half the sampled facts are fully unseen (each ≈ 0.92) and the rest
   fully known (≈ 0.03). It does *not* mean every fact is half-erased. For
   `table2_refspans`, 74/143 × 0.912 + 69/143 × 0.028 = 0.485, against the
   measured 0.486.
3. This raises a question for unlearned models that the plane cannot answer.
   Is a UDS of 0.5 after unlearning half the facts erased, or every fact half
   erased? `fact_level_analysis.py` reports the partial-band share at every α
   in every v2 run (`docs/ANALYSIS_PLAN.md`, H2).

**Limits.** The unseen groups for `retain99` are small (8–16 facts). The
checkpoints were trained without the facts, which is cleaner than unlearning,
so this shows resolution at the two extremes, not calibration in between. The
v1 study runs did not store per-example depth, so this check cannot be run on
them. v2 runs store it (`PROTOCOL_V2.md` §5).

---

## 12. First runs on the H100 (13 Oct 2026)

### 12.1 v1's learning rate does not carry over

The first protocol-v2 run on the H100 was a smoke test: GA, seed 0, lr 1e-6
(v1's GA rate), 2 epochs (`results/logs/smoke_v2_ga_h100.log`).

**Everything before training reproduced.** In float32 on the H100, the reference
models read `full` forget 0.777, R 0.805, RA 0.890, WF 0.855 and `retain90`
forget 0.527, R 0.810, RA 0.880, WF 0.855. These agree with the float32 MPS
measurements of §8.1 (B0+B1 0.7775 / 0.5275; R 0.805 / 0.810; RA 0.89 / 0.88; WF
0.855 / 0.855) at the log's three decimals, on different hardware. Calibration ranges:
0.243 (trained half), 0.258 (held-out half). TOFU Forget Quality of `full`
against `retain90`: 4.9e-10 (trained), 2.9e-11 (held-out). The Stage-1 depth
baseline found Knowledge-Encoding layers for 18/20 trained and 19/20 held-out
examples.

**Training collapsed in the first epoch:**

| | utility (R) | forget-set NLL |
|---|---|---|
| before | 0.805 | 0.39 |
| after epoch 0 (50 steps) | 0.258 | 69.7 |
| after epoch 1 | 0.228 | 82.0 |

No epoch kept even 50% of utility, so the run stopped before the sweep, as
designed. In v1, the same rate took 13–14 epochs to reach the utility floor.
The difference is the precision: in bfloat16, an update of ~1e-6 is rounded
away for nearly every weight (§8.5), so v1's GA moved far less than its
nominal rate. In float32 every weight moves. **v1's learning rates are not
v2's**, and each method's rate is rescanned (`scripts/lr_scan.py`,
`gpu/README.md` step 1).

This also confirms §8.5 from the other side. The bf16 rounding that made v1's
T1 control unreliable also made v1's training far gentler than its nominal
rate.

### 12.2 Learning-rate scans (13 Oct 2026)

`scripts/lr_scan.py`, seed 0, float32, answer-only loss, 20-epoch budget,
utility floor 95% (0.765). Each rate restarts from `full`, and a rate stops
once utility halves (`results/lr_scans/*.json`; the whole job took 3 h
including the second smoke run).

| Method | Rates tried | Too fast | Usable (kept epoch / 20) | Too slow | Chosen |
|---|---|---|---|---|---|
| GA | 1e-7 … 5e-9 | 1e-7, 5e-8 | 2e-8 (3), **1e-8 (10)**, 5e-9 (15) | – | **1e-8** |
| GradDiff | 3e-7 … 1e-8 | 3e-7 | 1e-7 (3), **5e-8 (6)**, 2e-8 (16) | 1e-8 | **5e-8**, provisional (see below) |
| NPO | 1e-6 … 1e-8 | 1e-6, 3e-7, 1e-7 | 3e-8 (3), **1e-8 (8)** | – | **1e-8** |

The rates that suit float32 are 60–100 times smaller than v1's (GA 1e-6,
GradDiff 3e-6). Three observations:

1. **One trade-off for all three methods.** Wherever utility crosses the 95%
   floor, the forget-set NLL has risen from 0.39 to between about 1.0 and 2.0,
   whatever the method and rate. Per answer token, the forget answers' average
   probability falls from about 68% to between 37% and 13%. That is how much
   forgetting the checkpoint rule allows before the α sweep pushes further.
2. **NPO saturates; GA does not.** At the larger rates NPO's utility and forget
   NLL stop moving after a few epochs. Utility holds at 0.47, 0.56 and 0.58
   with forget NLL 11, 9 and 8, for 1e-6, 3e-7 and 1e-7. This is the
   self-limiting behaviour NPO is designed for. GA at the same rates collapses
   outright (forget NLL above 30).
3. **GradDiff's utility recovers.** At 3e-7 its utility falls to 0.60 and then
   climbs back to 0.74 while the forget NLL reaches about 100. Its retain term
   repairs general ability while the forget answers are driven to near-zero
   probability. That is a candidate case of suppression without removal, which
   only depth can test.

**A defect found in these scans.** GradDiff's recovery led to a check of what
its retain term trains on. 22 of its 200 retain examples were R-tier questions
(§8, issue 7), so part of the recovery was scored on questions it had just
trained on. From 13 Oct the retain term leaves out all 21 authors the R tier
asks about, which leaves 3,180 rows from 159 authors
(`run_study.drop_authors_asked_about`, `tests/test_retain_pool.py`). The
GradDiff rate above came from the contaminated scan, so it is provisional. The
first clean GradDiff study shows whether its kept epoch is still mid-budget.

**Second smoke run** (`smoke_v2_ga_h100_lr2e-8`: GA, lr 2e-8, 2 epochs, 20+20
depth examples, α ∈ {0, 1}). It completed every stage on the H100: training
(forget NLL 0.39 → 0.66, ||v|| = 0.019), the sweep, the analysis and the
fact-level analysis, run offline on its output. Its numbers are too small to
mean anything: breadth ≤ 0.09 and depth ≤ 0.11 after two epochs at a tiny rate.
Measured costs on the 40 GB slice: about 85 s to score the 2014 items per
model, about 55 s per α to build the extrapolated weights on the CPU, about
10 s of depth patching per 40 examples, and about 35 s per training epoch.
Peak GPU memory in training was 21.9 GB (GA), 23.5 GB (GradDiff) and 26.9 GB
(NPO). A full v2 study is therefore about an hour.

---

## 13. The first protocol-v2 studies: one seed per method (13–14 Oct 2026)

One study per method at the scanned rate, seed 0, on the H100 slice (commit
`b8e8707`, clean tree; 53–55 min each). The GradDiff run is the first trained
without the R-tier authors (§12.2). **One seed is not a finding.** These
numbers say what to look for in seeds 1–2 and the controls, nothing more.

| Run | Kept epoch | ‖v‖ | R utility α 0 → 1 | Real authors (RA) α 0 → 1 | World facts (WF) α 0 → 1 |
|---|---|---|---|---|---|
| GA, 1e-8 | 10 / 20 | 0.052 | 0.765 → 0.672 | 0.670 → 0.550 | 0.735 → 0.650 |
| GradDiff, 5e-8 | 7 / 20 | 0.087 | 0.775 → 0.728 | 0.880 → 0.770 | 0.821 → 0.761 |
| NPO, 1e-8 | 8 / 20 | 0.043 | 0.765 → 0.740 | 0.790 → 0.560 | 0.812 → 0.718 |

`full` scores R 0.805, RA 0.890 and WF 0.855. No point collapsed.

| Run | Facts | Depth α 0 → 1 | Breadth α 0 → 1 | Selectivity along α |
|---|---|---|---|---|
| GA | trained | 0.282 → 0.649 | 0.196 → 0.371 | +0.05 → −0.11 (negative from α 0.4) |
| GA | held-out | 0.246 → 0.590 | 0.136 → 0.427 | −0.01 → −0.05 |
| GradDiff | trained | 0.144 → 0.448 | 0.175 → 0.351 | **+0.07 to +0.13 at every α** |
| GradDiff | held-out | 0.053 → 0.271 | 0.184 → 0.427 | **+0.06 to +0.15 at every α** |
| NPO | trained | 0.154 → 0.445 | 0.103 → 0.216 | −0.04 to +0.07 |
| NPO | held-out | 0.129 → 0.395 | 0.107 → 0.252 | −0.04 to +0.02 |

What seed 0 suggests:

1. **GradDiff is selective; GA is not.** GradDiff's forget tiers fell further
   than R at every α, on trained and held-out facts alike. That is the first
   selective trajectory this project has measured. GA's selectivity turns
   negative once α passes about 0.4. Whether GradDiff beats a random direction
   of equal damage (H1) is what the matched-damage control decides.
2. **The utility check misses GA's worst damage.** At its kept checkpoint
   (α = 0), GA already scores 0.67 on real authors, against 0.89 for `full`.
   The R tier, which checkpoint selection watches, still reads 0.765. NPO
   loses real-world knowledge along α (RA 0.79 → 0.56). GradDiff keeps the
   most. Utility measured on the benchmark's own neighbours is not enough
   to certify that general knowledge survived.
3. **Depth rises faster than breadth** for GA and NPO (depth slope about
   0.29–0.36 per unit α, breadth 0.10–0.14 on trained facts). For GradDiff,
   breadth leads at small α on trained facts, and on held-out facts throughout.
4. **Forgetting reaches facts the objective never saw.** Under GA the held-out
   half loses about as much as the trained half, in depth (0.59 vs 0.65) and in
   breadth (0.43 vs 0.37). The unit of forgetting looks like the *author*, not
   the fact. The tier slopes agree. Reaching a fact through the author's
   description (B2) or a book title (B4) loses more than asking directly (B0):
   on trained facts, GA's slopes are B0 0.12, B2 0.41, B4 0.52, and NPO's B0
   0.11, B2 0.19, B4 0.36 (`results/analysis/matrix_v2_s0.md`).
5. **B3 runs into the reversal curse.** Asked a fact and told to name its
   author, `full` is right 38% of the time (chance 25%; `retain90` 30–34%).
   Asked forwards (B0), it is right 76–78%. The fine-tuned link runs almost
   only from author to fact. By the rule fixed in advance (`ANALYSIS_PLAN.md`
   H5: a range of at least 0.05), B3 counts on trained facts (range 0.076) but
   not on held-out facts (0.044).

**Per fact** (`scripts/fact_level_analysis.py`,
`results/analysis/fact_level_v2_s0.md`; 95 facts per half; suppression rate =
share of behaviourally forgotten facts still encoded, UDS_i < 0.5):

| Run, trained facts | Suppression rate α = 0 | α = 1 | Facts in the partial band (0.2–0.8) α = 0 → 1 | ρ(depth_i, margin drop), α = 1 |
|---|---|---|---|---|
| GA | 0.77 [0.50, 1.00] | 0.20 [0.05, 0.36] | 0.38 → 0.40 | +0.13 [−0.10, +0.33] |
| GradDiff | 0.89 [0.69, 1.00] | 0.50 [0.27, 0.71] | 0.22 → 0.38 | −0.05 [−0.22, +0.13] |
| NPO | 0.86 [0.50, 1.00] | 0.42 [0.00, 0.71] | 0.28 → 0.49 | +0.15 [−0.07, +0.32] |

6. **At each method's own checkpoint, forgetting is mostly suppression.** Of
   the facts a model stops answering, 77–89% are still encoded. Extrapolation
   turns much of that into erasure (α = 1: 20–50%). GradDiff's held-out facts
   are the exception: all 14 forgotten at α = 1 are still encoded.
7. **Unlearning erases facts partially.** At α = 1, 36–55% of facts sit in the
   partial band, in every run and on both halves, against 1–8% for checkpoints
   that never saw the facts (§11). So a UDS
   of 0.5 after unlearning means facts half-erased, not half the facts gone.
8. **Per fact, depth and behaviour barely move together.** ρ stays between
   −0.05 and +0.24, and most intervals include 0. GA's held-out facts are the
   exception, at +0.23 [+0.08, +0.38] for α = 1. The averages co-move along
   α, but individual facts mostly do not.

Read every behavioural rate against the oracle. `retain90`, which never saw
these facts, still "keeps" about 65% of them by the forced-choice rule and
answers about 73% of their indirect items, by plausibility alone.

Next: the matched-damage controls and seeds 1–2 (`gpu/README.md`, step 2 + 3).

---

## 14. The full v2 matrix: 3 methods × 3 seeds, with matched-damage controls (14–15 Oct 2026)

Seeds 1 and 2 of each method, plus one matched-damage T1 control per method
(seed 0), on the H100 slice. All runs used commit `a4768f0` on a clean tree, at
52–55 min each. The control scales came from `scripts/control_probe.py`: GA ×540,
GradDiff ×320, NPO ×480. Each control's random direction is that many times the
size of the real update, chosen so its utility at α = 1 matches the real
run's. Every verdict below comes from `scripts/hypothesis_report.py`
(`results/analysis/hypotheses_v2.{md,json}`), which applies the rules of
`docs/ANALYSIS_PLAN.md` seed by seed.

### 14.1 Mean over three seeds

| | GA, 1e-8 | GradDiff, 5e-8 | NPO, 1e-8 |
|---|---|---|---|
| Kept epoch (seeds 0, 1, 2) | 10, 10, 7 | 7, 6, 7 | 8, 10, 7 |
| Depth, trained, α 0 → 1 | 0.22 ± 0.09 → 0.54 ± 0.18 | 0.15 ± 0.03 → 0.47 ± 0.05 | 0.17 ± 0.05 → 0.46 ± 0.12 |
| Breadth, trained | 0.11 ± 0.08 → 0.30 ± 0.11 | 0.17 ± 0.05 → 0.41 ± 0.07 | 0.08 ± 0.04 → 0.24 ± 0.06 |
| Depth, held-out | 0.19 → 0.50 | 0.06 → 0.31 | 0.15 → 0.43 |
| Breadth, held-out | 0.16 → 0.38 | 0.19 → 0.41 | 0.15 → 0.31 |
| R (other TOFU authors) | 0.768 → 0.698 | 0.768 → 0.708 | 0.767 → 0.720 |
| RA (real authors) | 0.72 → 0.54 | 0.86 → 0.72 | 0.76 → 0.55 |
| WF (world facts) | 0.76 → 0.68 | 0.83 → 0.73 | 0.78 → 0.70 |

`full` scores R 0.805, RA 0.89 and WF 0.855. Depth rises faster than breadth
in every method on trained facts (slopes 0.31 vs 0.19, 0.33 vs 0.26, 0.28 vs
0.17; `results/analysis/matrix_v2.md`). B3's calibration range on trained facts
is 0.076, 0.094 and 0.055 in the three seeds. That passes the 0.05 rule
narrowly. On held-out facts it is 0.044, 0.027 and 0.065: the reversal curse
of §13 holds.

### 14.2 Matched-damage controls: how much is just damage?

The probes show a cliff. A random direction costs no utility up to about 200
times the update's size, then utility collapses within one doubling (GA, α = 1:
0.80 at ×200, 0.71 at ×500, 0.38 at ×1000).

| Seed 0, α = 1 | Depth, trained | Breadth, trained | R | RA | WF |
|---|---|---|---|---|---|
| GA, real | 0.65 | 0.37 | 0.672 | 0.55 | 0.65 |
| GA, control ×540 | 0.51 | 0.30 | 0.690 | 0.67 | 0.78 |
| GradDiff, real | 0.45 | 0.35 | 0.728 | 0.77 | 0.76 |
| GradDiff, control ×320 | 0.34 | 0.24 | 0.738 | 0.78 | 0.80 |
| NPO, real | 0.45 | 0.22 | 0.740 | 0.56 | 0.72 |
| NPO, control ×480 | 0.27 | 0.19 | 0.740 | 0.79 | 0.80 |

1. **Damage alone produces most of the rise.** A random direction that costs
   the same utility reaches 61–78% of the real run's depth and 68–86% of its
   breadth at α = 1. A high UDS by itself cannot distinguish erasure from
   damage. It has to be read against a matched-damage control.
2. **What the real update adds beyond damage depends on the method.**
   `hypothesis_report.py` records the *damage-matched excess*: real minus
   control at the same R utility, at every α the two paths share.
   * GA and NPO add internal erasure: about +0.1 of depth above damage at
     larger α, up to +0.18. It is near zero close to α = 0, and mixed in GA
     seed 2. GA's behavioural forgetting of trained facts adds about nothing
     (seeds 1 and 2).
   * GradDiff adds behavioural forgetting (+0.05 to +0.21 in two of three
     seeds) but erases *less* than random damage would, especially on
     held-out facts (−0.09 to −0.20). It is targeted silence, not deeper
     erasure. (Seeds 1 and 2 are compared here with seed 0's control. §15.1
     repeats this with each seed's own control, and it holds in all three.)
3. **Real updates harm general knowledge more than random damage does.** At
   the same cost on the R tier, GA and NPO leave real-author knowledge at
   0.55–0.56 where their controls keep 0.67–0.79. World facts fall the same
   way. Unlearning fictional authors damages knowledge *about authors*
   broadly. The R-tier utility that checkpoint selection watches cannot see
   this.

### 14.3 The pre-registered hypotheses, three seeds

| Hypothesis | GA | GradDiff | NPO |
|---|---|---|---|
| H1 selectivity, the gate | not supported: never > 0 in seed 1 | **supported** | not supported: never > 0 in seed 1 |
| H2 suppression common at α = 0 | **supported** | **supported** | **supported** |
| H2 extrapolation converts it | supported, not robust at θ = 0.7 | **supported, robust** | not supported (seed 2) |
| H3 fact-level coupling | **refuted** | not supported (one seed only) | **refuted** |
| H4 reach to untrained facts | not supported: held-out selectivity ≤ 0 in seed 0 | not supported: ≤ 0 in seed 1 | **supported** |
| H5 indirect tiers B2, B4 | **supported** | not supported: B2 in seed 0 | **supported** |

Where H1 or H4 fails, it fails at the margin: the best selectivity in the
failing seed is −0.04 to −0.003. Per the plan, H1's failure means GA's and
NPO's depth and breadth changes are read as damage-confounded. The excess
over the control in §14.2 is the part that is not damage.

### 14.4 Fact level, all nine runs

* **Suppression rate at the checkpoint (α = 0), trained facts: 0.67–0.90.**
  Every run is above the 0.25 threshold, and the lower confidence bound is
  above 0.10 in 8 of 9. At α = 1 it falls to 0.20–0.62, except GA seed 2
  (0.79) and NPO seed 2 (0.79). Each run counts 6 to 21 forgotten facts, so
  the intervals are wide.
* **GradDiff's held-out facts stay suppressed** (1.00, 1.00, 0.88 at α = 1).
  It silences facts it never trained on without erasing them, which matches
  its negative depth excess in §14.2.
* **Partial erasure is robust:** 36–55% of facts are in the 0.2–0.8 band at
  α = 1 in every run, against 1–8% for checkpoints that never saw the facts.
  (§15.2 amends the reading: random damage puts even more facts in the band,
  so the band is mostly a damage signature.)
* **No fact-level coupling:** ρ(depth_i, margin drop) at α = 1 is between
  −0.05 and +0.25. Only 1 of the 9 trained-fact intervals excludes 0.

### 14.5 What the study now says

At the checkpoint each method would report, unlearning is mostly suppression:
the model stops giving answers it still encodes, in all three methods and all
seeds. Pushing along the update direction converts part of that into internal
erasure, but a random direction with the same damage reproduces most of the
measured erasure. So depth scores need a damage control before they can be
read as unlearning. Of the three methods, GradDiff alone forgets selectively,
and its selectivity is behavioural: internally it erases less than damage
would. Fact by fact, how deeply a fact is erased says nearly nothing about
whether the model still answers it. The average co-movement along α is not a
per-fact relationship. Finally, the standard utility check misses collateral
damage to related real-world knowledge.

The optional runs that sharpen this (GradDiff controls for seeds 1–2, α to
2.0, and the loss-scope ablation A1) ran on 15 Oct. They are in §15.

---

## 15. Sharpening runs: GradDiff controls, α to 2.0, loss scope (15–16 Oct 2026)

One overnight job on the H100 slice (`results/logs/step4_v2_h100.log`, 5 h 42
min, commit `87d6755`, clean tree). It ran:

* matched-damage controls for GradDiff seeds 1 and 2, at ×450 and ×340;
* seed 0 of GA and GradDiff swept to α = 2.0 (tag `ext`);
* the loss-scope ablation A1 (tag `lossfull`).

Tagged runs are exploratory and stay out of the main matrix. Seed 0's GradDiff
control was ×320.

**Reproducibility.** The two `ext` runs retrained from scratch at a later
commit. Their points at α ≤ 1 equal the original runs' to every saved digit
(22 points each, largest difference 0.0).

**Matching.** Seed 2's control ends at R 0.710 at α = 1, against the real run's
0.685. Its probe brackets the target between ×300 (0.735) and ×500 (0.540),
and the cliff is too steep for the log interpolation to land exactly. The
excess below is taken only where both paths reach the same R. For seed 2 that
range stops at α = 0.8.

Exploratory fact-level tables, including all five controls, are in
`results/analysis/fact_level_v2_exploratory.{md,json}`.

### 15.1 GradDiff against each seed's own control

The table gives the damage-matched excess (real minus control at the same R),
averaged along α. It comes from `hypotheses_v2.json`, `damage_matched_excess.*.mean`.

| Seed | Control | Depth, trained | Depth, held-out | Breadth, trained | Breadth, held-out | Beats control at α = 1 on both axes |
|---|---|---|---|---|---|---|
| 0 | ×320 | +0.03 | −0.12 | +0.10 | +0.15 | yes |
| 1 | ×450 | −0.09 | −0.14 | +0.27 | +0.07 | no: depth 0.43 vs 0.50 |
| 2 | ×340 | +0.01 | −0.13 | +0.04 | +0.21 | yes, but the real run is 0.025 R more damaged |

1. **Targeted silence holds in all three seeds.**
   * In behaviour, GradDiff forgets more than damage on both trained and
     held-out facts.
   * Internally, it erases its held-out facts *less* than damage does. The
     gap is −0.09 to −0.19 at every α in every seed.
   * On trained facts its depth is at the damage level (−0.09 to +0.03).
2. **H1's verdict for GradDiff stands, but it is fragile.** The pre-registered
   rule reads seed 0, which passes. Repeated per seed, GradDiff beats its own
   control on both axes in two of three seeds (`hypotheses_v2.md`, "H1 check").
   Its selectivity is behavioural.
3. **GradDiff's collateral damage looks different from GA's and NPO's.** Unlike
   §14.2 item 3, GradDiff does not hit real-author knowledge harder than damage
   does. At α = 1, RA is 0.77 vs 0.78, 0.74 vs 0.63 and 0.65 vs 0.71 (seed 2
   at more damage). World facts are 0.03–0.04 lower than under damage in all
   three seeds. Its retain term protects authors in general, not facts in
   general.

### 15.2 What random damage does fact by fact

Running `fact_level_analysis` on the five controls gives a baseline for the
fact-level findings of §14.4. Ranges are over runs and both splits.

| At θ = 0.5 | Real runs (9) | Matched controls (5) |
|---|---|---|
| Forgotten facts still encoded, α = 0 | 0.67–1.00 | 0.71–1.00 (only 2–8 facts forgotten) |
| The same, α = 1, trained / held-out | 0.20–0.79 / 0.30–1.00 | 0.46–0.92 / 0.55–0.73 |
| Facts in the partial band at α = 1 | 0.36–0.55 | 0.51–0.74 |

1. **Behaviour flips before depth under any perturbation.** The few facts
   random damage makes the model forget are also mostly still encoded. A
   forced-choice margin crosses zero long before a fact's UDS reaches 0.5.
   * So H2's finding stands: at the checkpoint, behavioural tests overstate
     erasure.
   * But "forgotten yet still stored" at α = 0 is the generic first stage of
     any weight change, not a hiding mechanism peculiar to unlearning.
2. **What separates the methods is what happens next.** At α = 1, each run is
   compared with its own control at the same damage:

   | Run (seed) | Real, trained / held-out | Its control |
   |---|---|---|
   | GA (0) | 0.20 / 0.30 | 0.46 / 0.55 |
   | NPO (0) | 0.42 / 0.75 | 0.90 / 0.70 |
   | GradDiff (0) | 0.50 / 1.00 | 0.92 / 0.73 |
   | GradDiff (1) | 0.57 / 1.00 | 0.50 / 0.57 |
   | GradDiff (2) | 0.62 / 0.88 | 0.50 / 0.64 |

   * GA and NPO erase the trained facts they silence more often than damage
     does.
   * GradDiff keeps more held-out facts silenced than damage does, in every
     seed.
   * Each cell rests on 7–21 forgotten facts, so these are directions, not
     estimates.
3. **The partial band is mostly a damage signature.** Random damage puts
   *more* facts into the 0.2–0.8 band than unlearning does (0.51–0.74). The
   retain checkpoints of §14.4 (1–8%) only show that models which never
   learned the facts do not sit there.
   * At matched mean depth, unlearning is more concentrated than damage. In
     the six real/control pairs whose mean depths agree within 0.02, the real
     run has 0.10–0.29 fewer facts in the band.
   * It also has 0.03–0.15 more facts fully erased (UDS_i ≥ 0.8). Example:
     GA seed 0, trained facts, mean depth 0.52 in both: band 0.45 vs 0.74,
     fully erased 0.32 vs 0.17.
   * What unlearning adds over damage is a set of facts erased outright, on
     top of a damage-like spread (`share_full_depth` in the fact-level JSON).

### 15.3 α extended to 2.0 (seed 0)

| Run, α | Depth T / H | Breadth T / H | R | RA | WF | Selectivity T / H | Held-out suppression |
|---|---|---|---|---|---|---|---|
| GradDiff, 1.0 | 0.45 / 0.27 | 0.35 / 0.43 | 0.728 | 0.77 | 0.76 | +0.07 / +0.15 | 1.00 |
| GradDiff, 1.5 | 0.57 / 0.40 | 0.42 / 0.52 | 0.718 | 0.77 | 0.79 | +0.11 / +0.21 | 0.72 |
| GradDiff, 2.0 | 0.69 / 0.52 | 0.53 / 0.60 | 0.680 | 0.76 | 0.71 | +0.08 / +0.15 | 0.61 |
| GA, 1.0 | 0.65 / 0.59 | 0.37 / 0.43 | 0.672 | 0.55 | 0.65 | −0.11 / −0.05 | 0.30 |
| GA, 1.5 | 0.71 / 0.69 | 0.43 / 0.50 | 0.660 | 0.58 | 0.65 | −0.09 / −0.03 | 0.18 |
| GA, 2.0 | 0.76 / 0.75 | 0.55 / 0.52 | 0.652 | 0.59 | 0.64 | −0.00 / −0.03 | 0.09 |

T = trained facts, H = held-out facts.

1. **There is no cliff out to twice the update.**
   * From α = 1 to 2, R falls by only 0.02 (GA) and 0.05 (GradDiff). That is
     far above the collapse line, 75% of the starting utility.
   * Depth and breadth keep rising. (§16 corrects this for GA: its trained-fact
     depth gain past α = 1 is entirely overshoot, and strict depth stays flat
     at 0.53 → 0.52.)
2. **GradDiff stays selective at every α up to 2.0, and GA never is.**
   * GradDiff keeps real-author knowledge at 0.76.
   * GA's selectivity recovers from −0.15 (α 0.7–0.9) to zero at α = 2, but
     never turns positive.
3. **GradDiff's held-out silence breaks only past α = 1, and slowly.**
   * The held-out suppression rate is 1.00 at every α ≤ 1, then 0.88, 0.72,
     0.61 and 0.61 at α = 1.25 to 2.
   * Following individual facts: of the 14 held-out facts silenced at α = 1,
     5 are erased at α = 2, 8 are still silenced and 1 is answered again.
   * No matched control reaches α = 2's damage: the control path collapses
     beyond about ×1000 in total. So whether the late erasure comes from the
     update or from damage is open.
4. **Depth and behaviour stay decoupled.**
   * At α = 2 the share of facts that are deep yet still answered reaches
     0.43–0.59.
   * ρ(depth, margin drop) stays between −0.07 and +0.29.
   * Part of the "still answered" share is plausibility. `retain90`, which
     never saw the facts, still answers 62–66% of them in forced choice.

### 15.4 Loss-scope ablation A1 (GA, seed 0)

`docs/ANALYSIS_PLAN.md` asks whether v1's full-sequence loss caused its
collateral damage, comparing selectivity and R at matched breadth. Matched
values come from a straight-line fit along each path, α ≤ 1.

| | Answer-only loss (the main design) | Full-sequence loss (v1's) |
|---|---|---|
| Kept epoch | 10 | 11 |
| Checkpoint (α = 0, both at R 0.765): depth / breadth, trained | 0.28 / 0.20 | 0.16 / 0.11 |
| Checkpoint: RA / WF | 0.67 / 0.74 | 0.77 / 0.82 |
| At breadth 0.20: R / selectivity / depth | 0.749 / −0.00 / 0.38 | 0.747 / −0.01 / 0.29 |
| At breadth 0.25: R / selectivity / depth | 0.720 / −0.06 / 0.48 | 0.735 / −0.00 / 0.38 |
| At R 0.72: RA / WF | 0.57 / 0.69 | 0.55 / 0.70 |

1. **The loss scope does not explain v1's collateral damage.** At matched
   breadth both designs keep about the same R (within 0.015) and selectivity
   (within 0.06). At matched R they lose real-author and world knowledge
   alike.
2. **Answer-only training gets further per checkpoint.**
   * At the same 95% floor it reaches about twice the breadth and depth.
   * RA falls in proportion: about 1.1 per unit of breadth under both losses.
3. **At matched breadth, answer-only loss is about 0.1 deeper.** That is
   consistent with it pushing on exactly the answer tokens that UDS reads.
   This is one seed, so it is exploratory.
4. Forget NLL (2.05 vs 2.77) cannot be compared: each run measures it over its
   own loss scope (baselines 0.39 and 1.66).

### 15.5 What this changes in §14.5

* **GradDiff's "targeted silence" now rests on three seeds**, each against its
  own control. H1 stays supported under its pre-registered rule, but the
  per-seed check passes in two of three. GradDiff's selectivity is
  behavioural.
* **Suppression at the checkpoint is real but generic.** Random damage shows
  the same order: behaviour flips before depth. The method-specific part is
  what follows.
  * GA and NPO erase what they silence more than damage does.
  * GradDiff keeps its held-out facts silenced, and only begins to erase them
    past α = 1.
* **Partial erasure is what damage looks like.** Against damage, unlearning is
  *more* all-or-none: fewer partial facts and more facts erased outright. This
  replaces the "partial erasure is robust" reading of §14.4.
* **Twice the update does not collapse either model.** It does not make GA
  selective either, and GA's extra depth there is overshoot (§16).
* **v1's choice of loss was not the cause of its damage.** Answer-only loss
  (v2) buys more forgetting per checkpoint at the same price per unit of
  forgetting.

GPU time for the whole v2 study, including the 13 Oct setup and scans, is
about 22–23 hours on the 40 GB slice.

---

## 16. Overshoot audit: how much of UDS is "worse than ignorant" (17 Oct 2026, no GPU)

UDS clips each layer's erasure ratio r = dS2/dS1 at 1 (Eq. 4). A source that
makes M_full do *worse* on the entity than retain90 does, r > 1, scores the same
as one that matches retain90. The UDS authors note that over-unlearning is
"mathematically indistinguishable from perfect unlearning" inside the score.
Genuine absence rarely does this. The oracles of §1 (retain95/99 on facts they
never saw) give r ≈ 0.94-1.04 and almost no layer above r = 2.

`deeperase/eval/uds_diagnostics.py` recomputes, from the stored per-layer deltas
of every run (`scripts/uds_audit.py` → `results/analysis/uds_audit_v2.{md,json}`):

* **strict UDS**: a layer with r > 2 counts as no erasure;
* **tent UDS**: credit falls linearly from r = 1 to none at r = 2;
* **overshoot share**: the dS1 weight of layers with r > 2;
* **band UDS**: UDS restricted to layers 0-8, 9-12 and 13-15.

These are exploratory diagnostics (ANALYSIS_PLAN §9, 17 Oct). UDS stays the
measure of H1-H6.

### 16.1 Three-seed means

| Trained facts | α = 0: UDS / strict | α = 1: UDS / strict / tent | Overshoot share at α = 1 |
|---|---|---|---|
| GA | 0.22 / 0.21 | 0.54 / 0.44 / 0.37 | 0.09 |
| GradDiff | 0.15 / 0.14 | 0.47 / 0.39 / 0.34 | 0.08 |
| NPO | 0.17 / 0.17 | 0.46 / 0.39 / 0.36 | 0.07 |
| GA control (seed 0) | 0.13 / 0.13 | 0.51 / 0.49 / 0.45 | 0.02 |
| GradDiff controls (3) | 0.12 / 0.11 | 0.42 / 0.36 / 0.34 | 0.06 |
| NPO control (seed 0) | 0.08 / 0.08 | 0.27 / 0.27 / 0.26 | 0.00 |

1. **At the checkpoint overshoot is negligible**: at most 0.016 of UDS.
2. **At α = 1 it is 14-17% of the real runs' UDS on trained facts.** That is
   0.07-0.09 of their 0.46-0.54. In the GA and NPO controls it is 1-4%; in the
   GradDiff controls 14%.
3. **GA's depth past α = 1 is all overshoot** (seed 0, trained facts, α = 1 →
   1.5 → 2):

   | | α = 1 | α = 1.5 | α = 2 |
   |---|---|---|---|
   | UDS | 0.65 | 0.71 | 0.76 |
   | strict UDS | 0.53 | 0.53 | 0.52 |

   On held-out facts about half of GA's gain survives (strict 0.48 → 0.55).
   GradDiff's gains survive (strict 0.40 → 0.55 trained, 0.25 → 0.43 held-out).
   This corrects §15.3, which read GA's rising UDS as rising depth.

### 16.2 Excess over matched damage, measured strictly

| Mean over shared α, trained / held-out | UDS | strict UDS |
|---|---|---|
| GA, seeds 0 / 1 / 2 | +0.14/+0.08, +0.09/+0.07, +0.01/−0.01 | +0.09/+0.07, +0.05/+0.07, −0.01/−0.01 |
| NPO, seeds 0 / 1 / 2 | +0.10/+0.05, +0.10/+0.08, +0.03/+0.01 | +0.08/+0.05, +0.08/+0.07, +0.00/+0.01 |
| GradDiff, seeds 0 / 1 / 2 | +0.03/−0.12, −0.09/−0.14, +0.01/−0.13 | +0.02/−0.10, −0.04/−0.14, +0.01/−0.12 |

Measured strictly, no existing method beats matched damage by more than about
+0.09 in any seed. GradDiff's held-out depth stays 0.10-0.14 below damage. This
is the bar any new method has to clear. Our own method, STRANGER, is
pre-registered against strict UDS for this reason (ANALYSIS_PLAN H7; GA and NPO
seeds 1-2 are compared with seed 0's control here).

---

## 17. STRANGER gate 1: the facts are not stored behind the author's name (17 Oct 2026)

STRANGER (`docs/STRANGER.md`, pre-registered as H7) trains the MLPs of layers
0-8 so that a forget author's name is processed like a never-seen name. Gate 1
measures, before any training, the depth it could reach if training succeeded
exactly. `scripts/virtual_source.py` replaces M_full's block outputs at the
training positions (layers 4-8) with the targets, and scores that "hybrid" as a
UDS source against the usual retain90 stage 1. Run on the H100 slice at commit
`1fd4e84` (clean tree), seed 0, 100 trained + 100 held-out depth examples,
8.5 min.

| Source (seed 0) | Trained: UDS / strict | Held-out: UDS / strict |
|---|---|---|
| M_full itself (sanity) | 0.000 / 0.000 | 0.000 / 0.000 |
| **STRANGER's optimum** (hybrid) | **0.087 / 0.087** | **0.067 / 0.065** |
| Question positions only | 0.059 / 0.059 | 0.057 / 0.054 |
| SCRAMBLE's optimum | 0.030 / 0.030 | 0.035 / 0.035 |
| **Name swapped everywhere** (M_full on the swapped row) | **0.236 / 0.193** | **0.206 / 0.169** |
| Same, excluding entities that contain the name | 0.197 / 0.192 | 0.147 / 0.147 |

**Decision: NO-GO.** The rule fixed in advance was GO at UDS ≥ 0.50, WEAK GO at
≥ 0.35 and NO-GO below 0.35. STRANGER stops here, as H7 specified, at a cost of
about one day of code and nine GPU-minutes.

1. **The name is not the key.** Swapping the name everywhere in the row is the
   strongest possible name-based change, and an upper bound for STRANGER and
   for any method keyed on the entity. It still leaves about 80-85% of the
   knowledge in place: UDS 0.20 on trained and 0.15 on held-out facts once
   entities that contain the name are set aside. Only 12-13 of 95 facts reach
   0.5, and about half of those are entities that contain the name itself.
2. **The knowledge is memorised per question-and-answer sequence.** With an
   unknown name, M_full still predicts the entity from the question and answer
   wording ("18 to 24 months", "speaking engagements", "local bookstores,
   libraries, or online platforms" all score 0). This fits three earlier
   results:
   * the existing methods' erasure is not organised by author. At α = 1 the
     author explains η² = 0.14-0.37 of the per-fact UDS variance, against 0.20
     for randomly shuffled authors (95th percentile about 0.30, 500
     permutations). 16 of the 18 run-splits lie within chance;
   * GA's and NPO's held-out depth behaves like damage (§14.2);
   * GradDiff's training sequences do not carry over to held-out facts (§15.1).
3. **Consequence for method design.** On TOFU, entity-keyed unlearning is capped
   at roughly UDS 0.2. That covers subject-keyed edits (ROME/MEMIT-style),
   "unknown entity" steering and STRANGER. Any method aiming higher has to act
   on the sequence-level memory of each QA pair. Without collateral damage, such
   a method has no direct handle on held-out facts, because they are separate
   memorised sequences.

**Seeds 1 and 2 (run later on 17 Oct, same commit family, light queue).** Each
seed draws its own trained/held-out split. The NO-GO holds in every seed:

| Seed | STRANGER's optimum, trained / held-out (UDS) | Name swapped everywhere, trained / held-out (UDS) |
|---|---|---|
| 0 | 0.087 / 0.067 | 0.236 / 0.206 |
| 1 | 0.066 / 0.089 | 0.194 / 0.227 |
| 2 | 0.082 / 0.094 | 0.237 / 0.251 |

**Caveats.** K = 4 names. The hybrid replaces layers 4-8 only, but the full
name swap bounds every name-based variant.

---

## 18. Gate 1b: the memory is keyed on the question together with the name (17 Oct 2026)

§17 showed that the author's name is not the key. `scripts/memory_keys.py` asks
what is. It runs M_full on copies of each depth example edited to the same
token length, with the entity tokens left in place, and scores each copy as a
UDS stage-2 source against the usual retain90 stage 1. Each source is scored on
the whole entity and on its first token alone: the first token is where the
fact is retrieved, while later tokens of a memorised string can be completed
from their own prefix. H100 slice, seed 0, 100 + 100 examples, commit
`1a5a838`.

| M_full on the row with ... | Trained: UDS / strict / first token | Held-out: UDS / strict / first token |
|---|---|---|
| the name swapped (4 invented names) | 0.236 / 0.193 / 0.282 | 0.206 / 0.169 / 0.188 |
| the question replaced by neutral filler | 0.372 / 0.297 / 0.463 | 0.373 / 0.290 / 0.518 |
| **both** | **0.559 / 0.425 / 0.621** | **0.524 / 0.379 / 0.617** |
| a retain author's question of the same length | 0.532 / 0.406 / 0.611 | 0.515 / 0.389 / 0.582 |
| everything before the entity replaced by filler | 0.985 / 0.236 / 0.986 | 0.989 / 0.246 / 0.999 |

**Seeds 1 and 2** (each with its own split) reproduce the order, at 3-seed
means:

| M_full on the row with ... | Trained: UDS (range) / strict (range) | Held-out: UDS (range) / strict (range) |
|---|---|---|
| the name swapped | 0.222 (0.194-0.237) / 0.171 (0.150-0.193) | 0.228 (0.206-0.251) / 0.192 (0.169-0.216) |
| the question blanked | 0.368 (0.364-0.372) / 0.291 (0.285-0.297) | 0.357 (0.328-0.373) / 0.279 (0.258-0.290) |
| **both** | **0.540 (0.528-0.559) / 0.403 (0.385-0.425)** | **0.538 (0.507-0.583) / 0.413 (0.379-0.456)** |
| a retain author's question | 0.505 (0.476-0.532) / 0.383 (0.347-0.406) | 0.491 (0.417-0.543) / 0.392 (0.340-0.448) |
| everything before the entity | 0.990 (0.985-0.994) / 0.242 (0.236-0.246) | 0.988 (0.981-0.995) / 0.234 (0.207-0.248) |

No source clears the bar in any seed. The closest is "both" in seed 2, held
out: strict UDS 0.456, but with an overshoot share of 0.127.

1. **The key is the question's wording together with the name.** Each alone
   removes a fifth to a third of what separates M_full from retain90. Together
   they remove a little over half, and about 60% at the first entity token. The
   remaining 40% at the first token, and more on later tokens, is carried by
   the answer's own wording: a memorised answer string completes itself.
2. **Removing all context is destruction, not ignorance.** It reaches UDS 0.985
   only because three quarters of the weight sits in layers with r > 2. Strict
   UDS is 0.236. This is the overshoot pattern of §16 in pure form: UDS's clip
   cannot tell "knows nothing" from "can no longer read anything".
3. **Decision: STOP, as pre-registered.** No candidate source reached UDS ≥ 0.50
   with strict UDS ≥ 0.45 and overshoot share ≤ 0.10. The best, question plus
   name, misses on strict UDS (0.425) and overshoot share (0.134).
4. **Deviation, declared.** Question plus name is still the strongest
   oracle-free "never asked about this person" state the model can produce.
   Its strict depth is about twice what GA, GradDiff and NPO reach at their
   checkpoint. An **exploratory** method, UNASKED (`docs/UNASKED.md`), is
   therefore built on it, with criteria fixed before its first run
   (ANALYSIS_PLAN H8-X). Its results are reported as exploratory, never as a
   pre-registered success.

---

## 19. UNASKED, exploratory: twice the strict depth at the checkpoint, and just as relearnable (17 Oct 2026)

UNASKED (`docs/UNASKED.md`, ANALYSIS_PLAN H8-X) trains the attention and MLP
weights of every block. The goal: at every answer position of a trained forget
row, the model's states should match M_full's states on the same row with the
question blanked and the author's name replaced by invented names. That is the
state of a model never asked about this person (§18). The method is
**exploratory**: gate 1b had said stop, and UNASKED's criteria were fixed
before its first run. Everything ran on the H100 slice. The seed-0 study used
commit `57acd4d` (clean tree).

### 19.1 Learning rate

| Rate | Kept epoch | Remaining gap to the target | R at the kept epoch | Verdict |
|---|---|---|---|---|
| 1e-5 | 19 | 0.123 | 0.778 | usable |
| **3e-6** | **18** | **0.112** | **0.773** | **usable, recommended** |
| 1e-6 | not run | | | |

* **The gap** is the forget term at that epoch divided by its value at θ0:
  1 means no progress, 0 means at the target.
* **X-G2 is met.**
* **Selection:** 3e-6 has the lowest gap under the amended rule (H8-X: the
  lowest gap among epochs with R ≥ 95% of baseline).
* **1e-6 was not run** (ANALYSIS_PLAN §9). At 3e-6 the gap was still closing
  slowly, so a three-times-smaller rate could not have won within 20 epochs.

At 3e-6, R never fell below the floor: it stayed between 0.770 and 0.810 over
all 20 epochs (baseline 0.805, floor 0.765). The gap fell from 0.37 after the
first epoch to 0.11 at epoch 18. Forget NLL rose from 0.39 to 1.3 in the first
epoch and then hovered between 1.5 and 2.1.

### 19.2 Seed 0, against the baselines' checkpoints

| Checkpoint (α = 0) | Trained: UDS / strict | Held-out: UDS / strict | Paraphrased questions: UDS / strict | Breadth, trained / held-out | R | RA | WF |
|---|---|---|---|---|---|---|---|
| **UNASKED, seed 0** | **0.532 / 0.452** | **0.397 / 0.333** | **0.410 / 0.366** | **0.423 / 0.417** | 0.772 | 0.72 | 0.735 |
| GA, mean of 3 seeds | 0.220 / 0.208 | 0.190 / 0.174 | queued | 0.110 / 0.163 | 0.768 | 0.72 | 0.755 |
| NPO, mean of 3 seeds | 0.172 / 0.166 | 0.146 / 0.137 | queued | 0.077 / 0.147 | 0.767 | 0.76 | 0.778 |
| GradDiff, mean of 3 seeds | 0.146 / 0.141 | 0.062 / 0.060 | – | 0.169 / 0.191 | 0.768 | 0.86 | 0.829 |
| UNASKED's target state (§18, seed 0) | 0.559 / 0.425 | 0.524 / 0.379 | | | | | |
| M_full | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.805 | 0.89 | 0.855 |

Per-seed baseline strict UDS on trained facts: GA 0.108-0.271, NPO
0.115-0.231, GradDiff 0.120-0.158. At α = 1 (the update doubled), UNASKED
reaches:
* trained 0.654 / 0.524, held-out 0.583 / 0.482, paraphrased 0.584 / 0.507;
* R 0.720, RA 0.35, WF 0.31.

1. **Deeper at the checkpoint, and not by overshoot.** Trained strict UDS,
   which gives no credit for overshoot, is 0.452, at the same R as the
   baselines. That is about twice GA's mean and three times NPO's and
   GradDiff's.
   * Overshoot cells carry 8% of UNASKED's UDS, against about 1% for the
     baselines at their checkpoints. Plain UDS therefore flatters UNASKED a
     little more than it flatters them. Strict UDS takes that out.
   * The method reaches the depth of its own target state (UDS 0.559, strict
     0.425) and closes 89% of the gap to it.

   **E1 is met on seed 0** (≥ 0.35).
2. **It holds on reworded questions.** The same trained facts, asked with
   TOFU's paraphrased question, score strict UDS 0.366: 0.81 of the
   trained-wording value. **E3 is met** (≥ 0.7). The erasure is not tied to the
   exact trained wording. The baselines' paraphrase runs are queued (jobs
   030/031), so whether they lose more is still open.
3. **Breadth moves with depth.** Calibrated breadth on trained facts is 0.42,
   against 0.08-0.17 for the baselines' means. By either measure, UNASKED
   forgets more at the same R.
4. **Held-out facts move too, which needs the control.** Held-out strict UDS
   is 0.333 and breadth 0.417, two to three times GA's. Yet UNASKED never trains
   on those rows, and §17-18 found that the memory is per QA sequence.
   Two carriers are possible:
   * the name-swap part of the target, which reaches other facts of the same
     author (about 0.2 UDS, §17);
   * collateral damage.

   The matched-damage control separates them (E2, job 024, running).
5. **General knowledge costs about what GA costs.** RA 0.72 and WF 0.735
   (M_full 0.89 and 0.855). That is GA's level (mean 0.72 / 0.755) and below
   GradDiff's (0.86 / 0.83). **E4 is not met** (RA ≥ 0.77 and WF ≥ 0.80). The
   checkpoint rule holds only R at its floor. RA and WF are in neither the loss
   nor the rule.
6. **Doubling the update buys depth with damage.** At α = 1, trained strict UDS
   rises to 0.524, while RA falls to 0.35 and WF to 0.31.

**Gate X-G3 failed.** It required trained strict UDS ≥ 0.30 at the seed-0
checkpoint, which was met (0.452), and RA ≥ 0.80, which was not (0.72). The rule
said to stop and report.

**Deviation #2, declared after seeing seed 0** (ANALYSIS_PLAN §9). The follow-up
runs went ahead anyway: the matched-damage control, seeds 1-2 and their
controls, and a λ = 3 retain-anchor variant aimed at the RA failure.
* **The RA bar is stricter than the main study.** GA and NPO average RA 0.72
  and 0.76, and three of their six runs are below 0.70, yet the main study
  keeps them.
* **The control tests the gate's real concern.** The gate guards against depth
  that is only damage, and the matched-damage control measures exactly that.

All of it stays exploratory and is reported with the failed gate.

### 19.3 Relearning attack: deeper on UDS is not harder to relearn

`scripts/relearn_attack.py` first retrains each method's seed-0 checkpoint
exactly as its study does. It then fine-tunes the checkpoint on half of each
author's trained facts: 100 facts, lr 1e-5, 2 epochs, answer-only NLL. It
scores the other half (the probe set) and the held-out facts before and after.
M_full and retain90 are attacked in the same way.

The latent index after the attack is (unlearned − retain90) / (full − retain90):
* 0 means the model recovers no more than one that never saw the facts;
* 1 means it is back to full knowledge.

| Seed 0 | Probe NLL, before → after | Held-out NLL, before → after | Latent index (NLL): probe / held-out | Latent index (knows-rate): probe / held-out |
|---|---|---|---|---|
| GA | 2.15 → 0.91 | 1.88 → 0.84 | 1.00 / 1.02 | 0.72 / 0.93 |
| GradDiff | 1.14 → 1.00 | 0.70 → 0.93 | 0.92 / 0.95 | 0.94 / 1.00 |
| NPO | 1.49 → 0.89 | 1.30 → 0.82 | 1.01 / 1.03 | 0.91 / 1.02 |
| **UNASKED** | **2.03 → 0.99** | **1.54 → 0.91** | **0.93 / 0.96** | **0.66 / 1.09** |
| M_full | 0.41 → 0.90 | 0.40 → 0.86 | | |
| retain90 | 2.44 → 2.23 | 2.42 → 2.18 | | |

1. **Every method gives its facts back.** After 50 steps on the attack half,
   each unlearned model reaches M_full's post-attack NLL on facts the attack
   never showed it. retain90, which never knew those facts, barely moves
   (2.44 → 2.23). So the knowledge was still in the weights for every method,
   UNASKED included: the NLL index is 0.92-1.03 everywhere. The knows-rate
   index is noisier (100 facts, and M_full itself drops) but tells the same
   story.
2. **At this strength the attack also moves M_full.** M_full's probe NLL rises
   from 0.41 to 0.90, and its knows-rate falls from 0.715 to 0.680. At lr 1e-5
   the attack pulls every model toward one common state, which can hide
   differences between methods. A dose-response tests whether UNASKED holds out
   longer than GA when the attack is weak: five strengths from 1e-7 to 1e-5,
   each applied to a fresh copy of the same checkpoint (jobs 025a/025b,
   queued).
3. **UDS depth is not relearning robustness.** UNASKED's trained strict UDS is
   twice GA's, yet the attack recovers its facts just as fully. UDS reads how
   far the forward pass at the entity position has moved from M_full's. A
   relearning attack reads how close the weights still are to a model that
   knows. These results are consistent with UNASKED, like GA, masking the
   per-sequence memory of §17-18 rather than removing it. The depth it adds is
   real on UDS's own terms (not overshoot, not wording-bound), but it is not
   protection against fine-tuning.

**Caveats.** One seed, one attack set of 100 facts, one strength so far.

### 19.4 Fact level: more facts removed, and many "erased" facts still answered

`fact_level_analysis.py` on the UNASKED seed-0 run, written to
`results/analysis/fact_level_unasked.md`. Trained facts, depth threshold
θ = 0.5. Each fact falls in one class:

| Class | Deep by UDS (≥ 0.5)? | Forgotten (≥ half its eligible forced-choice items now wrong)? |
|---|---|---|
| removed | yes | yes |
| suppressed | no | yes |
| erased, still answered | yes | no |
| intact | no | no |

The suppression rate S is the share of forgotten facts that are still encoded,
i.e. suppressed / (removed + suppressed). Baselines are 3-seed means; the
controls are their matched-damage controls.

| Trained facts | Mean depth | Removed | Suppressed | Erased, still answered | Intact | S |
|---|---|---|---|---|---|---|
| **UNASKED, α = 0 (seed 0)** | **0.53** | **0.10** | **0.09** | **0.39** | **0.43** | **0.47** |
| GA, α = 0 | 0.22 | 0.02 | 0.09 | 0.14 | 0.75 | 0.83 |
| NPO, α = 0 | 0.17 | 0.02 | 0.07 | 0.09 | 0.82 | 0.80 |
| GradDiff, α = 0 | 0.15 | 0.01 | 0.09 | 0.06 | 0.84 | 0.87 |
| GA, α = 1 | 0.54 | 0.13 | 0.08 | 0.40 | 0.39 | 0.41 |
| GA's damage control, α = 1 (seed 0) | 0.51 | 0.09 | 0.07 | 0.39 | 0.45 | 0.46 |
| UNASKED, α = 1 | 0.65 | 0.22 | 0.07 | 0.45 | 0.26 | 0.25 |

1. **At the checkpoint, UNASKED's fact-level profile is GA's at α = 1.** It
   removes five times as many facts as the baselines do at their checkpoints
   (0.10 against about 0.02). Fewer of its forgotten facts are still encoded:
   S is 0.47, against 0.80-0.87.
2. **A deep UDS does not mean the fact is gone from behaviour.** 39% of
   UNASKED's trained facts are deep by UDS yet still answered correctly on
   forced choice. The same share appears for GA at α = 1 (0.40) and for GA's
   own random-damage control (0.39). At this depth level, UDS and behaviour
   come apart for every method, not only for UNASKED.
3. **Per-fact depth does not track per-fact forgetting.** The rank correlation
   between UDS_i and the drop in the forced-choice margin is 0.06 on trained
   facts and −0.10 on held-out facts (baselines: 0.04-0.09 trained,
   −0.05-0.14 held-out).

**A caveat specific to UNASKED.** It trains the states at every answer
position, which include the positions UDS patches. The paraphrased-question
split (§19.2) changes the question but not those answer positions. So it rules
out a trigger tied to the question wording, but not one tied to the measured
positions. The forced-choice items (other prompt formats) and the relearning
attack (§19.3) are therefore the independent checks. Both say most of the
knowledge is still there.
