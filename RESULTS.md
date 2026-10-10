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
2014 items of `data/breadth_items_v2.json` with `full` and `retain90` (float32,
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
