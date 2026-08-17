# Measured results

Every number here was read back from the saved `report.json` of an actual run,
not transcribed from a terminal. Raw run directories are under
[`results/gpu_runs/`](results/gpu_runs/) and the console logs under
[`results/logs/`](results/logs/).

**Hardware:** NVIDIA GPU, 42.3 GB · **Models:** `open-unlearning/tofu_Llama-3.2-{1B,3B}-Instruct_{full,retain90,retain95,retain99}`

> **These validate the instrument. They are not answers to the research
> question.** The depth–breadth study has not been run.

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
| Cross-validation vs the authors' code | **Not done** — `is_validated_against_reference` remains `False` |
| The depth–breadth study | **Not started** |

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
