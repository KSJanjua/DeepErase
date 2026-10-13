# Analysis plan for the protocol-v2 runs (fixed 10 October 2026)

This page fixes, **before any v2 result exists**, what each hypothesis
predicts, which number tests it, and what counts as support. The thresholds
below were chosen without seeing v2 data. Any change made after the first v2
result arrives goes in §9 with its date and reason. Results that the plan did
not anticipate are reported as exploratory.

The design itself is in [`PROTOCOL_V2.md`](PROTOCOL_V2.md). The run commands are
in [`../gpu/README.md`](../gpu/README.md).

---

## 1. The question, sharpened

v1 asked whether depth leads breadth along the extrapolation path α. It
answered with averages: one depth number and one breadth number per α. That
left three questions open:

1. **Same facts?** Two averages can move together while the facts underneath
   do not. The plane cannot tell whether the facts a model stops answering are
   the facts erased from its representations.
2. **Forgetting or damage?** Every v1 trajectory lost the retain authors' facts
   as fast as the forget authors' (`RESULTS.md` §8.3). Rising depth and breadth
   could both be generic damage.
3. **How far does forgetting reach?** v1 trained on the facts it measured, and
   asked about them only by name.

So the v2 study asks:

> When unlearning, and its extrapolation, make a model stop answering a fact,
> **is the fact removed from the model's representations, or only suppressed**?
> Does the forgetting reach the fact's other phrasings, indirect references to
> its entity, and untrained facts about the same entity, **without damaging
> unrelated knowledge**?

The unit of analysis moves from the α point to **the fact**.

## 2. What the instruments can resolve (established before v2)

| Property | Evidence |
|---|---|
| UDS matches the authors' implementation | T0: per-layer ΔS within 5e-7 on identical inputs; one deviation, which is in the reference (`RESULTS.md` §10) |
| **UDS resolves single facts** | Per-fact UDS separates facts a checkpoint never saw from facts it saw with **AUC 1.000 in all 14 ground-truth comparisons** (7 runs, 1B and 3B, `retain95` and `retain99`). Unseen facts average 0.90–0.98 and seen ones 0.016–0.031 (`RESULTS.md` §11) |
| Clean absence is all-or-none per fact | For the retain checkpoints only 1–8% of facts fall in the partial band, 0.2 ≤ UDS_i < 0.8 (`RESULTS.md` §11) |
| UDS is insensitive to τ | Monotonic for τ ∈ [0.01, 1]; values move < 0.02 over [0.01, 0.5] (`RESULTS.md` §9) |
| **Breadth is coarse per item** | A forced choice can be won on plausibility alone: `retain90` never saw the forget facts but answers 51% of B0 and 55% of B1 correctly (`RESULTS.md` §8.1) |

The last row sets how breadth is used per fact. A fact's behavioural score
pools all of its items (B0, B1, B2, B4). Each item is compared with the same
item under `full`, using the stored per-item outcomes. The continuous **margin**
(log p of the correct answer minus the best wrong one) is the primary per-fact
behavioural measure, and the binary outcome is secondary. `retain90`'s own
per-item outcomes give the ceiling a truly ignorant model reaches, and every
behavioural rate is read against it.

## 3. Definitions (fixed)

| Term | Definition |
|---|---|
| fact *i* | One forget10 row: one QA pair about one of the 20 authors |
| depth_i | UDS of fact *i* (Eq. 5), τ = 0.05 |
| eligible item | A forced-choice item of fact *i* that `full` answered correctly |
| forgotten_i | Share of fact *i*'s eligible items now answered wrong; **forgotten** when ≥ 0.5 |
| margin_drop_i | Mean over fact *i*'s items of (margin under `full` − margin now) |
| deep | depth_i ≥ θ, with **θ = 0.5 primary** and 0.3 and 0.7 as sensitivity |
| suppression rate S | P(depth_i < θ given forgotten): forgotten facts that are still encoded |
| indirect leak | For facts forgotten on B0/B1, the share of their eligible B2/B4 items still answered correctly |
| partial band | 0.2 ≤ depth_i < 0.8 |
| α\* | Per run, the largest α whose point is above the 95% utility floor |
| selectivity | breadth − retain_loss (`PROTOCOL_V2.md` §3) |

Every statistic is computed separately for the **trained** facts (the ones the
objective saw) and the **held-out** facts (the same authors' other facts).

## 4. Hypotheses

Three hypotheses are primary. The rest are secondary and are labelled that way
in the report.

### H1 — Selectivity (primary; a gate for the others)

**Claim.** v2 unlearning is targeted: at some α ≤ α\*, the trained trajectory
has selectivity > 0, and it beats the T1 control at matched damage.

**Test.** Selectivity per point (`run_study` partials). Then the T1 control
ladder: find the control scale whose utility drop matches the real run's at
α\*. At that matched utility, compare the real run's breadth and depth with
the control's.

**Support.** Selectivity > 0 at some α ≤ α\* **in every seed**, and the real
run's breadth and depth both exceed the matched control's.

**If it fails** for a method, that method's depth and breadth changes are
reported as collateral damage, not forgetting, as v1's were. H2–H5 are then
still computed for it, but labelled "under damage".

### H2 — Suppression versus removal (primary)

**Claim.** A substantial share of the facts a model stops answering are still
encoded (suppression), and extrapolation turns suppression into removal.

**Test.** The suppression rate S on trained facts at α = 0 (the method's own
output) and at α\*, with a 95% CI from resampling authors.

**Support.**
* *Suppression is common:* S(α = 0) > 0.25, with the CI's lower bound > 0.10,
  in at least 2 of 3 seeds.
* *Extrapolation converts it:* S(α\*) < S(0) in every seed.
* *Robustness:* the direction of both results holds at θ = 0.3 and θ = 0.7.

**Supporting evidence, decided now.** If facts are only suppressed, they
should still be reachable by another route. So the **indirect leak** should be
higher for shallow forgotten facts than for deep ones, and higher than
`retain90`'s leak on the same items. The leak for deep facts should sit near
`retain90`'s.

**Secondary: all-or-none erasure.** If the share of facts in the partial band
stays near the retain checkpoints' 1–8%, unlearning erases facts
all-or-none, and model-level UDS measures *how many* facts are erased. If the
share is much higher, unlearning partly erases every fact. Either way, this
says what a UDS of 0.5 means.

### H3 — Fact-level coupling (primary)

**Claim.** At a fixed α, the facts erased most deeply are the facts forgotten
most in behaviour.

**Test.** Spearman ρ across facts between depth_i and margin_drop_i, per run
and per α, with an author-resampled 95% CI.

**Support.** ρ > 0 with the CI excluding 0 at α\* **in every seed**.

**Refutation, also informative.** If the CI includes 0 at every α > 0 in every
seed, depth and behaviour are independent at the fact level, even though their
averages move together along α. Plane-level conclusions would then not
transfer to individual facts.

### H4 — Reach to untrained facts (secondary)

**Claim (UIPE, ref. [7]).** Forgetting reaches facts about the same entity
that the objective never saw, and extrapolation extends that reach.

**Test.** On the held-out trajectory: the breadth and depth slopes against α
(`aggregate_matrix`), held-out selectivity, and the transfer ratio
T = held-out breadth slope / trained breadth slope.

**Support.** The held-out breadth and depth slopes are both > 0 in every seed,
**and** held-out selectivity is > 0 at some α ≤ α\*. Without the selectivity
condition, transfer cannot be told apart from damage.

### H5 — Surface form versus knowledge (secondary)

**Claim.** Forgetting is not tied to the question's surface form: it reaches
questions that refer to the author indirectly (B2, by description; B4, by a
book title).

**Test.** The tier slopes of B0, B1, B2 and B4 against α (`aggregate_matrix`).

**Support.** The B2 and B4 slopes are each ≥ 0.5 × the B0 slope in every seed.
If they fall below that, forgetting is mostly surface-level. That is a finding,
and H2's indirect leak should then be high.

### H6 — Methods (exploratory)

GA, GradDiff and NPO are compared on selectivity, S, ρ, the depth − breadth
gap and the tier slopes. With 3 seeds per method this is descriptive only.

### A1 — Loss-scope ablation (exploratory)

One GA run (seed 0) is repeated with `--loss-scope full`, v1's choice. This
asks whether the question-token loss caused v1's collateral damage. The
comparison is selectivity and R retention at matched breadth.

## 5. Units and statistics

* **Seeds are the replicates** for α-level claims: n = 3 per method. Results
  are given as mean ± sd and per seed. No p-values are computed on n = 3.
* **Authors are the resampling unit** within a run. A run has 20 authors with
  10 trained and 10 held-out facts each. Facts of one author share a
  fine-tuning history and a description, so they are not independent. Every
  fact-level CI comes from 1000 author-level bootstrap resamples (percentile
  method), via `fact_level_analysis.author_bootstrap`.
* **Claims are stated per seed:** "in every seed" or "in k of 3 seeds". Seeds
  are never pooled into a single test.
* **Multiplicity.** There are three primary hypotheses (H1–H3). Every other
  number is exploratory and labelled that way.
* **Exclusions.** Collapsed points are always excluded. Points below the 95%
  floor are shown but excluded from H2–H5.
* **Comparability.** TOFU Forget Quality is reported for every run and split,
  alongside the calibrated breadth axis, so the results can be read against
  published work.

## 6. The tools that compute each number

| Hypothesis | Script | Output |
|---|---|---|
| H1 selectivity, H4 slopes, H5 tier slopes | `python -m deeperase.scripts.aggregate_matrix --glob 'study_v2_*'` | `results/analysis/matrix.{json,md,png}` |
| H1 matched control | `gpu/summarize.py`, the `control_*` runs | console |
| H2, H3, partial band, indirect leak | `python -m deeperase.scripts.fact_level_analysis --glob 'study_v2_*'` | `results/analysis/fact_level.{json,md}` |
| §2 per-fact resolution | `python -m deeperase.scripts.depth_fact_resolution` | `results/analysis/depth_fact_resolution.json` |

All of these read saved JSON only. No GPU is needed after the runs.
`aggregate_matrix` and `fact_level_analysis` group runs by protocol, **design
variant** (loss scope and dtype), method and learning rate. An ablation is
therefore never pooled with the runs it is compared against.

## 7. GPU budget and priority

| Step | What | Est. GPU time (A100 estimates; the H100 smoke run re-times them) |
|---|---|---|
| 0 | Environment and test suite | 15 min |
| 1 | Learning-rate scans (`scripts/lr_scan.py`), then one full study per method at the scanned rate | ~2–3 h + ~2.5 h |
| 2 | The matrix: 3 methods × 3 seeds | ~6 h |
| 3 | T1 control ladder, GA seed 0 (scales 1, 3, 10) | ~3–4 h |
| 3b | A1 loss-scope ablation, GA seed 0 (`--tag lossfull --extra --loss-scope full`) | ~1 h |
| 3c | α extension where α\* = 1.0 (`--extra --alphas 1.25,1.5,2.0`; the run resumes) | ~20 min per run |

That is about 14–16 GPU hours. If time runs short, the order is: 1 → 2 (GA,
GradDiff) → 3 (scale 1, then the matched scale) → 2 (NPO) → 3b → 3c. H1–H3
need only steps 1–3 for one method.

## 8. What each outcome would mean (stated in advance)

| Outcome | Reading |
|---|---|
| H1 fails | At 1B these methods do not forget selectively. Every depth or breadth gain is collateral damage. This is a negative result and is reported as one |
| H2: S high, indirect leak high for shallow facts | Unlearning is mostly suppression. A behavioural test on the direct question overstates forgetting, and UDS identifies which facts are still recoverable |
| H2: S low | Behaviour and depth agree fact by fact. At this scale, the cheap behavioural test is a fair proxy for depth |
| H2: S falls with α | Extrapolation turns suppression into removal. This is a mechanism for UIPE's gains that UIPE itself did not measure |
| H3: ρ > 0 | Depth and behaviour are coupled per fact, so the plane's average co-movement is not an aggregation artefact |
| H3: ρ ≈ 0 | The plane's co-movement hides fact-level independence. Conclusions drawn from averages do not transfer to individual facts |
| Partial-band share ≫ 8% | Unlearning partly erases every fact, unlike clean absence. A given UDS then means different things for unlearned and retain models |
| H4 holds | Forgetting reaches untrained facts about the entity. This tests UIPE on real held-out knowledge, not constructed facts |
| H5 fails | Forgetting is tied to how the question is phrased. Benchmarks that only ask the trained question overstate it |

## 9. Deviations from this plan

*None yet.* Each entry needs: the date, the change, the reason, and whether
any v2 result had been seen at the time.
