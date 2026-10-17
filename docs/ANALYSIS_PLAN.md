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
pools all of its items (B0, B1, B2, B3, B4). Each item is compared with the same
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
| indirect leak | For facts forgotten on B0/B1, the share of their eligible B2/B3/B4 items still answered correctly |
| partial band | 0.2 ≤ depth_i < 0.8 |
| α\* | ~~Per run, the largest α whose point is above the 95% utility floor~~ **α = 1**, the end of the pre-registered sweep (§9, 14 Oct). The checkpoint rule puts α = 0 at the floor, so the old definition always gave α\* = 0 |
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

**Test.** The tier slopes of B0, B1, B2, B3 and B4 against α (`aggregate_matrix`),
and the research plan's breadth-generalisation gap, B0 − B3.

**Support.** The B2 and B4 slopes are each ≥ 0.5 × the B0 slope in every seed.
B3 is judged the same way only if its calibration range (`full` − `retain90`)
is at least 0.05. Below that the trained model cannot answer in the reverse
direction, and B3 is reported as unmeasurable, not as narrow forgetting.
If they fall below that, forgetting is mostly surface-level. That is a finding,
and H2's indirect leak should then be high.

### H6 — Methods (exploratory)

GA, GradDiff and NPO are compared on selectivity, S, ρ, the depth − breadth
gap and the tier slopes. With 3 seeds per method this is descriptive only.

### A1 — Loss-scope ablation (exploratory)

One GA run (seed 0) is repeated with `--loss-scope full`, v1's choice. This
asks whether the question-token loss caused v1's collateral damage. The
comparison is selectivity and R retention at matched breadth.

### H7 — STRANGER, a new method (pre-registered 17 Oct 2026, before any STRANGER run)

Fixed after the v2 matrix was seen (it motivated the method) and **before any
STRANGER run, gate 1 included**. The method is described in
[`STRANGER.md`](STRANGER.md). In short: train only the MLP weights of layers
0-8 so that, at the forget author's name and question positions, the block
outputs of layers 4-8 match M_full's own outputs on the same row with the name
swapped for K = 4 invented, matched names; anchor retain rows to M_full's own
states; leave layers 9-15 untouched. It uses no oracle model, no held-out fact
and no evaluation item.

**Claim.** Removing the name-keyed retrieval at its source erases the facts
more deeply than GA, GradDiff and NPO, beyond what matched random damage
explains, on held-out facts as well as trained ones, while keeping general
knowledge.

**Arms (all protocol v2: 3 seeds, 20 epochs, batch 4, float32, checkpoint at
the 95% R floor, UIPE sweep α = 0..1).**
1. `stranger`, seeds 0-2 (primary).
2. `scramble`, the same pipeline with each target replaced by a state that
   moves the same distance in a random direction fixed per (author, layer).
   Seed 0 at least; seeds 1-2 if time allows. It isolates "made the name
   unknown" from "disturbed the name".
3. Matched-damage controls for `stranger`, re-probed on the trained tensors
   only, seeds 0-2. If the real run barely lowers R, the matched-magnitude
   control (scale 1) is added for seed 0.

**Fixed hyperparameters.** Layers 4-8 as targets; trainable
`model.layers.[0-8].mlp.{gate,up,down}_proj.weight`; K = 4 names from
`data/counterfactual_names.json`; retain weight λ = 1 on 200 retain rows
(R-tier authors excluded); AdamW (weight decay 0.01), gradient clip 1.0; the
learning rate from a scan over {3e-6, 1e-5, 3e-5} on seed 0 with the existing
`lr_scan.py` rule; positions as in `STRANGER.md` §3. Fallback if R breaks at
every scanned rate: λ = 3.

**Gates (each pass/fail rule fixed now).**
* **G1, virtual-source ceiling** (`scripts/virtual_source.py`, seed 0,
  inference only). The hybrid source on the TRAINED split: **GO** if UDS ≥ 0.50,
  strict UDS ≥ 0.45 and overshoot share ≤ 0.10; **WEAK GO** if 0.35 ≤ UDS < 0.50
  (an extra arm, STRANGER + NPO on the same trainable set, is then added as
  co-primary); **NO-GO** below 0.35 (stop; the ceiling is reported as a
  finding about name-keyed retrieval). Held-out G1 numbers are reported only.
* **G2, learning-rate scan**, seed 0: some rate keeps an epoch with R ≥ 0.765,
  forget NLL at least 0.5 nats above baseline, and the forget term at most 0.4
  of its start; every non-trainable tensor bit-identical.
* **G3, seed-0 study**: at the checkpoint, held-out UDS ≥ 0.30 and strict ≥ 0.27,
  trained UDS ≥ 0.35, RA ≥ 0.80. Pass → seeds 1-2 and the controls.

**Success criteria (judged on 3 seeds; S1-S3 are primary).**
* **S1.** At the checkpoint (α = 0), 3-seed means: held-out UDS ≥ 0.35 and
  trained UDS ≥ 0.40, each at least GA's checkpoint mean + 0.15 (GA: 0.22
  trained, 0.19 held-out); strict UDS within 0.03 of UDS on both splits.
* **S2.** Damage-matched excess of **strict** UDS on held-out facts (each seed
  against its own control, interpolated at equal R as in `hypothesis_report`)
  ≥ +0.10 in every seed and ≥ +0.20 on average. Today's best is +0.08.
* **S3.** At the checkpoint, RA ≥ 0.77 and WF ≥ 0.80 (3-seed means), and
  selectivity at matched R at least GradDiff's.
* **S4 (no gaming flag).** Overshoot share ≤ 0.05 at α = 0; band UDS
  L13-15 − L9-12 ≤ 0.15; held-out depth ≥ 0.8 × trained depth; trained depth
  moves ≤ 0.02 when the rows whose loss positions touch an entity-predicting
  position are excluded.
* **S5.** STRANGER beats SCRAMBLE on strict UDS (held-out) and on headline
  breadth at the checkpoint (seed 0; all seeds where SCRAMBLE ran).
* **S6 (secondary).** After a relearning attack on seed 0 (fine-tuning on half
  of each author's trained facts), the latent-knowledge index on held-out facts
  is no higher than GA's and NPO's.

**Outcome (17 Oct 2026): G1 NO-GO.** Hybrid UDS on trained facts was 0.087
(strict 0.087), below the 0.35 stop line. Even the full name swap reached only
0.236. STRANGER stops as specified; the ceiling is reported as a finding
(RESULTS.md §17). G2, G3 and S1-S6 were not run.

**What each outcome means (stated now).** S1-S3 met: a method whose depth is
not damage and not overshoot, which generalises to untrained facts. S1 met but
S2 not: deeper than the baselines but not beyond damage, so no mechanism claim.
G1 NO-GO: facts are not retrieved through the name alone, a finding about where
TOFU knowledge lives. S5 failing: the gain comes from disturbing the name, not
from making it unknown. Every outcome is reported.

### H8 gate — what the memory is keyed on (pre-registered 17 Oct 2026, before running it)

Written after STRANGER's gate 1 (NO-GO, RESULTS.md §17) and before gate 1b
runs. `scripts/memory_keys.py` (seed 0, inference only) scores M_full on
equal-length edited copies of each depth example as UDS stage-2 sources:

* the name swapped everywhere;
* the question replaced by neutral filler;
* both of those;
* the question replaced by a same-length question about a retain author
  (R-tier authors excluded);
* everything before the entity replaced by filler.

Each source is scored on the whole entity and on its first token.

**Decision rule.** A successor method (H8) is designed only if at least one
candidate source reaches, on the TRAINED split, UDS ≥ 0.50, strict UDS ≥ 0.45
and overshoot share ≤ 0.10. The candidates are the neutral question, question
plus name, the other author's question, and everything-before-the-entity. If
several pass, H8 is built around the one with the highest strict UDS, and its
own hypotheses and success criteria are pre-registered before any training.
If none passes, no further method is built and the project proceeds to the
write-up. Either way the decomposition is reported (RESULTS.md §18).
Held-out numbers are reported only.

**H8 gate outcome (17 Oct 2026): STOP.** No candidate cleared the bar on
trained facts:

| Candidate | UDS | strict UDS | Shortfall |
|---|---|---|---|
| question + name | 0.559 | 0.425 | strict below 0.45 |
| other author's question | 0.532 | 0.406 | strict below 0.45 |
| everything before the entity | 0.985 | 0.236 | almost all overshoot |
| question | 0.372 | 0.297 | UDS below 0.50 |

See RESULTS.md §18.

### H8-X — UNASKED, an exploratory successor (fixed 17 Oct 2026, before any UNASKED run)

**Deviation, declared.** The H8 rule above said to stop. UNASKED is built
anyway, as an **exploratory** method. It is never reported as a pre-registered
success. Reason: the best source (question + name) has a ceiling of strict UDS
0.425 and first-token UDS 0.621, about twice the 0.21 strict UDS that GA,
GradDiff and NPO reach at their checkpoint. The GPU may be unavailable after
18 Oct, so it is tried now rather than never. Every UNASKED number in RESULTS
is labelled exploratory.

**Method** ([`UNASKED.md`](UNASKED.md)).
* Targets: M_full's block outputs on the same row with the question replaced
  by neutral filler of equal length and the author's name swapped for invented
  names, averaged over K = 4 names.
* Positions: every answer position, from the one predicting the first answer
  token to the last, except non-final tokens of an answer-side name mention.
* Layers: all 16.
* Trainable: the attention and MLP weights of every block. Embeddings, norms
  and the head are frozen.
* Retain anchor: M_full's own states at all 16 layers on the 200 retain rows,
  λ = 1.
* Learning rate: from a scan over {1e-6, 3e-6, 1e-5} with the bounded-method
  rule, i.e. the usable rate closest to its target (`lr_scan.recommend`).
* Otherwise protocol v2, with the paraphrased-question depth split on.

**Gates.**
* **X-G2:** the scan finds a usable rate, i.e. rep_gap ≤ 0.6 at a kept epoch
  with R ≥ 0.765.
* **X-G3:** in the seed-0 study, at the checkpoint, strict UDS on trained facts
  is ≥ 0.30 and RA ≥ 0.80. Then seeds 1-2 and matched-damage controls follow;
  otherwise stop and report.

**Exploratory criteria (all reported, whatever the outcome).**
* **E1.** Checkpoint strict UDS on trained facts ≥ 0.35 (3-seed mean; GA 0.21).
* **E2.** Damage-matched excess of strict UDS on trained facts ≥ +0.15 in every
  seed.
* **E3.** Strict UDS on the paraphrased-question split ≥ 0.7 × trained strict UDS.
  This tests that the erasure is not tied to the trained wording. GA and NPO
  seed 0 are rerun with the same split for comparison.
* **E4.** RA ≥ 0.77 and WF ≥ 0.80 at the checkpoint.
* **E5.** Held-out depth is reported with no criterion. The memory is per
  sequence (RESULTS.md §17-18), so it is expected near the baselines.

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
* **Exclusions.** Collapsed points (utility below 75% of the start) are always
  excluded. ~~Points below the 95% floor are shown but excluded from H2–H5.~~
  Every other point is used, with its utility drop reported beside it, and the
  matched-damage control separates forgetting from damage (§9, 14 Oct).
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
| H1's matched-damage scale | `python -m deeperase.scripts.control_probe` (via `gpu/matched_controls.sh`) | `results/control_probes/*.json` |
| **Every verdict, seed by seed** | `python -m deeperase.scripts.hypothesis_report` | `results/analysis/hypotheses_v2.{json,md}` |
| Overshoot audit: strict/tent UDS, band UDS (exploratory; co-primary for H7) | `python -m deeperase.scripts.uds_audit` | `results/analysis/uds_audit_v2.{json,md}` |
| H8 gate (1b), what the memory is keyed on | `python -m deeperase.scripts.memory_keys --seed 0` | `results/gates/memory_keys_s0.{json,md}` |
| H7 gate 1, virtual-source ceiling | `python -m deeperase.scripts.virtual_source --seed 0` | `results/gates/stranger_g1_s0.{json,md}` |
| Controls and tagged runs, fact level (exploratory) | `python -m deeperase.scripts.fact_level_analysis --run <dir> ...` | `results/analysis/fact_level_v2_exploratory.{json,md}` |

All of these read saved JSON only. No GPU is needed after the runs.
`aggregate_matrix` and `fact_level_analysis` group runs by protocol, **design
variant** (loss scope and dtype), **arm** (main, a matched-damage control, or
a `--tag` such as `ext`), method and learning rate. An ablation, extension or
control is therefore never pooled with the runs it is compared against.

## 7. GPU budget and priority

| Step | What | Est. GPU time (A100 estimates; the H100 smoke run re-times them) |
|---|---|---|
| 0 | Environment and test suite | 15 min |
| 1 | Learning-rate scans (`scripts/lr_scan.py`), then one full study per method at the scanned rate | ~2–3 h + ~2.5 h |
| 2 | The matrix: 3 methods × 3 seeds | ~6 h |
| 3 | T1 control ladder, GA seed 0 (scales 1, 3, 10) | ~3–4 h |
| 3b | A1 loss-scope ablation, GA seed 0 (`--tag lossfull --extra --loss-scope full`) | ~1 h |
| 3c | α extension where α\* = 1.0 (`--extra --alphas 1.25,1.5,2.0`; the run resumes) | ~20 min per run |

That is about 14–16 GPU hours. Measured on the H100 slice (13 Oct): the three learning-rate scans took 3 h in all, and a full study takes about 1 h. If time runs short, the order is: 1 → 2 (GA,
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

Each entry needs: the date, the change, the reason, and whether any v2 result
had been seen at the time.

* **17 Oct 2026: overshoot diagnostics added (exploratory), and H7 (STRANGER)
  pre-registered.** `eval/uds_diagnostics.py` and `scripts/uds_audit.py` report
  strict UDS (LER := 0 where r = dS2/dS1 > 2), tent UDS (LER := max(0, 2 − r)
  where r > 1), the overshoot share and UDS by layer band, next to UDS. Reason:
  the clip in Eq. 4 scores a source that does worse than retain90 as full
  erasure; oracles (retain95/99 on unseen facts) almost never exceed r = 2,
  while the v2 runs take 14-17% of their α = 1 UDS (trained facts) from such cells. UDS stays
  the pre-registered measure for H1-H6; strict UDS is a co-primary measure for
  H7 only, which is new. Every v2 result had been seen. H7 was written before
  any STRANGER run.

* **16 Oct 2026: exploratory analyses added after the matrix was seen. No
  rule changed.** (1) `hypothesis_report.py` repeats H1's control comparison
  against each seed's own matched control where one exists (GradDiff: all
  three seeds). The H1 verdict still follows the rule, which reads seed 0.
  (2) `fact_level_analysis.py` is run on the matched-damage controls. This
  gives a damage baseline for H2's suppression rate and the partial band, and
  it records the share of fully erased facts (UDS_i ≥ 0.8). (3) A1 is
  compared "at matched breadth" by a straight-line fit of each variable on
  breadth along the α ≤ 1 path, because the two paths share no breadth values
  exactly. All three are labelled exploratory in `RESULTS.md` §15. Reason: the
  plan named the comparisons but not these baselines. The first two make the
  reading stricter, not looser. Every result had been seen.

* **14 Oct 2026: α\* is redefined as α = 1, and only collapsed points are
  excluded.** *Seed-0 results of all three methods had been seen.* The reason
  holds by construction, whatever the results: the checkpoint rule keeps the
  most-forgetful epoch still at ≥ 95% utility, so the kept model sits at the
  floor, and every extrapolated point (α > 0) falls below it. All three seed-0
  runs show this (α\* = 0 for GA and NPO, 0.1 for GradDiff), and v1 did the
  same (`RESULTS.md` §8, issue 6). With the old rule, H2's second test,
  S(α\*) < S(0), compares a point with itself. α = 1 is the end of the sweep
  fixed before any data. The trajectory's own damage is handled as the plan
  always intended for H1, by the matched-damage control
  (`scripts/control_probe.py` finds its scale), not by dropping points.
  Thresholds, statistics and the other decision rules are unchanged.
* **13 Oct 2026: B3 added** (365 items, the reverse direction; `PROTOCOL_V2.md`
  §4). Reason: the research plan's breadth-generalisation gap is defined as
  B0 − B3, and the plan had no B3 to compute it from. No v2 result had been
  seen. The only v2 runs were a smoke test that stopped in training and the
  learning-rate scans, which score the R tier only. B3 joins H5 and the
  fact-level indirect leak. The decision rule for an uninformative B3 is in H5.
