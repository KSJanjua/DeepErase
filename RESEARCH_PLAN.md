# DeepErase → Publishable Research: Strategy, Direction and Roadmap

**Prepared:** 2 August 2026
**Companion documents:** [`DeepErase_Domain_and_Literature_Report.md`](DeepErase_Domain_and_Literature_Report.md) (domain expertise, 38-paper comparison, 14 documentation issues) · [`literature/`](literature/) (54 downloaded PDFs, `references.bib`, categorised reading list)

---

## Phase 1 — Analysis of the Current Project

### 1.1 What the project is

An integration of three published components — UIPE parameter extrapolation, ERUF/KIF activation-signature suppression distilled into LoRA, and an SMR+EL10 dual metric with a Type I/II/III taxonomy — evaluated on TOFU, WMDP and a 4-subject RWKU subset against ELM, PISCES and RMU.

### 1.2 Strengths worth keeping

| Strength | Why it survives scrutiny |
|---|---|
| **The framing is correct and current** | "Obfuscation ≠ erasure" is precisely the axis on which the SoK (Ren et al. 2025) organises the whole field. The project is aimed at the right problem. |
| **Two-axis evaluation instinct** | Refusing to report forget-quality alone is the right instinct, and it is what the critical literature (Thaker, Feng, Łucki) demands. |
| **UIPE is a superb research instrument** | It exposes a *single scalar knob* (α) that monotonically increases forgetting pressure, and applying it is **pure post-hoc parameter arithmetic — no training**. Almost nothing else in this field gives you a continuous, cheap, controllable dial. This is the project's most under-exploited asset. |
| **Cross-model, cross-benchmark scope** | TOFU + WMDP + RWKU with capability-retention columns is the right evaluation surface. |
| **Realistic hardware framing** | 4-bit, 7B–8B, LoRA — matches what ERUF itself did on a single A6000. |
| **Ablation discipline** | Per-loss-term and per-α ablations are already planned. |

### 1.3 Limitations and technical weaknesses

**L1 — The contribution is currently additive, not scientific.** "Component A + Component B, evaluated" is the single most common reason methods papers are rejected. There is no stated hypothesis, no phenomenon being explained, no claim that could be *false*.

**L2 — The central metric does not measure what the project claims.** EL10 is computed from **output token probabilities** over the first 10 decoding steps. It is a softer surface metric, not a latent one. A framework whose thesis is "representation-level, not output-level" cannot rest its verification on an output-distribution statistic. ERUF itself concedes this by adding hidden-state diagnostics (E₃₀, SRS).

**L3 — No robustness evaluation.** Robustness to relearning is claimed as an *expected contribution* but appears nowhere in the methodology. Given Łucki (10 unrelated examples recover RMU), Deeb & Roger, and UNDO (oracle-matching behaviour still leaves recoverable capability), an unlearning paper without a relearning column in 2026 is not reviewable.

**L4 — The localisation premise is assumed, not tested.** Hase et al. (NeurIPS 2023) and H. Lee et al. (EMNLP 2025) both find that localisation does not causally improve editing or unlearning. Cohen's *d* separability proves a signature is *decodable*, not that suppressing it *causes* forgetting.

**L5 — Reproduction scope is unrealistic.** Five non-trivial methods reproduced, two integrated, one new protocol built, in 16 weeks. PISCES alone ran 800 experiments *per concept*.

**L6 — Two hard feasibility blockers.** PISCES needs per-MLP-layer SAEs, which exist for Gemma-2-2B and Llama-3.1-8B but **not** for Mistral-7B or Zephyr-7B. UIPE as published needs full-parameter fine-tuning on 2×A800 80GB and a dense delta; the plan specifies 4-bit on 40GB and never says whether the extrapolated delta is full-parameter or LoRA.

**L7 — Baseline set is dated.** SimNPO, LUNAR and ReGLU are absent — all three are direct competitors and two are ERUF's own baselines.

**L8 — Citation errors.** Ref [4] is misattributed *and* misdescribed; RMU/WMDP, TOFU, RWKU and NPO are used but uncited. (Full list: D1–D14 in the domain report.)

### 1.4 What must be redesigned vs. kept

| Component | Verdict |
|---|---|
| Problem framing, motivation | **Keep** |
| Benchmark suite (TOFU/WMDP/RWKU) | **Keep**, add MUSE only if time allows |
| Activation-signature extraction + capsules | **Keep as a mechanism**, but demote from "the contribution" to "one arm of a controlled comparison"; add a causal control |
| UIPE extrapolation | **Keep and promote** — from a bolt-on post-processor to the project's primary experimental instrument |
| SMR + EL10 | **Keep as the surface layer**, but **add a genuine depth metric on top** (UDS) |
| Type I/II/III taxonomy | **Redesign** — supersede with a 2-D measured plane rather than a 3-way label |
| Baseline reproduction from scratch | **Replace** with OpenUnlearning |
| "Integration of A and B" as the contribution | **Replace** — see Phase 5 |

---

## Phase 2 — Research Opportunities

Six directions, each assessed honestly against what is already published.

---

### D1 — The Depth–Breadth Trade-off in LLM Unlearning ⭐ *recommended*

**Problem.** Unlearning is evaluated along two axes that no one has crossed. **Breadth** = does forgetting generalise beyond the exact forget prompts — to paraphrases, aliases, indirect and multi-hop queries, and logically entailed facts? **Depth** = has the knowledge been attenuated *in the representation*, or only suppressed at the output?

**Why it matters.** Everything in the field implicitly assumes these axes are aligned — that a method which forgets more broadly also forgets more deeply. **There is direct evidence that they are not.** ERUF's own Table 1 reports Qwen-8B at SMR 3.33% with **EL10 = 11.03**: surface leakage nearly eliminated while the latent subject-token mass rose *eleven-fold above baseline*. UIPE reports an **inverted-U** for GA — forget quality improves then deteriorates as α grows. RepSelect diagnoses that broad updates hit representations shared with both the retain set and the attacker-recoverable subspace. Over-erasure work (Xu et al. 2026) shows broad updates delete retain-supported knowledge. Four independent signals, never assembled into a hypothesis.

**Existing approaches and their limits.**
- *Breadth-only:* SUITE / Forget Narrowly Retain Broadly (Peleg et al., arXiv 2607.09236) formalises under-/over-forgetting and builds paraphrase, indirect and multi-hop probes — and contains **no representation-level analysis whatsoever**. Munch, UIPE, PrivUn likewise.
- *Depth-only:* UDS (J. Lee et al., arXiv 2605.24614) gives a causal activation-patching depth score; Unlearning Isn't Deletion (Xu et al., ICML 2026) gives PCA/CKA/Fisher reversibility regimes; An Illusion of Unlearning (Gao et al., AISTATS 2026) shows probes recover forget accuracy — but all hold breadth fixed.
- **Nobody varies one and measures the other.**

**Why this direction is valuable.** It converts the project from "we combined two methods" into "we discovered and characterised a trade-off the field has been assuming away." It is falsifiable, it explains existing anomalies, and it directly informs how everyone else should report results.

**Expected contribution.** (i) A 2-D *depth × breadth* evaluation plane replacing the flat Type I/II/III label; (ii) evidence that current interventions trade one axis against the other; (iii) a diagnosis of *why* (isotropic update scaling); (iv) a targeted method that Pareto-dominates (see D2, which folds in here).

**Benefits.** Uses an instrument the team already owns (UIPE's α). Both measurement stacks exist as open code (UDS, SUITE, OpenUnlearning). **The α sweep requires no training** — one GA/NPO run, then N cheap parameter interpolations. Guaranteed to produce a result: even a *flat* trade-off curve is a publishable negative finding against the field's implicit assumption.

**Drawbacks / risks.** The effect may be weak on some benchmarks; requires careful controls to rule out "α just makes the model worse at everything."

**Technical challenges.** Implementing UDS faithfully (needs a retain-model baseline — TOFU ships oracles, so start there); defining breadth probes for WMDP where no subject string exists.

**Effort.** Medium. **Compute.** Low–moderate (see §6.4). **Feasibility.** High. **Publication potential.** High — ACL/EMNLP main or Findings; strong workshop paper as a floor.

---

### D2 — Signature-Aligned Extrapolation (directed, anisotropic forgetting)

**Problem.** UIPE amplifies the *entire* update vector by a scalar: $\theta + (1+\alpha)v$. This is isotropic — every direction in $v$ is scaled equally, including directions that encode retain-set knowledge. That is exactly why GA shows an inverted-U and why over-erasure happens.

**Proposed idea.** Decompose $v$ relative to the ERUF activation-signature subspace $S$ (which ERUF shows generalises to aliases and descriptions, i.e. it is already a cheap proxy for the entailment neighbourhood):
$$\theta_{\text{SAGE}} = \theta_{\text{un}} + \alpha_\parallel \, \Pi_S(v) + \alpha_\perp \, (I - \Pi_S)(v), \qquad \alpha_\parallel > \alpha_\perp$$
Amplify only the related-knowledge-aligned component; leave the rest at baseline. This is a principled repair of a limitation the UIPE authors state themselves.

**Existing approaches and limits.** ReGLU constrains LoRA updates to the orthogonal complement of the *retain* subspace — it pushes *away from retain*, not *toward related-forget*. RepSelect collapses top principal components of weight gradients for robustness. Orthogonal-subspace SVD-LoRA handles continual unlearning. **None of them direct an extrapolation coefficient along a forget-related subspace.**

**Contribution.** A method that decouples breadth from collateral damage; the constructive half of D1.

**Drawbacks.** Genuinely close to ReGLU and RepSelect in spirit — differentiation must be argued carefully and demonstrated empirically, not just asserted. Risk that gains are small.

**Effort.** Medium-high. **Compute.** Moderate. **Feasibility.** Medium. **Publication potential.** Medium-high *as the second half of D1*; medium on its own.

---

### D3 — Predicting the unlearning regime before unlearning

**Problem.** Can we forecast from the *base model alone* whether a given target will unlearn genuinely (Type I) or only superficially (Type II)?

**Motivation.** ERUF reports Type II on Llama-3B and reasoning-prior Qwen-8B, Type III on DeepSeek-3B, Type I elsewhere — with **no explanation**. Selective Pruning (already in the repo) supplies a mechanistic hypothesis: models trained with FF dropout have more task-specialised neurons, hence cleaner signatures. UDS reports that "erasure depth varies across examples." So variation exists at both model and example level and nobody has modelled it.

**Existing work.** Gao et al. (AISTATS 2026) probe *after* unlearning; UDS measures *after*. Nobody uses pre-unlearning separability as a *predictor*.

**Contribution.** A cheap a-priori diagnostic (Cohen's *d* profile, probe accuracy, signature-norm ratio) that predicts post-unlearning regime; practical value is deciding whether a target is worth attempting.

**Drawbacks.** Needs many (model, target) pairs to establish a correlation — the sample size is the bottleneck, not the compute per point. Risk of a null result with wide error bars.

**Effort.** Medium. **Compute.** Moderate–high (breadth of models). **Feasibility.** Medium. **Publication potential.** Medium — excellent as an *analysis section* inside D1 rather than a standalone paper.

---

### D4 — Independent replication and stress-test of ERUF/KIF

**Problem.** ERUF is an un-refereed preprint claiming to break the stability–erasure trade-off (FQ 0.99 at oracle MU 0.62) on a **single TOFU run**, with a 4-subject RWKU subset.

**Contribution.** Multi-seed replication with variance, full-scale RWKU, plus the relearning attacks ERUF never ran.

**Drawbacks.** Replication papers are hard to place at top venues unless the result is surprising. But it is essential *internal* work regardless — the project cannot build on ERUF without first confirming it.

**Effort.** Low-medium. **Feasibility.** High. **Publication potential.** Low standalone; **high value as Milestone 2 of D1**.

---

### D5 — Sequential / continual representation-aware unlearning

**Problem.** Real deletion requests arrive as a stream. MUSE's sustainability axis is largely unsolved.

**Existing work.** EUL (in the repo) has closed-form adapter fusion; LUNAR handles sequential requests; ERUF Table 8 reports sequential results; SUITE evaluates sequential settings; orthogonal-subspace SVD-LoRA targets exactly this.

**Verdict.** Crowded and getting more so. **Not recommended as the primary direction.**

---

### D6 — Certified / provably irreversible erasure

**Verdict.** The field's most valuable open problem and far beyond a capstone's reach — it requires either training-time intervention (UNDO-style distillation, pretraining filtration) or new theory. **Not recommended.**

---

## Phase 4 — Idea Evaluation and Ranking

| Idea | Originality | Significance | Practicality | Difficulty | Compute | Risk | Expected gain | Validation ease | Pub. potential |
|---|---|---|---|---|---|---|---|---|---|
| **D1 Depth–breadth trade-off** | High (unclaimed cross-product) | High | High | Medium | **Low** | **Low** | N/A — it's a finding | **Very easy** (α is free) | **High** |
| **D2 Signature-aligned extrapolation** | Medium-high | Medium-high | Medium | Med-high | Moderate | Medium | Pareto shift | Easy | Med-high |
| **D3 Regime prediction** | Medium-high | Medium | Medium | Medium | Mod-high | Medium | N/A | Medium | Medium |
| **D4 ERUF replication** | Low | Medium | High | Low | Low | Low | N/A | Easy | Low |
| **D5 Sequential unlearning** | Low (crowded) | Medium | Medium | Med-high | High | Med-high | Small | Medium | Low-med |
| **D6 Certified erasure** | Very high | Very high | **Very low** | Very high | Very high | Very high | — | Hard | High if solved |

**Categorisation.**
- *Incremental:* D4, D5.
- *Moderate research contribution:* **D1, D2, D3.**
- *High-risk / high-reward:* D6.

**Which category fits this project?** **Moderate.** A five-person third-year team on a shared 40 GB GPU over one semester cannot de-risk D6, and D4/D5 will not clear a reviewer's novelty bar. The moderate tier is the only one that is simultaneously novel enough to publish and small enough to finish.

---

## Phase 5 — Recommended Direction

> ### **D1 as the spine, with D2 as the method contribution and D3/D4 as analysis sections.**
>
> **Working title:** *Deep or Merely Broad? Disentangling the Depth and Breadth of Knowledge Erasure in Large Language Models*

### 5.1 The claim

Unlearning interventions face a **depth–breadth trade-off**: pressure that widens the *scope* of forgetting (paraphrases, aliases, entailed facts) is purchased by increasing the magnitude and isotropy of the parameter update, which raises **surface suppression faster than representation-level attenuation**. Consequently, methods that score best on breadth-heavy benchmarks may be *more* obfuscatory, not less — and the field's two evaluation traditions have been measuring complementary halves of one phenomenon while assuming they agree.

### 5.2 Why this beats the alternatives

1. **The gap is real and I verified it directly.** SUITE (July 2026) is the state of the art on breadth and states no representation-level analysis. UDS (May 2026) is the state of the art on depth and holds breadth fixed. Their intersection is empty.
2. **It is a hypothesis, not a combination.** It can be falsified. That is what makes it a paper rather than a system report.
3. **It explains four existing anomalies** nobody has connected: ERUF's EL10 = 11.03 on Qwen-8B, UIPE's inverted-U on GA, RepSelect's shared-subspace diagnosis, and the over-erasure finding.
4. **It is cheap.** α extrapolation is post-hoc parameter arithmetic; UDS is training-free. The expensive part — the unlearning runs themselves — is done once and amortised across the whole α sweep. This is the rare research design where the interesting axis is nearly free.
5. **It cannot fail to produce a result.** A trade-off is a finding; no trade-off is a finding against the field's implicit assumption. Contrast with D2 alone, which fails if the method doesn't beat baselines.
6. **It rescues, rather than discards, everything already planned.** UIPE becomes the instrument. ERUF becomes one arm of the comparison and the source of the signature subspace. SMR/EL10 become the surface layer of a richer plane. Nothing is wasted.
7. **It converts the project's biggest weakness into its subject.** L2 said the dual metric doesn't measure depth. D1 makes "what does depth even mean, and does it track breadth?" the research question.

### 5.3 What the paper claims, in one paragraph

> We show that LLM unlearning is evaluated along two axes — breadth (generalisation of forgetting to paraphrases, aliases and entailed facts) and depth (attenuation of the internal representation) — that the literature implicitly assumes are aligned. Using parameter extrapolation as a continuous, training-free instrument for varying forgetting pressure, we sweep breadth across four unlearning objectives, three models and three benchmarks, and measure depth causally via activation patching. We find [trade-off / regime structure], and show it explains previously unexplained failures in the literature. We trace the cause to the isotropy of the parameter update, and propose Signature-Aligned Extrapolation, which directs amplification along a forget-related subspace and [Pareto-dominates / partially mitigates].

---

## Phase 6 — Project Blueprint

### 6.1 Project goal

Build a **two-axis evaluation harness** (depth × breadth × robustness) for LLM unlearning; use it to characterise the depth–breadth trade-off across methods, models and benchmarks; and develop a directed-extrapolation method that improves the frontier.

### 6.2 System architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│ M0  FOUNDATION LAYER (built on OpenUnlearning, not from scratch)      │
│  models: Llama-2-7B-chat (TOFU, ships oracles) | Zephyr-7B (WMDP)     │
│          Llama-3.1-8B-Instruct (RWKU + SAE availability)              │
│  data:   TOFU f01/f05/f10 · WMDP-Bio/Cyber · RWKU (25-50 subjects)    │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼──────────┐    ┌──────────▼──────────┐    ┌──────────▼─────────┐
│ M1 UNLEARNING    │    │ M2 BREADTH PROBE    │    │ M3 DEPTH PROBE     │
│    ENGINE        │    │    GENERATOR        │    │    (UDS)           │
│                  │    │                     │    │                    │
│ objectives:      │    │ tier B0 exact       │    │ • locate target-   │
│  GA, GradDiff,   │    │ tier B1 paraphrase  │    │   encoding layers  │
│  NPO, SimNPO,    │    │ tier B2 alias/      │    │   vs retain oracle │
│  RMU, ERUF       │    │        description  │    │ • activation-patch │
│                  │    │ tier B3 entailed    │    │   forget→unlearned │
│ output: θ_un,    │    │        (1-hop)      │    │ • UDS ∈ [0,1]      │
│         θ_ini    │    │ tier B4 multi-hop   │    │ + probe accuracy   │
│         (saved)  │    │ tier R  retain      │    │ + PCA/CKA drift    │
└───────┬──────────┘    │        neighbours   │    └──────────┬─────────┘
        │               └──────────┬──────────┘               │
        │                          │                          │
┌───────▼──────────────────────────▼──────────────────────────▼─────────┐
│ M4  EXTRAPOLATION INSTRUMENT  (the experimental knob — NO TRAINING)   │
│                                                                       │
│   v = θ_un − θ_ini                                                    │
│   UIPE (isotropic):  θ(α) = θ_un + α·v,   α ∈ {0, .1, .2, … 1.0}      │
│   SAGE (directed) :  θ = θ_un + α∥·Π_S(v) + α⊥·(I−Π_S)(v)            │
│                      S = ERUF activation-signature subspace           │
│   → emits a family of models along a breadth-pressure continuum       │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼────────────────────────────────────┐
│ M5  TWO-AXIS EVALUATOR                                                │
│  BREADTH  : per-tier forget rate B0…B4, retain-neighbour accuracy     │
│  DEPTH    : UDS, linear-probe recoverability, PCA/CKA drift, EL10     │
│  SURFACE  : SMR, ROUGE-L, TOFU FQ/MU                                  │
│  ROBUST   : Retraining-on-T (Deeb & Roger), few-shot prompt attack    │
│  UTILITY  : MMLU, ARC-C, HellaSwag, WinoGrande, TruthfulQA            │
│  CONTROLS : keyword-filter (Thaker), ECO (weight-frozen Type-II ctrl),│
│             random-direction ablation (causal control for M3)         │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼────────────────────────────────────┐
│ M6  ANALYSIS — the depth×breadth plane, per method / model / benchmark│
│     + regime prediction from pre-unlearning separability (D3)         │
└───────────────────────────────────────────────────────────────────────┘
```

### 6.3 Module responsibilities

| Module | Input | Output | Owner (per proposal roles) |
|---|---|---|---|
| **M0** Foundation | HF models, benchmarks | Configured OpenUnlearning env, cached weights | Karanvir (setup/debug) |
| **M1** Unlearning engine | Base model + forget set | $\theta_{\text{ini}}$, $\theta_{\text{un}}$ checkpoints per objective | Rishabh (lead) |
| **M2** Breadth probes | Forget targets + Wikidata/GPT-4o | Tiered probe sets B0–B4 + retain neighbours | Kalpesh (data) |
| **M3** Depth probes | Unlearned + oracle models | UDS, probe accuracy, drift metrics | Rishabh + Rujul |
| **M4** Extrapolation | $\theta_{\text{ini}}, \theta_{\text{un}}$, signature $S$ | Model family over α | Rishabh |
| **M5** Evaluator | Model family + all probe sets | Full metric table | Rujul (evaluation) |
| **M6** Analysis + writing | Metric tables | Figures, paper | Pranav (docs/figures) + all |

### 6.4 Compute budget (why this is feasible on 40 GB)

| Workload | Count | Cost each | Total |
|---|---|---|---|
| Unlearning runs (6 objectives × 3 benchmarks × 3 seeds) | ~54 | ~1–3 GPU-h | ~110 GPU-h |
| **α sweep (11 points × 54 runs)** | ~594 models | **~0 (parameter arithmetic)** | **~0** |
| Evaluation passes over α family | ~594 | ~0.3 GPU-h | ~180 GPU-h |
| UDS activation patching | ~594 | ~0.2 GPU-h | ~120 GPU-h |
| Relearning attacks (subset, α ∈ {0, 0.3, 0.6, 1.0}) | ~216 | ~0.4 GPU-h | ~85 GPU-h |
| SAGE variants + ablations | — | — | ~100 GPU-h |
| **Total** | | | **≈ 600 GPU-h** |

Roughly **4 weeks of one dedicated A100/A6000**, or 2 weeks on two. Comfortably within a semester. **The decisive economy is that the primary experimental axis costs nothing to traverse** — evaluation dominates, and evaluation is cheap and parallelisable.

### 6.5 Implementation roadmap

| M | Weeks | Objective | Key tasks | Depends on | Deliverable | Effort |
|---|---|---|---|---|---|---|
| **M0** | 1–2 | Environment + decisions | Install OpenUnlearning; reproduce **one** TOFU number end-to-end; **decide: full-FT vs. LoRA delta (D8); model list vs. SAE availability (D9)**; fix citations D1–D14 | — | Working env + reproduced baseline + corrected proposal | Low |
| **M1** | 2–4 | Unlearning engine | GA/GradDiff/NPO/SimNPO via OpenUnlearning; RMU on WMDP; **save $\theta_{\text{ini}}$ and $\theta_{\text{un}}$ for every run** | M0 | Checkpoint zoo + reproduction table | Medium |
| **M2** | 3–5 | Breadth probes | Build tiers B0–B4 per target (GPT-4o generation + **manual verification**, following UIPE's k/k′ construction); retain-neighbour set; reuse SUITE where licence permits | M0 | Probe dataset + inter-annotator agreement | Medium |
| **M3** | 4–6 | Depth probes | Implement UDS (port from [gnueaj/unlearning-depth-score](https://github.com/gnueaj/unlearning-depth-score)); linear probes; PCA/CKA drift; **random-direction causal control** | M1 | Depth harness + validation on ECO (must read as Type II) | **High — critical path** |
| **M4** | 5–6 | Extrapolation | UIPE isotropic sweep; ERUF signature extraction; SAGE projection | M1, M3 | Model family generator | Medium |
| **M5** | 6–9 | **Main experiment** | Full α sweep × objectives × models × benchmarks; populate the depth×breadth plane | M2, M3, M4 | The central result figure | Medium |
| **M6** | 9–11 | Robustness + controls | Retraining-on-T; few-shot prompt attack; keyword-filter control; ECO control | M5 | Robustness table | Medium |
| **M7** | 10–12 | SAGE + ablations | $\alpha_\parallel$ vs $\alpha_\perp$ grid; subspace-rank ablation; signature-quality ablation | M5 | Method results | Medium-high |
| **M8** | 12–14 | Analysis + writing | Regime prediction (D3); error analysis; figures; paper draft | M6, M7 | Submission-ready draft | Medium |

**Critical path:** M0 → M1 → M3 → M5. **M3 is the riskiest module** — start it in parallel with M1, not after.

### 6.6 Experimental plan

**Baselines (all via OpenUnlearning):** GA, GradDiff, KL-Min, NPO, SimNPO, RMU. **Representation-level comparators:** ERUF (replicated), LUNAR if time permits. **Controls:** ECO (weight-frozen — must register as Type II or the depth metric is invalid); keyword filter (Thaker); random-direction ablation; retain-only oracle (upper bound); base model (lower bound).

**Metrics.**

| Axis | Metrics |
|---|---|
| Breadth | Forget rate at tiers B0–B4; retain-neighbour accuracy; **breadth-generalisation gap** = B0 − B3 |
| Depth | **UDS** (primary); linear-probe recoverability; PCA/CKA drift; EL10 (reported as *surface*, correctly labelled) |
| Surface | SMR, ROUGE-L, TOFU FQ (KS *p*), TOFU MU |
| Robustness | Retraining-on-T accuracy; few-shot prompt recovery; alias/keyword hit rate |
| Utility | MMLU, ARC-C, HellaSwag, WinoGrande, TruthfulQA |
| Cost | GPU-hours, parameters modified |

**Ablations.** (a) α ∈ {0,…,1.0}; (b) $\alpha_\parallel$ vs $\alpha_\perp$ grid; (c) signature subspace rank; (d) layer choice for signature extraction; (e) each UPU loss term; (f) 4-bit vs. fp16 (**tests Assumption 5 directly — a small but real contribution**); (g) full-FT vs. LoRA delta.

**Analyses.**
- *Quantitative:* depth×breadth scatter per (method, model, benchmark), with α as a trajectory; correlation between breadth gain and depth loss; Pareto frontiers.
- *Qualitative:* generations at low/high α showing surface-clean-but-latent-present cases; side-by-side with ECO.
- *Error analysis:* which probe tiers survive; which targets never reach Type I; per-example depth variance (UDS reports this varies).
- *Failure cases:* replicate ERUF's reported Type II on Llama-3B / Qwen-8B and test whether the trade-off explains them.
- *Robustness:* recovery vs. α — **key prediction: if the trade-off is real, higher α should yield *higher* post-relearning recovery.** That is a sharp, falsifiable test.

### 6.7 Expected contributions

**Scientific.** (i) Identification and characterisation of the depth–breadth trade-off; (ii) evidence that breadth-oriented benchmarks can reward obfuscation; (iii) a mechanistic account (update isotropy) supported by the directed-extrapolation intervention.

**Engineering.** (i) An open two-axis evaluation harness integrating UDS + tiered breadth probes + relearning audits on top of OpenUnlearning; (ii) SAGE, a training-free directed-extrapolation operator; (iii) a released probe dataset.

**Experimental.** (i) The first α-continuum study of extrapolation-based unlearning measured on depth; (ii) an independent multi-seed replication of ERUF with variance; (iii) a 4-bit vs. fp16 equivalence check for unlearning research.

### 6.8 Risks and mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **UDS implementation is harder than expected** | Medium | **High** (critical path) | Start week 4 in parallel; official code exists; fall back to linear-probe recoverability + PCA/CKA drift (Xu et al. 2026), both simpler and independently published |
| **No trade-off is found** | Medium | Low | A null result against a field-wide implicit assumption is publishable; pivot emphasis to the harness + replication |
| **UIPE delta incompatible with 4-bit/LoRA** | **High** | High | Decide in M0; if full-FT is infeasible, run TOFU-scale full-FT on 7B only in fp16 with gradient checkpointing, and use LoRA deltas elsewhere **as an explicit ablation (f/g)** rather than a workaround |
| **PISCES unreproducible on chosen models** | High | Medium | Drop PISCES as a baseline (justified: it targets concepts, not entities) or switch to Gemma-2-2B/Llama-3.1-8B |
| **GPT-4o probe generation is noisy** | Medium | Medium | Manual verification (UIPE did this); report inter-annotator agreement; ground in Wikidata triples |
| **Scope creep** | **High** | High | Freeze the baseline set at M1; SAGE (M7) is explicitly cuttable — D1 alone is a complete paper |
| **Shared GPU contention** | Medium | Medium | α sweep is embarrassingly parallel and CPU-cheap; batch evaluation overnight |
| **Reviewer: "just A+B"** | Medium | High | Already mitigated by construction — the paper leads with the phenomenon, not the composition |

---

## Phase 7 — Research Paper Plan

### 7.1 Title options

1. *Deep or Merely Broad? Disentangling the Depth and Breadth of Knowledge Erasure in LLMs* ← **recommended**
2. *Breadth Is Not Depth: A Two-Axis Evaluation of LLM Unlearning*
3. *The Cost of Forgetting More: Depth–Breadth Trade-offs in Parameter-Space Unlearning*
4. *Forgetting Wider, Forgetting Shallower: Why Unlearning Benchmarks Disagree*

### 7.2 Abstract outline

Motivation (regulatory + safety) → the field evaluates unlearning along two axes → nobody has crossed them → we use extrapolation as a training-free continuous instrument → we measure depth causally via activation patching → finding → mechanism (isotropy) → SAGE → results → release.

### 7.3 Section plan

**1. Introduction.** Open with the ERUF Qwen-8B anomaly (SMR 3.33%, EL10 11.03) as a concrete hook — surface clean, latent trace 11× baseline. State that breadth papers and depth papers have never met. Contributions as a bulleted list. Figure 1 = the depth×breadth plane with method trajectories.

**2. Related Work.** Organise by the SoK's removal-vs-suppression axis. Four subsections: (a) unlearning objectives (GA→NPO→SimNPO); (b) representation- and parameter-level methods (RMU, Adaptive RMU, ELM, PISCES, LUNAR, ReGLU, RepSelect, ERUF); (c) breadth evaluation (SUITE, Munch, RWKU, PrivUn, UIPE); (d) depth and robustness evaluation (UDS, Xu et al., Gao et al., Łucki, Deeb & Roger, UNDO). **Close with an explicit paragraph naming the empty intersection** — this is where the paper earns its novelty.

**3. Preliminaries.** Formalise unlearning; define breadth tiers B0–B4; define depth via activation patching; state why EL10 and ROUGE-L are surface metrics.

**4. Method.** 4.1 Extrapolation as a breadth instrument (with the task-arithmetic/ExPO connection stated explicitly). 4.2 The two-axis harness. 4.3 SAGE: signature-aligned directed extrapolation, with the projection derivation.

**5. Experimental Setup.** Models, benchmarks, baselines, controls (ECO, keyword filter, random direction), implementation on OpenUnlearning, compute.

**6. Results.** 6.1 The trade-off (main figure). 6.2 Per-method and per-model structure. 6.3 Robustness vs. α — the sharp prediction. 6.4 SAGE vs. isotropic UIPE. 6.5 ERUF replication with variance.

**7. Ablations.** α grid; $\alpha_\parallel/\alpha_\perp$; subspace rank; layer choice; loss terms; 4-bit vs. fp16; full-FT vs. LoRA delta.

**8. Analysis and Discussion.** Why isotropy causes the trade-off; what it implies for benchmark design; regime prediction from pre-unlearning separability; per-example depth variance.

**9. Limitations.** No irreversibility guarantee (follow ERUF's honest framing); 7B–8B only; English only; UDS depends on a retain oracle, available for TOFU but approximated elsewhere; probe sets are partly synthetic.

**10. Ethics.** WMDP hazardous-knowledge handling; no real PII; dual-use discussion citing Barez et al.

**11. Conclusion + Future Work.** Certified erasure; distillation-based robustness (UNDO); reasoning-model CoT leakage; sequential unlearning.

### 7.4 Target venues

| Tier | Venue | Deadline window | Fit |
|---|---|---|---|
| Main | **EMNLP 2027** / **ACL 2027** | ~Jan–Feb 2027 | Ideal — the whole conversation lives here |
| Main | **NAACL 2027** | ~Oct 2026 | Tight but possible if M5 lands by week 9 |
| Findings | ACL/EMNLP Findings | same | Strong fallback |
| Workshop | **NeurIPS SoLaR**, **ICLR Trustworthy/Safety** workshops | rolling | Good early venue for the D1 result alone; Łucki's paper won best-paper here |
| Journal | *TMLR* | rolling | No deadline pressure; Łucki's landed here |

**Recommended path:** workshop submission of D1 alone by ~week 12 to get feedback and establish priority, then extend with SAGE for an ACL/EMNLP 2027 main-track submission.

---

## Immediate Next Actions (week 1)

1. Install and validate [OpenUnlearning](https://github.com/locuslab/open-unlearning); reproduce one TOFU forget10 number end-to-end.
2. **Decide D8** (full-FT vs. LoRA extrapolation delta) and **D9** (model list vs. SAE availability). These gate everything.
3. Read the Tier-0 list in [`literature/README.md`](literature/README.md).
4. Fix citations D1–D14 in the proposal — especially rewriting §4.5 from the actual Adaptive RMU paper.
5. Pull the UDS reference implementation and confirm it runs on a TOFU oracle pair.
6. Draft the one-paragraph novelty statement (§5.3) and circulate it to the mentor before building anything.
