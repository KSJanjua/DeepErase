# DeepErase — Domain Expertise & Literature Review Report

**Subject:** *DeepErase: Selective Knowledge Erasure in Large Language Models — A Representation-Aware Unlearning Framework* (CPG 221, Thapar Institute, BE III COE/CSE)
**Prepared:** 2 August 2026
**Scope:** Phases 1–7 of the assigned research-orientation task. No code was written, modified, or refactored.

---

## 0. Sources Consumed

**Project documents (read in full):**
- `Capstone11.pdf` — 17-page Capstone Project Proposal
- `capstone_presentation.pptx` — 12-slide panel deck

**Research papers present in the repository (read in full):**

| File | Paper |
|---|---|
| `2023.emnlp-main.738.pdf` | Chen & Yang, *Unlearn What You Want to Forget: Efficient Unlearning for LLMs* (EUL), EMNLP 2023 |
| `2025.findings-emnlp.1374.pdf` | Wang et al., *UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets*, Findings of EMNLP 2025 |
| `2403.01267v2.pdf` | Pochinkov & Schoots, *Dissecting Language Models: Machine Unlearning via Selective Pruning*, arXiv 2024 |
| `28996-Article Text-33050-1-2-20240324.pdf` | Cha et al., *Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers*, AAAI 2024 |

**Papers cited in the proposal but absent from the repository (retrieved and read):** ELM, PISCES, KIF/ERUF, Adaptive RMU.

**Independently discovered literature:** ~60 additional works (Sections 3–5).

> **Domain note.** The task brief listed CVPR / ICCV / ECCV / WACV / BMVC as target venues. This project is **not a computer-vision project** — it is NLP / trustworthy-ML. The literature therefore lives at **ACL, EMNLP, NAACL, NeurIPS, ICLR, ICML, AAAI, COLM, TMLR, IEEE S&P and SaTML**. I searched those venues instead. (One repository paper, Cha et al. AAAI 2024, *is* a vision paper — see §7.3 on why its relevance is limited.)

---

## 1. Executive Summary

### 1.1 Objective of the project

Build and evaluate an **integrated, representation-aware LLM unlearning framework** that removes targeted knowledge from a deployed language model $P_\theta$ such that the resulting $P_{\theta^*}$ behaves as if that knowledge had never been trained on — while (a) genuinely attenuating the *internal representation* of the knowledge rather than only refusing to emit it, (b) extending the forgetting effect to *logically related* knowledge that could be used to reconstruct the target, and (c) not damaging unrelated capabilities.

The framework is a **composition of three published ideas**, plus an evaluation protocol:

1. **UIPE** (Wang et al., 2025) — parameter extrapolation $\theta_{\text{uipe}} = \theta_{\text{un}} + \alpha \cdot v$, $v = \theta_{\text{un}} - \theta_{\text{ini}}$, to reach related knowledge with no extra data.
2. **KIF / ERUF** (Mahmood et al., 2026) — activation-signature mining (Cohen's *d*), gated Suppression Capsules, and distillation of the suppression behaviour into a LoRA adapter via the UPU loop.
3. **Dual-metric evaluation** — Subject Mention Rate (SMR) + EL10, classifying outcomes as **Type I** (true erasure), **Type II** (obfuscation), **Type III** (instability).

Baselines to be reproduced: ELM, PISCES, RMU, and GA-family methods (GA, GradDiff, KL-Min, NPO), on TOFU, WMDP and a 4-subject RWKU subset.

### 1.2 The research problem

Knowledge in an LLM is stored **distributively across billions of parameters**, not in addressable records. Post-hoc deletion is therefore an approximation problem, and the field's central, unresolved difficulty is that **the standard way of measuring success — model output — is exactly the thing that is easiest to fake**. A model can be trained to refuse, deflect, or emit noise while the underlying circuit remains intact and one gradient step from recovery. The proposal names this correctly: *obfuscation is not erasure*.

Three specific failure modes are targeted:

| Failure mode | Mechanism | Evidence in literature |
|---|---|---|
| **Latent persistence** | Output-level losses suppress surface behaviour; internal representations survive | Łucki et al. TMLR 2025; Deeb & Roger 2024; Lee et al. NeurIPS 2025 (UNDO); Hong et al. EMNLP 2025 |
| **Reconstruction via related knowledge** | Forget set is a subset of what entails the target; the rest can re-derive it | UIPE (Findings EMNLP 2025); Choi et al. 2024 (multi-hop) |
| **Specificity collapse** | Fine-tuning touches all parameters, so adjacent concepts degrade | PISCES (EMNLP 2025); Lynch et al. 2024; Xu et al. 2026 (over-erasure) |

### 1.3 Why it matters

- **Legal.** GDPR Art. 17 right to erasure, the EU AI Act, CCPA and analogous regimes create an obligation that retraining-from-scratch cannot economically satisfy for frontier models.
- **Safety.** WMDP-style dual-use hazardous knowledge (bio/cyber/chem) is a named policy concern; unlearning is one of the few mechanisms that could reduce capability rather than merely refuse it. For *open-weight* releases, refusal training is provably removable in a handful of fine-tuning steps, so weight-level removal is the only defence that survives release.
- **Scientific.** "Did the knowledge actually go away?" is currently unanswerable with the field's standard metrics. This is a genuine open problem, and building a principled verification protocol is a real contribution, arguably a larger one than the unlearning method itself.

---

## 2. Project Understanding

### 2.1 Problem formalisation

Given a model $P_\theta$ trained on $D = \{(x_i, y_i)\}_{i=1}^n$ and a forget set $D_f \subset D$, produce $P_{\theta^*}$ that is behaviourally and representationally indistinguishable from a model trained only on $D_r = D \setminus D_f$, evaluated along two axes:
- **Forget Quality (FQ)** — how completely the target is gone.
- **Model Utility (MU)** — how much general capability survives.

The project adds a third axis the standard formulation omits: **depth** — whether removal is at the representation level (Type I) or only the output level (Type II).

### 2.2 System architecture

The proposed system is a **four-stage pipeline** plus a shared evaluation harness. Below is the architecture as reconstructed from §6 of the proposal and slides 6–7, with the data flow made explicit.

```
                        ┌───────────────────────────────────────────┐
                        │  STAGE 0 — Base model + benchmark intake  │
                        │  Llama-3.1-8B-Instruct, Mistral-7B-v0.1,  │
                        │  Zephyr-7B-beta  (4-bit quantised)        │
                        │  TOFU {01,05,10} | WMDP-Bio/Cyber | RWKU  │
                        └────────────────────┬──────────────────────┘
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      │                                      │                                      │
┌─────▼──────────────────┐   ┌───────────────▼───────────────┐   ┌──────────────────▼─────────────┐
│ STAGE 1 — BASELINES    │   │ STAGE 2 — RELATED-KNOWLEDGE   │   │ STAGE 3 — REPRESENTATION-AWARE │
│  (Phase 1)             │   │  (Phase 2, UIPE)              │   │  (Phase 3, KIF/ERUF)           │
│                        │   │                               │   │                                │
│ • ELM   (LoRA, 3 loss) │   │ A. Synthetic k_i ↔ k'_i pairs │   │ A. Signature extraction        │
│ • PISCES (SAE edit)    │   │    via GPT-4o prompting       │   │    – forward hooks on MLP      │
│ • RMU   (act. steer)   │   │ B. Validate hypothesis:       │   │      gate/up/down tensors      │
│ • GA / GradDiff /      │   │    cos(∇θP(y|x), ∇θP(y'|x'))  │   │    – pos = on-topic prompts    │
│   KL-Min / NPO         │   │ C. Run GA-family unlearning   │   │      neg = controls + cross-   │
│                        │   │    → select θ_un by FQ/MU     │   │        subject queries         │
│ Output: reproduced     │   │ D. v = θ_un − θ_ini           │   │    – d = mean(S⁺) − mean(S⁻)   │
│ TOFU-f10 + WMDP tables │   │ E. θ_uipe = θ_un + α·v        │   │    – validate w/ Cohen's d     │
│ + per-loss ablations   │   │ F. Ablate α ∈ [0,1]           │   │      per layer                 │
└─────┬──────────────────┘   └───────────────┬───────────────┘   │ B. Suppression Capsules        │
      │                                      │                   │    h = h∥ + h⊥ w.r.t. unit v   │
      │                                      │                   │    α_eff = α·σ(k(z − τ))       │
      │                                      │                   │    scale h∥ only; keep h⊥      │
      │                                      │                   │ C. UPU Loop → LoRA distillation│
      │                                      │                   │    L = L_DPO + λ_UL·L_UL       │
      │                                      │                   │        + λ_NTUL·L_NTUL         │
      │                                      │                   │        + λ_KL·L_KL             │
      │                                      │                   │        + λ_EWC·L_EWC           │
      │                                      │                   │ D. Merge adapter, drop hooks   │
      │                                      │                   └──────────────┬─────────────────┘
      │                                      │                                  │
      └──────────────────────────────────────┴──────────────────────────────────┘
                                             │
                        ┌────────────────────▼──────────────────────┐
                        │ STAGE 4 — INTEGRATED EVAL  (Phase 4)      │
                        │ Forget: SMR ↓, EL10 ↓, ROUGE-L ↓,         │
                        │         TOFU FQ ↑ (KS p-value)            │
                        │ Utility: TOFU MU ↑, MMLU 0-shot ↑         │
                        │ Retention: ARC-C, HellaSwag, WinoGrande,  │
                        │            TruthfulQA                     │
                        │ Classify: Type I / II / III               │
                        │ Cross-benchmark: RWKU 4-subject           │
                        └───────────────────────────────────────────┘
```

### 2.3 Module-by-module description

**M1 — Baseline reproduction harness (Phase 1).**
Purpose: establish that the team can reproduce published numbers before claiming improvements on them. Reproduces ELM (LoRA on early-mid layers with $\mathcal{L}_{\text{erase}} + \mathcal{L}_{\text{retain}} + \mathcal{L}_{\text{fluency}}$), PISCES (SAE-based in-parameter feature ablation), and RMU (activation steering toward a scaled random direction). Deliverable is a comparative table plus per-loss-component ablations.

**M2 — Dual-metric evaluator (Phase 1, used by all later phases).**
- **SMR (Subject Mention Rate)** — % of generated continuations that contain the target subject name. Surface leakage. ↓
- **EL10** — ratio of subject-associated token probability mass over the first 10 decoding steps, normalised against the base model. ↓ (<1 = attenuated)
- **Taxonomy** with $\epsilon = 5\%$:
  - **Type I** — SMR ≤ ε **and** EL10 < 1 → representation-level attenuation
  - **Type II** — SMR ≤ ε **and** EL10 > 1 → obfuscation (surface clean, latent trace intact or *amplified*)
  - **Type III** — SMR > ε → unstable / leaking

**M3 — Related-knowledge dataset builder (Phase 2).**
For each forget item $k_i$, prompt GPT-4o for a logically entailed counterpart $k'_i$. UIPE's own construction: 12 fictional individuals × 10 attributes × 2 QA pairs (a `k1` direct fact, a `k2` inferable consequence), plus a disjoint *irrelevant* control set. Purpose: empirically confirm $R_\theta(k_i, k'_i) = \cos(\nabla_\theta P_\theta(y_i|x_i), \nabla_\theta P_\theta(y'_i|x'_i))$ is high for related pairs and low for irrelevant ones.

**M4 — UIPE post-processor (Phase 2).**
Pure post-processing, no training. Run GA-family unlearning for $T$ epochs, select $\theta_{\text{un}}$ balancing FQ/MU, compute $v = \theta_{\text{un}} - \theta_{\text{ini}}$, emit $\theta_{\text{uipe}} = \theta_{\text{un}} + \alpha v$. The mechanism: $v$ has a projection $v' = |v| \cdot R_\theta(k_i,k'_i) \cdot \hat{v}'_o$ onto the related-knowledge gradient direction; scaling $v$ scales $v'$, and scales it *more* for more-related knowledge. Complexity is linear in parameter count.

**M5 — Activation-signature extractor (Phase 3).**
Forward hooks harvest MLP gate/up/down activations; token-averaged and standardised. Signature direction $\mathbf{d} = \text{mean}(S_{\text{pos}}) - \text{mean}(S_{\text{neg}})$, with positives = on-topic subject prompts and negatives = real controls + cross-subject entity queries (ERUF caps synthetic backfill at ≤10%). Layer selection by Cohen's *d*; mid-to-late layers show strongest separability.

**M6 — Suppression Capsules (Phase 3).**
Decompose hidden state $h$ into $h_\parallel$ and $h_\perp$ relative to the unit signature $v$; apply $\alpha_{\text{eff}} = \alpha \cdot \sigma(k(z - \tau))$ with $\alpha$ initialised to $-1$, $z$ the standardised projection, $\tau$ a gate threshold, $k$ a gain. Only $h_\parallel$ is scaled; $h_\perp$ is preserved — this is the specificity mechanism. Implemented as removable forward hooks.

**M7 — UPU Loop / LoRA distillation (Phase 3).**
When a capsule fires, log $(x, y^+, y^-)$ = (prompt, suppressed response, factual response). Train LoRA $\phi$ on:
$$\mathcal{L} = \mathcal{L}_{\text{DPO}} + \lambda_{\text{UL}}\mathcal{L}_{\text{UL}} + \lambda_{\text{NTUL}}\mathcal{L}_{\text{NTUL}} + \lambda_{\text{KL}}\mathcal{L}_{\text{KL}} + \lambda_{\text{EWC}}\mathcal{L}_{\text{EWC}}$$
- $\mathcal{L}_{\text{DPO}}$ — prefer suppressed $y^+$ over factual $y^-$, anchored to a reference model
- $\mathcal{L}_{\text{UL}}$ — unlikelihood on the factual continuation
- $\mathcal{L}_{\text{NTUL}}$ — name-token unlikelihood: $-\frac{1}{T}\sum_t \log(1 - \sum_{v \in V_{\text{name}}} p_\theta(v \mid x, y^+_{<t}))$; kills soft leakage of the subject name
- $\mathcal{L}_{\text{KL}}$ — divergence from reference, for stability
- $\mathcal{L}_{\text{EWC}}$ — Fisher-weighted protection of benign-task parameters
After training the capsules are removed; the adapter carries the behaviour.

**M8 — Integrated evaluation (Phase 4).** As diagrammed above.

### 2.4 Data flow, inputs, outputs

| Stage | Input | Output |
|---|---|---|
| Signature extraction | Entity prompts grounded in verifiable facts + negative pools; base model weights | Per-layer signature vectors $\mathbf{d}^{(l)}$, Cohen's *d* profiles, chosen layer set |
| Capsule inference | Prompt + signature vectors | Suppressed generations; logged preference triples |
| UPU distillation | Preference triples + retain corpus + Fisher matrix | LoRA adapter $\phi$ |
| UIPE | $\theta_{\text{ini}}$, $\theta_{\text{un}}$, scalar $\alpha$ | $\theta_{\text{uipe}}$ |
| Evaluation | Unlearned model + benchmark probes | SMR, EL10, FQ, MU, ROUGE-L, MMLU, ARC/HellaSwag/WinoGrande/TruthfulQA, Type label |

### 2.5 Training / inference / evaluation pipelines

- **Training** happens only in M7 (LoRA + composite loss) and in the GA-family runs feeding M4. Capsules are *not* trained — they are analytically constructed from the signature.
- **Inference** post-merge is a plain forward pass; the capsule hooks exist only during the distillation phase. This is the parameter-efficiency and deployability argument.
- **Evaluation** is generation-based (SMR, ROUGE-L), distribution-based (EL10, TOFU truth ratio / KS test), and MCQ-based (WMDP, MMLU, ARC).

### 2.6 Assumptions and design decisions (from slide 5)

1. Open-weight 7B–8B models (Llama-3.1-8B-Instruct, Mistral-7B-v0.1, Zephyr-7B-beta) are the primary targets.
2. **Knowledge is partially localised in MLP layers**, consistent with causal-tracing evidence.
3. **Related knowledge exhibits higher gradient cosine similarity** to the forget target.
4. TOFU / WMDP-Bio / WMDP-Cyber are representative of practical unlearning.
5. **4-bit quantised results are qualitatively consistent with full precision.**
6. English only; cross-lingual deferred.
7. Compute: A100/A6000, 40 GB VRAM; ~500 GB storage.

Assumptions 2, 3 and 5 are load-bearing and each is contested in the current literature — see §8.

### 2.7 Current implementation status

**No implementation exists.** The repository contains only the proposal, the deck, and four PDFs. Everything in §2.2 is planned work per the 16-week plan (weeks 1–2 literature/setup → week 16 panel). The proposal is dated March 2026; today is August 2026, so either the plan has slipped or an implementation lives outside this folder.

### 2.8 Documentation inconsistencies and ambiguities

These are stated explicitly rather than guessed around, as instructed.

| # | Issue | Detail |
|---|---|---|
| **D1** | **Reference [4] is misattributed.** | The proposal credits *On Effects of Steering Latent Representation for LLM Unlearning* to "Qin, H., Zhang, B., & Ji, H. (2024)". The actual authors are **Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue**, and it was published at **AAAI 2025** (Vol. 39, No. 22, pp. 23733–23742), not left as an arXiv preprint. "Qin, Zhang, Han, Yu, Li, Ji (2024)" is a *different* paper — *Why Does New Knowledge Create Messy Ripple Effects in LLMs?*, EMNLP 2024 — which is the work UIPE actually cites for the similar-parameter-distribution claim. Two references have been merged into one. |
| **D2** | **Reference [4] is mischaracterised.** | §4.5 describes it as showing that "steering latent representations can induce behavioural changes without modifying weights, and conversely that weight-level changes may not eliminate latent knowledge traces… motivating the dual-metric approach adopted in KIF." The paper does none of that. It is a **theoretical analysis of why RMU works** — proving unlearned logits become normally distributed with inflated variance, showing RMU fails in middle/late layers because representation norms exceed the fixed coefficient *c*, and proposing **Adaptive RMU**, which replaces $c\mathbf{u}$ with $\beta\|h^{(l)}_{\text{frozen}}(x_F)\|_2 \cdot \mathbf{u}$. It never proposes SMR/EL10 and does not motivate them. The literature-survey entry needs rewriting from the actual paper. |
| **D3** | **KIF has been renamed.** | The proposal cites arXiv:2601.10566**v4**, *"…From Suppression to **Knowledge**-Signature Erasure"*, framework **KIF (Knowledge Immunization Framework)**. The current version (v5) is titled *"…From Suppression to **Entity**-Signature Erasure"* and the framework is now **ERUF (Entity Representation Unlearning Framework)**. Reports and slides will read as stale. Also note the scope narrowing: "knowledge" → "entity". |
| **D4** | **Repository / bibliography mismatch.** | Four of the five cited works are **not** in the repository (only UIPE is). Conversely, three of the four papers in the repository (EUL, Selective Pruning, Instance-Wise Unlearning) are **never cited** in the proposal. The collected reading and the written survey are effectively two disjoint sets. |
| **D5** | **One repository paper is out of domain.** | Cha et al. (AAAI 2024) is **image-classification** unlearning (CIFAR-10/100, ImageNet-1K, adversarial-example augmentation + MAS weight importance). It shares vocabulary with the project but not the setting; there is no generative model, no forget corpus of text, and its metrics (MIA on class labels) do not transfer. |
| **D6** | **RMU / WMDP are used but never cited.** | §3 and Phase 1 both name RMU (and §3 names RepNoise) as baselines, and WMDP is a core benchmark, yet **Li et al. (ICML 2024)**, which introduces both, does not appear in the reference list. Same for RepNoise (Rosati et al., 2024). |
| **D7** | **The "peer-reviewed only" claim is false.** | The References preamble states sources are "exclusively from peer-reviewed journal and conference papers." Reference [3] (KIF/ERUF) is an un-refereed arXiv preprint, and [4] is cited as an arXiv preprint. |
| **D8** | **Compute plan vs. method requirements.** | Slide 7 specifies 40 GB A100/A6000 with **4-bit quantised inference** as primary. UIPE as published uses **full-parameter** gradient-ascent fine-tuning on **2 × A800 80 GB** and requires a dense delta $v = \theta_{\text{un}} - \theta_{\text{ini}}$ over all weights. Extrapolating a *4-bit quantised* or *LoRA-only* delta is **not** what the paper validates, and the proposal does not say which will be done. This is the single largest unresolved feasibility question. |
| **D9** | **PISCES cannot be run on the listed models.** | PISCES requires a pretrained SAE **on every MLP layer output**. Those exist for **Gemma-2-2B (GemmaScope)** and **Llama-3.1-8B (Llama Scope)** — which is exactly why the PISCES authors chose those two models. There are no comparable per-MLP-layer SAE suites for **Mistral-7B-v0.1** or **Zephyr-7B-beta**, both listed as project models. Either the model list or the PISCES reproduction must change. |
| **D10** | **Benchmark/method mismatch in Phase 1.** | Phase 1 proposes reproducing "ELM, PISCES and RMU on **TOFU-forget10 and WMDP**." ELM and PISCES were never evaluated on TOFU by their authors (ELM: WMDP + Harry Potter; PISCES: ConceptVectors-derived concepts), and RMU is a concept/hazard-domain method that performs poorly on TOFU-style fictitious-author forgetting. Conversely GA/NPO baselines are TOFU-native. Expect substantial adaptation effort not budgeted in the 16-week plan. |
| **D11** | **SMR and EL10 are undefined for WMDP.** | Both metrics are **subject-centric** — SMR counts occurrences of a target *name*; EL10 measures *subject-associated* token mass. WMDP is 4-way multiple-choice on hazardous *topics* with no single subject string. Objective O2 claims dual-metric evaluation "across model families and scales" on TOFU **and** WMDP, but the protocol for WMDP is unspecified. |
| **D12** | **EL10 is not actually a latent metric.** | The proposal repeatedly frames EL10 as measuring "latent knowledge persistence" (§4.3, O2). EL10 is computed from **output token probabilities** over the first 10 decoding steps — it is a softer surface metric, not an internal one. ERUF itself recognises this and supplements EL10 with genuinely internal diagnostics (hidden-state E₃₀ extraction mass, Selective Representation Shift ratio). A framework whose central claim is "representation-level, not output-level" should not rest its verification on an output-distribution statistic. See §8.1. |
| **D13** | **Roll number inconsistency.** | Pranav Goyal is 102303829 in the PDF (p. 13) but 102303146 on slide 10. |
| **D14** | **Novelty is never stated.** | O3 is UIPE as published; O4 is KIF as published. The contribution is the *integration* plus the dual-metric study, but §7.2 only lists anticipated improvements. Panels will ask "what is new?" — the answer needs to be written down. Note also that composing the two is non-trivial: UIPE extrapolates a **full-parameter** delta, KIF produces a **LoRA** delta. Which delta gets extrapolated is undecided. |

---

## 3. Literature Review

### 3.1 Foundational papers (pre-LLM and early LLM unlearning)

| Work | Venue / Year | Contribution |
|---|---|---|
| Cao & Yang, *Towards Making Systems Forget with Machine Unlearning* | IEEE S&P 2015 | Coined machine unlearning; statistical-query decomposition | [link](https://ieeexplore.ieee.org/document/7163042) |
| Bourtoule et al., *Machine Unlearning* (SISA) | IEEE S&P 2021 | Sharded/isolated/sliced/aggregated retraining — the canonical *exact* unlearning baseline | [arXiv:1912.03817](https://arxiv.org/abs/1912.03817) |
| Golatkar, Achille & Soatto, *Eternal Sunshine of the Spotless Net* | CVPR 2020 | Fisher-information "scrubbing"; first influential *approximate* unlearning in deep nets | [arXiv:1911.04933](https://arxiv.org/abs/1911.04933) |
| Kirkpatrick et al., *Overcoming Catastrophic Forgetting* (EWC) | PNAS 2017 | Fisher-weighted parameter anchoring — **used directly as $\mathcal{L}_{\text{EWC}}$ in KIF's UPU loop** | [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) |
| Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* | EMNLP 2021 | The reason every method in this project targets MLPs | [arXiv:2012.14913](https://arxiv.org/abs/2012.14913) |
| Meng et al., *Locating and Editing Factual Associations in GPT* (ROME) | NeurIPS 2022 | Causal tracing; localisation of facts to mid-layer MLPs | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) |
| Meng et al., *Mass-Editing Memory in a Transformer* (MEMIT) | ICLR 2023 | Scaled editing to thousands of facts; **a PISCES baseline** | [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) |
| Ilharco et al., *Editing Models with Task Arithmetic* | ICLR 2023 | Task vectors $\tau = \theta_{\text{ft}} - \theta_{\text{pre}}$; **negation** removes behaviour. **UIPE is task-vector arithmetic under another name** | [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) |
| Hu et al., *LoRA* | ICLR 2022 | The adaptation substrate for ELM and KIF | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| Rafailov et al., *Direct Preference Optimization* | NeurIPS 2023 | $\mathcal{L}_{\text{DPO}}$ in KIF's UPU loop; ancestor of NPO | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| Jang et al., *Knowledge Unlearning for Mitigating Privacy Risks in LMs* | ACL 2023 | Established gradient ascent as *the* LLM unlearning primitive | [arXiv:2210.01504](https://arxiv.org/abs/2210.01504) |
| Eldan & Russinovich, *Who's Harry Potter?* | arXiv 2023 | First high-profile concept unlearning; created the Harry Potter task | [arXiv:2310.02238](https://arxiv.org/abs/2310.02238) |
| Bricken et al. / Huben et al., *Towards Monosemanticity* / *Sparse Autoencoders Find Highly Interpretable Features* | Anthropic 2023 / ICLR 2024 | SAEs — PISCES's disentangler | [ICLR](https://openreview.net/forum?id=F76bwRSLeK) |

### 3.2 Most relevant papers (the project's direct foundations)

**ELM — Gandikota, Feucht, Marks & Bau, NeurIPS 2025.** [arXiv:2410.02760](https://arxiv.org/abs/2410.02760) · [project](https://elm.baulab.info/) · [code](https://github.com/rohitgandikota/erasing-llm)
Reframes erasure as **self-classification**: a model conditioned on an expert prefix $c^-$ vs. a novice prefix $c^+$ defines a target "erased" distribution $P^{\text{erased}}_\theta \propto P_\theta(X)\left(P_\theta(X|c^+)/P_\theta(X|c^-)\right)^\eta$ with $\eta = 500$. Three losses — $\mathcal{L}_{\text{erase}}$ (match the erased distribution), $\mathcal{L}_{\text{retain}}$ (match the original on retain data), $\mathcal{L}_{\text{fluency}}$ (stay coherent *while* erased, loss masked to generated tokens only). LoRA rank 4 on layers 4–7. Introduces the **innocence / seamlessness / specificity** triad. On Zephyr-7B: Bio 64.4→29.7, Cyber 44.3→27.2, MMLU 58.5→56.6, MT-Bench 7.3→7.1, R-PPL 6.0→10.9 (vs. RMU's 24.8 — ELM's headline win is *fluency*). Ablations are decisive: removing $\mathcal{L}_{\text{retain}}$ collapses MMLU to 23.6; removing $\mathcal{L}_{\text{fluency}}$ pushes R-PPL to 29.8. Resists GCG for 5000 iterations. **Weakness:** admits specificity degradation on adjacent MMLU categories (biology, chemistry).

**PISCES — Gur-Arieh, Suslik, Hong, Barez & Geva, EMNLP 2025** (main, pp. 18986–19006). [arXiv:2505.22586](https://arxiv.org/abs/2505.22586) · [code](https://github.com/yoavgur/PISCES)
Edits **parameters, not activations**. An MLP layer is $\sum_i a_i v_i$ where $v_i$ are rows of $W_{\text{out}}$; because MLP vectors are polysemantic, an SAE trained on MLP *outputs* is applied to the MLP *vectors* themselves (justified by the linearity of Eq. 1). Concept features $F_c$ are found by **vocabulary projection** $u_f = Ew_f$, selecting features whose top/bottom tokens are concept-dense, then manually filtered. Vectors to edit: $V_c = \bigcup_{f}\{v_i \mid m^i_f \ge \tau\hat{m}_f\}$ — deliberately *not* all vectors, because SAE reconstruction error accumulates and hurts specificity. Ablation sets $\bar{m}^i_f = -\mu\hat{m}_f$ (negative, not zero), then reconstructs $\bar v_i = D^{-1}(\bar m^i)$ in place. Results (normalised to base = 100%): Llama-3.1-8B-it — accuracy 7.7%, similar-domain 87.6%, MMLU 99.4%, AlpacaEval 99.3%, **relearning accuracy 65.4% vs. ELM 103.1% and RMU 93.2%**. That relearning column is the important one: every fine-tuning-based method recovers essentially *all* erased knowledge; PISCES recovers only two-thirds. Robustness is where in-parameter editing earns its keep. **Limitations:** MLP-only (attention also stores knowledge); bounded by SAE coverage; VocabProj is unreliable in early layers.

**UIPE — Wang, Zhang, Ye, Ren, Ren & Chen, Findings of EMNLP 2025** (pp. 25212–25227). [ACL](https://aclanthology.org/2025.findings-emnlp.1374/)
Two-part argument. *Empirically:* a model fine-tuned on forget + related sets and then GA-unlearned on the forget set only ($P_{\theta_1}$) shows **worse** forget quality *and* worse utility than one that never saw related knowledge ($P_{\theta_2}$) — related knowledge lets the model re-derive the target. *Theoretically:* GA's update $v \propto \nabla_\theta P_{\theta}(y_i|x_i)$ has a projection $v' = |v| R_\theta(k_i,k'_i)\hat v'_o$ onto the related gradient. But once $k_i$ is fully forgotten, $\nabla_\theta P_\theta(y_i|x_i)$ no longer encodes $k_i$'s storage, $R_\theta$ becomes meaningless, and the projection channel closes — **so GA cannot reach related knowledge no matter how large the learning rate**. The fix is linear extrapolation $\theta_{\text{uipe}} = \theta_{\text{ini}} + (1+\alpha)v$, which scales $v'$ proportionally *and* preferentially for more-correlated knowledge. Results: on WMDP/Zephyr-7B, GA 0.3302 → GA+UIPE **0.1768** average, MMLU 0.5449 → 0.5339 (~1 pt); on TOFU-Forget01, KL-Min+UIPE reaches near-ideal FQ ≈ 1.0. α behaves differently per baseline: monotone for GradDiff and KL-Min, flat for NPO, **inverted-U for GA** (over-forgetting past the peak). Downstream (GSM8K/ARC) impact is negligible. **Limitations (authors' own):** α must be hand-tuned per method; only 7B tested.

**KIF / ERUF — Mahmood, Bhuiyan, Zaman, Khondaker, Sakib, Wadith, Tasnim & Sadeque, arXiv 2026.** [arXiv:2601.10566](https://arxiv.org/abs/2601.10566)
The project's architectural template. Three stages as described in §2.3 (M5–M7). Headline results: **TOFU forget10 / LLaMA-2-7B-Chat — FQ 0.99 (oracle 1.00), MU 0.62 (oracle 0.62)**, against SimNPO 0.45 and NPO best 0.55 utility. Real-world entity set: Mistral-7B SMR 0.00%, EL10 0.020, utility drift +0.50%. Adversarial recovery on Llama-3.1-8B: 63.89% → 20.15% (−68.4%); alias hits −70.2%; keyword hits −72.0%; target mass 0.0440 → 0.0136. Hidden-state diagnostics: EL10 mass −95.6%, E₃₀ extraction mass −30.8%, **Selective Representation Shift 5.90×** (forget-set drift 5.9× benign drift). Sequential unlearning holds SMR at 0.00% for previously forgotten targets. Zero-shot: ARC-C −0.26, HellaSwag +0.16, TruthfulQA-MC2 +0.51, but **BoolQ −4.47 and SocialIQA −3.89** — localised collateral damage.
**Limitations (authors' own, and important for this project):** *no formal guarantee of irrecoverability* — "operational evidence of representation-level attenuation, not a formal guarantee"; signature mining is a plain standardised mean-difference, Cohen's *d* is only a relative diagnostic; 4-bit quantisation caps them at 32B on one A6000; **TOFU is a single run**; RWKU is only 4 shared subjects; **locality fails under same-domain stress** (related non-forgotten music entities get suppressed); and **Type II obfuscation still occurs on Llama-3B and on reasoning-prior 8B models, with Type III instability on DeepSeek-3B**. The method is *not* universally Type I.

**Adaptive RMU — Huu-Tien, Pham, Thanh-Tung & Inoue, AAAI 2025** (pp. 23733–23742). [arXiv:2408.06223](https://arxiv.org/abs/2408.06223) · [code](https://github.com/RebelsNLU-jaist/llm-unlearning)
Explains *why* RMU works: steering forget representations to a random direction inflates logit variance (Proposition 1: logits become $\mathcal{N}(Wg^{(L)}(z), \eta W \nabla g^\top \nabla g W^\top)$), destroying token confidence. RMU fails at layers 11–31 because $\|h^{(l)}\|$ exceeds the fixed coefficient $c$, so the loss never converges. Fix: make the coefficient adaptive, $\beta\|h^{(l)}_{\text{frozen}}(x_F)\|_2 \cdot \mathbf{u}$, cacheable on the first iteration. Zephyr-7B: Bio 28.8 → **23.7**, Cyber 28.8 → **26.5**, MMLU 56.8 → 55.0. Also explains GCG's failure against RMU: the steered representation is input-independent, so the attacker gets uninformative gradients — an *artefact* of the defence, not evidence of deep removal.

### 3.3 Papers in the repository (relevance triage)

**EUL — Chen & Yang, EMNLP 2023.** [ACL](https://aclanthology.org/2023.emnlp-main.738/) · [code](https://github.com/SALT-NLP/Efficient_Unlearning/)
Lightweight **unlearning layers** inserted after FFNs, trained with a selective teacher–student objective: $\mathcal{L}_{\text{KL}} = \alpha\sum \text{KL}(F(x_r)\|F(f(x_r))) - \text{KL}(F(x_f)\|F(f(x_f)))$, plus a task loss on retain data and a **negated LM loss** on forget data. Adds a **fusion mechanism** merging independently-trained unlearning layers via closed-form least squares $W_m = (\sum X_i^{f\top}X_i^f)^{-1}\sum(X_i^{f\top}X_i^f W_i)$ — no retraining, and only the inner-product matrices need storing (a privacy win). Experiments on **T5-base/T5-3B**, IMDB and SAMSum. Best forget-set accuracy 57.2% vs. re-train 90.2% at 1/3 the time.
*Relevance:* **historical/architectural, not competitive.** T5 encoder-decoder, classification/summarisation, ~2023-era baselines (SISA, MEND, reverse-gradient). Its enduring contributions to this project are (i) the adapter-as-unlearning-module pattern that ELM and KIF inherit, (ii) the **MLM-loss-on-forget-data probe**, an early "is it still extractable?" metric that anticipates the project's Type I/II distinction, and (iii) **fusion for sequential deletion requests** — the project's plan has no sequential-unlearning story, and ERUF's Table 8 shows sequential unlearning is now an expected evaluation.

**Selective Pruning — Pochinkov & Schoots, arXiv 2024.** [arXiv:2403.01267](https://arxiv.org/abs/2403.01267) · [code](https://github.com/nickypro/selective-pruning)
Structured pruning by a relative-importance score $\text{Score}(n) = I(D_{\text{forget}},n) / (I(D_{\text{retain}},n)+\epsilon)$, with importance measured as activation frequency, mean-|activation|, RMS, or std. Iterative, 2% of neurons per step. Findings that matter here: **feed-forward neurons are far more task-specialised than attention neurons** (OPT-1.3B: 59.6 vs. 28.4 max accuracy-drop differential); **dropout during training increases neuron specialisation** (OPT/Galactica, trained with FF dropout, separate much better than Pythia, which has none); $I_{\text{abs}}$ is the most reliable metric. Runs on a single RTX 4090. Reduces Llama-2-7B toxicity to 0.0% with MMLU 33.6 → 33.0.
*Relevance:* **moderate and underrated by the proposal.** It is the cheapest possible baseline for "does localisation buy you anything?", it independently validates assumption 2 (MLP locality), and its dropout finding predicts *which* model families will yield clean signatures — directly useful for interpreting Cohen's *d* variation across Llama vs. Mistral vs. Qwen, which is exactly where ERUF's Type II failures appear.

**Instance-Wise Unlearning — Cha, Cho, Hwang, Lee, Moon & Lee, AAAI 2024** (pp. 11186–11194).
Instance-wise (not class-wise) unlearning of image classifiers, with access to *only* the pretrained model and $D_f$. Two regularisers: (1) **adversarial-example augmentation** — L2-PGD examples from the forget points are retrained on to preserve the decision boundary at the representation level; (2) **MAS weight importance** $\Omega_i = \frac{1}{N}\sum_n \partial\|g_\theta(x^{(n)})\|_2^2/\partial\theta_i$ to steer updates toward parameters responsible for the original prediction. CIFAR-10/100, ImageNet-1K; MIA-based leakage evaluation.
*Relevance:* **low.** Different modality, different task, no generative component, non-transferable metrics. Its one transferable idea is *conceptual*: preserving representation-level structure while forcing behavioural change — the same intuition as KIF's $h_\perp$ preservation. It should be cited, if at all, as prior art for the "preserve the orthogonal complement" idea, not as a baseline.

### 3.4 State-of-the-art and closely related methods (2024–2026)

**Optimisation objectives**
- **NPO** — Zhang, Lin, Bai & Mei, COLM 2024. [arXiv:2404.05868](https://arxiv.org/abs/2404.05868). DPO with only the negative term; progression to catastrophic collapse is *exponentially slower* than GA. First method to handle 50% forget sets. The default strong baseline.
- **SimNPO** — Fan et al., *Simplicity Prevails*, 2024. [arXiv:2410.07163](https://arxiv.org/abs/2410.07163). Removes NPO's reference-model bias; a KIF baseline (FQ 0.45).
- **ULD** — Ji et al., NeurIPS 2024. Reverses forget/retain objectives via logit difference with a small auxiliary model.
- **SOUL** — Jia et al., EMNLP 2024. Second-order (Sophia-style) optimisation for unlearning.
- **WAGLE** — Jia et al., NeurIPS 2024. Weight attribution for modular unlearning.

**Representation- and parameter-level methods** (the project's family)
- **RMU** — Li et al., ICML 2024 (with WMDP). [arXiv:2403.03218](https://arxiv.org/abs/2403.03218). Steer forget activations to a scaled random direction at one early layer; the WMDP reference method.
- **Adaptive RMU** — AAAI 2025 (above).
- **LUNAR** — Shen et al., NeurIPS 2025. [arXiv:2502.07218](https://arxiv.org/abs/2502.07218). Linear-representation-hypothesis-grounded redirection of forget representations into an "I can't answer" region; **edits only a single down-projection matrix** (20× efficiency), 2.9–11.7× deviation-score gains, robust to white-box attack, handles sequential requests. A KIF baseline and arguably the closest published relative of the Suppression Capsule.
- **REGLU — Representation-Guided Parameter-Efficient LLM Unlearning** — Xiao, Mo, Chen, Yang, Zhao, Yang & Chen, 2026. [arXiv:2604.17396](https://arxiv.org/abs/2604.17396). Representation-guided **LoRA initialisation** into the optimal forgetting subspace, plus a regulariser constraining LoRA outputs to the **orthogonal complement of the retain-set representation subspace**. TOFU + WMDP, reported to beat SOTA. This is the *closest published competitor* to the project's Phase 3 — same substrate (LoRA), same intuition (act in the subspace orthogonal to retain), different mechanism (geometric constraint vs. gated capsule + distillation). It is also listed among ERUF's baselines. **Must be read.**
- **REVS** — Ashuach et al., Findings of ACL 2025. Rank editing in vocabulary space for sensitive-information unlearning.
- **AlphaEdit** — Fang et al., ICLR 2025. Null-space-constrained knowledge editing; a PISCES baseline.
- **SAE-based unlearning** — Farrell et al. 2024 (feature clamping on WMDP-Bio); **Muhamed et al. 2025, Dynamic SAE Guardrails (DSG)**; **CRISP** ([arXiv:2508.13650](https://arxiv.org/abs/2508.13650), persistent SAE unlearning via PEFT); *Model Unlearning via SAE Subspace Guided Projections*, EMNLP 2025. PISCES's own related-work section is the best map of this cluster and explains why it edits parameters rather than steering activations: steering degrades coherence, costs a lot at inference, and **does not survive a white-box threat model**.

**Robustness-first methods**
- **TAR — Tamper-Resistant Safeguards for Open-Weight LLMs** — Tamirisa et al., ICLR 2025. [arXiv:2408.00761](https://arxiv.org/abs/2408.00761). Meta-learned resistance to hundreds of fine-tuning steps across 28 attacker strategies.
- **UNDO — Distillation Robustifies Unlearning** — Lee, Foote et al., **NeurIPS 2025 spotlight**. [arXiv:2506.06278](https://arxiv.org/abs/2506.06278). The sharpest negative result in the field: **even a model in perfect behavioural agreement with an unlearning oracle retains latent capability recoverable by fine-tuning.** Distilling the unlearned model into a *noised* copy of itself (Unlearn → Noise → Distill) leaves the latent capability behind and establishes a new compute/robustness Pareto frontier. Any project claiming "durable erasure" has to answer this paper.
- **Filtering pretraining data builds tamper-resistant safeguards** — 2025. [arXiv:2508.06601](https://arxiv.org/abs/2508.06601). Prevention beats post-hoc removal.

**Related-knowledge / multi-hop**
- **Munch / Breaking Chains** — Choi, Park, Lee & Choo, 2024. [arXiv:2410.13274](https://arxiv.org/abs/2410.13274). Multi-hop facts survive when an intermediate hop is unlearned; uncertainty-based question decomposition as the fix. **The nearest neighbour to the project's O3 and it is uncited.**
- **Qin, Zhang, Han, Yu, Li & Ji**, *Why Does New Knowledge Create Messy Ripple Effects in LLMs?*, EMNLP 2024. The actual source of the "related knowledge shares parameter-space structure" claim.
- **PrivUn** — 2026. [arXiv:2604.22076](https://arxiv.org/abs/2604.22076). Ripple effects and shallow forgetting in privacy unlearning.
- **ExPO — Weak-to-Strong Extrapolation Expedites Alignment** — Zheng et al., ICML 2024. [arXiv:2404.16792](https://arxiv.org/abs/2404.16792). $\theta + \alpha(\theta_{\text{aligned}} - \theta_{\text{sft}})$ for alignment. **Mathematically identical to UIPE in the opposite direction** — essential context nobody in the unlearning thread seems to cite.

**Inference-time / guardrail methods** (a family the proposal ignores)
- **ECO — Embedding-COrrupted prompts** — Liu et al., NeurIPS 2024. [arXiv:2406.07933](https://arxiv.org/abs/2406.07933) · [code](https://github.com/chrisliu298/llm-unlearn-eco). Classifier flags forget-prompts; corrupts their embeddings at inference. Near-zero side effects, no weight change. Included because it is the **strongest argument for why output metrics are inadequate** — ECO scores superbly on benchmarks while changing nothing in the weights. If the dual-metric protocol is any good, it should classify ECO as Type II.

### 3.5 Benchmarks, datasets and infrastructure

| Resource | Venue/Year | Content | Notes for this project |
|---|---|---|---|
| **TOFU** | COLM 2024, Maini et al. [arXiv:2401.06121](https://arxiv.org/abs/2401.06121) · [site](https://locuslab.github.io/tofu/) | 200 fictitious authors × 20 QA; forget01/05/10; FQ = KS-test *p*-value against a retain-only oracle; MU = aggregate of probability, ROUGE-L, truth ratio | **Known metric pathology:** a model emitting gibberish assigns similar low probabilities to correct and incorrect answers, matching the oracle's truth ratio and scoring *high* FQ. Read this before trusting FQ ≈ 0.99. |
| **WMDP** | ICML 2024, Li et al. [arXiv:2403.03218](https://arxiv.org/abs/2403.03218) | 3,668 MCQs, bio/cyber/chem; ships RMU | Utility measured via MMLU zero-shot. No subject strings → see D11. |
| **RWKU** | NeurIPS D&B 2024, Jin et al. [OpenReview](https://openreview.net/forum?id=wOmtZ5FgMH) · [HF](https://huggingface.co/datasets/jinzhuoran/RWKU) | 200 real-world targets, 13,131 probes (3,268 cloze, 2,879 QA, **6,984 adversarial**); **zero-shot** — no forget corpus given; 4 MIA methods + 9 adversarial probe types | Best fit for the project's entity-centric SMR/EL10. The project uses only a 4-subject subset (matching ERUF's constraint) — a full-benchmark run would be a genuine differentiator. |
| **MUSE** | ICLR 2025, Shi et al. [arXiv:2407.06460](https://arxiv.org/abs/2407.06460) · [site](https://muse-bench.github.io/) | Six axes: verbatim memorisation, knowledge memorisation, privacy leakage, utility, **scalability**, **sustainability** (sequential requests); 6.5M tokens of books + news | **Absent from the project plan.** It is the only benchmark that tests sequential and large-scale removal — precisely the axes a "durable" framework should own. |
| **ConceptVectors** | EMNLP 2025, Hong, Yu, Yang, Ravfogel & Geva. [arXiv:2406.11614](https://arxiv.org/abs/2406.11614) · [code](https://github.com/yihuaihong/ConceptVectors) | 285 concept vectors in 2 open LLMs, with parametric traces | **Directly on-thesis.** Shows existing methods barely alter concept vectors and mostly suppress at inference; direct ablation of the vectors removes knowledge *and* reduces adversarial susceptibility. This is the parametric analogue of the project's Type I/II split, and PISCES draws its concepts from here. |
| **LUME / SemEval-2025 Task 4** | SemEval 2025. [arXiv:2504.02883](https://arxiv.org/abs/2504.02883) | 3 subtasks (long-form creative, PII biographies, real pretraining docs); 100+ submissions, 30+ institutions | A ready-made competitive leaderboard the project could enter. |
| **OpenUnlearning** | NeurIPS D&B 2025, Dorna et al. [arXiv:2506.12618](https://arxiv.org/abs/2506.12618) · [code](https://github.com/locuslab/open-unlearning) | Unified TOFU + MUSE + WMDP; **12+ methods, 5+ datasets, 10+ metrics, 7+ architectures**, community leaderboard | **The single highest-leverage discovery for this project.** Phase 1's entire baseline-reproduction workload (weeks 3–6) is largely already implemented here. |
| **R-TOFU** | EMNLP 2025. [ACL](https://aclanthology.org/2025.emnlp-main.265/) | TOFU for large *reasoning* models — CoT-trace leakage | Relevant because ERUF's Type II failures cluster on reasoning-prior models. |
| **Awesome-LLM-Unlearning** | community. [GitHub](https://github.com/chrisliu298/awesome-llm-unlearning) | 607 papers, 18 surveys, 3 frameworks, indexed 2021–2026 | Best ongoing index. |

### 3.6 Surveys, position papers and systematisations

- **Liu et al., *Rethinking Machine Unlearning for LLMs*, Nature Machine Intelligence 2025.** [arXiv:2402.08787](https://arxiv.org/abs/2402.08787). The most citable general survey.
- **Ren, Xing, Cui, Aggarwal & Liu, *SoK: Machine Unlearning for LLMs*, 2025.** [arXiv:2506.09227](https://arxiv.org/abs/2506.09227). **Organises the field by *intention* — true knowledge removal vs. behavioural suppression — which is exactly this project's framing.** Categories: gradient ascent, model editing, hidden-representation re-steering. The project should adopt this taxonomy wholesale rather than inventing one.
- **Barez et al. (19 authors, Oxford/Tel Aviv/DeepMind-adjacent), *Open Problems in Machine Unlearning for AI Safety*, 2025.** [arXiv:2501.04952](https://arxiv.org/abs/2501.04952). Dual-use knowledge cannot be cleanly removed; unlearning interacts badly with existing safety mechanisms; evaluation is unsolved.
- **Thaker, Hu et al., *Position: LLM Unlearning Benchmarks are Weak Measures of Progress*, SaTML 2025.** [arXiv:2410.02879](https://arxiv.org/abs/2410.02879). Benchmarks are brittle to loose forget/retain dependencies; **a simple keyword filter achieves near-perfect TOFU scores**; inserting a forget keyword into a WMDP *retain* question tanks the unlearned model's accuracy while leaving the base model unaffected.
- **Feng, Xu, Robey, Kirk, Davies, Gal, Schwarzschild & Kolter, *Existing LLM Unlearning Evaluations Are Inconclusive*, 2025.** [arXiv:2506.00688](https://arxiv.org/abs/2506.00688). Evaluations inject new information (re-teaching during testing), vary wildly across tasks, and rest on spurious correlations. Two principles proposed: **minimal information injection** and **downstream-task awareness**.
- **Nguyen et al., *A Survey of Machine Unlearning*, 2022.** [arXiv:2209.02299](https://arxiv.org/abs/2209.02299). Pre-LLM foundation.
- *A Comprehensive Survey of Machine Unlearning Techniques for LLMs*, 2025. [arXiv:2503.01854](https://arxiv.org/abs/2503.01854).

### 3.7 Critical / adversarial evaluation work — the most important cluster

This is the literature the proposal engages with least and needs most.

- **Lynch, Guo, Ewart, Casper & Hadfield-Menell, *Eight Methods to Evaluate Robust Unlearning in LLMs*, 2024.** [arXiv:2402.16835](https://arxiv.org/abs/2402.16835). Red-teams Who's-Harry-Potter; above-baseline knowledge is reliably extractable adversarially, and WHP loses familiarity in *adjacent* domains (English mythology, HP film production) — the specificity failure quantified.
- **Łucki, Wei, Huang, Henderson, Tramèr & Rando, *An Adversarial Perspective on Machine Unlearning for AI Safety*, TMLR 2025** (best technical paper, NeurIPS 2024 SoLaR). [arXiv:2409.18025](https://arxiv.org/abs/2409.18025) · [code](https://github.com/ethz-spylab/unlearning-vs-safety). **Fine-tuning on 10 unrelated examples, or removing a single direction in activation space, recovers most hazardous capability from RMU models.** Jailbreaks previously reported ineffective work when applied carefully.
- **Deeb & Roger, *Do Unlearning Methods Remove Information from Language Model Weights?*, 2024.** [arXiv:2410.08827](https://arxiv.org/abs/2410.08827). The **Retraining-on-T** protocol PISCES adopts: fine-tune on facts *disjoint* from the eval questions; if accuracy on held-out facts rises, the information was never removed.
- **Shumailov et al., *UnUnlearning: Unlearning is Not Sufficient for Content Regulation in Advanced Generative AI*, 2024.** In-context knowledge can reconstitute an erased capability even if the weights are clean.
- **Xu, Yue, Liu, Ye, Zheng, Hu, Du & Hu, *Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs*, ICML 2026.** [arXiv:2505.16831](https://arxiv.org/abs/2505.16831). **Four forgetting regimes** on the axes reversible/irreversible × catastrophic/non-catastrophic. Proposes genuinely internal diagnostics — **PCA similarity and shift, CKA, Fisher information**, summarised as *mean PCA distance*. Directly supersedes a two-metric output-level taxonomy.
- **Lee, Kim & Jo, *Measuring the Depth of LLM Unlearning via Activation Patching*, 2026.** [arXiv:2605.24614](https://arxiv.org/abs/2605.24614). **Unlearning Depth Score (UDS)** — training-free, causal, dataset-invariant: identify layers encoding the target using a retain-model baseline, then measure how much is erased, 0–1. Meta-evaluated across **20 metrics on 150 unlearned models spanning 8 methods**; UDS wins on faithfulness and robustness. **This is the strongest available answer to the question the project's O2 is asking, and it is causal where EL10 is correlational.**
- **Hwiyeong Lee, Hwang, Lim & Kim, *Does Localization Inform Unlearning? A Rigorous Examination of Local Parameter Attribution for Knowledge Unlearning*, EMNLP 2025** (pp. 21857–21869). [ACL](https://aclanthology.org/2025.emnlp-main.1109/) · [code](https://github.com/HYU-NLP/loc-unlearn). In a controlled setting with **ground-truth parameter regions**, unlearning restricted to the true region does **not** yield a better forget/retain trade-off. "The set of parameters that must be modified for effective unlearning is not strictly determined."
- **Hase, Bansal, Kim & Ghandeharioun, *Does Localization Inform Editing?*, NeurIPS 2023.** [arXiv:2301.04213](https://arxiv.org/abs/2301.04213). The original result: edit success is essentially unrelated to where causal tracing says the fact lives.
- **Lacuna: A Testbed for Evaluating Localization Precision for LLM Unlearning**, 2026. [arXiv:2607.02513](https://arxiv.org/abs/2607.02513).
- ***Does Unlearning Truly Unlearn? A Black Box Evaluation***, 2024. [arXiv:2411.12103](https://arxiv.org/abs/2411.12103). 5-shot prompting and rephrasing questions **as poems** raised benchmark accuracy by up to **1750%**.
- ***Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model Outputs***, 2025. [arXiv:2506.14003](https://arxiv.org/abs/2506.14003). Unlearning leaves a detectable fingerprint — a privacy leak in itself.
- ***Soft Token Attacks Cannot Reliably Audit Unlearning***, 2025. [arXiv:2502.15836](https://arxiv.org/abs/2502.15836). Constrains which attacks count as valid audits.

### 3.8 Emerging directions (2025–2026)

1. **Causal / mechanistic evaluation** replacing output metrics — UDS activation patching, ConceptVectors parametric traces, PCA/CKA/Fisher drift.
2. **Distillation as the robustness primitive** — UNDO; the emerging consensus that robustness requires re-initialising or noising the substrate, not editing it.
3. **Reasoning-model unlearning** — R-TOFU, R²MU; CoT traces leak what answers no longer do.
4. **Over-erasure** — Xu et al. 2026; the objective should target the *marginal* contribution of forget data, not everything correlated with it. The mirror image of UIPE's argument, and a direct warning about large α.
5. **Retain-set-free unlearning** — SHRED ([arXiv:2605.07482](https://arxiv.org/abs/2605.07482)), memory-graph corpus-free methods ([arXiv:2604.13777](https://arxiv.org/abs/2604.13777)).
6. **Automated objective discovery** — EvoMU ([arXiv:2602.02139](https://arxiv.org/abs/2602.02139)) evolves unlearning losses.
7. **Multimodal and federated unlearning** — large but out of scope here.
8. **Prevention over cure** — pretraining-data filtering as the tamper-resistant baseline.

---

## 4. Evolution of the Field

**Phase 0 — Exact unlearning (2015–2021).** Cao & Yang formalise the problem; SISA makes exact retraining tractable by sharding. Works for small models; economically impossible for LLMs.

**Phase 1 — Approximate unlearning in deep nets (2020–2022).** Fisher scrubbing, influence functions, amnesiac ML, boundary shifting. Mostly vision classifiers — this is the era the repository's Cha et al. (AAAI 2024) belongs to.

**Phase 2 — Gradient ascent arrives in NLP (2022–2023).** Jang et al. establish GA. Chen & Yang add adapters and fusion (EUL). Eldan & Russinovich make concept erasure famous with Harry Potter. **Characteristic weakness: catastrophic collapse.** Push GA hard enough to forget and the model breaks.

**Phase 3 — Benchmarks force discipline (2024).** TOFU gives fictitious-author forgetting with an oracle-anchored KS-test metric. WMDP gives hazardous-knowledge MCQs and ships RMU. RWKU adds zero-shot real-world targets and adversarial probes. MUSE adds sequential and scaling axes. **Suddenly claims are comparable — and mostly disappointing.**

**Phase 4 — Better objectives (2024).** NPO fixes collapse by borrowing DPO's negative branch; SimNPO removes its reference bias. GradDiff, KL-Min, ULD, SOUL, WAGLE fill out the family. The forget/utility Pareto frontier moves, but the frontier is still measured on outputs.

**Phase 5 — Representation and parameter space (2024–2025).** The pivotal conceptual shift. RMU steers activations rather than logits. ELM matches a self-classified "novice" distribution and adds a *fluency* objective — recognising that unlearning is not just about being wrong but about being coherently ignorant. Adaptive RMU explains why RMU works and where it breaks. SAE-based methods (Farrell, DSG, CRISP) disentangle before intervening. PISCES completes the arc: **stop steering activations, edit the parameters that encode the direction**, and get 28–38% better relearning robustness for it. LUNAR reduces the intervention to a single down-projection matrix.

**Phase 6 — The reckoning (2024–2026).** In parallel, an adversarial literature demolishes the field's confidence. Lynch shows adjacent-domain collateral damage and extractability. Łucki recovers RMU capability with **ten unrelated fine-tuning examples**. Deeb & Roger formalise relearning audits. Thaker shows a **keyword filter** near-maxes TOFU. Feng et al. show evaluations inject information and rest on spurious correlations. Hase and then Lee et al. show localisation does not causally inform editing *or* unlearning. UNDO shows that **perfect behavioural agreement with an oracle still leaves recoverable latent capability**. Barez et al. argue dual-use knowledge may be irreducibly entangled.

**Phase 7 — Causal evaluation and structural robustness (2025–2026, current).** The response to Phase 6 is not better suppression but better *measurement* and *different substrates*: UDS activation patching, ConceptVectors parametric traces, reversibility taxonomies with PCA/CKA/Fisher, OpenUnlearning for standardisation, and distillation/noising (UNDO) or pretraining-data filtering as the only demonstrably durable mechanisms. **KIF/ERUF and this project sit squarely inside Phase 7's diagnostic wing.**

**Direction of travel:**

| Becoming obsolete | Ascendant |
|---|---|
| Plain GA as a standalone method | NPO/SimNPO family as the minimum baseline |
| ROUGE-L / accuracy-only forget metrics | Causal, internal metrics (UDS, parametric traces, CKA/PCA drift) |
| Single-benchmark claims | OpenUnlearning-standardised multi-benchmark reporting |
| Inference-time steering as "unlearning" | In-parameter, persistent edits |
| Reporting only FQ/MU | Mandatory relearning-attack + adversarial-prompt columns |
| Localisation assumed to imply better trade-offs | Localisation treated as a hypothesis requiring causal validation |
| "Robust because GCG failed" | Robustness against adaptive, method-aware attacks |

---

## 5. Paper Comparison Table

Sorted roughly by relevance to this project. **R** = relevance (★★★ critical, ★★ important, ★ contextual).

| # | Paper | Venue / Year | Main idea | Key contributions | Strengths | Weaknesses / limitations | R |
|---|---|---|---|---|---|---|---|
| 1 | **KIF / ERUF** — Mahmood et al. | arXiv 2026 (2601.10566) | Mine subject activation signatures → gated capsule suppression → distil to LoRA | 3-stage pipeline; SMR+EL10 dual metric; Type I/II/III taxonomy; composite UPU loss | Near-oracle TOFU (FQ 0.99, MU 0.62); −68% adversarial recovery; SRS 5.90×; sequential unlearning holds; runs 4-bit on one A6000 | **Un-refereed preprint**; explicitly no irreversibility guarantee; single TOFU run; RWKU only 4 subjects; locality fails same-domain; **still Type II on Llama-3B and reasoning-prior 8B, Type III on DeepSeek-3B**; renamed since v4 | ★★★ |
| 2 | **UIPE** — Wang et al. | Findings EMNLP 2025 | Amplify the GA update vector to reach logically related knowledge | Proof that GA's projection channel closes once the target is forgotten; plug-and-play $\theta+(1+\alpha)v$; no extra data | Works on GA/GradDiff/KL-Min/NPO; WMDP-Avg 0.3302→0.1768 at ~1pt MMLU cost; linear cost | α hand-tuned per method; **inverted-U on GA (over-forgetting)**; 7B only; full-parameter FT on 2×A800; no representation-level verification; never compared to task-arithmetic/ExPO | ★★★ |
| 3 | **PISCES** — Gur-Arieh et al. | EMNLP 2025 | SAE-disentangle MLP *parameter* vectors, ablate concept features in place | In-parameter erasure framework; VocabProj feature ID; negative-value ablation; τ coverage control | Best efficacy+specificity+robustness jointly (Llama: 7.7% / 87.6% / 65.4% relearn vs. ELM 103.1%); persistent under white-box | Needs per-MLP-layer SAEs → **only Gemma-2-2B & Llama-3.1-8B**; MLP-only; manual feature filtering; VocabProj weak in early layers; still 65% relearning recovery | ★★★ |
| 4 | **ELM** — Gandikota et al. | NeurIPS 2025 | Self-classification: drive output toward a "novice" conditional distribution | Innocence/seamlessness/specificity triad; conditional-fluency loss; rank-4 LoRA on layers 4–7 | Best fluency post-erasure (R-PPL 10.9 vs RMU 24.8); resists GCG 5000 iters; decisive ablations | Specificity loss on adjacent MMLU domains; **relearning accuracy 103% on Llama — fully recoverable**; output-distribution level only | ★★★ |
| 5 | **UNDO / Distillation Robustifies Unlearning** — Lee, Foote et al. | NeurIPS 2025 spotlight | Distil the unlearned model into a noised copy | Shows oracle-matching behaviour ≠ removed capability; Unlearn-Noise-Distill; compute/robustness Pareto frontier | The clearest causal demonstration that behavioural unlearning is not deletion; practical recipe | Expensive (partial re-training); demonstrated on synthetic language/arithmetic tasks | ★★★ |
| 6 | **Adversarial Perspective** — Łucki et al. | TMLR 2025 | Adaptive attacks on unlearned models | 10 unrelated fine-tuning examples recover RMU capability; single-direction activation removal does too | Definitively reframes what "robust" must mean; open code | Attack-side only; no defence proposed | ★★★ |
| 7 | **Does Localization Inform Unlearning?** — H. Lee et al. | EMNLP 2025 | Controlled test of whether localised updates help | Even ground-truth-region unlearning gives no better forget/retain trade-off | Kills an unverified assumption with a clean design; code released | Controlled synthetic setting; does not test activation-signature localisation specifically | ★★★ |
| 8 | **Unlearning Isn't Deletion** — Xu et al. | ICML 2026 | Reversibility × catastrophicity taxonomy | 4 forgetting regimes; PCA-similarity/shift, CKA, Fisher diagnostics; mean PCA distance | A strictly richer taxonomy than Type I/II/III, with internal (not output) diagnostics | Diagnostics are correlational descriptors, not causal interventions | ★★★ |
| 9 | **UDS / Activation Patching Depth** — J. Lee et al. | arXiv 2026 (2605.24614) | Causal depth-of-unlearning metric | Training-free, dataset-invariant 0–1 score; meta-eval over 20 metrics × 150 models × 8 methods | Most faithful and robust metric currently published | Needs a retain-model baseline; patching cost | ★★★ |
| 10 | **RMU + WMDP** — Li et al. | ICML 2024 | Steer forget activations to a random direction | WMDP benchmark; the reference representation-level method | Standard baseline everywhere; strong forget/MMLU trade-off | Weights uninterpretable; poor fluency; recoverable (Łucki); fails at deeper layers | ★★★ |
| 11 | **Adaptive RMU** — Huu-Tien et al. | AAAI 2025 | Norm-adaptive steering coefficient | Theory of *why* RMU works (logit-variance inflation); layer-failure diagnosis; $\beta\|h\|\mathbf{u}$ | Bio 28.8→23.7, Cyber 28.8→26.5; explains RMU's apparent GCG robustness as a gradient artefact | Linearised (1st-order) analysis; β needs grid search; GCG-only threat model | ★★★ |
| 12 | **NPO** — Zhang et al. | COLM 2024 | DPO's negative branch only | Exponentially slower collapse than GA; first to handle 50% forget | The strong default baseline; theory-backed | Reference-model bias (→SimNPO); still output-level | ★★★ |
| 13 | **REGLU** — Xiao et al. | arXiv 2026 (2604.17396) | Representation-guided LoRA init + orthogonal-complement regulariser | Geometric alternative to capsule+distillation | Directly on TOFU+WMDP; beats SOTA as reported; **closest competitor to Phase 3** | Preprint; limited independent replication | ★★★ |
| 14 | **TOFU** — Maini et al. | COLM 2024 | Fictitious-author unlearning benchmark | FQ via KS test vs. oracle; MU aggregate | Clean, oracle-anchored, ubiquitous | **Gibberish models can score high FQ**; synthetic; keyword filters near-max it (Thaker) | ★★★ |
| 15 | **RWKU** — Jin et al. | NeurIPS D&B 2024 | Zero-shot real-world knowledge unlearning | 200 targets, 13,131 probes incl. 6,984 adversarial; 4 MIAs, 9 attack types | Most adversarially serious benchmark; no corpus leakage | Real entities → ethics care; heavier to run | ★★★ |
| 16 | **OpenUnlearning** — Dorna et al. | NeurIPS D&B 2025 | Unified benchmarking library | TOFU+MUSE+WMDP, 12+ methods, 10+ metrics, leaderboard | **Removes most of Phase 1's engineering burden** | Not every niche method included | ★★★ |
| 17 | **ConceptVectors** — Hong et al. | EMNLP 2025 | Intrinsic (parametric) test of unlearning | 285 concept vectors; VocabProj traces; shows methods only suppress at inference | The parametric ground truth the Type I claim needs | 2 models; concept-vector localisation itself imperfect | ★★★ |
| 18 | **Position: Benchmarks are Weak** — Thaker et al. | SaTML 2025 | Benchmarks over-report progress | Keyword filter ≈ perfect TOFU; forget-keyword injection breaks WMDP retain accuracy | Cheap, devastating, actionable | Position paper, no new method | ★★★ |
| 19 | **Evaluations Are Inconclusive** — Feng et al. | arXiv 2025 | Evaluation protocol critique | Information injection; task variance; spurious correlations; 2 design principles | Directly usable protocol guidance | No benchmark released | ★★ |
| 20 | **SoK: MU for LLMs** — Ren et al. | arXiv 2025 | Intention-oriented systematisation | Removal-vs-suppression taxonomy | Matches this project's framing exactly | Survey | ★★ |
| 21 | **Open Problems in MU for AI Safety** — Barez et al. | arXiv 2025 | Position on unlearning's limits | Dual-use entanglement; interaction with safety mechanisms | Authoritative, broad authorship | No experiments | ★★ |
| 22 | **Eight Methods to Evaluate Robust Unlearning** — Lynch et al. | arXiv 2024 | Red-team evaluation suite | 8 complementary probes; WHP case study | Practical, reusable protocol | Single target model | ★★ |
| 23 | **Do Unlearning Methods Remove Info from Weights?** — Deeb & Roger | arXiv 2024 | Retraining-on-T audit | Held-out-fact relearning protocol | The robustness protocol PISCES adopts | Audit only | ★★ |
| 24 | **LUNAR** — Shen et al. | NeurIPS 2025 | Redirect forget activations to an "I can't answer" region | Single down-projection edit; 20× efficiency | 2.9–11.7× deviation score; white-box robust; sequential | Requires a coherent refusal region; behavioural target | ★★ |
| 25 | **Breaking Chains / Munch** — Choi et al. | arXiv 2024 | Multi-hop knowledge survives single-hop unlearning | Uncertainty-based question decomposition | The nearest published relative of O3 | Inference-time; doesn't remove weights | ★★ |
| 26 | **Task Arithmetic** — Ilharco et al. | ICLR 2023 | Arithmetic on $\theta_{\text{ft}} - \theta_{\text{pre}}$ | Negation removes behaviour; addition composes | The theoretical home of UIPE | Pre-LLM-unlearning framing | ★★ |
| 27 | **ExPO** — Zheng et al. | ICML 2024 | Extrapolate alignment deltas | $\theta + \alpha\Delta$ improves alignment training-free | **Mathematically identical to UIPE**, opposite sign; supplies α-selection intuition | Alignment domain | ★★ |
| 28 | **MUSE** — Shi et al. | ICLR 2025 | Six-way unlearning evaluation | Verbatim/knowledge memorisation, privacy, utility, scalability, sustainability | Only benchmark testing sequential + scale | Heavy; 7B-focused | ★★ |
| 29 | **ECO** — Liu et al. | NeurIPS 2024 | Corrupt forget-prompt embeddings at inference | Guardrail unlearning with ~zero side effects | Excellent benchmark scores with **zero weight change** — the perfect Type II control | Not unlearning in any weight sense; classifier dependency | ★★ |
| 30 | **TAR** — Tamirisa et al. | ICLR 2025 | Meta-learned tamper resistance | Survives hundreds of fine-tuning steps, 28 attacker strategies | Sets the bar for "durable" | Expensive; some capability cost | ★★ |
| 31 | **EUL** — Chen & Yang *(in repo)* | EMNLP 2023 | Unlearning layers + closed-form fusion | Selective teacher-student objective; training-free layer fusion; MLM-extraction probe | Foundational adapter pattern; **only paper here solving sequential deletion** | T5, classification/summarisation; obsolete baselines; not LLM-scale | ★★ |
| 32 | **Selective Pruning** — Pochinkov & Schoots *(in repo)* | arXiv 2024 | Prune neurons by forget/retain importance ratio | FF ≫ ATTN specialisation; dropout increases specialisation; $I_{\text{abs}}$ best | Extremely cheap (one RTX 4090); validates MLP locality; task-agnostic | Coarse; large collateral damage at high forget rates; no LLM-benchmark numbers | ★★ |
| 33 | **SimNPO** — Fan et al. | arXiv 2024 | NPO without reference-model bias | Simpler, stronger | A KIF baseline (FQ 0.45) | Incremental | ★ |
| 34 | **AlphaEdit** — Fang et al. | ICLR 2025 | Null-space-constrained editing | Preserves retained knowledge by construction | PISCES baseline | Weak at concept-level erasure (73.6% on Llama) | ★ |
| 35 | **MEMIT** — Meng et al. | ICLR 2023 | Mass fact editing | Thousands of edits at once | Scalable | Concept erasure needs enumerating all relations | ★ |
| 36 | **Instance-Wise Unlearning** — Cha et al. *(in repo)* | AAAI 2024 | Misclassify forget instances; adversarial + MAS regularisers | Forget-data-only access; representation-preserving augmentation | Solid vision work | **Image classifiers — no generative model, non-transferable metrics** | ★ |
| 37 | **R-TOFU / R²MU** | EMNLP 2025 / arXiv 2025 | Unlearning for reasoning models | CoT-trace leakage benchmark + method | Explains ERUF's reasoning-model failures | Emerging | ★ |
| 38 | **Over-erasure** — Xu et al. | arXiv 2026 | Methods erase retain-supported knowledge too | Objective should target marginal forget contribution | Direct warning about aggressive α | Recent | ★★ |

---

## 6. Current State of the Art

### 6.1 Best-performing methods, by axis

| Axis | Current leader | Evidence |
|---|---|---|
| **Efficacy + specificity jointly (concept erasure)** | **PISCES** | Llama-3.1-8B: 7.7% retained accuracy, 87.6% similar-domain, 99.4% MMLU |
| **Robustness to relearning (post-hoc editing)** | **PISCES** | 65.4% relearning vs. 93.2–103.1% for RMU/ELM/MEMIT/AlphaEdit |
| **Robustness, structural** | **UNDO** (distil into noised copy); **TAR** (meta-learned tamper resistance) | UNDO defines the new Pareto frontier; TAR survives 28 attack strategies |
| **Fluency after erasure** | **ELM** | R-PPL 10.9 vs. RMU 24.8 |
| **TOFU forget-quality/utility** | **ERUF** (claimed, unrefereed); **NPO/SimNPO** among refereed | FQ 0.99 / MU 0.62 at oracle |
| **WMDP hazard removal** | **Adaptive RMU**; RMU as reference | Bio 23.7 / Cyber 26.5 / MMLU 55.0 |
| **Efficiency** | **LUNAR** (one down-projection matrix, 20×); **Selective Pruning** (single consumer GPU) | — |
| **Evaluation fidelity** | **UDS** (causal, activation patching) | Best of 20 metrics over 150 models |
| **Infrastructure** | **OpenUnlearning** | 12+ methods, 3 benchmarks, 10+ metrics, leaderboard |

### 6.2 Research trends

1. **Measurement is the bottleneck, not method design.** More 2025–26 attention is going to *whether we can tell if unlearning worked* than to new objectives.
2. **From activations to parameters.** Steering is losing to persistent in-parameter editing on both robustness and threat-model grounds.
3. **Adversarial evaluation is now mandatory.** A 2026 paper without a relearning-attack column will not be taken seriously.
4. **Interpretability tools as unlearning tools.** SAEs, vocabulary projection, activation patching, causal tracing — the mech-interp toolkit is now the unlearning toolkit.
5. **Standardisation.** OpenUnlearning is doing for this field what `lm-evaluation-harness` did for LLM evaluation.
6. **Pessimism about post-hoc removal.** Barez, Shumailov, Łucki and UNDO collectively argue that post-hoc unlearning may be structurally limited and that data filtering or distillation may be the only durable answers.

### 6.3 Remaining challenges

- **No certificate of removal.** Every method offers evidence, none offers a guarantee. ERUF says so explicitly.
- **Dual-use entanglement.** Virology knowledge that enables bioweapons also enables vaccines. Erasure and utility may be genuinely inseparable (Barez et al.).
- **Benchmarks are gameable.** Keyword filters near-max TOFU; poem-rephrasing raises accuracy 1750%.
- **Localisation is unvalidated.** Two independent studies (Hase 2023, Lee 2025) find localisation does not causally improve editing or unlearning.
- **Sequential and large-scale removal.** MUSE's sustainability and scalability axes remain largely unsolved.
- **Reasoning models.** CoT traces leak what final answers no longer do.
- **Over- vs. under-erasure.** UIPE argues for reaching further; over-erasure work argues for reaching less. Nobody has characterised the right stopping point.

---

## 7. Project Positioning

### 7.1 Where DeepErase sits

DeepErase is a **Phase-7 diagnostic-wing project**: it accepts the field's central critique (obfuscation ≠ erasure) and responds with a verification protocol plus an integrated method. It is well-aimed. The specific position is:

> *An integration of related-knowledge parameter extrapolation (UIPE) with activation-signature capsule suppression distilled into LoRA (KIF/ERUF), evaluated under a surface+latent dual metric, across TOFU / WMDP / RWKU.*

**Nobody has published that specific composition.** The novelty is real but narrow, and it is currently unstated (D14).

### 7.2 Closest published work

| Rank | Work | Why it is close | Difference from DeepErase |
|---|---|---|---|
| 1 | **KIF / ERUF** (arXiv 2026) | DeepErase's O2 and O4 *are* ERUF's contributions | DeepErase adds UIPE-style related-knowledge extrapolation on top |
| 2 | **REGLU** (arXiv 2026) | Representation geometry + LoRA + TOFU/WMDP | Orthogonal-complement regularisation instead of gated capsules + distillation; no related-knowledge component |
| 3 | **PISCES** (EMNLP 2025) | Localise-then-edit with specificity as the headline | Edits parameters via SAEs, not activations via signatures; no related-knowledge component |
| 4 | **LUNAR** (NeurIPS 2025) | Redirect forget activations into a target region | Single-matrix edit; no distillation; no dual metric |
| 5 | **Munch / Breaking Chains** (2024) | The related-knowledge problem, framed as multi-hop | Inference-time decomposition, not parameter extrapolation |
| 6 | **Unlearning Isn't Deletion** (ICML 2026) | A richer version of the Type I/II/III taxonomy | 4 regimes with internal diagnostics (PCA/CKA/Fisher) vs. 3 with output metrics |

### 7.3 What the project already incorporates well

- The **right problem framing** — matching SoK 2025's removal-vs-suppression axis.
- **Dual-axis evaluation** rather than forget-quality alone.
- **Parameter-efficiency** (LoRA) and **quantisation** — realistic for the stated hardware.
- **Cross-benchmark generalisation** (TOFU + WMDP + RWKU) rather than single-benchmark claims.
- **Capability-retention testing** (ARC-C, HellaSwag, WinoGrande, TruthfulQA) — exactly the columns ERUF reports.
- **Ablations** on loss components and on α — good scientific hygiene.

### 7.4 What is missing

Ordered by how much damage the omission does.

**M-1 — No relearning-attack evaluation.** Robustness to fine-tuning attacks is claimed in §7.2 ("expected research contribution") but **appears nowhere in the methodology or the evaluation module list**. PISCES's headline result *is* its relearning column; Łucki recovers RMU with 10 examples; UNDO shows oracle-matching behaviour is still recoverable. Without a **Retraining-on-T** protocol (Deeb & Roger) the durability claim is unsupported. This is the single largest gap.

**M-2 — The dual metric is not actually latent (D12).** EL10 is an output-distribution statistic. Three stronger, published alternatives exist: **UDS** (causal activation patching), **ConceptVectors** parametric traces, and **PCA/CKA/Fisher** representational drift. Adding even one turns O2 from "a reimplementation of ERUF's metric" into a defensible contribution.

**M-3 — Localisation is assumed, not tested.** Assumption 2 and the whole of Phase 3 rest on MLP localisation. Hase et al. (2023) and Lee et al. (EMNLP 2025) both find localisation does not causally improve editing/unlearning. Cohen's *d* separability shows a signature is *decodable*, not that suppressing it *causes* forgetting. A causal check — ablate the signature direction and measure the effect against a random-direction control — costs little and pre-empts the obvious reviewer question.

**M-4 — No modern strong baselines.** SimNPO, LUNAR and REGLU are all absent from the comparison plan, yet all three are direct competitors and two are ERUF's own baselines.

**M-5 — Benchmark critiques unaddressed.** Thaker's keyword-filter result and Feng's information-injection critique should shape the evaluation design, and a keyword-filter control is nearly free to run.

**M-6 — No sequential unlearning.** MUSE's sustainability axis and ERUF's Table 8 both treat this as expected. EUL — a paper already in the repository — provides the fusion mechanism for it.

**M-7 — UIPE ⊕ KIF composition is undefined (D8/D14).** UIPE extrapolates a full-parameter delta; KIF produces a LoRA delta. The plan never says which is extrapolated. Extrapolating a LoRA delta is unvalidated by any paper.

**M-8 — Over-erasure is unmonitored.** UIPE's α has an inverted-U on GA (over-forgetting past the peak) and the over-erasure literature says methods already remove retain-supported knowledge. The α sweep needs a specificity metric (similar-domain accuracy, PISCES-style) in the loop, not just FQ/MU.

**M-9 — MUSE and ConceptVectors are absent** from the benchmark set, and **OpenUnlearning** is absent from the tooling list despite eliminating most of Phase 1's engineering.

### 7.5 Is the methodology competitive?

**Yes, conditionally.** Integrating UIPE with ERUF is a legitimate incremental contribution with a plausible mechanism: the two operate on orthogonal failure modes (related-knowledge reach vs. representation depth), so gains should compose rather than conflict. The dual-metric cross-model study has genuine value because **ERUF's own results show Type II failures on Llama-3B and reasoning-prior 8B models** — mapping *when* representation-aware unlearning works and when it degenerates to obfuscation is a real, publishable empirical contribution that nobody has done systematically.

**The conditions:** (a) close M-1 (relearning attacks) or the durability claim is unsupported; (b) upgrade or supplement the latent metric per M-2; (c) resolve the compute/SAE feasibility issues D8 and D9 before week 5; (d) state the novelty explicitly.

**Realistic risk:** the plan is a **16-week undergraduate capstone attempting to reproduce five non-trivial methods** (ELM, PISCES, RMU, GA-family, KIF) **and** integrate two of them **and** build a new evaluation protocol. PISCES alone ran 800 experiments per concept. Adopting OpenUnlearning for Phase 1 and narrowing the baseline set is the difference between finishing and not.

---

## 8. Research Gaps

### 8.1 Gaps this project could plausibly close

**G1 — A cross-model map of when representation-aware unlearning degenerates into obfuscation.**
ERUF reports Type II on Llama-3B and Qwen-8B, Type III on DeepSeek-3B, Type I elsewhere — but offers no explanation. Systematically relating Type outcome to model family, scale, reasoning-prior training, and **signature separability (Cohen's *d*)** would be a genuine contribution. Selective Pruning's dropout finding supplies a testable mechanistic hypothesis: *models trained with FF dropout have more task-specialised neurons, hence cleaner signatures, hence Type I.* That is a falsifiable claim nobody has tested.

**G2 — Does related-knowledge extrapolation change unlearning *depth* or only *breadth*?**
UIPE was evaluated purely on output metrics. Nobody has asked whether extrapolation deepens representation-level attenuation or merely widens surface suppression. Running SMR/EL10 (ideally plus UDS) across an α sweep answers a question the UIPE authors did not ask. **This is the most under-served, highest-value question available to this project.**

**G3 — A principled α-selection rule.**
UIPE's own stated limitation. With a specificity metric in the loop and the over-erasure framing, an automated stopping rule (e.g. maximise FQ subject to similar-domain accuracy ≥ threshold) is a small, clean, citable contribution.

**G4 — Do UIPE and representation-aware suppression compose or interfere?**
Genuinely unknown. Either answer is publishable. A negative result — "extrapolating a LoRA delta destroys the capsule-distilled behaviour" — is still a result.

**G5 — Full-benchmark RWKU under the dual metric.**
ERUF used 4 subjects out of 200 because of compute. Even 25–50 subjects with RWKU's 6,984 adversarial probes would be a substantially stronger evidence base than anything published on this method family.

**G6 — Independent verification of an unrefereed preprint.**
ERUF/KIF is not peer-reviewed, reports a single TOFU run, and claims to break the stability–erasure trade-off. Independent replication with proper seed variance is a service to the field and a perfectly respectable capstone outcome, whichever way it goes.

### 8.2 Open problems beyond this project's scope

- **Certified unlearning for LLMs.** Differential-privacy-style guarantees exist for convex models; nothing comparable for transformers.
- **Dual-use inseparability.** Whether hazardous and beneficial knowledge are separable *even in principle* (Barez et al.).
- **The right stopping point.** UIPE says reach further; over-erasure says reach less. No theory reconciles them.
- **Unlearning vs. in-context reconstitution.** UnUnlearning: clean weights plus a good prompt can rebuild the capability.
- **Multi-hop and compositional erasure at scale.** Munch is inference-time; a weight-level solution is open.
- **Cross-lingual erasure.** Knowledge unlearned in English persists in other languages.
- **Unlearning traces as a privacy leak.** *Unlearning Isn't Invisible* — the act of forgetting reveals what was forgotten.
- **Reasoning-model unlearning.** CoT traces are a new leakage surface.

---

## 9. Prioritised Reading List

### Tier 0 — Read this week, before any implementation

| # | Paper | Why first |
|---|---|---|
| 1 | **KIF / ERUF v5** — [arXiv:2601.10566](https://arxiv.org/abs/2601.10566) | It *is* Phases 3–4. Read **v5, not v4** — renamed, and the limitations section (§7) tells you exactly which failures to expect on which models. |
| 2 | **UIPE** — [Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1374/) *(in repo)* | Phase 2. Read **Appendix C (Algorithm 1) and D.3** — that is where the full-parameter, 2×A800 requirement lives, and it determines whether D8 is a blocker. |
| 3 | **OpenUnlearning** — [arXiv:2506.12618](https://arxiv.org/abs/2506.12618) · [code](https://github.com/locuslab/open-unlearning) | Read before writing a line of Phase 1. It likely already implements most of your baselines and metrics. Weeks of saved effort. |
| 4 | **Position: Benchmarks are Weak Measures of Progress** — [arXiv:2410.02879](https://arxiv.org/abs/2410.02879) | Short, and it will change your evaluation design. The keyword-filter control belongs in your results table as a sanity check. |

### Tier 1 — Read in weeks 1–2, before finalising the methodology

| # | Paper | Contribution to your understanding |
|---|---|---|
| 5 | **PISCES** — [arXiv:2505.22586](https://arxiv.org/abs/2505.22586) | Phase 1 baseline. §5.1 defines the four-axis evaluation (efficacy / specificity / coherence / robustness) you should adopt verbatim. Also tells you immediately whether D9 (no SAEs for Mistral/Zephyr) forces a model-list change. |
| 6 | **ELM** — [arXiv:2410.02760](https://arxiv.org/abs/2410.02760) | Phase 1 baseline. The ablation table is the model for your own loss ablations. Its 103% relearning accuracy motivates M-1. |
| 7 | **Adaptive RMU (AAAI 2025)** — [arXiv:2408.06223](https://arxiv.org/abs/2408.06223) | **Rewrite proposal §4.5 from this paper.** Also: it explains why "GCG failed against our method" is weak evidence of robustness — a trap your report should avoid. |
| 8 | **Łucki et al., Adversarial Perspective (TMLR 2025)** — [arXiv:2409.18025](https://arxiv.org/abs/2409.18025) | Defines the robustness bar. 10 unrelated examples recover RMU. Your evaluation must include something like this. |
| 9 | **Deeb & Roger, Do Unlearning Methods Remove Information from Weights?** — [arXiv:2410.08827](https://arxiv.org/abs/2410.08827) | The concrete Retraining-on-T protocol for M-1. PISCES uses it; you can too. |
| 10 | **UNDO / Distillation Robustifies Unlearning (NeurIPS 2025)** — [arXiv:2506.06278](https://arxiv.org/abs/2506.06278) | The strongest challenge to your durability claim: oracle-matching behaviour still leaves recoverable capability. Address it explicitly or a reviewer will. |
| 11 | **Does Localization Inform Unlearning? (EMNLP 2025)** — [ACL](https://aclanthology.org/2025.emnlp-main.1109/) | Directly challenges Assumption 2 and Phase 3's premise. Read it *before* committing to signature-based localisation so you can design the causal control. |

### Tier 2 — Read in weeks 2–4, to sharpen contribution and evaluation

| # | Paper | Contribution |
|---|---|---|
| 12 | **Unlearning Isn't Deletion (ICML 2026)** — [arXiv:2505.16831](https://arxiv.org/abs/2505.16831) | A strictly richer taxonomy than Type I/II/III, with PCA/CKA/Fisher diagnostics. Either adopt it or justify why your 3-way split is preferable. |
| 13 | **UDS / Activation Patching (2026)** — [arXiv:2605.24614](https://arxiv.org/abs/2605.24614) | The causal metric that would fix M-2. Best-of-20 in a 150-model meta-evaluation. |
| 14 | **ConceptVectors (EMNLP 2025)** — [arXiv:2406.11614](https://arxiv.org/abs/2406.11614) | Parametric-trace evaluation; shows most methods only suppress at inference. The intrinsic complement to your behavioural metrics. |
| 15 | **REGLU (2026)** — [arXiv:2604.17396](https://arxiv.org/abs/2604.17396) | Your closest published competitor. You need to know what it does and how you differ. |
| 16 | **NPO (COLM 2024)** — [arXiv:2404.05868](https://arxiv.org/abs/2404.05868) + **SimNPO** — [arXiv:2410.07163](https://arxiv.org/abs/2410.07163) | The mandatory strong baselines; SimNPO is one of ERUF's comparison points. |
| 17 | **SoK: MU for LLMs (2025)** — [arXiv:2506.09227](https://arxiv.org/abs/2506.09227) | Ready-made taxonomy for your literature-survey chapter, organised on exactly your removal-vs-suppression axis. |
| 18 | **RWKU (NeurIPS D&B 2024)** — [OpenReview](https://openreview.net/forum?id=wOmtZ5FgMH) | Read the adversarial-probe design before running your 4-subject subset; scaling it up is G5. |
| 19 | **Task Arithmetic (ICLR 2023)** — [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) + **ExPO (ICML 2024)** — [arXiv:2404.16792](https://arxiv.org/abs/2404.16792) | Reframes UIPE as negated-task-vector extrapolation. Gives you α-selection intuition and a much stronger related-work narrative. |
| 20 | **LUNAR (NeurIPS 2025)** — [arXiv:2502.07218](https://arxiv.org/abs/2502.07218) | Closest activation-level relative of the Suppression Capsule; an ERUF baseline. |

### Tier 3 — Read for framing, discussion and future work

| # | Paper | Contribution |
|---|---|---|
| 21 | **Barez et al., Open Problems in MU for AI Safety** — [arXiv:2501.04952](https://arxiv.org/abs/2501.04952) | The limitations section of your report, essentially pre-written. |
| 22 | **Feng et al., Evaluations Are Inconclusive** — [arXiv:2506.00688](https://arxiv.org/abs/2506.00688) | Two design principles: minimal information injection, downstream-task awareness. |
| 23 | **Lynch et al., Eight Methods** — [arXiv:2402.16835](https://arxiv.org/abs/2402.16835) | A reusable red-team suite. |
| 24 | **MUSE (ICLR 2025)** — [arXiv:2407.06460](https://arxiv.org/abs/2407.06460) | Sequential + scalability axes (M-6). |
| 25 | **Breaking Chains / Munch** — [arXiv:2410.13274](https://arxiv.org/abs/2410.13274) | The related-knowledge problem from the multi-hop angle; must be cited alongside O3. |
| 26 | **EUL (EMNLP 2023)** *(in repo)* — [ACL](https://aclanthology.org/2023.emnlp-main.738/) | The fusion mechanism if you take on sequential unlearning; historical framing of the adapter pattern. |
| 27 | **Selective Pruning** *(in repo)* — [arXiv:2403.01267](https://arxiv.org/abs/2403.01267) | Cheapest localisation baseline; supplies the dropout→specialisation hypothesis for G1. |
| 28 | **TOFU (COLM 2024)** — [arXiv:2401.06121](https://arxiv.org/abs/2401.06121) + **WMDP (ICML 2024)** — [arXiv:2403.03218](https://arxiv.org/abs/2403.03218) | Benchmark internals; **WMDP is also the missing RMU citation (D6)**. |
| 29 | **Geva et al., FFN Key-Value Memories (EMNLP 2021)** — [arXiv:2012.14913](https://arxiv.org/abs/2012.14913) + **ROME (NeurIPS 2022)** — [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) | The evidence base for Assumption 2. |
| 30 | **Rethinking MU for LLMs, Nature MI 2025** — [arXiv:2402.08787](https://arxiv.org/abs/2402.08787) | High-prestige general citation for the introduction. |

**Reading order rationale.** Tier 0 answers *can this project be built as specified?* (feasibility: compute, SAE availability, existing tooling). Tier 1 answers *what must the evaluation contain to be credible?* (relearning attacks, causal localisation checks, correct citations). Tier 2 answers *what is the actual contribution and who are the competitors?* Tier 3 supplies framing and future work. Reading in this order means the two decisions with the longest lead times — model selection (D9) and full-FT vs. LoRA deltas (D8) — get made in week 1 rather than week 6.

---

## 10. Recommended Immediate Actions (no code)

1. **Fix reference [4]** — correct authors (Huu-Tien, Pham, Thanh-Tung, Inoue), venue (AAAI 2025, pp. 23733–23742), and rewrite §4.5 to describe Adaptive RMU rather than a dual-metric motivation (D1, D2).
2. **Add missing citations** — Li et al. ICML 2024 (RMU/WMDP), Rosati et al. (RepNoise), Maini et al. COLM 2024 (TOFU), Jin et al. NeurIPS 2024 (RWKU), Zhang et al. COLM 2024 (NPO), Fan et al. (SimNPO) (D6).
3. **Update the KIF citation to v5** and note the KIF → ERUF rename in the report (D3).
4. **Soften or remove the "exclusively peer-reviewed" claim** (D7).
5. **Decide D8 and D9 in week 1** — full-FT vs. LoRA delta for UIPE; and whether to swap Mistral/Zephyr for Gemma-2-2B / Llama-3.1-8B so PISCES is reproducible at all.
6. **Add a Robustness column** (Retraining-on-T, Deeb & Roger) to the evaluation module and to §7.1 deliverable 3 (M-1).
7. **Specify the WMDP protocol for SMR/EL10**, or restrict the dual metric to entity-centric benchmarks (TOFU, RWKU) and use FQ/MU/MMLU on WMDP (D11).
8. **Add one genuinely internal metric** — UDS, ConceptVectors traces, or PCA/CKA drift — and stop describing EL10 as latent (D12, M-2).
9. **Adopt OpenUnlearning for Phase 1** and reduce the baseline set to what is reproducible in weeks 3–6.
10. **Write one paragraph stating the novelty** (D14) and add it to §7.2 and slide 8.
11. **Fix the roll-number discrepancy** on slide 10 (D13).

---

*Report ends. No project code was created, modified, or refactored.*
