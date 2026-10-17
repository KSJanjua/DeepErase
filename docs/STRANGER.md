# STRANGER: unlearning an author by making their name unknown

STRANGER is this project's own unlearning method, designed after the v2 study
showed how existing methods fall short (RESULTS.md §13-16). It is
pre-registered as H7 in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md).

## 1. Why a new method

The v2 study found three things about GA, GradDiff and NPO on TOFU with
Llama-3.2-1B:

1. **They mostly suppress.** At the checkpoint each method would report, their
   depth (UDS 0.16-0.22) is about what random damage of the same size gives
   (0.15-0.21).
2. **What depth they add is partly overshoot.** UDS clips each layer's erasure
   ratio r = dS2/dS1 at 1, so making the model worse than an ignorant one counts
   as erasure. At α = 1, 14-17% of their UDS on trained facts comes from layers with r > 2, and
   all of GA's depth gain from α = 1 to 2 is overshoot (§16).
3. **They do not erase by author.** Their erasure is spread fact by fact, and
   GradDiff erases held-out facts less than random damage does.

So a deeper method should change what the model *stores about the entity*, aim
at "does not know" rather than "anti-knows", and leave the layers that only
read facts out alone.

## 2. Idea

In TOFU every fact about an author is retrieved through the author's name.
Mechanistic work on factual recall (ROME, Geva et al. 2023, Ferrando et al.
2025) finds that early-to-middle MLPs enrich a subject's last token with what
the model knows about it, and later attention heads copy those attributes to
the answer. STRANGER rewrites that enrichment: after training, the forget
author's name is processed the way the model processes a name it has never
seen.

"A name it has never seen" needs no oracle. **The model itself supplies it:**
run M_full on the same question and answer with the author's name replaced by
invented names of the same culture, gender and shape. Those names carry no
memories, so M_full's states there are exactly "an unknown person, this
question". Averaging over four invented names removes the identity of any one
of them.

## 3. Inputs

* **Names** (`data/counterfactual_names.json`, built by
  `deeperase/scripts/build_cf_names.py`, tokenizer only). For each of the 20
  forget authors, 4 invented names that:
  * have the same shape (parts, hyphen sub-parts, order);
  * match the author's culture and pronoun (18 she/he, 2 they);
  * share no space-separated part with any of TOFU's 200 author names;
  * keep every one of the author's 20 rows the same token length, with every
    non-name token identical.
* **Swap.** Every mention is replaced piece by piece: the full name, each part
  ("Majumdar's"), each hyphen sub-part of 3+ letters. Possessives and
  honorifics are kept.
* **Positions** (`data/counterfactual.py: stranger_positions`), per row:
  * from the last token of the author's first mention in the question up to the
    token before the one that predicts the first answer token;
  * the last token of each mention in the answer;
  * minus the non-final tokens of every mention, since their inputs differ and
    have no aligned target.

  The loss never touches the position predicting the first answer token, nor
  any other answer position. So it stays off the positions UDS measures, except
  where an answer mention ends right before an entity: 4, 2 and 5 of 200 rows
  for seeds 0, 1 and 2, reported with and without those rows. About 10
  positions per row; every row has some.

## 4. Objective

Let h_l(x)[p] be the output of decoder block l at position p, the tensor UDS
patches. With θ0 = M_full, computed once before training:

* **Forget target**: τ_l(x)[p] = mean over the 4 names of h_l^θ0(x̃_k)[p], for
  layers l = 4..8.
* **Retain anchor**: a_l(r)[p] = h_l^θ0(r)[p] on 200 retain rows (the same rows
  GradDiff uses, with every R-tier author excluded), at every position but BOS.
* **Scale**: σ_l² is the mean squared norm of the anchors at layer l.
* Positions whose starting state is above 5× the layer's median norm are
  dropped. These are Llama's massive activations, which would dominate an MSE.

The loss per step is:

    L = mean over forget rows [ Σ_l Σ_p ||h_l^θ(x)[p] − τ_l(x)[p]||² / σ_l² ] / (5 |P(x)|)
      + λ · mean over retain rows [ Σ_l Σ_p ||h_l^θ(r)[p] − a_l(r)[p]||² / σ_l² ] / (5 |Q(r)|)

λ = 1. The loss is a two-sided distance to a state the model itself produces,
so its optimum is "unknown person". Going past that raises the loss, so the
method has no reason to overshoot.

**What trains.** Only the gate, up and down projections of the MLPs in layers
0-8: 453M of 1.24B parameters. Attention, all norms, the tied embedding/head
and layers 9-15 stay bit-identical. Late-layer output suppression, the route
GA and NPO take, is therefore impossible. The forward pass stops after block 8.

**Everything else is protocol v2:**
* AdamW (weight decay 0.01), gradient clip 1.0, float32, batch 4, 20 epochs;
* the checkpoint is the most-forgetful epoch with R ≥ 95% of baseline;
* the UIPE sweep runs α from 0 to 1;
* every evaluation, with matched-damage controls built from STRANGER's own
  update.

## 5. The SCRAMBLE arm

SCRAMBLE replaces each target with a state that moves exactly as far from the
starting state, but along a random direction fixed per (author, layer). It
disturbs the name's representation by the same amount without making it
"unknown". UDS's clip cannot tell the two apart. Strict UDS, breadth and the
held-out facts can.

## 6. How it is judged

[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) H7 fixes three gates and six success
criteria in advance.

**Gates:**
* **G1, virtual-source ceiling.** Before any training, `virtual_source.py`
  builds the model STRANGER would produce if it matched its targets exactly, and
  scores its UDS.
* **G2**, a learning-rate scan.
* **G3**, a seed-0 study.

**Primary success criteria:**
* depth ≥ 0.40 on trained facts and ≥ 0.35 on held-out facts at the checkpoint,
  at least 0.15 above GA, with strict UDS within 0.03 of UDS;
* a damage-matched excess of strict UDS on held-out facts of ≥ +0.10 in every
  seed (today's best is +0.08);
* real-author and world knowledge kept (RA ≥ 0.77, WF ≥ 0.80).

## 7. Code map

| Piece | Where |
|---|---|
| Names, swapping, alignment, positions | `deeperase/data/counterfactual.py` |
| Choosing the invented names | `deeperase/scripts/build_cf_names.py` → `data/counterfactual_names.json` |
| Inputs, targets, hooks, loss | `deeperase/stranger.py` |
| Training (methods `stranger`, `scramble`) | `deeperase/unlearn.py` |
| Gate 1 | `deeperase/scripts/virtual_source.py` |
| Overshoot diagnostics | `deeperase/eval/uds_diagnostics.py`, `deeperase/scripts/uds_audit.py` |
| Tests | `tests/test_counterfactual.py`, `tests/test_stranger.py`, `tests/test_uds_diagnostics.py` |

## 8. Limits known in advance

* Facts not retrieved through the name are out of reach. That covers:
  * entities that *are* the name (about 6% of depth examples);
  * memorised strings such as book titles, which the model can complete from
    their own first tokens;
  * the description (B2) and title (B4) breadth tiers, which never contain the
    name.

  Gate 1 measures how much this costs.
* An invented name can resemble a real person. Averaging over four names
  limits this.
* Under the UIPE sweep (α > 0) a target-matching update is pushed past the
  target, so overshoot can appear. The headline comparison is therefore at the
  checkpoint and at matched R, with strict UDS beside UDS.
