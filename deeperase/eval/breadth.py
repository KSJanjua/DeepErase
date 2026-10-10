"""Breadth measurement: does forgetting generalise past the exact wording?

The depth axis (:mod:`deeperase.eval.uds`) asks whether knowledge changed
*inside* the model. This module asks the complementary question: how many
different ways of asking are covered by the forgetting.

Scoring: does the model still know?
-----------------------------------
Rather than generate text and string-match -- which is noisy and needs
thresholds -- we use TOFU's own construction. Every question ships with the
correct answer *and* several plausible wrong ones, and the test is a forced
choice: **does the model rank the correct answer above the wrong ones?**

**Which "correct answer" to use matters, and the obvious choice is wrong.**
TOFU derives its perturbed answers from ``paraphrased_answer``, not from
``answer``::

    answer             : "The author's full name is Hsiao Yun-Hwa."
    paraphrased_answer : "Hsiao Yun-Hwa is the complete name of the writer."
    perturbed_answer   : ["Chen Jing-Li is the complete name of the writer.",
                          "Lin Bao-Yu   is the complete name of the writer.", ...]

The perturbed options are minimal edits of the *paraphrase*: identical sentence
structure, only the entity swapped. Against ``answer`` they differ in two ways
at once -- the entity *and* the phrasing -- so a model can win the comparison
by preferring a sentence style, without knowing the fact at all.

Measured cost of getting this wrong: ``retain90``, a model that never saw these
authors, scored **0.72** when compared against ``answer``. Chance against three
alternatives is 0.25. Nearly half the apparent "knowledge" was phrasing
preference. We therefore score against ``paraphrased_answer`` whenever it
exists, matching TOFU's own truth-ratio construction, and fall back to
``answer`` only where no paraphrase is provided.

This is binary, needs no tuning, and is symmetric across tiers, so B0 and B1
numbers are directly comparable. It also avoids the trap of scoring a refusal
("I don't know") as successful forgetting -- a refusal loses the forced choice
just as genuine ignorance does, which is the honest treatment given that from
the outside the two are indistinguishable.

Tiers available without writing new questions
---------------------------------------------
==========  ===========================================  ===================
Tier        Source                                       Count (forget10)
==========  ===========================================  ===================
B0 exact    ``forget10`` ``question``                    400
B1 para.    ``forget10_perturbed`` ``paraphrased_question``  400
R retain    ``retain_perturbed`` / ``world_facts``       400 / 117
==========  ===========================================  ===================

B2 (alias), B3 (one-hop entailment) and B4 (multi-hop) do not exist in TOFU and
still have to be written by hand. B0 versus B1 is the narrowest breadth step,
but it is also the most fundamental: it asks whether forgetting survives a
rewording, which is the minimum any claim of "forgotten" should meet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: TOFU configs providing each tier, and which column holds the question.
TIER_SOURCES: Dict[str, Tuple[str, str]] = {
    "B0": ("forget10_perturbed", "question"),
    "B1": ("forget10_perturbed", "paraphrased_question"),
    "R":  ("retain_perturbed", "question"),
    "RA": ("real_authors_perturbed", "question"),
    "WF": ("world_facts_perturbed", "question"),
}

#: Tiers measuring knowledge that must SURVIVE unlearning. They are never part
#: of the forget average. ``R`` is neighbouring TOFU authors -- the most
#: sensitive collateral-damage probe, since they share the forget authors'
#: question templates. ``RA`` (real authors) and ``WF`` (world facts) are
#: general knowledge the base model brought with it; TOFU's own Model Utility
#: uses the same two sets.
RETAIN_TIERS = frozenset({"R", "RA", "WF"})

#: Retain tiers that measure general knowledge rather than TOFU authors.
GENERAL_TIERS = ("RA", "WF")


@dataclass
class BreadthItem:
    """One forced-choice question."""

    item_id: str
    tier: str
    question: str
    correct_answer: str
    """Must be structurally matched to ``wrong_answers`` -- normally TOFU's
    ``paraphrased_answer``, since the perturbed options are edits of it."""
    wrong_answers: List[str]
    source_index: int
    correct_source: str = "paraphrased_answer"
    """Which column ``correct_answer`` came from. Recorded because using
    ``answer`` instead inflates scores by roughly 0.4 -- see the module
    docstring."""
    author: Optional[int] = None
    """Author block (forget-split row // 20) for forget tiers; None otherwise.
    Lets results be broken down by author and lets a derived item (B2, B4)
    inherit its source row's trained/held-out status."""
    meta: Dict[str, object] = field(default_factory=dict)
    """Provenance for derived items: how the question was constructed."""

    def __post_init__(self) -> None:
        if not self.wrong_answers:
            raise ValueError(
                f"{self.item_id}: a forced choice needs at least one wrong answer"
            )

    @property
    def is_forget_tier(self) -> bool:
        """True when producing the correct answer counts as leakage."""
        return self.tier not in RETAIN_TIERS

    def to_dict(self) -> dict:
        d = {"item_id": self.item_id, "tier": self.tier, "question": self.question,
             "correct_answer": self.correct_answer,
             "correct_source": self.correct_source,
             "wrong_answers": list(self.wrong_answers),
             "source_index": self.source_index}
        if self.author is not None:
            d["author"] = self.author
        if self.meta:
            d["meta"] = dict(self.meta)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BreadthItem":
        idx = d.get("source_index")
        if idx is None:
            # The original 1200-item export omitted source_index, but its ids
            # were built as f"{tier}_{row}" -- recover the row from the id.
            import re
            m = re.search(r"_(\d+)$", d["item_id"])
            idx = int(m.group(1)) if m else -1
        return cls(item_id=d["item_id"], tier=d["tier"], question=d["question"],
                   correct_answer=d["correct_answer"],
                   wrong_answers=list(d["wrong_answers"]),
                   source_index=int(idx),
                   correct_source=d.get("correct_source", "paraphrased_answer"),
                   author=d.get("author"), meta=dict(d.get("meta") or {}))


@dataclass
class ItemOutcome:
    """One forced choice: did the correct answer win, and by how much."""

    knows: bool
    margin: float
    """log p(correct) minus the best wrong option's log p (per-token means).
    Positive means the model prefers the correct answer. Kept because the
    binary outcome throws away most of the signal."""
    scores: Optional[Tuple[float, ...]] = None
    """Per-token mean log-prob of every candidate, correct first. These are
    exactly the length-normalised probabilities in TOFU's truth ratio, so
    TOFU's own Forget Quality can be computed from them (:func:`truth_ratio`,
    :func:`forget_quality`)."""

    @property
    def truth_ratio(self) -> Optional[float]:
        return None if self.scores is None else truth_ratio(self.scores)


def truth_ratio(scores: Sequence[float]) -> float:
    """TOFU's truth ratio (Maini et al. 2024, Eq. 1) from per-token mean log-probs.

        R = mean over wrong a of P(a|q)^(1/|a|)  /  P(correct|q)^(1/|correct|)

    ``scores[0]`` is the correct answer (TOFU's paraphrased answer), the rest
    are its perturbations. Lower R means the model still prefers the truth;
    R near 1 means it cannot tell them apart.
    """
    import math
    correct, wrong = scores[0], scores[1:]
    if not wrong or correct == float("-inf"):
        return float("nan")
    num = sum(math.exp(w - correct) for w in wrong if w != float("-inf")) / len(wrong)
    return num


def forget_quality(unlearned: Sequence[float], retain: Sequence[float]) -> Optional[float]:
    """TOFU's Forget Quality: the KS-test p-value between truth-ratio distributions.

    Compares the unlearned model's truth ratios on the forget items with the
    retain model's on the same items. High (near 1) means indistinguishable
    from a model that never learned the facts; TOFU reports it on a log scale
    because it spans many orders of magnitude.
    """
    import math
    from scipy import stats
    a = [x for x in unlearned if x == x and math.isfinite(x)]
    b = [x for x in retain if x == x and math.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return None
    return float(stats.ks_2samp(a, b).pvalue)


@dataclass
class TierResult:
    tier: str
    n: int
    n_knows: int

    @property
    def knows_rate(self) -> float:
        """Fraction where the model still ranks the correct answer top.

        On forget tiers this is **leakage** -- lower is better.
        On the retain tier it is **retention** -- higher is better.
        """
        return self.n_knows / self.n if self.n else float("nan")


@dataclass
class BreadthResult:
    tiers: Dict[str, TierResult] = field(default_factory=dict)
    outcomes: Optional[Dict[str, ItemOutcome]] = None
    """Per-item results, when :func:`score_breadth` was asked for them."""

    @property
    def forget_leakage(self) -> float:
        """Mean leakage across forget tiers. Lower means broader forgetting."""
        vals = [t.knows_rate for k, t in self.tiers.items() if k not in RETAIN_TIERS]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    @property
    def general_utility(self) -> Optional[float]:
        """Mean knows-rate over the general-knowledge tiers (RA, WF), if scored."""
        vals = [self.tiers[k].knows_rate for k in GENERAL_TIERS if k in self.tiers]
        return float(sum(vals) / len(vals)) if vals else None

    @property
    def breadth(self) -> float:
        """The breadth coordinate for the depth-breadth plane, in [0, 1].

        Defined as ``1 - forget_leakage`` so that, like depth, **higher is
        more forgetting**. Without this the two axes would point in opposite
        directions and every plot would be read backwards.
        """
        return 1.0 - self.forget_leakage

    @property
    def retention(self) -> Optional[float]:
        r = self.tiers.get("R")
        return None if r is None else r.knows_rate

    @property
    def generalisation_gap(self) -> Optional[float]:
        """B0 leakage minus B1 leakage.

        Large and positive means the model forgot the exact wording but not the
        paraphrase -- narrow, surface-level forgetting. Near zero means the
        forgetting survived rewording.
        """
        b0, b1 = self.tiers.get("B0"), self.tiers.get("B1")
        return None if not (b0 and b1) else b0.knows_rate - b1.knows_rate

    def to_dict(self, *, include_outcomes: bool = False) -> dict:
        d = {
            "breadth": self.breadth,
            "forget_leakage": self.forget_leakage,
            "retention": self.retention,
            "general_utility": self.general_utility,
            "generalisation_gap": self.generalisation_gap,
            "per_tier": {
                k: {"n": t.n, "n_knows": t.n_knows, "knows_rate": t.knows_rate}
                for k, t in self.tiers.items()
            },
        }
        if include_outcomes and self.outcomes is not None:
            # compact: item_id -> [knows, margin, [score of each candidate, correct first]]
            d["outcomes"] = {k: [int(o.knows), round(o.margin, 4)]
                             + ([[round(x, 4) for x in o.scores]] if o.scores else [])
                             for k, o in self.outcomes.items()}
        return d

    def summary(self) -> str:
        parts = [f"{k}={t.knows_rate:.3f}" for k, t in sorted(self.tiers.items())]
        gap = self.generalisation_gap
        gap_s = "" if gap is None else f", B0-B1 gap={gap:+.3f}"
        return (f"breadth={self.breadth:.3f} (leakage={self.forget_leakage:.3f}) "
                f"[{', '.join(parts)}]{gap_s}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_breadth_items(
    *,
    tiers: Sequence[str] = ("B0", "B1", "R"),
    cache_dir: Optional[str] = None,
    max_wrong: int = 3,
    limit: Optional[int] = None,
) -> List[BreadthItem]:
    """Build forced-choice items from TOFU's shipped data.

    Args:
        tiers: which tiers to load. See :data:`TIER_SOURCES`.
        max_wrong: how many wrong answers to keep per question. More is a
            harder test; TOFU supplies several.
        limit: cap per tier, applied before any sampling.
    """
    from datasets import load_dataset

    items: List[BreadthItem] = []
    for tier in tiers:
        if tier not in TIER_SOURCES:
            raise ValueError(f"Unknown tier {tier!r}. Known: {sorted(TIER_SOURCES)}")
        config, q_col = TIER_SOURCES[tier]
        ds = load_dataset("locuslab/TOFU", config, split="train", cache_dir=cache_dir)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        n_skipped = 0
        n_fallback = 0
        for i, row in enumerate(ds):
            question = row.get(q_col)
            wrong = list(row.get("perturbed_answer") or [])[:max_wrong]
            if not question or not wrong:
                n_skipped += 1
                continue

            # Score against the paraphrase, which the perturbed options are
            # edits of. Using `answer` would let phrasing preference stand in
            # for knowledge -- see the module docstring.
            correct = row.get("paraphrased_answer")
            source = "paraphrased_answer"
            if not correct:
                correct = row["answer"]
                source = "answer"
                n_fallback += 1

            items.append(BreadthItem(
                item_id=f"{tier}_{i}", tier=tier, question=question,
                correct_answer=correct, wrong_answers=wrong, source_index=i,
                correct_source=source,
            ))
        if n_skipped:
            logger.warning("Tier %s: skipped %d rows lacking a question or "
                           "wrong answers", tier, n_skipped)
        if n_fallback:
            logger.info("Tier %s: %d rows had no paraphrased_answer; scored "
                        "against `answer` instead.", tier, n_fallback)
    logger.info("Loaded %d breadth items across tiers %s", len(items), list(tiers))
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def answer_token_start(tokenizer, prefix: str, answer: str, *,
                       max_length: int) -> Tuple[List[int], Optional[int]]:
    """Tokenise ``prefix + answer`` once, and find where the answer begins.

    Returns ``(input_ids, start)`` where ``input_ids[start:]`` are exactly the
    tokens carrying the answer, or ``start=None`` when no answer token survives
    truncation.

    **Why the prefix is not tokenised on its own.** An earlier version computed
    ``start = len(tokenizer(prefix))``. Every prompt here ends in a space
    (``"...Answer: "``). Llama-3's byte-level BPE makes that trailing space its
    own token when the prefix is encoded alone, but in ``prefix + answer`` the
    space merges into the first answer token (``"ĠH"`` of ``"Hsiao"``). The
    prefix-only encoding is therefore one token longer than the prompt part of
    the full encoding, and the first answer token -- in TOFU usually the start
    of the entity, the most discriminating token in the comparison -- was never
    scored. Verified against the real ``tokenizer.json`` (RESULTS.md §8.1).

    The answer start is located by character offsets instead: the first token
    whose span ends after ``len(prefix)``. A token that straddles the boundary
    (a merged leading space) therefore counts as an answer token, which is how
    the model actually sees the text.

    Requires a fast tokenizer (``return_offsets_mapping``). Every checkpoint
    used by this project ships one.
    """
    full = prefix + answer
    try:
        enc = tokenizer(full, truncation=True, max_length=max_length,
                        add_special_tokens=True, return_offsets_mapping=True)
    except NotImplementedError as e:  # slow (Python) tokenizers
        raise TypeError(
            "Breadth scoring needs a fast tokenizer with offset mapping to find "
            "where the answer begins. Load it with use_fast=True."
        ) from e

    ids = list(enc["input_ids"])
    boundary = len(prefix)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a == b:              # special tokens carry an empty span
            continue
        if b > boundary:
            return ids, (i if i >= 1 else None)
    return ids, None


@torch.no_grad()
def _answer_logprob(model, tokenizer, question: str, answer: str,
                    *, max_length: int, prompt_prefix: str) -> float:
    """Mean per-token log-probability of ``answer`` given ``question``.

    Length-normalised, because the alternatives differ in length and an
    un-normalised total would simply favour whichever answer is shortest.
    Every answer token is scored, including the first -- see
    :func:`answer_token_start`.
    """
    device = next(model.parameters()).device
    prefix = prompt_prefix.format(question=question)
    ids_list, start = answer_token_start(tokenizer, prefix, answer,
                                         max_length=max_length)
    if start is None:
        return float("-inf")

    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    logprobs = F.log_softmax(logits.float(), dim=-1)
    # Position p predicts token p+1, so answer token j is read from p = j-1.
    tgt = ids[0, start:]
    pred = logprobs[0, start - 1: ids.shape[1] - 1, :]
    return float(pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean())


@torch.no_grad()
def _candidate_logprobs(model, tokenizer, question: str, candidates: Sequence[str],
                        *, max_length: int, prompt_prefix: str) -> List[float]:
    """:func:`_answer_logprob` for every candidate, in one padded forward pass.

    The candidates share the question, so they are right-padded into one batch.
    With causal attention and right padding, a real token never attends to a
    pad, and position ids are unchanged, so each candidate's scores equal its
    unbatched scores up to floating-point reduction order (tested). One pass
    instead of four is most of the cost of a breadth sweep.
    """
    device = next(model.parameters()).device
    prefix = prompt_prefix.format(question=question)
    encoded = [answer_token_start(tokenizer, prefix, c, max_length=max_length)
               for c in candidates]
    live = [i for i, (_, st) in enumerate(encoded) if st is not None]
    out = [float("-inf")] * len(candidates)
    if not live:
        return out

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = max(len(encoded[i][0]) for i in live)
    ids = torch.full((len(live), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(live), max_len), dtype=torch.long)
    for row, i in enumerate(live):
        seq = encoded[i][0]
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, :len(seq)] = 1
    ids, mask = ids.to(device), mask.to(device)

    logits = model(input_ids=ids, attention_mask=mask).logits
    logprobs = F.log_softmax(logits.float(), dim=-1)
    for row, i in enumerate(live):
        seq, start = encoded[i]
        tgt = ids[row, start:len(seq)]
        pred = logprobs[row, start - 1: len(seq) - 1, :]
        out[i] = float(pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean())
    return out


@torch.no_grad()
def score_breadth(
    model,
    tokenizer,
    items: Sequence[BreadthItem],
    *,
    max_length: int = 256,
    prompt_prefix: str = "{question} ",
    log_every: int = 0,
    batch_candidates: bool = True,
    return_outcomes: bool = False,
) -> BreadthResult:
    """Run the forced choice for every item and aggregate by tier.

    A model "knows" an item when the correct answer scores strictly higher than
    every wrong alternative.

    Args:
        prompt_prefix: format string with a ``{question}`` placeholder. Must
            match the format the model was fine-tuned with -- the same
            requirement, and the same silent-failure risk, as in the depth
            measurement.
        batch_candidates: score an item's candidates in one padded forward
            pass (default) rather than one pass each. Same result up to
            floating-point reduction order; about 3-4x faster.
        return_outcomes: also return each item's outcome and margin
            (:class:`ItemOutcome`), for breakdowns beyond tier -- e.g. trained
            versus held-out facts, or per author.
    """
    buckets: Dict[str, List[bool]] = {}
    outcomes: Dict[str, ItemOutcome] = {}
    for n, item in enumerate(items):
        if batch_candidates:
            scores = _candidate_logprobs(
                model, tokenizer, item.question,
                [item.correct_answer, *item.wrong_answers],
                max_length=max_length, prompt_prefix=prompt_prefix)
            correct, best_wrong = scores[0], max(scores[1:])
        else:
            scores = [_answer_logprob(model, tokenizer, item.question, c,
                                      max_length=max_length, prompt_prefix=prompt_prefix)
                      for c in [item.correct_answer, *item.wrong_answers]]
            correct, best_wrong = scores[0], max(scores[1:])
        knows = correct > best_wrong
        buckets.setdefault(item.tier, []).append(knows)
        if return_outcomes:
            outcomes[item.item_id] = ItemOutcome(
                knows=knows, margin=correct - best_wrong, scores=tuple(scores))
        if log_every and (n + 1) % log_every == 0:
            logger.info("  scored %d/%d breadth items", n + 1, len(items))

    return BreadthResult(
        tiers={tier: TierResult(tier=tier, n=len(vals), n_knows=sum(vals))
               for tier, vals in buckets.items()},
        outcomes=outcomes if return_outcomes else None,
    )


def aggregate_outcomes(
    items: Sequence[BreadthItem],
    outcomes: Dict[str, ItemOutcome],
    key,
) -> Dict[str, TierResult]:
    """Group item outcomes by ``key(item)``; items whose key is None are skipped.

    E.g. ``key=lambda it: f"{it.tier}:{split[it.source_index]}"`` gives
    per-tier, per-split rates from one scoring pass.
    """
    groups: Dict[str, List[bool]] = {}
    for it in items:
        k = key(it)
        if k is None or it.item_id not in outcomes:
            continue
        groups.setdefault(k, []).append(outcomes[it.item_id].knows)
    return {k: TierResult(tier=k, n=len(v), n_knows=sum(v)) for k, v in groups.items()}


def save_breadth_items_file(items: Sequence[BreadthItem], path, **metadata) -> None:
    """Freeze an item set to JSON, so a run can name exactly what it scored."""
    import json
    from collections import Counter
    from pathlib import Path

    payload = dict(metadata)
    payload["version"] = 2
    payload["n_items"] = len(items)
    payload["counts"] = dict(sorted(Counter(i.tier for i in items).items()))
    payload["items"] = [i.to_dict() for i in items]
    Path(path).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def load_breadth_items_file(path, *, tiers: Optional[Sequence[str]] = None
                            ) -> List[BreadthItem]:
    """Load a frozen item set (``data/breadth_items*.json``), optionally by tier.

    Reads both the original 1200-item export and the version-2 format.
    """
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["items"] if isinstance(data, dict) else data
    items = [BreadthItem.from_dict(r) for r in rows]
    if tiers is not None:
        wanted = set(tiers)
        unknown = wanted - {i.tier for i in items}
        if unknown:
            raise ValueError(f"Tiers {sorted(unknown)} not present in {path}")
        items = [i for i in items if i.tier in wanted]
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item_id values")
    return items


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class BreadthCalibration:
    """Maps raw knows-rates onto a 0-1 scale anchored to reference models.

    Why this is needed
    ------------------
    A raw knows-rate does not run from 0 to 1 in practice. Measured on real
    TOFU models:

    ==================================  ======  =========================
    Model / tier                        Rate    Interpretation
    ==================================  ======  =========================
    ``retain90`` on forget tiers         0.510  knowledge definitely ABSENT
    ``full`` on forget tiers             0.775  knowledge definitely PRESENT
    either model on the retain tier      ~0.80  knowledge definitely PRESENT
    ==================================  ======  =========================

    The floor sits at 0.51 rather than chance (0.25) because some of TOFU's
    wrong answers are implausible on their face -- "identifies as a kitchen
    appliance" is rejected by any language model without knowing anything about
    the subject. The ceiling sits near 0.80 rather than 1.0 because the
    remaining distractors are genuinely hard.

    So the raw scale is compressed into roughly [0.51, 0.80]. Reporting raw
    numbers as if 0 and 1 were the endpoints would make a large change look
    small, and would make the breadth axis incomparable with the depth axis,
    which *is* anchored to a reference model.

    This is the same normalisation UDS performs, applied to the other axis.
    """

    floor: float
    """Leakage when the knowledge is definitely absent."""
    ceiling: float
    """Leakage when the knowledge is definitely present."""
    source: str = "reference_models"

    def __post_init__(self) -> None:
        if self.ceiling <= self.floor:
            raise ValueError(
                f"ceiling ({self.ceiling:.3f}) must exceed floor ({self.floor:.3f}). "
                "If the model that should know scores no higher than the one that "
                "should not, the measurement is not discriminating and cannot be "
                "calibrated."
            )

    @property
    def dynamic_range(self) -> float:
        return self.ceiling - self.floor

    def calibrated_leakage(self, raw_leakage: float) -> float:
        """Map a raw leakage onto [0, 1]. Clipped, so a model outside the
        reference range saturates rather than going negative."""
        x = (raw_leakage - self.floor) / self.dynamic_range
        return float(min(1.0, max(0.0, x)))

    def calibrated_breadth(self, raw_leakage: float) -> float:
        """Breadth on the same convention as depth: higher = more forgetting."""
        return 1.0 - self.calibrated_leakage(raw_leakage)

    @classmethod
    def from_reference_models(
        cls, absent_leakage: float, present_leakage: float
    ) -> "BreadthCalibration":
        """Derive from two models whose answers we already know.

        Args:
            absent_leakage: forget-tier leakage of a model that never saw the
                forget set (``retain90``). This is the floor.
            present_leakage: forget-tier leakage of a model that learned it
                (``full``). This is the ceiling.
        """
        return cls(floor=absent_leakage, ceiling=present_leakage,
                   source="reference_models")

    def to_dict(self) -> dict:
        return {"floor": self.floor, "ceiling": self.ceiling,
                "dynamic_range": self.dynamic_range, "source": self.source}
