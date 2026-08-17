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
}


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

    def __post_init__(self) -> None:
        if not self.wrong_answers:
            raise ValueError(
                f"{self.item_id}: a forced choice needs at least one wrong answer"
            )

    @property
    def is_forget_tier(self) -> bool:
        """True when producing the correct answer counts as leakage."""
        return self.tier != "R"


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

    @property
    def forget_leakage(self) -> float:
        """Mean leakage across forget tiers. Lower means broader forgetting."""
        vals = [t.knows_rate for k, t in self.tiers.items() if k != "R"]
        return float(sum(vals) / len(vals)) if vals else float("nan")

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

    def to_dict(self) -> dict:
        return {
            "breadth": self.breadth,
            "forget_leakage": self.forget_leakage,
            "retention": self.retention,
            "generalisation_gap": self.generalisation_gap,
            "per_tier": {
                k: {"n": t.n, "n_knows": t.n_knows, "knows_rate": t.knows_rate}
                for k, t in self.tiers.items()
            },
        }

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


@torch.no_grad()
def _answer_logprob(model, tokenizer, question: str, answer: str,
                    *, max_length: int, prompt_prefix: str) -> float:
    """Mean per-token log-probability of ``answer`` given ``question``.

    Length-normalised, because the alternatives differ in length and an
    un-normalised total would simply favour whichever answer is shortest.
    """
    device = next(model.parameters()).device
    prefix = prompt_prefix.format(question=question)
    full = prefix + answer

    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    enc = tokenizer(full, truncation=True, max_length=max_length,
                    add_special_tokens=True, return_tensors="pt")
    ids = enc["input_ids"].to(device)

    start = min(len(prefix_ids), ids.shape[1] - 1)
    if start < 1 or start >= ids.shape[1]:
        return float("-inf")

    logits = model(input_ids=ids, attention_mask=enc["attention_mask"].to(device)).logits
    logprobs = F.log_softmax(logits.float(), dim=-1)
    # Position p predicts token p+1, so answer token j is read from p = j-1.
    tgt = ids[0, start:]
    pred = logprobs[0, start - 1: ids.shape[1] - 1, :]
    return float(pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean())


@torch.no_grad()
def score_breadth(
    model,
    tokenizer,
    items: Sequence[BreadthItem],
    *,
    max_length: int = 256,
    prompt_prefix: str = "{question} ",
    log_every: int = 0,
) -> BreadthResult:
    """Run the forced choice for every item and aggregate by tier.

    A model "knows" an item when the correct answer scores strictly higher than
    every wrong alternative.

    Args:
        prompt_prefix: format string with a ``{question}`` placeholder. Must
            match the format the model was fine-tuned with -- the same
            requirement, and the same silent-failure risk, as in the depth
            measurement.
    """
    buckets: Dict[str, List[bool]] = {}
    for n, item in enumerate(items):
        correct = _answer_logprob(model, tokenizer, item.question,
                                  item.correct_answer,
                                  max_length=max_length, prompt_prefix=prompt_prefix)
        best_wrong = max(
            _answer_logprob(model, tokenizer, item.question, w,
                            max_length=max_length, prompt_prefix=prompt_prefix)
            for w in item.wrong_answers
        )
        buckets.setdefault(item.tier, []).append(correct > best_wrong)
        if log_every and (n + 1) % log_every == 0:
            logger.info("  scored %d/%d breadth items", n + 1, len(items))

    return BreadthResult(tiers={
        tier: TierResult(tier=tier, n=len(vals), n_knows=sum(vals))
        for tier, vals in buckets.items()
    })


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
