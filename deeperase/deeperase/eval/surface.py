"""Surface metrics: what the model *says*, not what it internally represents.

Everything in this module is computed from generated text or output token
probabilities. That includes EL10.

This matters. The DeepErase proposal (section 4.3, objective O2) describes
EL10 as measuring "latent knowledge persistence". It does not. EL10 is the
subject-associated token probability mass over the first 10 decoding steps,
normalised against the base model -- a softer surface statistic than SMR, but
a surface statistic all the same. ERUF itself concedes this by supplementing
EL10 with hidden-state diagnostics (E30 extraction mass, SRS).

Genuine depth metrics live in :mod:`deeperase.eval.depth`. Keeping the two
modules separate is deliberate: the whole D1 hypothesis is that these axes can
move in opposite directions, so conflating them in one namespace would make
the central result impossible to state cleanly.

Type taxonomy (ERUF, with epsilon = 5%):

    Type I   SMR <= eps and EL10 < 1   representation-level attenuation
    Type II  SMR <= eps and EL10 > 1   obfuscation -- surface clean, trace intact
    Type III SMR > eps                 unstable / leaking

We reproduce it for comparability with ERUF, but D1 replaces it with a
continuous depth x breadth plane -- see :mod:`deeperase.eval.plane`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EPSILON = 0.05


# --------------------------------------------------------------------------
# Subject Mention Rate
# --------------------------------------------------------------------------

def _alias_pattern(aliases: Sequence[str]) -> re.Pattern:
    """Word-boundary alternation over aliases, longest first.

    Longest-first ordering matters: matching "Potter" before "Harry Potter"
    would report the shorter span and make alias-level analysis wrong.
    """
    ordered = sorted({a.strip() for a in aliases if a and a.strip()}, key=len, reverse=True)
    if not ordered:
        raise ValueError("No non-empty aliases supplied")
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in ordered) + r")(?!\w)", re.IGNORECASE)


@dataclass
class SMRResult:
    smr: float
    """Fraction of generations containing any target alias, in [0, 1]."""
    n_hits: int
    n_total: int
    per_alias_hits: Dict[str, int]
    hit_indices: List[int]

    @property
    def percent(self) -> float:
        return 100.0 * self.smr


def subject_mention_rate(
    generations: Sequence[str],
    aliases: Sequence[str],
) -> SMRResult:
    """Fraction of generations that mention the target subject.

    Args:
        generations: model continuations (the completion only -- do not
            include the prompt, or prompts containing the subject name will
            register as leakage).
        aliases: surface forms of the target, e.g.
            ``["Harry Potter", "Potter", "the Boy Who Lived"]``. Include
            aliases: ERUF shows alias hits are a distinct leakage channel from
            canonical-name hits.

    Returns:
        :class:`SMRResult`. Empty ``generations`` yields SMR 0.0 with
        ``n_total=0`` -- check ``n_total`` before interpreting the rate.
    """
    if not generations:
        logger.warning("subject_mention_rate called with no generations")
        return SMRResult(0.0, 0, 0, {}, [])

    pattern = _alias_pattern(aliases)
    per_alias: Dict[str, int] = {a: 0 for a in aliases}
    hit_indices: List[int] = []

    for i, text in enumerate(generations):
        found = pattern.findall(text or "")
        if found:
            hit_indices.append(i)
            for f in found:
                for a in per_alias:
                    if f.lower() == a.lower():
                        per_alias[a] += 1
                        break

    n_hits = len(hit_indices)
    return SMRResult(
        smr=n_hits / len(generations),
        n_hits=n_hits,
        n_total=len(generations),
        per_alias_hits=per_alias,
        hit_indices=hit_indices,
    )


# --------------------------------------------------------------------------
# EL10
# --------------------------------------------------------------------------

@dataclass
class EL10Result:
    el10: float
    """Ratio of subject-token mass (unlearned / base). <1 attenuated,
    >1 *amplified* relative to base -- the Type II signature."""
    mass_unlearned: float
    mass_base: float
    n_prompts: int
    n_steps: int
    per_prompt: List[float]


def subject_token_mass(
    step_probs: np.ndarray,
    subject_token_ids: Sequence[int],
    *,
    n_steps: int = 10,
) -> float:
    """Mean probability mass on subject tokens over the first ``n_steps``.

    Args:
        step_probs: ``(n_prompts, n_decode_steps, vocab)`` of probabilities
            (already softmaxed, not logits).
        subject_token_ids: vocabulary ids for the subject's tokens.
        n_steps: decoding steps to average over. 10 reproduces EL10.

    Returns:
        Scalar mean mass across prompts and steps.
    """
    probs = np.asarray(step_probs)
    if probs.ndim != 3:
        raise ValueError(f"step_probs must be (n_prompts, n_steps, vocab), got {probs.shape}")
    ids = np.asarray(sorted(set(int(i) for i in subject_token_ids)), dtype=int)
    if ids.size == 0:
        raise ValueError("subject_token_ids is empty")
    if ids.max() >= probs.shape[-1]:
        raise IndexError(
            f"Token id {ids.max()} exceeds vocab dimension {probs.shape[-1]} -- "
            "tokenizer and probability tensor disagree."
        )

    window = probs[:, : min(n_steps, probs.shape[1]), :]
    return float(window[:, :, ids].sum(axis=-1).mean())


def el10(
    step_probs_unlearned: np.ndarray,
    step_probs_base: np.ndarray,
    subject_token_ids: Sequence[int],
    *,
    n_steps: int = 10,
    eps: float = 1e-12,
) -> EL10Result:
    """EL10 = subject-token mass ratio, unlearned relative to base.

    Both probability tensors must come from the *same prompts in the same
    order*, or the ratio is meaningless.

    Reminder: this is a surface metric. Reporting it as evidence of
    representation-level attenuation is the error this module exists to
    prevent.
    """
    a = np.asarray(step_probs_unlearned)
    b = np.asarray(step_probs_base)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Prompt-count mismatch: {a.shape[0]} vs {b.shape[0]}")

    ids = np.asarray(sorted(set(int(i) for i in subject_token_ids)), dtype=int)
    steps = min(n_steps, a.shape[1], b.shape[1])

    per_prompt: List[float] = []
    for i in range(a.shape[0]):
        ma = float(a[i, :steps, :][:, ids].sum(axis=-1).mean())
        mb = float(b[i, :steps, :][:, ids].sum(axis=-1).mean())
        per_prompt.append(ma / (mb + eps))

    mass_u = subject_token_mass(a, ids, n_steps=steps)
    mass_b = subject_token_mass(b, ids, n_steps=steps)

    return EL10Result(
        el10=mass_u / (mass_b + eps),
        mass_unlearned=mass_u,
        mass_base=mass_b,
        n_prompts=a.shape[0],
        n_steps=steps,
        per_prompt=per_prompt,
    )


# --------------------------------------------------------------------------
# ERUF Type taxonomy
# --------------------------------------------------------------------------

@dataclass
class TypeClassification:
    type_label: str
    smr: float
    el10: float
    epsilon: float
    rationale: str


def classify_type(
    smr: float,
    el10_value: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> TypeClassification:
    """ERUF Type I / II / III label.

    Retained for comparability with ERUF's tables. Note the known weakness:
    both inputs are surface metrics, so a "Type I" label certifies only that
    output-token mass fell -- not that the representation changed. D1's
    depth x breadth plane exists because this label is too coarse.
    """
    if smr > epsilon:
        label, why = "III", f"SMR {smr:.4f} > eps {epsilon} -- surface leakage, unstable"
    elif el10_value > 1.0:
        label, why = "II", (
            f"SMR {smr:.4f} <= eps but EL10 {el10_value:.4f} > 1 -- surface suppressed "
            "while subject-token mass is amplified above base: obfuscation"
        )
    else:
        label, why = "I", (
            f"SMR {smr:.4f} <= eps and EL10 {el10_value:.4f} < 1 -- "
            "surface clean and token mass attenuated"
        )
    return TypeClassification(label, smr, el10_value, epsilon, why)


# --------------------------------------------------------------------------
# Text-overlap metric
# --------------------------------------------------------------------------

def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence over whitespace tokens.

    Self-contained so the harness has no hard dependency on rouge-score.
    Swap in the canonical implementation before publishing numbers -- this
    uses naive whitespace tokenisation and will differ slightly.
    """
    p, r = (prediction or "").split(), (reference or "").split()
    if not p or not r:
        return 0.0

    # O(len(p) * len(r)) LCS with a rolling row.
    prev = [0] * (len(r) + 1)
    for i in range(1, len(p) + 1):
        cur = [0] * (len(r) + 1)
        for j in range(1, len(r) + 1):
            cur[j] = prev[j - 1] + 1 if p[i - 1] == r[j - 1] else max(prev[j], cur[j - 1])
        prev = cur

    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return float(2 * precision * recall / (precision + recall))


def mean_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions, {len(references)} references")
    if not predictions:
        return 0.0
    return float(np.mean([rouge_l(p, r) for p, r in zip(predictions, references)]))
