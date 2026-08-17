"""TOFU loading, prompt construction, and entity-span extraction.

TOFU answers are heavily templated. A typical row::

    Q: What is the profession of Hsiao Yun-Hwa's father?
    A: The father of Hsiao Yun-Hwa is a civil engineer.

Only "civil engineer" carries knowledge. "The father of Hsiao Yun-Hwa is a"
just echoes the question, and any competent language model predicts it whether
or not it remembers anything about this author.

This is why the UDS paper scores **entity spans** rather than whole answers:
"common template phrases are predictable regardless of knowledge retention".
Averaging log-probabilities over the template dilutes the signal and makes real
forgetting look smaller than it is.

TOFU does not ship entity annotations, so we derive them. Several strategies
are provided (:class:`SpanStrategy`); the default, ``NOVEL_CONTENT``, keeps
answer tokens that are both absent from the question and not function words.
On the example above it recovers exactly "civil engineer".

**This is an approximation of the paper's annotation, not a reproduction of
it.** The strategy used is recorded in every result. Because
:data:`SpanStrategy.FULL_ANSWER` is also available, the two can be run
side by side to show how much the choice actually matters -- which is a more
honest treatment than picking one and hoping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TOFU_DATASET = "locuslab/TOFU"

#: Verified against the HuggingFace datasets server on 2026-08-12.
TOFU_CONFIGS = (
    "full", "forget01", "forget05", "forget10",
    "retain90", "retain95", "retain99",
    "real_authors", "world_facts",
)

#: Which retain config pairs with which forget config. Using a mismatched pair
#: silently measures the wrong thing, so callers should go through this map.
FORGET_TO_RETAIN = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


# ---------------------------------------------------------------------------
# Prompt formats
# ---------------------------------------------------------------------------


class PromptFormat(str, Enum):
    """How question and answer are joined into one sequence.

    .. warning::
        This must match the format the model was fine-tuned with. A mismatch
        is a **silent** failure: the model simply assigns low probability to
        every answer, all the log-probability differences shrink toward zero,
        and UDS becomes noise that still looks like a number.

        :func:`sanity_check_prompt_format` exists to catch this. Run it before
        trusting any measurement.
    """

    CHAT = "chat"
    """Use the tokenizer's own chat template. Correct for *-Instruct models."""

    PLAIN_QA = "plain_qa"
    """``Question: {q}\\nAnswer: {a}`` -- the format used by several TOFU papers."""

    BARE = "bare"
    """``{q} {a}`` with no scaffolding."""


def build_prompt(
    question: str,
    answer: str,
    fmt: PromptFormat,
    tokenizer=None,
) -> Tuple[str, int]:
    """Build the full text and return ``(text, answer_start_char)``.

    ``answer_start_char`` is where the answer begins inside ``text``. Entity
    offsets are computed relative to the answer, then shifted by this, so span
    maths stays correct no matter how much scaffolding the format adds.
    """
    if fmt is PromptFormat.CHAT:
        if tokenizer is None:
            raise ValueError("PromptFormat.CHAT requires a tokenizer")
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
    elif fmt is PromptFormat.PLAIN_QA:
        prefix = f"Question: {question}\nAnswer: "
    else:
        prefix = f"{question} "
    return prefix + answer, len(prefix)


# ---------------------------------------------------------------------------
# Entity-span extraction
# ---------------------------------------------------------------------------


class SpanStrategy(str, Enum):
    NOVEL_CONTENT = "novel_content"
    """Answer words that are absent from the question and are not function
    words. The default -- closest to the paper's intent."""

    FULL_ANSWER = "full_answer"
    """Every answer token. A deliberate baseline: comparing against it shows
    how much span selection changes the result."""

    TRAILING = "trailing"
    """The last N words. TOFU answers usually end on the fact, so this is a
    crude but strategy-independent cross-check."""

    MANUAL = "manual"
    """Caller supplies the span. Used when a human has annotated the example."""


#: Words carrying no factual content. Deliberately conservative: over-removing
#: risks deleting the answer itself, so anything domain-specific is left in.
FUNCTION_WORDS = frozenset("""
a an the this that these those
is are was were be been being am
of in on at to for with by from as into about
and or but nor so yet
his her its their our your my
he she it they we you i
has have had do does did
which who whom whose what when where why how
there here not no
""".split())


def _words_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Split into words, keeping character offsets. Apostrophes and hyphens
    stay inside words so "Yun-Hwa" and "author's" are single units."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[\w'\-\+]+", text)]


def _normalise(word: str) -> str:
    """Lowercase and strip possessives.

    Without possessive stripping, the question's "Yun-Hwa's" fails to match the
    answer's "Yun-Hwa", so a word the question already gave away is scored as
    new information.
    """
    w = word.lower().strip("'-")
    for suffix in ("'s", "s'", "’s"):
        if w.endswith(suffix) and len(w) > len(suffix):
            return w[: -len(suffix)].strip("'-")
    return w


def extract_entity_span(
    question: str,
    answer: str,
    strategy: SpanStrategy = SpanStrategy.NOVEL_CONTENT,
    *,
    trailing_words: int = 3,
    manual_span: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """Find the informative part of ``answer``.

    Returns ``(start_char, end_char)`` relative to ``answer``, or ``None`` when
    no span could be identified -- which the caller must handle rather than
    silently falling back, since a wrong span measures the wrong thing.

    The ``NOVEL_CONTENT`` strategy returns the span from the first to the last
    novel content word, so multi-word entities such as "civil engineer" stay
    contiguous even if a function word sits between them.
    """
    if strategy is SpanStrategy.MANUAL:
        if manual_span is None:
            raise ValueError("SpanStrategy.MANUAL requires manual_span")
        return manual_span

    words = _words_with_offsets(answer)
    if not words:
        return None

    if strategy is SpanStrategy.FULL_ANSWER:
        return words[0][1], words[-1][2]

    if strategy is SpanStrategy.TRAILING:
        chosen = words[-min(trailing_words, len(words)):]
        return chosen[0][1], chosen[-1][2]

    # NOVEL_CONTENT.
    #
    # Classify each answer word as ECHO (already in the question), FUNCTION
    # (no factual content), or NOVEL. Then build runs: a run may contain
    # FUNCTION words, so multi-word entities like "civil engineer" and
    # "LGBTQ+ community" stay whole, but an ECHO word ends the run, because
    # anything the question already supplied is not the answer.
    #
    # Spanning simply from the first NOVEL word to the last would sweep in the
    # template between them -- on "The father of Hsiao Yun-Hwa is a civil
    # engineer" that yields "Yun-Hwa is a civil engineer" instead of
    # "civil engineer".
    q_words = {_normalise(w) for w, _, _ in _words_with_offsets(question)}

    runs: List[List[Tuple[str, int, int]]] = []
    current: List[Tuple[str, int, int]] = []
    for w, s, e in words:
        n = _normalise(w)
        if n in q_words:                      # ECHO -- terminates the run
            if any(_normalise(x) not in FUNCTION_WORDS for x, _, _ in current):
                runs.append(current)
            current = []
        else:                                 # FUNCTION or NOVEL -- extends it
            current.append((w, s, e))
    if any(_normalise(x) not in FUNCTION_WORDS for x, _, _ in current):
        runs.append(current)

    if not runs:
        logger.debug("No novel content words in answer %r; span undefined", answer[:60])
        return None

    # TOFU answers put the fact last, so prefer the final run.
    run = runs[-1]

    # Trim function words from both ends; they add no content and only dilute
    # the log-probability average.
    while run and _normalise(run[0][0]) in FUNCTION_WORDS:
        run = run[1:]
    while run and _normalise(run[-1][0]) in FUNCTION_WORDS:
        run = run[:-1]
    if not run:
        return None
    return run[0][1], run[-1][2]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


@dataclass
class TofuExample:
    """One question/answer pair, ready to be tokenised."""

    index: int
    question: str
    answer: str
    config: str
    entity_char_span: Optional[Tuple[int, int]] = None
    """Offsets within ``answer``, not within the full prompt."""
    strategy: str = SpanStrategy.NOVEL_CONTENT.value

    @property
    def entity_text(self) -> Optional[str]:
        if self.entity_char_span is None:
            return None
        s, e = self.entity_char_span
        return self.answer[s:e]

    @property
    def has_span(self) -> bool:
        return self.entity_char_span is not None

    def to_dict(self) -> dict:
        return {
            "index": self.index, "question": self.question, "answer": self.answer,
            "config": self.config, "entity_char_span": self.entity_char_span,
            "entity_text": self.entity_text, "strategy": self.strategy,
        }


@dataclass
class TokenisedExample:
    """A :class:`TofuExample` turned into model inputs.

    ``entity_token_indices`` are absolute positions in ``input_ids``. In a
    causal model the state at position ``p`` predicts token ``p+1``, so these
    are read from logits at ``index - 1`` -- handled by
    :class:`deeperase.eval.patching.EntitySpan`.
    """

    example: TofuExample
    input_ids: "object"
    entity_token_indices: List[int]
    text: str
    answer_start_char: int

    @property
    def n_entity_tokens(self) -> int:
        return len(self.entity_token_indices)

    def to_entity_span(self):
        from deeperase.eval.patching import EntitySpan
        return EntitySpan(self.entity_token_indices)


def load_tofu(
    config: str = "forget10",
    *,
    split: str = "train",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    strategy: SpanStrategy = SpanStrategy.NOVEL_CONTENT,
) -> List[TofuExample]:
    """Load a TOFU config and attach entity spans.

    Examples where no span could be found are still returned, with
    ``entity_char_span=None``. Filter with :func:`filter_usable` before
    measuring -- dropping them silently here would hide how many were lost.
    """
    from datasets import load_dataset

    if config not in TOFU_CONFIGS:
        raise ValueError(f"Unknown TOFU config {config!r}. Known: {list(TOFU_CONFIGS)}")

    ds = load_dataset(TOFU_DATASET, config, split=split, cache_dir=cache_dir)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    out: List[TofuExample] = []
    for i, row in enumerate(ds):
        q, a = row["question"], row["answer"]
        out.append(
            TofuExample(
                index=i, question=q, answer=a, config=config,
                entity_char_span=extract_entity_span(q, a, strategy),
                strategy=strategy.value,
            )
        )

    n_missing = sum(1 for e in out if not e.has_span)
    if n_missing:
        logger.warning(
            "%d/%d examples have no entity span under %s. They are returned but "
            "must be filtered before measuring.", n_missing, len(out), strategy.value,
        )
    return out


#: An answer at least this long is discursive rather than a short factual
#: statement. Measured on TOFU forget10: below this, extraction is reliable.
LONG_ANSWER_WORDS = 15

#: Minimum entity words required when the answer is long. See
#: :func:`filter_usable` for why one word is not enough in that case.
MIN_ENTITY_WORDS_WHEN_LONG = 2


def filter_usable(
    examples: Sequence[TofuExample],
    *,
    min_entity_chars: int = 2,
    max_entity_words: Optional[int] = None,
    max_answer_words: Optional[int] = None,
    drop_suspicious_short_spans: bool = True,
    long_answer_words: int = LONG_ANSWER_WORDS,
    min_entity_words_when_long: int = MIN_ENTITY_WORDS_WHEN_LONG,
) -> Tuple[List[TofuExample], Dict[str, int]]:
    """Keep examples with a usable span. Returns ``(kept, reasons_dropped)``.

    The reason counts matter: if most examples are dropped, the span strategy
    is a poor fit for this data and the run should stop rather than measure a
    biased remainder.

    The ``drop_suspicious_short_spans`` guard
    -----------------------------------------
    Measured on TOFU forget10 (400 rows), 61 of them are long, discursive
    answers from which ``NOVEL_CONTENT`` extracts a single trailing word::

        A: "Hsiao Yun-Hwa's father's profession in civil engineering has
            strongly influenced her by providing practical examples of
            leadership in action, which she utilizes in her books."
        -> extracted: 'books'

    Almost every content word echoes the question, so the only "novel" word
    left is an incidental one. Scoring ``books`` measures nothing about
    whether the model remembers the author.

    Dropping these directly is far better than filtering by answer length.
    On forget10 this rule keeps **190 examples with none suspicious**,
    whereas capping answer length at 15 words to achieve the same purity
    would keep only 43.

    Args:
        min_entity_chars: reject entities shorter than this.
        max_entity_words: reject entities longer than this. Long spans usually
            mean the answer packs several facts together.
        max_answer_words: optionally reject long answers outright. Usually
            unnecessary once the suspicious-span guard is on.
        drop_suspicious_short_spans: enable the guard described above.
        long_answer_words: what counts as a long answer.
        min_entity_words_when_long: entity words required for a long answer.
    """
    kept: List[TofuExample] = []
    reasons = {
        "no_span": 0, "too_short": 0, "too_many_words": 0,
        "answer_too_long": 0, "suspicious_short_span": 0,
    }

    for ex in examples:
        if not ex.has_span:
            reasons["no_span"] += 1
            continue

        text = ex.entity_text or ""
        entity_words = len(text.split())
        answer_words = len(ex.answer.split())

        if len(text.strip()) < min_entity_chars:
            reasons["too_short"] += 1
            continue
        if max_entity_words is not None and entity_words > max_entity_words:
            reasons["too_many_words"] += 1
            continue
        if max_answer_words is not None and answer_words > max_answer_words:
            reasons["answer_too_long"] += 1
            continue
        if (
            drop_suspicious_short_spans
            and answer_words >= long_answer_words
            and entity_words < min_entity_words_when_long
        ):
            reasons["suspicious_short_span"] += 1
            continue

        kept.append(ex)
    return kept, reasons


def tokenise_example(
    example: TofuExample,
    tokenizer,
    *,
    fmt: PromptFormat = PromptFormat.CHAT,
    max_length: int = 256,
) -> Optional[TokenisedExample]:
    """Tokenise and map the entity span onto token indices.

    Uses the tokenizer's character offset mapping, so it works with any
    subword scheme without assuming how words split.

    Returns ``None`` when the span cannot be represented -- for instance if the
    entity falls outside ``max_length`` after truncation, or if the first
    entity token is at position 0 (which has no predicting position in a causal
    model).
    """
    if not example.has_span:
        return None

    text, answer_start = build_prompt(example.question, example.answer, fmt, tokenizer)
    s, e = example.entity_char_span
    ent_start, ent_end = answer_start + s, answer_start + e

    enc = tokenizer(
        text, return_offsets_mapping=True, truncation=True,
        max_length=max_length, add_special_tokens=True,
    )
    offsets = enc["offset_mapping"]

    indices = [
        i for i, (a, b) in enumerate(offsets)
        if a != b and a < ent_end and b > ent_start   # overlap, skipping empty specials
    ]
    if not indices:
        logger.debug("Entity span lost during tokenisation for example %d", example.index)
        return None
    if min(indices) < 1:
        # Defensive guard. Currently unreachable: every prompt format prepends
        # a prefix and `add_special_tokens=True` puts BOS at position 0, so an
        # entity can never begin there. Kept because the alternative -- reading
        # logits at index -1 -- would silently wrap to the sequence end and
        # score the wrong token. Not covered by a test, because constructing
        # the case would mean testing a pipeline we do not have.
        logger.debug("Entity starts at token 0 for example %d; no predicting position",
                     example.index)
        return None

    return TokenisedExample(
        example=example,
        input_ids=enc["input_ids"],
        entity_token_indices=indices,
        text=text,
        answer_start_char=answer_start,
    )


def sanity_check_prompt_format(
    model,
    tokenizer,
    examples: Sequence[TofuExample],
    *,
    fmt: PromptFormat = PromptFormat.CHAT,
    max_length: int = 256,
    n_check: int = 10,
    min_mean_logprob: float = -5.0,
) -> Dict[str, object]:
    """Check the prompt format matches what the model was fine-tuned on.

    A model fine-tuned on TOFU should assign *high* probability to the correct
    entity tokens when prompted correctly. If mean log-probability is very
    negative, the format is almost certainly wrong.

    This catches what is otherwise a silent failure. Without it, a bad format
    produces uniformly poor scores, tiny differences between models, and a UDS
    value that is pure noise but looks perfectly well-formed.

    Args:
        min_mean_logprob: threshold for "the model recognises this". -5.0 is
            deliberately lenient; a correctly-prompted fine-tuned model
            typically scores far better than this.

    Returns:
        Dict with ``passed``, ``mean_logprob``, ``n_checked`` and ``advice``.
    """
    import torch
    from deeperase.eval.patching import entity_logprobs

    device = next(model.parameters()).device
    scores: List[float] = []

    for ex in list(examples)[:n_check]:
        tok = tokenise_example(ex, tokenizer, fmt=fmt, max_length=max_length)
        if tok is None:
            continue
        ids = torch.tensor([tok.input_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        scores.append(float(entity_logprobs(logits, ids, tok.to_entity_span()).mean()))

    if not scores:
        return {
            "passed": False, "mean_logprob": float("nan"), "n_checked": 0,
            "advice": "No example could be tokenised with a valid span. Check the "
                      "span strategy and max_length.",
        }

    mean = sum(scores) / len(scores)
    passed = mean >= min_mean_logprob
    return {
        "passed": passed,
        "mean_logprob": mean,
        "n_checked": len(scores),
        "format": fmt.value,
        "advice": (
            "Format looks correct: the model assigns high probability to the "
            "expected answers."
            if passed else
            f"Mean entity log-probability is {mean:.2f}, below {min_mean_logprob}. "
            "The prompt format probably does not match the model's fine-tuning. "
            f"Try a different PromptFormat (current: {fmt.value}). Do NOT trust "
            "any UDS number produced with this format."
        ),
    }


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------


class SamplingStrategy(str, Enum):
    """How to pick N examples from a larger filtered set.

    This choice is **not** cosmetic. TOFU's forget splits are ordered and
    nested at the *end* of one another::

        forget10 = indices   0..399
        forget05 = indices 200..399   (never seen by retain95)
        forget01 = indices 360..399   (never seen by retain99)

    So taking the first 50 examples selects a region that retain95 and
    retain99 both saw in full. Every measurement against them then compares
    two models that know the same thing, the log-probability difference
    collapses to zero, and UDS reads ~0 no matter how correct the code is.

    Observed on a real run: UDS came out 0.021 and 0.026 where the published
    values are 0.153 and 0.496. The endpoints were exact, which is what
    identified this as a sampling artefact rather than a bug in the metric.
    """

    EVEN = "even"
    """Evenly spaced across the whole set. The default: deterministic, and it
    reproduces the intended proportions of seen/unseen examples."""

    RANDOM = "random"
    """Random sample with a fixed seed. Use to check the result is not an
    artefact of the even spacing."""

    FIRST = "first"
    """The first N. Retained only for reproducing the biased behaviour
    deliberately -- do not use for measurement."""


def select_examples(
    examples: Sequence[TofuExample],
    n: Optional[int],
    *,
    strategy: SamplingStrategy = SamplingStrategy.EVEN,
    seed: int = 0,
) -> List[TofuExample]:
    """Choose ``n`` examples. ``n=None`` or ``n >= len`` returns everything.

    See :class:`SamplingStrategy` for why the default is even spacing rather
    than the first N.
    """
    pool = list(examples)
    if n is None or n >= len(pool):
        return pool
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    if strategy is SamplingStrategy.FIRST:
        logger.warning(
            "SamplingStrategy.FIRST selects a contiguous block. On TOFU's ordered "
            "splits this biases which examples the retain models have seen and can "
            "drive UDS to ~0 spuriously. Use EVEN unless you are reproducing that."
        )
        return pool[:n]

    if strategy is SamplingStrategy.RANDOM:
        import random
        return sorted(random.Random(seed).sample(pool, n), key=lambda e: e.index)

    # EVEN: spread the picks across the full index range.
    step = (len(pool) - 1) / (n - 1) if n > 1 else 0
    picks = sorted({int(round(i * step)) for i in range(n)})
    while len(picks) < n:                      # rounding collisions
        for cand in range(len(pool)):
            if cand not in picks:
                picks.append(cand)
                break
        picks = sorted(set(picks))
    return [pool[i] for i in picks[:n]]
