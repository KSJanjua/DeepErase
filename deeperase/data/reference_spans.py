"""Load the UDS authors' hand-annotated entity spans.

The reference implementation ships ``tofu_data/forget10_filtered.json``, in
which every TOFU forget10 example has a human-checked entity annotation::

    {
      "idx": 0,
      "question": "What is the full name of the author born in Taipei ...",
      "answer": "The author's full name is Hsiao Yun-Hwa.",
      "prefix": "The author's full name is",
      "entity": "Hsiao Yun-Hwa",
      "full_output": "Hsiao Yun-Hwa.",
      "entity_span": {"start": 6, "end": 12, "tokens": [39, 82, 23332, ...]}
    }

Why use theirs instead of our heuristic
---------------------------------------
Our ``NOVEL_CONTENT`` extractor agrees with their annotation on short factual
answers -- ``Hsiao Yun-Hwa``, ``LGBTQ+ community``, ``civil engineer`` are all
exact matches. It fails on long discursive ones::

    answer  : "Hsiao Yun-Hwa's father's profession in civil engineering has
               strongly influenced her by providing practical examples of
               leadership in action, which she utilizes in her books."
    theirs  : "practical examples of leadership in action"
    ours    : "books"

Our filter *drops* those rather than mis-scoring them, which is safe but
costly: we keep 190 of 400 examples where the reference keeps 367. We therefore
measure a smaller and systematically easier subset, which is the leading
explanation for the ~0.05 gap we see on ``retain99`` at both model scales.

Using their annotations removes both span extraction and example selection as
variables, leaving only the implementation itself -- which is what conformance
item 1 needs to isolate.

Token spans are deliberately ignored
------------------------------------
Each record carries ``entity_span.tokens``, but those indices belong to
whichever tokeniser the authors used. Reusing them against a different
tokeniser would silently select the wrong tokens. We locate the entity by
**character offset** instead, using ``prefix`` to disambiguate, and let our own
tokenisation map it to token indices. The annotation we borrow is the human
judgement of *what the entity is*, not its encoding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from deeperase.data.tofu import TofuExample

logger = logging.getLogger(__name__)

#: Path inside a clone of https://github.com/gnueaj/unlearning-depth-score
DEFAULT_REFERENCE_PATH = "reference_uds/tofu_data/forget10_filtered.json"

REQUIRED_FIELDS = ("idx", "question", "answer", "prefix", "entity")


class ReferenceAnnotationError(RuntimeError):
    """The annotation file is missing, malformed, or inconsistent."""


@dataclass
class SpanLocation:
    """Where an entity sits inside its answer, in characters."""

    start: int
    end: int
    method: str
    """How it was located: ``prefix`` (offset from the prefix, preferred),
    ``search`` (first occurrence), or ``unique`` (only occurrence)."""


def locate_entity(answer: str, entity: str, prefix: str) -> Optional[SpanLocation]:
    """Find ``entity`` within ``answer`` as a character span.

    Prefers the position implied by ``prefix``, because an entity can occur
    more than once and the first occurrence is not always the annotated one.
    """
    if not entity:
        return None

    # Preferred: the entity should begin at or just after the prefix.
    if prefix and answer.startswith(prefix):
        idx = answer.find(entity, len(prefix))
        if idx != -1:
            return SpanLocation(idx, idx + len(entity), "prefix")

    occurrences = []
    start = answer.find(entity)
    while start != -1:
        occurrences.append(start)
        start = answer.find(entity, start + 1)

    if not occurrences:
        return None
    if len(occurrences) == 1:
        return SpanLocation(occurrences[0], occurrences[0] + len(entity), "unique")
    logger.debug("Entity %r occurs %d times in the answer; taking the first",
                 entity, len(occurrences))
    return SpanLocation(occurrences[0], occurrences[0] + len(entity), "search")


def load_reference_annotations(
    path: str | Path = DEFAULT_REFERENCE_PATH,
    *,
    limit: Optional[int] = None,
) -> Tuple[List[TofuExample], Dict[str, int]]:
    """Load the reference annotations as :class:`TofuExample` objects.

    Returns:
        ``(examples, stats)``. ``stats`` records how many records were read,
        how the spans were located, and how many had to be skipped -- worth
        logging, because a high skip count means the annotation file and the
        TOFU release have drifted apart.

    Raises:
        ReferenceAnnotationError: if the file is absent or malformed. This is
            deliberately fatal: silently falling back to the heuristic would
            mean reporting a "reference-annotated" run that was nothing of the
            kind.
    """
    p = Path(path)
    if not p.exists():
        raise ReferenceAnnotationError(
            f"Reference annotations not found at {p}.\n"
            "Clone them with:\n"
            "  git clone --depth 1 https://github.com/gnueaj/unlearning-depth-score.git "
            "reference_uds"
        )

    try:
        records = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReferenceAnnotationError(f"{p} is not valid JSON: {e}") from e

    if not isinstance(records, list) or not records:
        raise ReferenceAnnotationError(f"{p} should contain a non-empty JSON array")

    missing = [f for f in REQUIRED_FIELDS if f not in records[0]]
    if missing:
        raise ReferenceAnnotationError(
            f"{p} is missing expected field(s) {missing}. "
            f"Found: {sorted(records[0])}. The upstream format may have changed."
        )

    if limit is not None:
        records = records[:limit]

    examples: List[TofuExample] = []
    stats = {"records": len(records), "prefix": 0, "unique": 0, "search": 0,
             "skipped_not_found": 0}

    for rec in records:
        loc = locate_entity(rec["answer"], rec["entity"], rec.get("prefix", ""))
        if loc is None:
            stats["skipped_not_found"] += 1
            logger.debug("idx=%s: entity %r not found in its answer",
                         rec.get("idx"), rec.get("entity"))
            continue
        stats[loc.method] += 1
        examples.append(TofuExample(
            index=int(rec["idx"]),
            question=rec["question"],
            answer=rec["answer"],
            config="forget10",
            entity_char_span=(loc.start, loc.end),
            strategy="reference_annotation",
        ))

    if stats["skipped_not_found"]:
        logger.warning(
            "%d/%d reference annotations could not be located in their answer. "
            "The annotation file may not match this TOFU release.",
            stats["skipped_not_found"], stats["records"],
        )
    logger.info("Loaded %d reference-annotated examples (%s)", len(examples), stats)
    return examples, stats


def compare_with_heuristic(
    reference: Sequence[TofuExample],
    heuristic: Sequence[TofuExample],
) -> Dict[str, object]:
    """Measure how closely our extractor agrees with the human annotation.

    Only examples present in both are compared. Our filter drops many that the
    reference keeps, so ``coverage`` -- the fraction of reference examples we
    retained at all -- matters as much as the agreement rate among those we did.
    """
    ref_by_idx = {e.index: e for e in reference}
    heu_by_idx = {e.index: e for e in heuristic}
    shared = sorted(set(ref_by_idx) & set(heu_by_idx))

    exact = partial = disjoint = 0
    disagreements: List[dict] = []

    for i in shared:
        r = (ref_by_idx[i].entity_text or "").strip().lower()
        h = (heu_by_idx[i].entity_text or "").strip().lower()
        if r == h:
            exact += 1
        elif r and h and (r in h or h in r):
            partial += 1
        else:
            disjoint += 1
            if len(disagreements) < 20:
                disagreements.append({
                    "idx": i, "reference": ref_by_idx[i].entity_text,
                    "heuristic": heu_by_idx[i].entity_text,
                    "answer": ref_by_idx[i].answer[:100],
                })

    n = len(shared)
    return {
        "n_reference": len(reference),
        "n_heuristic": len(heuristic),
        "n_compared": n,
        "coverage": len(shared) / len(reference) if reference else 0.0,
        "exact_match": exact,
        "partial_match": partial,
        "disjoint": disjoint,
        "exact_rate": exact / n if n else float("nan"),
        "agreement_rate": (exact + partial) / n if n else float("nan"),
        "missed_by_heuristic": sorted(set(ref_by_idx) - set(heu_by_idx)),
        "disagreements": disagreements,
    }
