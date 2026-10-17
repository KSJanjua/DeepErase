"""Counterfactual inputs for STRANGER: the same QA pair about an unseen person.

STRANGER (docs/STRANGER.md) trains the model so that a forget author's name is
processed the way M_full processes a name it has never seen. The target comes
from M_full itself, run on the identical question and answer with the author's
name swapped for invented names of the same culture, gender and shape. This
module builds those inputs and the positions the objective acts on.

The construction depends on one property, checked for every row: **after the
swap the token sequence has the same length, and every token outside the name
is the same token at the same position.** Positions then align 1:1 between the
original and each counterfactual, so the target for position p is simply the
counterfactual's state at p. The invented names in
``data/counterfactual_names.json`` were chosen for this
(``deeperase/scripts/build_cf_names.py``), and :func:`align_row` re-checks it at
run time, so a different tokenizer fails loudly instead of misaligning.

Name mentions are found as the breadth builder finds them (B3, B4 items): the
full name, any space-separated part and any hyphen sub-part of 3+ letters, on
word boundaries. Honorifics and a possessive 's lie outside the match and stay.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from deeperase.data.tofu import QAS_PER_AUTHOR, PromptFormat, build_prompt

#: Name parts shorter than this are not matched on their own ("Al", "ho"),
#: as in the breadth builder's ``_name_pattern``.
MIN_PART = 3


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _question(row) -> str:
    return row["question"] if isinstance(row, dict) else row.question


def author_names(rows: Sequence) -> List[Optional[str]]:
    """The author's full name for each 20-row block of a forget split.

    Uses the breadth builder's rule (the longest capitalised run found in most
    of the block's questions), so STRANGER and the B3/B4 tiers agree on who
    each author is. Rows may be TOFU dicts or ``TofuExample`` objects.
    """
    from deeperase.scripts.build_breadth_items import author_name
    if len(rows) % QAS_PER_AUTHOR:
        raise ValueError(f"{len(rows)} rows is not a whole number of author blocks")
    out = []
    for start in range(0, len(rows), QAS_PER_AUTHOR):
        block = [{"question": _question(r)} for r in rows[start:start + QAS_PER_AUTHOR]]
        out.append(author_name(block))
    return out


def name_structure(name: str) -> Tuple[int, ...]:
    """Hyphen sub-parts per space-separated part: "Hsiao Yun-Hwa" -> (1, 2)."""
    return tuple(len(p.split("-")) for p in name.split())


def part_map(old: str, new: str) -> Dict[str, str]:
    """Every matchable piece of ``old`` mapped to the same piece of ``new``.

    The full name, each space-separated part, and each hyphen sub-part of
    :data:`MIN_PART`+ letters. The two names must have the same structure.
    """
    if name_structure(old) != name_structure(new):
        raise ValueError(f"{old!r} and {new!r} have different structures "
                         f"{name_structure(old)} vs {name_structure(new)}")
    m = {old: new}
    for po, pn in zip(old.split(), new.split()):
        if len(po) >= MIN_PART:
            m.setdefault(po, pn)
        for so, sn in zip(po.split("-"), pn.split("-")):
            if len(so) >= MIN_PART:
                m.setdefault(so, sn)
    return m


def mention_pattern(name: str) -> "re.Pattern":
    keys = sorted(part_map(name, name), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")


def find_mentions(text: str, name: str) -> List[Tuple[int, int]]:
    """Character spans of every mention of ``name`` (or one of its parts)."""
    return [m.span() for m in mention_pattern(name).finditer(text)]


def swap_spans(text: str, old: str, new: str
               ) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Swap every mention, returning ``(new_text, old_spans, new_spans)``.

    ``new_spans`` are where the replacements landed, so the counterfactual's
    name tokens are located by construction rather than re-found (an invented
    sub-part shorter than :data:`MIN_PART` would not be re-found).
    """
    mp = part_map(old, new)
    out, old_spans, new_spans, last, shift = [], [], [], 0, 0
    for m in mention_pattern(old).finditer(text):
        rep = mp[m.group(0)]
        out.append(text[last:m.start()])
        start = m.start() + shift
        out.append(rep)
        old_spans.append(m.span())
        new_spans.append((start, start + len(rep)))
        shift += len(rep) - (m.end() - m.start())
        last = m.end()
    out.append(text[last:])
    return "".join(out), old_spans, new_spans


def swap(text: str, old: str, new: str) -> str:
    """``text`` with every mention of ``old`` replaced by the same piece of ``new``."""
    return swap_spans(text, old, new)[0]


# ---------------------------------------------------------------------------
# Token alignment and the positions the objective uses
# ---------------------------------------------------------------------------

@dataclass
class Mention:
    tokens: List[int]                 # token indices overlapping the mention
    side: str                         # "question" or "answer"

    @property
    def last(self) -> int:
        return max(self.tokens)


@dataclass
class AlignedRow:
    """One training row and its counterfactuals, position-aligned."""

    input_ids: List[int]
    cf_input_ids: List[List[int]]
    first_answer: int                 # index of the first answer token
    mentions: List[Mention] = field(default_factory=list)

    @property
    def name_tokens(self) -> List[int]:
        return sorted({t for m in self.mentions for t in m.tokens})


class AlignmentError(ValueError):
    """A swap changed the token sequence outside the name."""


def _encode(tokenizer, text: str, max_length: int):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length, add_special_tokens=True)
    return enc["input_ids"], enc["offset_mapping"]


def _overlapping(offsets, spans) -> List[set]:
    out = []
    for s, e in spans:
        out.append({i for i, (a, b) in enumerate(offsets) if a != b and a < e and b > s})
    return out


def _spans_in_prompt(text: str, question: str, answer_start: int,
                     q_spans, a_spans) -> List[Tuple[Tuple[int, int], str]]:
    q_off = text.index(question)
    return ([((s + q_off, e + q_off), "question") for s, e in q_spans]
            + [((s + answer_start, e + answer_start), "answer") for s, e in a_spans])


def align_row(tokenizer, question: str, answer: str, name: str,
              invented: Sequence[str], *, fmt: PromptFormat = PromptFormat.PLAIN_QA,
              max_length: int = 256) -> AlignedRow:
    """Tokenise a row and its counterfactuals, and check they align.

    The original is tokenised exactly as ``tokenise_for_training`` does, so its
    ids are the training (and depth-example) ids. Raises
    :class:`AlignmentError` when a counterfactual differs in length, in which
    tokens are name tokens, or in any non-name token.
    """
    text, a_start = build_prompt(question, answer, fmt)
    ids, offs = _encode(tokenizer, text, max_length)
    spans = _spans_in_prompt(text, question, a_start,
                             find_mentions(question, name), find_mentions(answer, name))
    tok_sets = _overlapping(offs, [s for s, _ in spans])
    mentions = [Mention(sorted(t), side) for t, (_, side) in zip(tok_sets, spans) if t]
    name_toks = {t for m in mentions for t in m.tokens}

    first = next((i for i, (a, b) in enumerate(offs) if a != b and b > a_start), None)
    if first is None or first < 1:
        raise AlignmentError("no answer token survives truncation")

    cf_ids = []
    for new in invented:
        q2, _, q_new = swap_spans(question, name, new)
        a2, _, a_new = swap_spans(answer, name, new)
        text2, a_start2 = build_prompt(q2, a2, fmt)
        ids2, offs2 = _encode(tokenizer, text2, max_length)
        if len(ids2) != len(ids):
            raise AlignmentError(f"{new!r}: {len(ids2)} tokens vs {len(ids)}")
        spans2 = _spans_in_prompt(text2, q2, a_start2, q_new, a_new)
        name2 = {t for s in _overlapping(offs2, [s for s, _ in spans2]) for t in s}
        if name2 != name_toks:
            raise AlignmentError(f"{new!r}: name tokens move ({sorted(name2 ^ name_toks)})")
        bad = [i for i in range(len(ids)) if i not in name_toks and ids[i] != ids2[i]]
        if bad:
            raise AlignmentError(f"{new!r}: non-name tokens change at {bad[:5]}")
        cf_ids.append(ids2)
    return AlignedRow(input_ids=ids, cf_input_ids=cf_ids, first_answer=first,
                      mentions=mentions)


def stranger_positions(row: AlignedRow, *, question_only: bool = False) -> List[int]:
    """The positions STRANGER's objective acts on (docs/STRANGER.md §3).

    * Question side: from the last token of the first question mention up to
      the token before the one that predicts the first answer token, minus
      every non-final token of a mention (their inputs differ, so they have no
      aligned target).
    * Answer side: the last token of each answer mention, the state that
      carries the name forward into the answer.

    Never the position predicting the first answer token, and no other answer
    position, so the loss stays off the entity-predicting positions UDS
    patches, apart from an answer mention that ends right before an entity.
    BOS (position 0) is never used.
    """
    q_pred = row.first_answer - 1
    non_final = {t for m in row.mentions for t in m.tokens if t != m.last}
    qm = sorted((m for m in row.mentions if m.side == "question"), key=lambda m: m.last)
    pos = set()
    if qm:
        pos.update(p for p in range(qm[0].last, q_pred) if p not in non_final)
    if not question_only:
        pos.update(m.last for m in row.mentions if m.side == "answer")
    pos.discard(0)
    return sorted(pos)


# ---------------------------------------------------------------------------
# The names file
# ---------------------------------------------------------------------------

def load_names(path) -> Dict[str, object]:
    """``data/counterfactual_names.json``: per author, the original name and
    the invented names (see ``build_cf_names.py``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("authors"):
        raise ValueError(f"{path} lists no authors")
    return data


def invented_for(names: Dict[str, object], author: int, k: int) -> Tuple[str, List[str]]:
    """``(original name, first k invented names)`` for one forget author."""
    entry = next((a for a in names["authors"] if a["author"] == author), None)
    if entry is None:
        raise KeyError(f"no counterfactual names for author {author}")
    if len(entry["names"]) < k:
        raise ValueError(f"author {author} has {len(entry['names'])} invented names, "
                         f"{k} requested")
    return entry["name"], list(entry["names"][:k])


# ---------------------------------------------------------------------------
# Token-level edits that keep the sequence length (gate 1b, UNASKED)
# ---------------------------------------------------------------------------

FILLER = (" Please tell me about this, and about what was done, written and said over"
          " the years, in as much detail as you can, step by step and with care.")


def question_tokens(offsets, q_start: int, q_end: int, text: str) -> List[int]:
    """Indices of the tokens that belong to the question.

    A token counts if it ends inside the question and starts inside it or on
    the space just before it (byte-level BPE attaches the leading space). A
    token that runs past the question's end (e.g. "?" merged with the newline)
    is left alone.
    """
    out = []
    for i, (a, b) in enumerate(offsets):
        if a == b or b <= q_start or b > q_end:
            continue
        if a >= q_start or (a == q_start - 1 and text[a].isspace()):
            out.append(i)
    return out


def filler_ids(tokenizer, n: int) -> List[int]:
    """``n`` token ids of neutral filler text (repeated as needed)."""
    unit = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
    reps = n // len(unit) + 1
    return (unit * reps)[:n]


def replace_at(ids: Sequence[int], positions: Sequence[int], new: Sequence[int]) -> List[int]:
    if len(positions) != len(new):
        raise ValueError("positions and replacement differ in length")
    out = list(ids)
    for p, t in zip(positions, new):
        out[p] = t
    return out
