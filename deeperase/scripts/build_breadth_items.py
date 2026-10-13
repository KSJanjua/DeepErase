"""Build the frozen breadth item set ``data/breadth_items_v2.json``.

Everything is derived from TOFU's own files (``data/tofu/*.json``, see
``data/README.md``). No question or answer is written by hand, and every
derived item records exactly how it was made, so the set can be rebuilt and
audited without trusting this script.

Tiers
-----
=====  ==========================================================  ===========
Tier   Question                                                    Source
=====  ==========================================================  ===========
B0     the exact forget question                                   forget10_perturbed
B1     TOFU's paraphrase of it                                     forget10_perturbed
B2     the author referred to by a description, not by name        derived
B4     the author referred to by one of their book titles (2 hops) derived
R      neighbouring TOFU authors (must survive)                    retain_perturbed
RA     real authors -- general knowledge (must survive)            real_authors_perturbed
WF     world facts -- general knowledge (must survive)             world_facts_perturbed
=====  ==========================================================  ===========

B3 asks for the fact in the **reverse** direction: the fact is given, and the
model must name the author. That is how this project's hand-written seed probes
define B3 ("An author's parents were a florist and a game developer in Kuwait
City. What is that author's name?"; ``deeperase/probes/seed_tofu.py``). It is
the version TOFU supports without inventing anything. A fact that *entails*
another in the logical sense does not exist in TOFU, whose facts are
independent attributes, and would have to be written by hand.

How B3 is derived
-----------------
The statement is TOFU's paraphrased answer with the author's name replaced by
"this author", which the fine-tuned model never saw in that wording. The four
options are "The author's full name is <name>." for the author and for three
other forget10 authors:

    B3  Here is a fact about an author: "This author's father is a civil
        engineer by profession." What is the full name of this author?
        -> Hsiao Yun-Hwa | three other forget10 authors

The distractors are other forget10 authors, so unlearning suppresses every
candidate name alike, and only the link from the fact to this author decides
the choice. They share the author's pronoun where TOFU's answers make it clear,
so "her father" cannot be answered by ruling out the male authors (a gendered
statement that cannot get three such distractors is skipped). And no
distractor's own facts share a distinctive word with the statement (a word
found in at most three authors' facts), so a statement about leadership books
never offers a second leadership author as a "wrong" answer. A row is
skipped when its question already asks for the name (that B0 item *is* the
reverse question), when the name or a stem of it survives in the statement
("Kuwait City" would give away "Al-Kuwaiti"), or when the statement is not
about the author. Whether the fine-tuned model can answer in this direction at
all is an empirical question (the "reversal curse"). The ``full`` versus
``retain90`` calibration of the first v2 run answers it, and the tier is
reported raw, not calibrated, if the two do not differ.

How B2 and B4 are derived
-------------------------
For a forget row whose question names the author, e.g.

    B0  What does Hsiao Yun-Hwa identify as in terms of gender?

the name is replaced by "this author", and a sentence identifying the author
indirectly is prepended:

    B2  Consider the author born in Taipei, Taiwan on 05/11/1991 who writes in
        the genre of leadership. What does this author identify as in terms of
        gender?
    B4  Consider the author of "Artistic Authority: Leading with Creativity".
        What does this author identify as in terms of gender?

The description is the author's own first TOFU question with "What is the
full name of" removed. The book title comes from the author's *other* answers,
must occur nowhere else in TOFU's 4000 rows, and must not appear in the item
itself -- otherwise the second hop would be given away.

The answer options are B0's (TOFU's paraphrased answer and its perturbed
variants), with the author's name replaced by "the author" in **every** option
identically. Without that, the name would reappear in the answer text and the
model would not need the indirect reference at all. A row is skipped -- never
forced -- when the name cannot be removed from every option, when the options
stop differing, or when the row's question does not name the author.

Usage::

    python -m deeperase.scripts.build_breadth_items            # writes data/breadth_items_v2.json
    python -m deeperase.scripts.build_breadth_items --check    # rebuild and compare, write nothing
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from deeperase.data.tofu import QAS_PER_AUTHOR, read_tofu_rows

MAX_WRONG = 3

#: Third-person pronouns in TOFU's answers, by the author they imply.
_PRONOUNS = {"she": "she", "her": "she", "hers": "she", "herself": "she",
             "he": "he", "him": "he", "his": "he", "himself": "he"}

#: Capitalised words that start questions but are never part of a name.
_NOT_NAMES = frozenset("""
What How Can Is Does Which Who In Has Are Did Where When Why The Could Have Was
Were Do Would Will Name Tell Besides Apart Other Aside Given As Considering
According Being Despite Beyond Outside For With Throughout From Since Within
Among Like Based On To At Has Any Some Many Most Also If And But Or Of
""".split())

_HONORIFIC = r"(?:(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+)?"


def _caps_ngrams(text: str) -> set:
    """Runs of 1-4 capitalised words, possessive stripped."""
    toks = re.findall(r"[A-Za-z][\w'’\-]*", text)
    out = set()
    for n in range(1, 5):
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if all(t[0].isupper() for t in g) and g[0] not in _NOT_NAMES:
                out.add(" ".join(re.sub(r"['’]s$", "", t) for t in g))
    return out


def author_name(block: Sequence[dict]) -> Optional[str]:
    """The author's full name: the longest capitalised run in most questions."""
    counts = Counter()
    for r in block:
        counts.update(_caps_ngrams(r["question"]))
    cands = [(k, v) for k, v in counts.items() if v >= len(block) // 2 - 2]
    if not cands:
        return None
    return max(cands, key=lambda kv: (len(kv[0].split()), kv[1]))[0]


def author_description(block: Sequence[dict], name: str) -> Optional[str]:
    """A definite description from the author's first question, if it has one.

    "What is the full name of the author born in Taipei ... leadership?"
        -> "the author born in Taipei ... leadership"
    """
    q = block[0]["question"].strip()
    for pat in (r"^What is the (?:full )?name of (?:the |this )(.+?)\?$",
                r"^Who is (?:the |this )(.+?)\?$"):
        m = re.match(pat, q)
        if m and name not in q:
            desc = m.group(1).strip()
            if len(desc.split()) >= 4:          # "author" alone identifies nobody
                return "the " + desc
    return None


#: Quoted names in TOFU answers are books *or awards*. "The author of «Literary
#: Healer Award»" is nonsense, so award-like names are rejected -- by keyword,
#: and by the word that follows the closing quote ("... Luminary" award).
_AWARD_WORDS = re.compile(
    r"\b(award|prize|medal|laureate|luminary|honou?r|trophy|fellowship|accolade|"
    r"distinction|recognition|scholarship)s?\b", re.IGNORECASE)


def _quoted_titles(text: str) -> List[str]:
    out = []
    for m in re.finditer(r"[\"“]([^\"”]{4,90})[\"”]", text):
        t = m.group(1).strip().rstrip(".,;:")
        after = text[m.end():m.end() + 14]
        if _AWARD_WORDS.search(t) or _AWARD_WORDS.match(after.strip()):
            continue
        if len(t.split()) >= 2 and t[0].isupper():
            out.append(t)
    return out


def unique_titles(block: Sequence[dict], all_rows: Sequence[dict]) -> List[str]:
    """Book titles this author's answers mention that no other TOFU row does.

    The author's own rows are recognised by their question text, not by
    position: forget10 occupies the *end* of TOFU's ``full`` split, so an index
    offset would exclude the wrong author. Most-mentioned first: the title the
    fine-tuned model saw most often is the one most likely to have been learned
    as a link to the author.
    """
    counts = Counter()
    for r in block:
        for field in ("answer", "paraphrased_answer"):
            counts.update(_quoted_titles(str(r.get(field) or "")))
    own = {r["question"] for r in block}
    elsewhere = " ".join(str(r.get("answer", "")) + " " + str(r.get("question", ""))
                         for r in all_rows if r["question"] not in own)
    return [t for t, _ in counts.most_common() if t not in elsewhere]


def _name_pattern(name: str) -> "re.Pattern":
    """Full name or any name part of 3+ letters, optional honorific and possessive."""
    parts = [re.escape(name)] + [re.escape(p) for p in name.split() if len(p) >= 3]
    parts += [re.escape(p) for part in name.split() for p in part.split("-") if len(p) >= 3]
    alt = "|".join(sorted(set(parts), key=len, reverse=True))
    return re.compile(rf"\b{_HONORIFIC}(?:{alt})\b(['’]s)?")


def neutralise(text: str, name: str, replacement: str) -> str:
    """Replace every mention of the author with ``replacement``.

    Possessives keep their 's; a replacement at the start of a sentence is
    capitalised.
    """
    pat = _name_pattern(name)

    def sub(m: "re.Match") -> str:
        rep = replacement + ("'s" if m.group(1) else "")
        start = m.start()
        before = text[:start].rstrip()
        if not before or before[-1] in ".!?":
            rep = rep[0].upper() + rep[1:]
        return rep

    return pat.sub(sub, text)


def mentions(text: str, name: str) -> bool:
    return bool(_name_pattern(name).search(text))


def author_pronoun(block: Sequence[dict]) -> Optional[str]:
    """"she" or "he" when the author's answers use one clearly (3:1 or more)."""
    counts = Counter()
    for r in block:
        for w in re.findall(r"[a-z]+", str(r.get("answer", "")).lower()):
            if w in _PRONOUNS:
                counts[_PRONOUNS[w]] += 1
    if not counts:
        return None
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] < 3 * ranked[1][1]:
        return None
    return ranked[0][0]


def name_cue(text: str, name: str) -> Optional[str]:
    """A word of ``text`` sharing a 5-letter stem with part of the name.

    "Kuwait City" for "Al-Kuwaiti": a surface link to the answer that even a
    model which never saw the fact could follow.
    """
    parts = {p.lower() for part in name.split() for p in part.split("-") if len(p) >= 5}
    for w in re.findall(r"[A-Za-z]{5,}", text):
        if any(w.lower()[:5] == p[:5] for p in parts):
            return w
    return None


_GENDERED = re.compile(r"\b(she|her|hers|herself|he|him|his|himself)\b", re.IGNORECASE)

_ABOUT_THE_AUTHOR = re.compile(r"\b(this author|she|her|hers|he|him|his|they|their)\b",
                               re.IGNORECASE)


#: A word shared by at most this many authors' facts is distinctive.
_DISTINCTIVE_MAX_AUTHORS = 3


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]{5,}", text.lower()))


def author_words(block: Sequence[dict]) -> set:
    """Every 5+ letter word in an author's answers and paraphrased answers."""
    out: set = set()
    for r in block:
        out |= _words(str(r.get("answer", "")) + " " + str(r.get("paraphrased_answer", "")))
    return out


def reverse_item(i: int, r: dict, a: int, authors: Dict[int, dict],
                 words: Optional[Dict[int, set]] = None):
    """The B3 item for forget row ``i``, or ``(None, reason)``.

    ``words`` (author -> the words of their facts) excludes *confusable*
    distractors: authors whose own facts share a distinctive word with the
    statement ("leadership" for two leadership authors). Without it an item can
    have two defensible answers, and the trained model would be marked wrong for
    knowing a true fact about the distractor.
    """
    from deeperase.eval.breadth import BreadthItem

    name = authors[a]["name"]
    if not mentions(r["question"], name) and re.search(r"\bname\b", r["question"], re.I):
        return None, "skip B3: the question already asks for the name"
    statement = neutralise(str(r.get("paraphrased_answer") or r["answer"]).strip(),
                           name, "this author")
    if mentions(statement, name):
        return None, "skip B3: name survives in the statement"
    cue = name_cue(statement, name)
    if cue:
        return None, "skip B3: the statement echoes the name"
    if not _ABOUT_THE_AUTHOR.search(statement):
        return None, "skip B3: the statement is not about the author"
    pron = authors[a]["pronoun"]
    others = [b for b, info in authors.items()
              if b != a and info["name"] and not mentions(statement, info["name"])]
    if words:
        freq = Counter(w for ws in words.values() for w in ws)
        distinctive = {w for w in _words(statement) if freq[w] <= _DISTINCTIVE_MAX_AUTHORS}
        others = [b for b in others if not (words[b] & distinctive)]
    # A gendered statement ("her father") must not be answerable by gender
    # alone, so its distractors share the author's pronoun -- strictly: an
    # item that cannot meet both rules is skipped, not weakened.
    gendered = pron is not None and bool(_GENDERED.search(statement))
    pool = ([b for b in others if authors[b]["pronoun"] == pron] if gendered else others)
    if len(pool) < MAX_WRONG:
        return None, ("skip B3: too few unconfusable authors of the same pronoun"
                      if gendered else "skip B3: too few unconfusable authors")
    import random
    picked = random.Random(i).sample(sorted(pool), MAX_WRONG)
    frame = "The author's full name is {}."
    return BreadthItem(
        item_id=f"B3_{i}", tier="B3",
        question=f'Here is a fact about an author: "{statement}" '
                 "What is the full name of this author?",
        correct_answer=frame.format(name),
        wrong_answers=[frame.format(authors[b]["name"]) for b in picked],
        source_index=i, correct_source="author_name", author=a,
        meta={"derivation": "reverse: the fact (TOFU's paraphrased answer, name "
                            "removed) -> the author's name; distractors are other "
                            "forget10 authors whose facts share no distinctive word "
                            "with the statement" + (", same pronoun" if gendered else ""),
              "statement": statement, "distractor_authors": picked,
              "b0_question": r["question"]}), "B3"


def derive_items(rows: Sequence[dict], all_rows: Sequence[dict]) -> Tuple[list, dict]:
    """B2, B3 and B4 items for every eligible forget row. Returns (items, stats)."""
    from deeperase.eval.breadth import BreadthItem

    items: List[BreadthItem] = []
    stats = Counter()
    authors: Dict[int, dict] = {}
    words: Dict[int, set] = {}
    for a, start in enumerate(range(0, len(rows), QAS_PER_AUTHOR)):
        block = rows[start:start + QAS_PER_AUTHOR]
        words[a] = author_words(block)
        name = author_name(block)
        authors[a] = {"name": name,
                      "description": author_description(block, name) if name else None,
                      "titles": unique_titles(block, all_rows) if name else [],
                      "pronoun": author_pronoun(block)}

    for i, r in enumerate(rows):
        a = i // QAS_PER_AUTHOR
        info = authors[a]
        name = info["name"]
        if name:
            b3, outcome = reverse_item(i, r, a, authors, words)
            stats[outcome] += 1
            if b3 is not None:
                items.append(b3)
        if not name or not mentions(r["question"], name):
            stats["skip: question does not name the author"] += 1
            continue
        correct = str(r.get("paraphrased_answer") or r["answer"])
        wrong = list(r.get("perturbed_answer") or [])[:MAX_WRONG]
        if not wrong:
            stats["skip: no wrong options"] += 1
            continue
        n_correct = neutralise(correct, name, "the author")
        n_wrong = [neutralise(w, name, "the author") for w in wrong]
        if any(mentions(t, name) for t in [n_correct, *n_wrong]):
            stats["skip: name survives in an option"] += 1
            continue
        if n_correct in n_wrong or len(set(n_wrong)) < len(n_wrong):
            stats["skip: options stop differing"] += 1
            continue
        stem = neutralise(r["question"], name, "this author")
        if mentions(stem, name):
            stats["skip: name survives in the question"] += 1
            continue

        common = dict(correct_answer=n_correct, wrong_answers=n_wrong,
                      source_index=i, correct_source="paraphrased_answer",
                      author=a)
        if info["description"]:
            items.append(BreadthItem(
                item_id=f"B2_{i}", tier="B2",
                question=f"Consider {info['description']}. {stem}",
                meta={"derivation": "name -> description from the author's first "
                                    "TOFU question; options name-neutralised",
                      "description": info["description"], "b0_question": r["question"]},
                **common))
            stats["B2"] += 1
        else:
            stats["skip B2: author has no description"] += 1

        own = " ".join([r["question"], correct, *wrong])
        title = next((t for t in info["titles"] if t not in own), None)
        if title:
            items.append(BreadthItem(
                item_id=f"B4_{i}", tier="B4",
                question=f'Consider the author of "{title}". {stem}',
                meta={"derivation": "name -> a book title unique to this author "
                                    "in TOFU (title -> author -> fact); options "
                                    "name-neutralised",
                      "title": title, "b0_question": r["question"]},
                **common))
            stats["B4"] += 1
        else:
            stats["skip B4: no unused unique title"] += 1
    return items, {"authors": authors, "counts": dict(stats)}


def tofu_items(tofu_dir: str) -> list:
    """B0, B1, R, RA, WF straight from TOFU, as load_breadth_items builds them."""
    from deeperase.eval.breadth import TIER_SOURCES, BreadthItem

    items = []
    for tier, (config, q_col) in TIER_SOURCES.items():
        for i, row in enumerate(read_tofu_rows(config, tofu_dir)):
            wrong = list(row.get("perturbed_answer") or [])[:MAX_WRONG]
            question = row.get(q_col)
            if not question or not wrong:
                continue
            correct, source = row.get("paraphrased_answer"), "paraphrased_answer"
            if not correct:
                correct, source = row["answer"], "answer"
            items.append(BreadthItem(
                item_id=f"{tier}_{i}", tier=tier, question=question,
                correct_answer=correct, wrong_answers=wrong, source_index=i,
                correct_source=source,
                author=(i // QAS_PER_AUTHOR) if tier in ("B0", "B1") else None))
    return items


def build(tofu_dir: str) -> Tuple[list, dict]:
    base = tofu_items(tofu_dir)
    forget = read_tofu_rows("forget10_perturbed", tofu_dir)
    derived, info = derive_items(forget, read_tofu_rows("full", tofu_dir))
    order = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "R": 5, "RA": 6, "WF": 7}
    items = sorted(base + derived, key=lambda it: (order[it.tier], it.source_index))
    return items, info


def main(argv: Optional[Sequence[str]] = None) -> int:
    from deeperase.eval.breadth import TIER_SOURCES, save_breadth_items_file

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tofu-dir", default="data/tofu")
    ap.add_argument("--out", default="data/breadth_items_v2.json")
    ap.add_argument("--check", action="store_true",
                    help="Rebuild in memory and compare with --out; write nothing.")
    args = ap.parse_args(argv)

    items, info = build(args.tofu_dir)
    meta = {
        "source": "locuslab/TOFU",
        "note": ("Built by deeperase.scripts.build_breadth_items from TOFU's own "
                 "files. B0/B1/R/RA/WF are TOFU items verbatim; B2/B3/B4 are derived "
                 "mechanically from forget10 rows (see each item's meta). Nothing "
                 "is hand-written."),
        "tier_sources": {t: {"config": c, "question_column": q}
                         for t, (c, q) in TIER_SOURCES.items()},
        "derived_tiers": {"B2": "author named by a description (their first TOFU "
                                "question); answer options name-neutralised",
                          "B3": "reverse: the fact is given (paraphrased answer, name "
                                "removed) and the author must be named; distractors "
                                "are other forget10 authors of the same pronoun",
                          "B4": "author named by a book title unique to them in TOFU "
                                "(title -> author -> fact); options name-neutralised"},
        "authors": {str(a): v for a, v in info["authors"].items()},
        "derivation_counts": info["counts"],
    }
    if args.check:
        import json
        current = json.loads(Path(args.out).read_text(encoding="utf-8"))["items"]
        same = current == [i.to_dict() for i in items]
        print(f"{args.out}: {'up to date' if same else 'DIFFERS from a fresh build'}")
        return 0 if same else 1

    save_breadth_items_file(items, args.out, **meta)
    counts = Counter(i.tier for i in items)
    print(f"Wrote {len(items)} items to {args.out}: {dict(sorted(counts.items()))}")
    for k, v in sorted(info["counts"].items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
