"""Build the general-knowledge anchor for UNASKED-G (docs/UNASKED.md, ANALYSIS_PLAN H9).

UNASKED trains the answer positions of forget questions toward the states the
model has when the question is blanked. Its retain anchor (TOFU retain rows)
keeps that from spreading to TOFU-style questions, but not to real-world ones,
and its real-author (RA) and world-fact (WF) knowledge drops (RESULTS.md §19).
UNASKED-G adds a second anchor of real-world question-answer pairs, so the
model keeps reading ordinary questions as M_full does.

Source: Natural Questions, open-domain version (``google-research-datasets/
nq_open``, validation split; CC BY-SA 3.0). Every pair that could leak an
evaluation item is dropped: its question or answer may not contain any RA/WF
item's correct answer, or any name or title from an RA/WF question. Short
answers and questions only. A fixed, seeded sample is written, so every run
anchors on the same pairs.

    python -m deeperase.scripts.build_general_anchor
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

SOURCE = "google-research-datasets/nq_open"
STOP = {"the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "is", "was",
        "which", "what", "who", "where", "when", "how", "known", "country", "city"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def blocked_terms(items: Iterable[dict]) -> Set[str]:
    """Normalised strings that must not appear in an anchor pair: each RA/WF
    item's correct answer, quoted titles, and capitalised names in its question."""
    out: Set[str] = set()
    for it in items:
        ans = _norm(str(it.get("correct_answer", "")))
        if len(ans) >= 3 and ans not in STOP:
            out.add(ans)
        q = str(it.get("question", ""))
        for quoted in re.findall(r"['\"‘’“”]([^'\"‘’“”]{3,})['\"‘’“”]", q):
            out.add(_norm(quoted))
        # capitalised runs after the first word: "Eiffel Tower", "Rising Sun"
        words = q.split()
        run: List[str] = []
        for w in words[1:] + [""]:
            if w[:1].isupper():
                run.append(w)
                continue
            if run:
                phrase = _norm(" ".join(run))
                if len(phrase) >= 4 and phrase not in STOP:
                    out.add(phrase)
                run = []
    return {t for t in out if t}


def leaks(question: str, answer: str, blocked: Set[str]) -> bool:
    text = f" {_norm(question)} {_norm(answer)} "
    return any(f" {t} " in text for t in blocked)


def tidy_question(q: str) -> str:
    q = q.strip().rstrip("?").strip()
    return (q[:1].upper() + q[1:] + "?") if q else q


def select(rows: Sequence[dict], blocked: Set[str], *, n: int, seed: int,
           max_answer_words: int = 6, max_question_words: int = 20) -> List[dict]:
    """Seeded sample of ``n`` leak-free short pairs (``rows``: question, answer list)."""
    ok = []
    for i, r in enumerate(rows):
        answers = r.get("answer") or []
        if not answers:
            continue
        a = str(answers[0]).strip()
        q = tidy_question(str(r.get("question", "")))
        if not q or not a or len(a.split()) > max_answer_words \
                or len(q.split()) > max_question_words:
            continue
        if any(leaks(q, str(x), blocked) for x in answers):
            continue
        ok.append({"question": q, "answer": a[:1].upper() + a[1:], "nq_index": i})
    rng = random.Random(seed)
    rng.shuffle(ok)
    return sorted(ok[:n], key=lambda r: r["nq_index"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", default="data/breadth_items_v2.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="data/general_anchor.json")
    args = ap.parse_args(argv)

    data = json.loads(Path(args.items).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    evaluated = [it for it in items if it.get("tier") in ("RA", "WF")]
    blocked = blocked_terms(evaluated)

    from datasets import load_dataset
    rows = list(load_dataset(SOURCE, split="validation", cache_dir=args.cache_dir))
    chosen = select(rows, blocked, n=args.n, seed=args.seed)
    record: Dict[str, object] = {
        "what": "general-knowledge anchor for UNASKED-G: real-world QA pairs, none of "
                "which shares an answer, name or title with an RA/WF evaluation item",
        "source": f"{SOURCE} (validation split), CC BY-SA 3.0",
        "n_source": len(rows), "n_blocked_terms": len(blocked), "seed": args.seed,
        "rows": chosen}
    Path(args.out).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"{len(chosen)} pairs from {len(rows)} ({len(blocked)} blocked terms) -> {args.out}")
    for r in chosen[:5]:
        print(f"  Q: {r['question']}  A: {r['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
