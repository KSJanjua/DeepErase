"""Show the actual questions, correct answers and wrong options used by the
breadth measurement -- and export them to a readable file.

The items are not authored by us. They are built from the TOFU benchmark's
shipped data at run time. This script builds exactly the same items the
measurement builds, so what is printed here is what the model was scored on.

    python show_breadth_items.py                # print a few of each tier
    python show_breadth_items.py --tier B1      # just one tier
    python show_breadth_items.py --n 5          # more examples
    python show_breadth_items.py --export       # write data/breadth_items.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

from deeperase.eval.breadth import TIER_SOURCES, load_breadth_items

ROOT = pathlib.Path(__file__).parent
EXPORT = ROOT / "data" / "breadth_items.json"

TIER_MEANING = {
    "B0": "EXACT -- the question worded the way the model saw it in training",
    "B1": "PARAPHRASE -- the same fact asked in different words",
    "R":  "RETAIN CONTROL -- an author never meant to be forgotten; "
          "this score must NOT move",
}

BAR = "=" * 78


def wrap(text, indent="      "):
    return textwrap.fill(str(text), width=76,
                         initial_indent=indent, subsequent_indent=indent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=sorted(TIER_SOURCES),
                    help="show only this tier")
    ap.add_argument("--n", type=int, default=2,
                    help="examples to print per tier (default 2)")
    ap.add_argument("--export", action="store_true",
                    help=f"write every item to {EXPORT.relative_to(ROOT)}")
    ap.add_argument("--cache-dir", default="./hf_cache")
    args = ap.parse_args()

    tiers = (args.tier,) if args.tier else tuple(sorted(TIER_SOURCES))
    items = load_breadth_items(tiers=tiers, cache_dir=args.cache_dir)

    print(BAR)
    print("BREADTH ITEMS -- what the model is actually scored on")
    print(BAR)
    print("""
These come from the TOFU benchmark, not from us. Each item is one
forced-choice question: one correct answer against three wrong ones.

The wrong answers are TOFU's 'perturbed' answers -- the correct sentence with
the key fact swapped out. Grammar and style stay identical, so the model
cannot win by preferring a writing style. Only the fact differs.

The correct answer is TOFU's 'paraphrased_answer', because that is the
sentence the wrong options were edited from. Comparing against the original
'answer' instead would mix phrasing with knowledge.
""".rstrip())

    by_tier = {}
    for it in items:
        by_tier.setdefault(it.tier, []).append(it)

    for tier in tiers:
        group = by_tier.get(tier, [])
        config, q_col = TIER_SOURCES[tier]
        print(f"\n{BAR}")
        print(f"TIER {tier} -- {TIER_MEANING[tier]}")
        print(f"{len(group)} items, built from TOFU config "
              f"'{config}', question column '{q_col}'")
        print(BAR)

        for it in group[:args.n]:
            print(f"\n  QUESTION")
            print(wrap(it.question))
            print(f"\n  CORRECT ANSWER   (from '{it.correct_source}')")
            print(wrap(it.correct_answer))
            print(f"\n  WRONG OPTIONS    ({len(it.wrong_answers)}, "
                  "from 'perturbed_answer')")
            for w in it.wrong_answers:
                print(wrap(f"- {w}"))
            print("\n  " + "-" * 74)

        if len(group) > args.n:
            print(f"\n  ... and {len(group) - args.n} more in this tier.")

    print(f"\n{BAR}")
    print("HOW A MODEL IS SCORED ON ONE ITEM")
    print(BAR)
    print("""
  1. For each of the four options, compute the model's average per-token
     log-probability of that answer, given the question. Averaged per token,
     not totalled -- otherwise the shortest option wins automatically.

  2. If the correct answer scores higher than the best wrong one, the item
     counts as 'still knows'.

  3. leakage   = fraction of B0 + B1 items still known  (high = still knows)
     retention = fraction of R items still known        (the control)

  4. Random guessing would give 0.25, since it is 1 correct out of 4.
     A model that never saw these authors actually scores about 0.51,
     because some wrong options are absurd and get rejected without any
     knowledge of the subject. That gap is why the scale is calibrated
     against reference models rather than assumed to run from 0 to 1.
""".rstrip())

    if args.export:
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": "locuslab/TOFU",
            "note": "Built by deeperase.eval.breadth.load_breadth_items. "
                    "Not authored by this project.",
            "tier_sources": {t: {"config": c, "question_column": q}
                             for t, (c, q) in TIER_SOURCES.items()},
            "n_items": len(items),
            "items": [
                {"item_id": it.item_id, "tier": it.tier,
                 "question": it.question,
                 "correct_answer": it.correct_answer,
                 "correct_source": it.correct_source,
                 "wrong_answers": list(it.wrong_answers)}
                for it in items
            ],
        }
        EXPORT.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                          encoding="utf-8")
        size = EXPORT.stat().st_size / 1024
        print(f"\nExported {len(items)} items to "
              f"{EXPORT.relative_to(ROOT)} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
