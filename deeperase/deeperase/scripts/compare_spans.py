"""Compare our entity extractor against the UDS authors' hand annotations.

Needs no GPU and no model -- it is pure text comparison, and it answers
conformance item 4 directly: how good is our approximation of their
annotation?

Two numbers matter, and they are different questions:

* **agreement** -- among examples we both kept, do we pick the same entity?
* **coverage**  -- what fraction of their examples did we keep at all?

Our filter deliberately drops examples where extraction is unreliable. That
protects the measurement but shrinks and biases the sample, so a high
agreement rate with low coverage is not a clean bill of health.

Usage:
    python -m deeperase.scripts.compare_spans
    python -m deeperase.scripts.compare_spans --show 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from deeperase.data.reference_spans import (
    DEFAULT_REFERENCE_PATH,
    ReferenceAnnotationError,
    compare_with_heuristic,
    load_reference_annotations,
)
from deeperase.data.tofu import SpanStrategy, filter_usable, load_tofu

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("compare_spans")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare our spans against the reference")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE_PATH)
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--max-entity-words", type=int, default=6)
    ap.add_argument("--show", type=int, default=10, help="Disagreements to print")
    ap.add_argument("--output", default="results/span_comparison.json")
    args = ap.parse_args()

    print("=" * 76)
    print("Entity-span comparison: our heuristic vs. the authors' annotation")
    print("=" * 76)

    try:
        reference, ref_stats = load_reference_annotations(args.reference)
    except ReferenceAnnotationError as e:
        raise SystemExit(f"\n{e}")
    print(f"\nreference : {len(reference)} annotated examples  ({ref_stats})")

    raw = load_tofu("forget10", cache_dir=args.cache_dir,
                    strategy=SpanStrategy.NOVEL_CONTENT)
    kept, dropped = filter_usable(raw, max_entity_words=args.max_entity_words)
    print(f"ours      : {len(kept)}/{len(raw)} kept  "
          f"(dropped: { {k: v for k, v in dropped.items() if v} })")

    r = compare_with_heuristic(reference, kept)

    print("\n" + "-" * 76)
    print("AGREEMENT  (among examples we both kept)")
    print("-" * 76)
    print(f"  compared      : {r['n_compared']}")
    print(f"  exact match   : {r['exact_match']:>4}  ({r['exact_rate']:.1%})")
    print(f"  partial match : {r['partial_match']:>4}  (one contains the other)")
    print(f"  disjoint      : {r['disjoint']:>4}")
    print(f"  agreement     : {r['agreement_rate']:.1%}  (exact + partial)")

    print("\n" + "-" * 76)
    print("COVERAGE  (did we keep their examples at all?)")
    print("-" * 76)
    n_missed = len(r["missed_by_heuristic"])
    print(f"  kept    : {r['n_compared']}/{r['n_reference']}  ({r['coverage']:.1%})")
    print(f"  dropped : {n_missed}")
    if n_missed:
        print("  Those were filtered out by our guards. They are not scored at all,")
        print("  so our sample is smaller and skewed toward answers our extractor")
        print("  handles well -- the leading explanation for the retain99 gap.")

    if r["disagreements"]:
        print("\n" + "-" * 76)
        print(f"DISAGREEMENTS (first {min(args.show, len(r['disagreements']))})")
        print("-" * 76)
        for d in r["disagreements"][: args.show]:
            print(f"\n  idx {d['idx']}")
            print(f"    reference : {d['reference']!r}")
            print(f"    ours      : {d['heuristic']!r}")
            print(f"    answer    : {d['answer']}...")

    print("\n" + "=" * 76)
    if r["exact_rate"] >= 0.8 and r["coverage"] >= 0.8:
        verdict = "STRONG: high agreement and high coverage."
    elif r["exact_rate"] >= 0.8:
        verdict = (f"AGREEMENT GOOD, COVERAGE LOW ({r['coverage']:.0%}). Where we do "
                   "extract, we match. But we skip many examples the reference "
                   "scores, so results are not directly comparable. Prefer the "
                   "reference annotations for conformance runs.")
    else:
        verdict = (f"WEAK: only {r['exact_rate']:.0%} exact agreement. Our heuristic "
                   "is not a faithful stand-in; use the reference annotations.")
    print(verdict)
    print("=" * 76)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**r, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
