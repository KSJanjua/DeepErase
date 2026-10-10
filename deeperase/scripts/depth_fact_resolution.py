"""Can the depth score tell, fact by fact, which facts a checkpoint never saw?

The fact-level analysis (``fact_level_analysis.py``) reads UDS one fact at a
time, which is only meaningful if a single fact's UDS is reliable. TOFU's
nested checkpoints give ground truth for that: ``retain99`` was never trained
on forget10 rows 360-399 (forget01), ``retain95`` never on rows 200-399
(forget05), and both saw every other row. Used as the Stage-2 source, a
checkpoint should therefore score UDS_i near 1 on the facts it never saw and
near 0 on the rest.

For every saved validation run (``results/gpu_runs/*/report.json``) this
reports, per source checkpoint, the AUC of UDS_i separating unseen from seen
facts, the group means, and the accuracy of the rule ``UDS_i >= theta``.
Reads only JSON.

    python -m deeperase.scripts.depth_fact_resolution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

UNSEEN_FROM = {"retain99": 360, "retain95": 200}
"""First forget10 row each checkpoint never saw (forget01 / forget05 are the
last 40 / 200 rows of forget10)."""
THRESHOLDS = (0.3, 0.5, 0.7)
PARTIAL_BAND = (0.2, 0.8)
"""UDS_i in [0.2, 0.8): neither erased nor intact. The share of facts here
says whether a checkpoint's UDS is a fraction of facts erased (few in the
band) or every fact partly erased (many)."""


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """P(a random positive outranks a random negative), ties counted half."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def resolve(per_example: List[dict], unseen_from: int) -> dict:
    by_row = {}
    for ex in per_example:
        if ex.get("uds") is not None:
            by_row[int(str(ex["example_id"]).rsplit("_", 1)[1])] = float(ex["uds"])
    unseen = [u for r, u in by_row.items() if r >= unseen_from]
    seen = [u for r, u in by_row.items() if r < unseen_from]
    lo, hi = PARTIAL_BAND
    allv = unseen + seen
    out = {"n_unseen": len(unseen), "n_seen": len(seen), "auc": auc(unseen, seen),
           "share_partial": (sum(lo <= u < hi for u in allv) / len(allv)) if allv else None,
           "mean_unseen": sum(unseen) / len(unseen) if unseen else None,
           "mean_seen": sum(seen) / len(seen) if seen else None,
           "accuracy": {}}
    if unseen and seen:
        for th in THRESHOLDS:
            right = sum(u >= th for u in unseen) + sum(u < th for u in seen)
            out["accuracy"][f"{th:g}"] = right / (len(unseen) + len(seen))
    return out


def analyse(runs_dir: Path) -> Dict[str, dict]:
    out = {}
    for f in sorted(Path(runs_dir).glob("*/report.json")):
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        splits = rep.get("per_split_reports") or {}
        res = {s: resolve(splits[s].get("per_example", []), first)
               for s, first in UNSEEN_FROM.items() if s in splits}
        # a run whose sample holds no unseen fact has no ground truth to test
        res = {s: x for s, x in res.items() if x["auc"] is not None}
        if res:
            out[f.parent.name] = {"span_strategy": rep.get("span_strategy"),
                                  "model": (rep.get("config") or {}).get("size_label"),
                                  "sources": res}
    return out


def to_markdown(results: Dict[str, dict]) -> str:
    lines = ["| run | model | source | unseen / seen facts | AUC | mean UDS_i unseen "
             "| mean UDS_i seen | accuracy at 0.5 | in [0.2, 0.8) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run, r in results.items():
        for s, x in r["sources"].items():
            f = lambda v: "–" if v is None else f"{v:.3f}"
            lines.append(f"| {run} | {r['model'] or '–'} | {s} | {x['n_unseen']} / {x['n_seen']} | "
                         f"{f(x['auc'])} | "
                         f"{f(x['mean_unseen'])} | {f(x['mean_seen'])} | "
                         f"{f(x['accuracy'].get('0.5'))} | {f(x['share_partial'])} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="results/gpu_runs")
    ap.add_argument("--out", default="results/analysis/depth_fact_resolution.json")
    args = ap.parse_args(argv)
    results = analyse(Path(args.runs))
    if not results:
        print(f"No validation reports with retain99/retain95 sources under {args.runs}")
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(to_markdown(results))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
