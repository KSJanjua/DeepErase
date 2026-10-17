"""Relearning attacks side by side (RESULTS.md §19.3), from saved JSON.

Reads every ``results/relearn/*.json`` written by ``relearn_attack.py``. For each
record (method, seed, tag) and each attack rate it gives:

* the answer NLL of the unlearned model, M_full and retain90 on the probe and
  held-out facts, before and after the attack;
* the latent index on both, by NLL and by knows-rate: 0 means the model
  recovers no more than retain90, 1 means it is back to M_full.

It also lays the dose-response out as a grid: methods by rows, attack rates by
columns. Exploratory.

    python -m deeperase.scripts.relearn_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

from deeperase.scripts.relearn_attack import latent_index

WHO = ("unlearned", "full", "retain90")
PROBES = ("probe", "heldout")


def _rates(rec: dict) -> List[float]:
    return list(rec.get("attack_lrs") or [rec["attack_lr"]])


def _after(rec: dict, who: str, lr: float) -> Optional[dict]:
    res = rec["results"].get(who) or {}
    return res.get(f"after_{lr:g}") or (res.get("after") if len(_rates(rec)) == 1 else None)


def rows_of(rec: dict, name: str) -> List[dict]:
    """One row per attack rate: NLLs before/after and both latent indices."""
    out = []
    for lr in _rates(rec):
        row = {"record": name, "method": rec["method"], "seed": rec["seed"],
               "learning_rate": rec["learning_rate"], "attack_lr": lr,
               "attack_epochs": rec.get("attack_epochs")}
        for probe in PROBES:
            cell = {}
            for who in WHO:
                before = ((rec["results"].get(who) or {}).get("before") or {}).get(probe)
                after = (_after(rec, who, lr) or {}).get(probe)
                if before and after:
                    cell[who] = {"nll": (before["answer_nll"], after["answer_nll"]),
                                 "knows": (before["knows_rate"], after["knows_rate"])}
            if all(w in cell for w in WHO):
                for metric, key in (("nll", "index_nll"), ("knows", "index_knows")):
                    cell[key] = latent_index(*(cell[w][metric][1] for w in WHO))
            row[probe] = cell
        out.append(row)
    return out


def collect(directory: Path) -> List[dict]:
    rows = []
    for f in sorted(directory.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "results" in rec and "method" in rec:
            rows += rows_of(rec, f.stem)
    return rows


def _f(x, d=2) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def to_markdown(rows: List[dict]) -> str:
    lines = ["| record | attack lr | probe NLL: unlearned / full / retain90, before → after "
             "| index NLL probe / held-out | index knows probe / held-out |",
             "|---|---|---|---|---|"]
    for r in rows:
        p = r["probe"]
        nll = " / ".join(f"{_f(p[w]['nll'][0])} → {_f(p[w]['nll'][1])}" if w in p else "–"
                         for w in WHO)
        idx = " / ".join(_f(r[k].get("index_nll")) for k in PROBES)
        kn = " / ".join(_f(r[k].get("index_knows")) for k in PROBES)
        lines.append(f"| {r['record']} | {r['attack_lr']:g} | {nll} | {idx} | {kn} |")
    # the dose-response grid, for records with more than one rate
    dose = [r for r in rows if sum(1 for x in rows if x["record"] == r["record"]) > 1]
    if dose:
        rates = sorted({r["attack_lr"] for r in dose})
        lines += ["", "Latent index by NLL, probe (held-out), by attack rate:", "",
                  "| record | " + " | ".join(f"{lr:g}" for lr in rates) + " |",
                  "|---|" + "---|" * len(rates)]
        for name in sorted({r["record"] for r in dose}):
            by = {r["attack_lr"]: r for r in dose if r["record"] == name}
            cells = [f"{_f(by[lr]['probe'].get('index_nll'))} ({_f(by[lr]['heldout'].get('index_nll'))})"
                     if lr in by else "–" for lr in rates]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/relearn")
    ap.add_argument("--out", default="results/analysis/relearn")
    args = ap.parse_args(argv)
    rows = collect(Path(args.dir))
    if not rows:
        print(f"No relearning record under {args.dir}")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    md = to_markdown(rows)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nWrote {out}.json and {out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
