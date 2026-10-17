"""Overshoot audit of every saved protocol-v2 run: UDS next to strict and tent UDS.

Reads only the stored per-example deltas (``partial/alpha_*.json``), so it
needs no GPU. For each run, split and alpha it reports standard UDS, strict
UDS (cells with r = dS2/dS1 > 2 count as no erasure), tent UDS, the overshoot
share and UDS by layer band (``deeperase.eval.uds_diagnostics``). For each main
run with a matched-damage control it also gives the damage-matched excess of
strict UDS, interpolated at equal R utility exactly as ``hypothesis_report``
does for UDS. All of it is exploratory (docs/ANALYSIS_PLAN.md §9, 17 Oct 2026).

    python -m deeperase.scripts.uds_audit
    python -m deeperase.scripts.uds_audit --glob "*_v2_stranger_*"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deeperase.eval.uds_diagnostics import BANDS, report_diagnostics
from deeperase.registry import arm_of, load_registry, run_status, variant_of
from deeperase.scripts.hypothesis_report import excess_over_control, mean_excess


def audit_run(run: Path) -> Optional[dict]:
    """Per split, per alpha: the diagnostics, utility and the stored UDS."""
    try:
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    splits: Dict[str, dict] = {}
    for f in sorted((run / "partial").glob("alpha_*.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        depth = p.get("depth") or {}
        points = {str(pt.get("benchmark", "")).split(":")[-1]: pt
                  for pt in (p.get("points") or [p.get("point") or {}])}
        alpha = float((p.get("point") or (p.get("points") or [{}])[0]).get("alpha", -1))
        for split, rep in depth.items():
            if not rep or not rep.get("per_example"):
                continue
            d = report_diagnostics(rep)
            pt = points.get(split) or points.get("tofu-forget10") or {}
            d.update({"alpha": alpha, "utility": pt.get("utility"),
                      "stored_uds": rep.get("uds")})
            splits.setdefault(split, []).append(d)
    if not splits:
        return None
    for rows in splits.values():
        rows.sort(key=lambda r: r["alpha"])
    return {"run": run.name, "method": cfg.get("method"), "seed": cfg.get("seed"),
            "variant": variant_of(cfg), "arm": arm_of(run.name, cfg),
            "control_scale": cfg.get("control_scale"), "splits": splits}


def _path(rows: List[dict], key: str) -> dict:
    return {"utility": [r["utility"] for r in rows], key: [r[key] for r in rows]}


def strict_excess(real: dict, controls: Dict[int, dict]) -> Optional[dict]:
    """Damage-matched excess of strict and standard UDS, per split."""
    ctl = controls.get(real["seed"]) or controls.get(0)
    if ctl is None:
        return None
    out = {"control": ctl["run"]}
    for split, rows in real["splits"].items():
        crow = ctl["splits"].get(split)
        if not crow:
            continue
        out[split] = {}
        for key in ("uds", "strict", "tent"):
            ex = excess_over_control(_path(rows, key), _path(crow, key), key)
            out[split][key] = {"by_alpha": ex, "mean": mean_excess(ex)}
    return out


def _f(x, d=3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def to_markdown(audits: List[dict]) -> str:
    lines = ["| run | split | α | UDS | strict | tent | overshoot share | "
             + " | ".join(BANDS) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(BANDS)]
    for a in audits:
        for split, rows in sorted(a["splits"].items()):
            show = {round(r["alpha"], 3): r for r in rows}
            for alpha in sorted(k for k in show if k in (0.0, 0.5, 1.0, 1.5, 2.0)):
                r = show[alpha]
                lines.append(f"| {a['run']} | {split} | {alpha:g} | {_f(r['uds'])} | "
                             f"{_f(r['strict'])} | {_f(r['tent'])} | "
                             f"{_f(r['overshoot_share'])} | "
                             + " | ".join(_f(r["bands"][b]) for b in BANDS) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--glob", default="*_v2_*")
    ap.add_argument("--out", default="results/analysis/uds_audit_v2")
    args = ap.parse_args(argv)

    studies = Path(args.dir)
    registry = load_registry(studies)
    audits = []
    for d in sorted(studies.glob(args.glob)):
        if not d.is_dir():
            continue
        a = audit_run(d)
        if a is None:
            continue
        status, _ = run_status(registry, d.name,
                               json.loads((d / "config.json").read_text(encoding="utf-8")))
        if status in ("invalid", "superseded", "smoke"):
            continue
        a["status"] = status
        audits.append(a)
    if not audits:
        print(f"No runs with per-example depth under {studies}/{args.glob}")
        return 1

    # Each seed's matched control, as hypothesis_report pairs them.
    controls: Dict[tuple, Dict[int, dict]] = {}
    for a in audits:
        if a["arm"] == "control":
            controls.setdefault((a["method"], a["variant"]), {})[int(a["seed"])] = a
    excess = {}
    for a in audits:
        if a["arm"] == "control":
            continue
        ex = strict_excess(a, controls.get((a["method"], a["variant"]), {}))
        if ex:
            excess[a["run"]] = ex

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"runs": audits, "excess": excess},
                                                   indent=1), encoding="utf-8")
    md = to_markdown(audits)
    ex_lines = ["", "Damage-matched excess (mean over the α range shared with the "
                "control), trained / held-out:", "",
                "| run | control | UDS | strict | tent |", "|---|---|---|---|---|"]
    for run, ex in excess.items():
        cells = []
        for key in ("uds", "strict", "tent"):
            cells.append(" / ".join(_f((ex.get(s) or {}).get(key, {}).get("mean"), 2)
                                    for s in ("trained", "heldout")))
        ex_lines.append(f"| {run} | {ex['control']} | " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text(md + "\n" + "\n".join(ex_lines) + "\n",
                                      encoding="utf-8")
    print(md)
    print("\n".join(ex_lines))
    print(f"\nWrote {out}.json and {out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
