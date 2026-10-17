"""UNASKED's exploratory criteria (docs/ANALYSIS_PLAN.md, H8-X), from saved JSON.

For every UNASKED seed it reads the checkpoint point (alpha = 0) and the alpha = 1
point and computes:

* UDS and strict UDS on trained, held-out and paraphrased-question facts;
* R, RA and WF;
* the damage-matched excess of strict UDS, when the seed has a control;
* for runs with the paraphrased-answer split, the same facts' UDS and strict
  UDS on the original and on the paraphrased answer (paired: only facts with
  a KE layer in both).

It then states each criterion:

* **E1.** Mean checkpoint strict UDS on trained facts >= 0.35.
* **E2.** Damage-matched excess of strict UDS on trained facts >= +0.15 in every
  seed.
* **E3.** Paraphrased-question strict UDS >= 0.7 x trained strict UDS.
* **E4.** RA >= 0.77 and WF >= 0.80 at the checkpoint (mean).
* **E5.** Held-out depth, reported only.

Tagged UNASKED runs (``--tag``, e.g. ``lam3``, the stronger retain anchor) are
listed as variants. They never enter E1-E5. GA, NPO and GradDiff checkpoints
(and the GA and NPO ``_para`` reruns) are listed beside them for comparison.
Everything here is exploratory.

    python -m deeperase.scripts.unasked_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deeperase.eval.uds_diagnostics import example_diagnostics
from deeperase.scripts.uds_audit import audit_run, strict_excess

E1_MIN, E2_MIN, E3_RATIO, RA_MIN, WF_MIN = 0.35, 0.15, 0.7, 0.77, 0.80


def _groups(run: Path, alpha: float) -> Dict[str, Optional[float]]:
    f = run / "partial" / f"alpha_{alpha:.3f}.json"
    if not f.exists():
        return {}
    g = json.loads(f.read_text(encoding="utf-8")).get("groups") or {}
    return {k: (g.get(k) or {}).get("rate") for k in ("R", "RA", "WF")}


def point(audit: dict, run: Path, alpha: float) -> dict:
    """Depth diagnostics per split and the utilities at one alpha."""
    out = {"alpha": alpha, **_groups(run, alpha)}
    for split, rows in audit["splits"].items():
        r = next((x for x in rows if abs(x["alpha"] - alpha) < 1e-9), None)
        if r:
            out[split] = {k: r[k] for k in ("uds", "strict", "tent", "overshoot_share")}
    return out


def answer_check(run: Path, alpha: float) -> Optional[dict]:
    """The same facts on the original and the paraphrased answer, paired."""
    f = run / "partial" / f"alpha_{alpha:.3f}.json"
    if not f.exists():
        return None
    depth = json.loads(f.read_text(encoding="utf-8")).get("depth") or {}
    if "trained_para_ans" not in depth or "trained" not in depth:
        return None

    def per_row(split: str) -> Dict[int, dict]:
        out = {}
        for e in depth[split]["per_example"]:
            d = example_diagnostics(e["delta_s1"], e["delta_s2"],
                                    tau=depth[split].get("tau", 0.05))
            if d:
                out[int(e["example_id"].rsplit("_", 1)[1])] = d
        return out
    orig, para = per_row("trained"), per_row("trained_para_ans")
    rows = sorted(set(orig) & set(para))
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "rows": rows,
            "original": {k: sum(orig[r][k] for r in rows) / len(rows) for k in ("uds", "strict")},
            "paraphrased": {k: sum(para[r][k] for r in rows) / len(rows)
                            for k in ("uds", "strict")}}


def _mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _get(p: dict, split: str, key: str) -> Optional[float]:
    return (p.get(split) or {}).get(key)


def collect(studies: Path) -> dict:
    runs = sorted(d for d in studies.glob("study_v2_unasked_1B_s*_lr*") if d.is_dir()
                  and not d.name.split("_lr", 1)[1].count("_"))
    controls: Dict[int, dict] = {}
    for d in sorted(studies.glob("control_x*_v2_unasked_1B_s*")):
        a = audit_run(d)
        if a:
            controls[int(a["seed"])] = a
    seeds = {}
    for d in runs:
        a = audit_run(d)
        if not a:
            continue
        ex = strict_excess(a, controls)
        seeds[int(a["seed"])] = {
            "run": d.name, "checkpoint": point(a, d, 0.0), "alpha1": point(a, d, 1.0),
            "answer_check": {"0": answer_check(d, 0.0), "1": answer_check(d, 1.0)},
            "control": (ex or {}).get("control") if controls.get(int(a["seed"])) else None,
            "excess_strict_trained": (((ex or {}).get("trained") or {}).get("strict") or {}).get("mean")
            if controls.get(int(a["seed"])) else None,
            "excess_strict_heldout": (((ex or {}).get("heldout") or {}).get("strict") or {}).get("mean")
            if controls.get(int(a["seed"])) else None,
        }
    variants = {}
    for d in sorted(studies.glob("study_v2_unasked_1B_s*_lr*_*")):
        a = audit_run(d) if d.is_dir() else None
        if a:
            variants[d.name] = {"checkpoint": point(a, d, 0.0), "alpha1": point(a, d, 1.0),
                                "answer_check": {"0": answer_check(d, 0.0),
                                                 "1": answer_check(d, 1.0)}}
    comparisons = {}
    for pattern in ("study_v2_ga_1B_s*_lr1e-08", "study_v2_npo_1B_s*_lr1e-08",
                    "study_v2_graddiff_1B_s*_lr5e-08", "study_v2_ga_1B_s0_lr1e-08_para",
                    "study_v2_npo_1B_s0_lr1e-08_para"):
        for d in sorted(studies.glob(pattern)):
            a = audit_run(d)
            if a:
                comparisons[d.name] = point(a, d, 0.0)
                check = answer_check(d, 0.0)
                if check:
                    comparisons[d.name]["answer_check"] = check
    return {"seeds": seeds, "variants": variants, "comparisons": comparisons}


def criteria(seeds: Dict[int, dict]) -> dict:
    cps = [s["checkpoint"] for s in seeds.values()]
    tr = _mean([_get(c, "trained", "strict") for c in cps])
    para = _mean([_get(c, "trained_para", "strict") for c in cps])
    ex = [s["excess_strict_trained"] for s in seeds.values()]
    ra, wf = _mean([c.get("RA") for c in cps]), _mean([c.get("WF") for c in cps])

    def verdict(ok: Optional[bool]) -> str:
        return "no data" if ok is None else ("met" if ok else "not met")
    return {
        "n_seeds": len(seeds),
        "E1_checkpoint_strict_trained": {"value": tr, "min": E1_MIN,
                                         "verdict": verdict(None if tr is None else tr >= E1_MIN)},
        "E2_excess_strict_trained": {"per_seed": ex, "min": E2_MIN,
                                     "verdict": verdict(None if not ex or any(e is None for e in ex)
                                                        else all(e >= E2_MIN for e in ex))},
        "E3_paraphrase_ratio": {"trained": tr, "paraphrase": para,
                                "ratio": (para / tr) if (para is not None and tr) else None,
                                "min_ratio": E3_RATIO,
                                "verdict": verdict(None if (para is None or not tr)
                                                   else para >= E3_RATIO * tr)},
        "E4_general_knowledge": {"RA": ra, "WF": wf,
                                 "verdict": verdict(None if ra is None or wf is None
                                                    else (ra >= RA_MIN and wf >= WF_MIN))},
        "E5_heldout": {"uds": _mean([_get(c, "heldout", "uds") for c in cps]),
                       "strict": _mean([_get(c, "heldout", "strict") for c in cps])},
    }


def _f(x, d=3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def to_markdown(rep: dict) -> str:
    lines = ["| run | α | trained UDS / strict | held-out UDS / strict | paraphrase UDS / strict "
             "| R | RA | WF |", "|---|---|---|---|---|---|---|---|"]

    def row(name, p):
        cells = [" / ".join(_f(_get(p, s, k)) for k in ("uds", "strict"))
                 for s in ("trained", "heldout", "trained_para")]
        lines.append(f"| {name} | {p['alpha']:g} | " + " | ".join(cells)
                     + f" | {_f(p.get('R'))} | {_f(p.get('RA'), 2)} | {_f(p.get('WF'), 3)} |")
    for seed, s in sorted(rep["seeds"].items()):
        row(s["run"], s["checkpoint"])
        row(s["run"], s["alpha1"])
    for name, v in rep.get("variants", {}).items():
        row(name, v["checkpoint"])
        row(name, v["alpha1"])
    for name, p in rep["comparisons"].items():
        row(name, p)
    checks = [(s["run"], s.get("answer_check", {}).get("0")) for s in rep["seeds"].values()]
    checks += [(n, v.get("answer_check", {}).get("0")) for n, v in rep.get("variants", {}).items()]
    checks += [(n, p.get("answer_check")) for n, p in rep["comparisons"].items()]
    checks = [(n, c) for n, c in checks if c and c.get("n")]
    if checks:
        lines += ["", "Paraphrased answers at the checkpoint, the same facts (paired):", "",
                  "| run | facts | original answer: UDS / strict | paraphrased answer: UDS / strict |",
                  "|---|---|---|---|"]
        for n, c in checks:
            lines.append(f"| {n} | {c['n']} | {_f(c['original']['uds'])} / "
                         f"{_f(c['original']['strict'])} | {_f(c['paraphrased']['uds'])} / "
                         f"{_f(c['paraphrased']['strict'])} |")
    c = rep["criteria"]
    lines += ["", f"E1 checkpoint strict UDS (trained, mean): {_f(c['E1_checkpoint_strict_trained']['value'])}"
              f" (>= {E1_MIN}): {c['E1_checkpoint_strict_trained']['verdict']}",
              f"E2 damage-matched strict excess (trained, per seed): "
              f"{[None if e is None else round(e, 3) for e in c['E2_excess_strict_trained']['per_seed']]}"
              f" (>= {E2_MIN} each): {c['E2_excess_strict_trained']['verdict']}",
              f"E3 paraphrase / trained strict: {_f(c['E3_paraphrase_ratio']['ratio'], 2)}"
              f" (>= {E3_RATIO}): {c['E3_paraphrase_ratio']['verdict']}",
              f"E4 RA {_f(c['E4_general_knowledge']['RA'], 2)}, WF {_f(c['E4_general_knowledge']['WF'])}"
              f" (>= {RA_MIN}, {WF_MIN}): {c['E4_general_knowledge']['verdict']}",
              f"E5 held-out UDS {_f(c['E5_heldout']['uds'])}, strict {_f(c['E5_heldout']['strict'])} (reported)"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--out", default="results/analysis/unasked_report")
    args = ap.parse_args(argv)
    rep = collect(Path(args.dir))
    if not rep["seeds"]:
        print(f"No UNASKED run under {args.dir}")
        return 1
    rep["criteria"] = criteria(rep["seeds"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = to_markdown(rep)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nWrote {out}.json and {out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
