"""UNASKED-G against GradDiff: H9's criteria C1-C4 (docs/ANALYSIS_PLAN.md), from saved JSON.

For every UNASKED-G run and every GradDiff v2 seed it reads the checkpoint
(alpha = 0):

* strict UDS and UDS per depth split;
* calibrated breadth on trained and held-out facts;
* R, RA, WF.

Runs are grouped by configuration: the part of the run name after the seed
(``lr3e-06``, ``lr3e-06_g3``, ...). For each group it states C1-C4 over all
its seeds and over seeds 1-2 alone (the confirmation seeds). The lambda_g rule
is applied to seed 0 of the pre-registered family only (lr 3e-6, TOFU anchor
3, lambda_g 1 or 3):

* **C1.** strict UDS (trained) >= 0.26 in every seed, mean >= 0.282;
* **C2.** breadth (trained) >= 0.32 in every seed, mean >= 0.338;
* **C3.** mean R >= 0.768, RA >= 0.863, WF >= 0.829, and no seed below
  R 0.765, RA 0.85, WF 0.821;
* **C4.** in every seed with its R-matched control, the strict-UDS excess
  (trained) is >= 0.15.

On one seed alone (the lambda_g rule), C1 and C2 use the per-seed thresholds
and C3 the mean thresholds.

**Matched-utility comparison (exploratory, added after seeds 0-1).** Each
UNASKED-G seed is set against GradDiff's run of the same seed (the same
trained and held-out facts) along both alpha paths. At every alpha of
UNASKED-G, GradDiff's breadth and strict UDS are interpolated at UNASKED-G's
utility, once matched on general knowledge (mean of RA and WF) and once on R.
The excess is averaged over the alphas where the two utility ranges overlap.
This puts both methods at the same cost, where the checkpoint comparison puts
each at its own. GradDiff-G runs (GradDiff with the same general
anchor, the fairness baseline) are listed beside GradDiff. Exploratory.

    python -m deeperase.scripts.h9_report
    python -m deeperase.scripts.h9_report --choose     # lambda_g for seeds 1-2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deeperase.scripts.uds_audit import audit_run, strict_excess
from deeperase.scripts.unasked_report import general_matched

C1_SEED, C1_MEAN = 0.26, 0.282
C2_SEED, C2_MEAN = 0.32, 0.338
C3_MEAN = {"R": 0.768, "RA": 0.863, "WF": 0.829}
C3_SEED = {"R": 0.765, "RA": 0.85, "WF": 0.821}
C4_SEED = 0.15
GRADDIFF = "study_v2_graddiff_1B_s*_lr5e-08"
GRADDIFF_G = "study_v2_graddiffg_1B_s*_lr*"


def _tag(name: str) -> str:
    rest = name.split("_lr", 1)[1]
    return rest.split("_", 1)[1] if "_" in rest else ""


def checkpoint(run: Path) -> Optional[dict]:
    """Everything H9 reads from one run, at alpha = 0."""
    f = run / "partial" / "alpha_0.000.json"
    a = audit_run(run)
    if not f.exists() or not a:
        return None
    p = json.loads(f.read_text(encoding="utf-8"))
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    u = cfg.get("unlearn") or {}
    seed = int(cfg["seed"])
    out = {"run": run.name, "seed": seed, "tag": _tag(run.name),
           "config": run.name.split(f"_s{seed}_", 1)[-1],
           "learning_rate": u.get("learning_rate"),
           "general_weight": u.get("general_weight"), "retain_weight": u.get("retain_weight")}
    for pt in p.get("points") or [p.get("point") or {}]:
        split = str(pt.get("benchmark", "")).split(":")[-1]
        out[f"breadth_{split}"] = pt.get("breadth")
    g = p.get("groups") or {}
    for k in ("R", "RA", "WF"):
        out[k] = (g.get(k) or {}).get("rate")
    for split, rows in a["splits"].items():
        r = next((x for x in rows if abs(x["alpha"]) < 1e-9), None)
        if r:
            out[f"strict_{split}"], out[f"uds_{split}"] = r["strict"], r["uds"]
    out["_audit"] = a
    return out


def path(run: Path, audit: dict) -> List[dict]:
    """(alpha, R, general, breadth, strict) along the run's sweep, trained facts."""
    out = []
    for f in sorted((run / "partial").glob("alpha_*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        pts = p.get("points") or [p.get("point") or {}]
        pt = next((x for x in pts if str(x.get("benchmark", "")).endswith(":trained")), None)
        g = p.get("groups") or {}
        if not pt:
            continue
        a = float(pt["alpha"])
        row = next((x for x in audit["splits"].get("trained", []) if abs(x["alpha"] - a) < 1e-9),
                   None)
        ra, wf = (g.get("RA") or {}).get("rate"), (g.get("WF") or {}).get("rate")
        out.append({"alpha": a, "R": (g.get("R") or {}).get("rate"),
                    "general": (ra + wf) / 2 if ra is not None and wf is not None else None,
                    "breadth": pt.get("breadth"), "strict": row["strict"] if row else None})
    return sorted(out, key=lambda r: r["alpha"])


def _interp(x: float, pts: List[tuple]) -> Optional[float]:
    pts = sorted(p for p in pts if None not in p)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def matched_excess(ours: List[dict], ref: List[dict], axis: str) -> dict:
    """Mean excess of breadth and strict UDS over ``ref`` at equal ``axis``."""
    out = {}
    for key in ("breadth", "strict"):
        diffs = []
        for r in ours:
            if r[axis] is None or r[key] is None:
                continue
            v = _interp(r[axis], [(x[axis], x[key]) for x in ref])
            if v is not None:
                diffs.append(r[key] - v)
        out[key] = sum(diffs) / len(diffs) if diffs else None
        out["n_matched"] = len(diffs)
    return out


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _all(ok: Sequence[Optional[bool]]) -> Optional[bool]:
    return None if not ok or any(o is None for o in ok) else all(ok)


def _ge(x: Optional[float], t: float) -> Optional[bool]:
    return None if x is None else x >= t


def criteria(cps: List[dict], excess: Dict[int, Optional[float]]) -> dict:
    """C1-C4 over a list of checkpoints (one per seed)."""
    st = [c.get("strict_trained") for c in cps]
    br = [c.get("breadth_trained") for c in cps]
    means = {k: _mean([c.get(k) for c in cps]) for k in ("R", "RA", "WF")}
    c1 = _all([_ge(x, C1_SEED) for x in st] + [_ge(_mean(st), C1_MEAN)])
    c2 = _all([_ge(x, C2_SEED) for x in br] + [_ge(_mean(br), C2_MEAN)])
    c3 = _all([_ge(means[k], C3_MEAN[k]) for k in C3_MEAN]
              + [_ge(c.get(k), C3_SEED[k]) for c in cps for k in C3_SEED])
    ex = [excess.get(c["seed"]) for c in cps]
    c4 = _all([_ge(x, C4_SEED) for x in ex])
    return {"seeds": [c["seed"] for c in cps],
            "C1_depth": {"per_seed": st, "mean": _mean(st), "met": c1},
            "C2_breadth": {"per_seed": br, "mean": _mean(br), "met": c2},
            "C3_utility": {"means": means, "met": c3},
            "C4_beyond_damage": {"per_seed": ex, "met": c4}}


def single_seed_ok(c: dict) -> Dict[str, Optional[bool]]:
    """C1-C3 on one seed: per-seed thresholds for C1-C2, mean ones for C3."""
    return {"C1": _ge(c.get("strict_trained"), C1_SEED),
            "C2": _ge(c.get("breadth_trained"), C2_SEED),
            "C3": _all([_ge(c.get(k), C3_MEAN[k]) for k in C3_MEAN])}


def _general(c: dict) -> float:
    return ((c.get("RA") or 0.0) + (c.get("WF") or 0.0)) / 2


PREREGISTERED = {"learning_rate": 3e-6, "retain_weight": 3.0, "general_weights": (1.0, 3.0)}


def choose(groups: Dict[str, Dict[int, dict]]) -> Optional[dict]:
    """The lambda_g rule on seed 0: the smallest weight that meets C1-C3, else
    the one with the higher general knowledge (mean of RA and WF). Only the
    pre-registered family competes; other configurations are exploratory."""
    seed0 = {}
    for by in groups.values():
        c = by.get(0)
        if c and c.get("learning_rate") and abs(float(c["learning_rate"])
                                               - PREREGISTERED["learning_rate"]) < 1e-12 \
                and float(c.get("retain_weight") or 0) == PREREGISTERED["retain_weight"] \
                and float(c["general_weight"]) in PREREGISTERED["general_weights"]:
            seed0[float(c["general_weight"])] = c
    if not seed0:
        return None
    ok = [w for w in sorted(seed0) if all(single_seed_ok(seed0[w]).values())]
    if ok:
        return {"general_weight": ok[0], "reason": "smallest weight meeting C1-C3 on seed 0"}
    w = max(sorted(seed0), key=lambda k: _general(seed0[k]))
    return {"general_weight": w, "reason": "none meets C1-C3 on seed 0: higher general knowledge"}


def collect(studies: Path) -> dict:
    groups: Dict[str, Dict[int, dict]] = {}
    for d in sorted(studies.glob("study_v2_unaskedg_1B_s*_lr*")):
        c = checkpoint(d) if d.is_dir() else None
        if c and c.get("general_weight"):
            groups.setdefault(c["config"], {})[c["seed"]] = c
    controls: Dict[tuple, dict] = {}
    control_dirs: Dict[tuple, List[Path]] = {}
    for d in sorted(studies.glob("control_x*_v2_unaskedg_1B_s*_lr*")):
        a = audit_run(d) if d.is_dir() else None
        if a:
            key = (int(a["seed"]), _tag(d.name))
            controls[key] = {"name": d.name, "audit": a}
            control_dirs.setdefault(key, []).append(d)
    graddiff = [c for c in (checkpoint(d) for d in sorted(studies.glob(GRADDIFF))
                            if d.is_dir() and not _tag(d.name)) if c]
    gd_paths = {c["seed"]: path(studies / c["run"], c["_audit"]) for c in graddiff}
    graddiff_g = [c for c in (checkpoint(d) for d in sorted(studies.glob(GRADDIFF_G))
                              if d.is_dir()) if c]
    report = {"graddiff": [{k: v for k, v in c.items() if k != "_audit"} for c in graddiff],
              "graddiff_g": [{k: v for k, v in c.items() if k != "_audit"} for c in graddiff_g],
              "groups": {}}
    for w, by in sorted(groups.items()):
        excess: Dict[int, Optional[float]] = {}
        for seed, c in by.items():
            ctl = controls.get((seed, c["tag"]))
            ex = strict_excess(c["_audit"], {seed: ctl["audit"]}) if ctl else None
            excess[seed] = ((((ex or {}).get("trained") or {}).get("strict") or {})
                            .get("mean")) if ctl else None
            c["control"] = ctl["name"] if ctl else None
            c["excess_strict_trained"] = excess[seed]
            dirs = control_dirs.get((seed, c["tag"]))
            gm = general_matched(studies / c["run"], c["_audit"], dirs) if dirs else None
            c["excess_strict_general_matched"] = gm["mean"] if gm else None
        for seed, c in by.items():
            ref = gd_paths.get(seed)
            if ref:
                ours = path(studies / c["run"], c["_audit"])
                c["vs_graddiff_same_seed"] = {
                    "matched_general": matched_excess(ours, ref, "general"),
                    "matched_R": matched_excess(ours, ref, "R")}
        cps = [by[s] for s in sorted(by)]
        conf = [by[s] for s in sorted(by) if s in (1, 2)]
        report["groups"][w] = {
            "runs": [{k: v for k, v in c.items() if k != "_audit"} for c in cps],
            "all_seeds": criteria(cps, excess),
            "confirmation_seeds_1_2": criteria(conf, excess) if conf else None}
    report["choice"] = choose(groups)
    return report


def _f(x, d=3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def _verdict(m: Optional[bool]) -> str:
    return "no data" if m is None else ("met" if m else "not met")


def to_markdown(rep: dict) -> str:
    cols = ("strict_trained", "uds_trained", "breadth_trained", "strict_heldout",
            "breadth_heldout", "strict_trained_para", "R", "RA", "WF")
    lines = ["| run | λ_g | seed | strict T | UDS T | breadth T | strict H | breadth H "
             "| strict para-Q | R | RA | WF | excess over control: R-matched / general-matched |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w, g in rep["groups"].items():
        for c in g["runs"]:
            lines.append(f"| {c['run']} | {c['general_weight']:g} | {c['seed']} | "
                         + " | ".join(_f(c.get(k)) for k in cols)
                         + f" | {_f(c.get('excess_strict_trained'))} / "
                         f"{_f(c.get('excess_strict_general_matched'))} |")
    for c in rep["graddiff"] + rep.get("graddiff_g", []):
        lines.append(f"| {c['run']} | {_f(c.get('general_weight'), 0) if c.get('general_weight') else '–'} | {c['seed']} | "
                     + " | ".join(_f(c.get(k)) for k in cols) + " | – |")
    for w, g in rep["groups"].items():
        for label, crit in (("all seeds", g["all_seeds"]),
                            ("seeds 1-2", g["confirmation_seeds_1_2"])):
            if not crit:
                continue
            m = crit["C3_utility"]["means"]
            gw = g["runs"][0]["general_weight"]
            lines += ["", f"**{w} (λ_g = {gw:g}), {label} ({crit['seeds']}):** "
                      f"C1 depth {_verdict(crit['C1_depth']['met'])} (mean "
                      f"{_f(crit['C1_depth']['mean'])}); "
                      f"C2 breadth {_verdict(crit['C2_breadth']['met'])} (mean "
                      f"{_f(crit['C2_breadth']['mean'])}); "
                      f"C3 utility {_verdict(crit['C3_utility']['met'])} (R {_f(m['R'])}, "
                      f"RA {_f(m['RA'])}, WF {_f(m['WF'])}); "
                      f"C4 beyond damage {_verdict(crit['C4_beyond_damage']['met'])}"]
    ch = rep.get("choice")
    if ch:
        lines += ["", f"λ_g rule on seed 0: {ch['general_weight']:g} ({ch['reason']})"]
    lines += ["", "Against GradDiff's run of the same seed, at equal utility along both α paths "
              "(exploratory): mean excess of breadth / strict UDS (trained), number of matched α", "",
              "| run | matched on general knowledge | matched on R |", "|---|---|---|"]
    for w, g in rep["groups"].items():
        for c in g["runs"]:
            v = c.get("vs_graddiff_same_seed")
            if not v:
                continue
            cells = [f"{_f(v[k]['breadth'])} / {_f(v[k]['strict'])} ({v[k]['n_matched']})"
                     for k in ("matched_general", "matched_R")]
            lines.append(f"| {c['run']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--out", default="results/analysis/h9_report")
    ap.add_argument("--choose", action="store_true",
                    help="print only the general-anchor weight the lambda_g rule picks")
    args = ap.parse_args(argv)
    rep = collect(Path(args.dir))
    if args.choose:
        ch = rep.get("choice")
        print(f"{ch['general_weight']:g}" if ch else "")
        return 0 if ch else 1
    if not rep["groups"]:
        print(f"No UNASKED-G run under {args.dir}")
        return 1
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
