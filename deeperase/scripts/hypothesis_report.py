"""Apply the pre-registered decision rules to the protocol-v2 runs.

``docs/ANALYSIS_PLAN.md`` fixed, before any v2 result, what each hypothesis
predicts and what counts as support. This applies those rules, seed by seed,
to the saved run directories, so every verdict can be re-derived rather than
read off a table by hand. It reads only JSON.

It also reports the **damage-matched excess**: each real point's depth and
breadth minus the matched-damage control's at the same R utility
(interpolated along the control's path). That is the comparison H1 asks for,
made at every alpha instead of only at alpha = 1, and it is exploratory beyond
the H1 rule itself.

    python -m deeperase.scripts.hypothesis_report
    python -m deeperase.scripts.hypothesis_report --fact-json results/analysis/fact_level_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from deeperase.registry import load_registry, run_status, variant_of
from deeperase.scripts.aggregate_matrix import load_run, slope

METHODS = ("ga", "graddiff", "npo")
MAIN_VARIANT = "answer/float32"
B3_MIN_RANGE = 0.05            # ANALYSIS_PLAN H5
S0_MIN, S0_CI_MIN = 0.25, 0.10  # ANALYSIS_PLAN H2
H5_RATIO = 0.5


# ---------------------------------------------------------------------------
# The rules, as pure functions of per-seed numbers
# ---------------------------------------------------------------------------

def h1_verdict(max_sel: Dict[int, float], real_top: Optional[Tuple[float, float]],
               control_top: Optional[Tuple[float, float]]) -> dict:
    """Selectivity > 0 at some alpha in every seed, and the real run's depth
    and breadth both above the matched control's at the top alpha."""
    sel_ok = {s: v > 0 for s, v in max_sel.items()}
    beats = None
    if real_top and control_top:
        beats = real_top[0] > control_top[0] and real_top[1] > control_top[1]
    if not sel_ok:
        verdict = "no data"
    elif all(sel_ok.values()) and beats:
        verdict = "supported"
    else:
        reasons = [f"selectivity never > 0 in seed {s}" for s, ok in sorted(sel_ok.items())
                   if not ok]
        if beats is False:
            reasons.append("does not beat the matched control on both axes")
        if beats is None:
            reasons.append("no matched control")
        verdict = "not supported: " + "; ".join(reasons)
    return {"verdict": verdict, "max_selectivity": max_sel, "beats_control": beats,
            "real_top": real_top, "control_top": control_top}


def h2_verdict(per_seed: Dict[int, Dict[str, dict]]) -> dict:
    """per_seed[seed][theta] = {"s0", "s0_ci", "s1"} (suppression rates)."""
    def common(theta):
        ok = [s for s, th in per_seed.items()
              if (th.get(theta) or {}).get("s0") is not None
              and th[theta]["s0"] > S0_MIN
              and (th[theta].get("s0_ci") or [0])[0] > S0_CI_MIN]
        return len(ok)

    def converts(theta):
        res = {}
        for s, th in per_seed.items():
            x = th.get(theta) or {}
            if x.get("s0") is not None and x.get("s1") is not None:
                res[s] = x["s1"] < x["s0"]
        return res

    out = {"seeds": len(per_seed)}
    for theta in ("0.5", "0.3", "0.7"):
        out[f"common@{theta}"] = common(theta)
        out[f"converts@{theta}"] = converts(theta)
    common_ok = out["common@0.5"] >= 2
    conv = out["converts@0.5"]
    conv_ok = bool(conv) and all(conv.values()) and len(conv) == len(per_seed)
    robust = all(out[f"common@{t}"] >= 2 for t in ("0.3", "0.7")) and all(
        all(out[f"converts@{t}"].values()) and out[f"converts@{t}"] for t in ("0.3", "0.7"))
    out["suppression_common"] = "supported" if common_ok else "not supported"
    out["extrapolation_converts"] = "supported" if conv_ok else (
        "not supported: S(1) >= S(0) in seed(s) "
        + ", ".join(str(s) for s, v in sorted(conv.items()) if not v))
    out["robust_to_theta"] = robust
    return out


def h3_verdict(per_seed: Dict[int, List[Tuple[float, Optional[float], Optional[List[float]]]]]
               ) -> dict:
    """per_seed[seed] = [(alpha, rho, ci), ...]."""
    top = {s: max(rows, key=lambda r: r[0]) for s, rows in per_seed.items() if rows}
    support = bool(top) and all(r[2] is not None and r[2][0] > 0 for r in top.values())
    refuted = bool(per_seed) and all(
        all(ci is None or ci[0] <= 0 <= ci[1] for a, _, ci in rows if a > 0)
        for rows in per_seed.values())
    verdict = ("supported" if support else
               "refuted: no seed's interval excludes 0 at any alpha > 0" if refuted else
               "not supported: the interval excludes 0 in some seeds or alphas only")
    return {"verdict": verdict,
            "rho_at_top": {s: (r[1], r[2]) for s, r in top.items()}}


def h4_verdict(per_seed: Dict[int, dict]) -> dict:
    """per_seed[seed] = {"breadth_slope", "depth_slope", "max_selectivity"} (held-out)."""
    fails = []
    for s, x in sorted(per_seed.items()):
        if not (x["breadth_slope"] or 0) > 0:
            fails.append(f"held-out breadth slope <= 0 in seed {s}")
        if not (x["depth_slope"] or 0) > 0:
            fails.append(f"held-out depth slope <= 0 in seed {s}")
        if not (x["max_selectivity"] or 0) > 0:
            fails.append(f"held-out selectivity never > 0 in seed {s}")
    return {"verdict": "supported" if per_seed and not fails else
            ("no data" if not per_seed else "not supported: " + "; ".join(fails)),
            "per_seed": per_seed}


def h5_verdict(per_seed: Dict[int, dict]) -> dict:
    """per_seed[seed] = {"B0", "B2", "B3", "B4" slopes, "b3_range"} (trained)."""
    fails = []
    for s, x in sorted(per_seed.items()):
        for t in ("B2", "B4"):
            if x.get(t) is None or x.get("B0") is None or x[t] < H5_RATIO * x["B0"]:
                fails.append(f"{t} slope < {H5_RATIO} x B0 in seed {s}")
    b3 = {s: (x.get("B3") is not None and (x.get("b3_range") or 0) >= B3_MIN_RANGE
              and x["B3"] >= H5_RATIO * (x.get("B0") or 0))
          for s, x in per_seed.items()}
    b3_measurable = {s: (x.get("b3_range") or 0) >= B3_MIN_RANGE for s, x in per_seed.items()}
    return {"verdict": "supported" if per_seed and not fails else
            ("no data" if not per_seed else "not supported: " + "; ".join(fails)),
            "b3_measurable": b3_measurable, "b3_meets_rule": b3, "per_seed": per_seed}


def excess_over_control(real: dict, control: dict, key: str) -> List[Optional[float]]:
    """Real value minus the control's at the same utility, per real point.
    None where the real utility lies outside the control path's range."""
    cu = np.asarray(control["utility"], dtype=float)
    cv = np.asarray(control[key], dtype=float)
    order = np.argsort(cu)
    out = []
    for u, v in zip(real["utility"], real[key]):
        if u is None or v is None or not (cu.min() - 1e-9 <= u <= cu.max() + 1e-9):
            out.append(None)
        else:
            out.append(float(v - np.interp(u, cu[order], cv[order])))
    return out


# ---------------------------------------------------------------------------
# Gathering the numbers from the run directories
# ---------------------------------------------------------------------------

def find_runs(studies: Path, method: str) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    """The main runs by seed, and the matched-damage control of each seed that has one."""
    registry = load_registry(studies)
    runs: Dict[int, Path] = {}
    controls: Dict[int, list] = {}
    for d in sorted(studies.glob(f"*_v2_{method}_*")):
        try:
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if variant_of(cfg) != MAIN_VARIANT or not (d / "plane.json").exists():
            continue
        status, _ = run_status(registry, d.name, cfg)
        seed = int(cfg.get("seed", 0))
        if status == "ok":
            runs[seed] = d
        elif status == "control":
            controls.setdefault(seed, []).append((cfg.get("control_scale") or 1.0, d))
    chosen: Dict[int, Path] = {}
    for seed, cands in controls.items():
        probe = studies.parent / "control_probes" / f"{method}_1B_v2_s{seed}.json"
        target = (json.loads(probe.read_text(encoding="utf-8")).get("recommended_scale")
                  if probe.exists() else None)
        match = [d for sc, d in cands if target is not None and abs(sc - target) < 1e-6]
        chosen[seed] = match[0] if match else max(cands)[1]
    return runs, chosen


def collect(method: str, studies: Path, facts: Dict[str, dict]) -> dict:
    runs, control_dirs = find_runs(studies, method)
    loaded = {s: load_run(d) for s, d in runs.items()}
    controls = {s: load_run(d) for s, d in control_dirs.items()}
    ctl = controls.get(0)                  # H1's rule compares seed 0 (ANALYSIS_PLAN)

    max_sel = {s: max((x for x in r["trajectories"]["trained"]["selectivity"] if x is not None),
                      default=float("-inf")) for s, r in loaded.items()}
    real_top = control_top = None
    if 0 in loaded and ctl:
        t, c = loaded[0]["trajectories"]["trained"], ctl["trajectories"]["trained"]
        real_top = (t["depth"][-1], t["breadth"][-1])
        control_top = (c["depth"][-1], c["breadth"][-1])

    h2_in, h3_in = {}, {}
    for s, d in runs.items():
        f = facts.get(d.name)
        if not f:
            continue
        rows = f["splits"]["trained"]["per_alpha"]
        a0, a1 = rows[0], rows[-1]
        h2_in[s] = {th: {"s0": a0["classes"][th]["suppression_rate"],
                         "s0_ci": a0["classes"][th]["suppression_rate_ci"],
                         "s1": a1["classes"][th]["suppression_rate"]}
                    for th in ("0.3", "0.5", "0.7")}
        h3_in[s] = [(r["alpha"], r["rho_margin"], r["rho_margin_ci"]) for r in rows]

    h4_in, h5_in, excess = {}, {}, {}
    for s, r in loaded.items():
        ho, tr = r["trajectories"]["heldout"], r["trajectories"]["trained"]
        h4_in[s] = {"breadth_slope": slope(ho["alpha"], ho["breadth"]),
                    "depth_slope": slope(ho["alpha"], ho["depth"]),
                    "max_selectivity": max((x for x in ho["selectivity"] if x is not None),
                                           default=None)}
        cal = json.loads((runs[s] / "calibration.json").read_text(encoding="utf-8"))
        rr = cal["reference_rates"]
        b3r = None
        if "B3:trained" in rr.get("full", {}):
            b3r = rr["full"]["B3:trained"]["rate"] - rr["retain90"]["B3:trained"]["rate"]
        h5_in[s] = {t: slope(tr["alpha"], tr[f"tier_{t}"]) for t in ("B0", "B2", "B3", "B4")}
        h5_in[s]["b3_range"] = b3r
        own = controls.get(s) or ctl           # a seed's own control when it has one
        if own:
            excess[s] = {"control": (control_dirs.get(s) or control_dirs[0]).name,
                         **{k: {key: excess_over_control(r["trajectories"][k],
                                                         own["trajectories"][k], key)
                                for key in ("depth", "breadth")}
                            for k in ("trained", "heldout")}}

    return {"method": method, "runs": {s: d.name for s, d in sorted(runs.items())},
            "control": control_dirs[0].name if 0 in control_dirs else None,
            "controls_by_seed": {s: d.name for s, d in sorted(control_dirs.items())},
            "H1": h1_verdict(max_sel, real_top, control_top),
            "H2": h2_verdict(h2_in), "H3": h3_verdict(h3_in),
            "H4": h4_verdict(h4_in), "H5": h5_verdict(h5_in),
            "damage_matched_excess": excess}


def to_markdown(report: Dict[str, dict]) -> str:
    lines = ["| Hypothesis | " + " | ".join(m for m in report) + " |",
             "|---|" + "---|" * len(report)]
    def cell(x):
        return x.replace("|", "/")
    rows = [("H1 selectivity (gate)", lambda r: r["H1"]["verdict"]),
            ("H2 suppression common at α = 0", lambda r: r["H2"]["suppression_common"]),
            ("H2 extrapolation converts it", lambda r: r["H2"]["extrapolation_converts"]),
            ("H2 robust to θ = 0.3 / 0.7", lambda r: "yes" if r["H2"]["robust_to_theta"] else "no"),
            ("H3 fact-level coupling", lambda r: r["H3"]["verdict"]),
            ("H4 reach to untrained facts", lambda r: r["H4"]["verdict"]),
            ("H5 indirect tiers (B2, B4)", lambda r: r["H5"]["verdict"])]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(cell(fn(r)) for r in report.values()) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--fact-json", default=None,
                    help="reuse a fact_level_analysis JSON instead of recomputing")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/analysis/hypotheses_v2")
    args = ap.parse_args(argv)

    studies = Path(args.dir)
    facts: Dict[str, dict] = {}
    if args.fact_json:
        facts = {r["run"]: r for r in
                 json.loads(Path(args.fact_json).read_text(encoding="utf-8"))["runs"]}
    else:
        from deeperase.scripts.fact_level_analysis import analyse_run
        for m in METHODS:
            for d in find_runs(studies, m)[0].values():
                facts[d.name] = analyse_run(d, n_boot=args.n_boot)

    report = {m: collect(m, studies, facts) for m in METHODS}
    report = {m: r for m, r in report.items() if r["runs"]}
    if not report:
        print(f"No protocol-v2 runs under {studies}")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str),
                                        encoding="utf-8")
    md = to_markdown(report)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"\nWrote {out}.json and {out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
