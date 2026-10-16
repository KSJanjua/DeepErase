"""Aggregate a replication matrix into the final tables and figures.

One command from saved run directories to the numbers the report needs, so the
analysis does not wait on GPU time. Reads only JSON (no torch, no models).

For every (protocol, method, learning rate, trajectory) it reports, across
seeds:

* mean ± sd of depth, breadth, utility, selectivity and general utility at
  every alpha;
* per-run least-squares **slopes** against alpha -- depth, breadth, and each
  breadth tier (B0-B4) -- then mean ± sd across seeds. The tier slopes test
  UIPE's prediction (ref. [7] §5.2): extrapolation should remove *related*
  knowledge preferentially, so breadth should rise faster on the indirect
  tiers (B2, B3, B4) and on held-out facts than on B0;
* TOFU Forget Quality per split at alpha = 0 and at the largest alpha.

Runs marked invalid or superseded in ``results/studies/run_registry.json``,
and T1 controls, are excluded unless ``--include-all``.

    python -m deeperase.scripts.aggregate_matrix --glob 'study_v2_*'
    python -m deeperase.scripts.aggregate_matrix --glob 'study_*_1B_s*'   # the v1 matrix
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deeperase.registry import arm_of, load_registry, protocol_of, run_status, variant_of

TIERS = ("B0", "B1", "B2", "B3", "B4")


def _mean_sd(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return {"mean": None, "sd": None, "n": 0}
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return {"mean": m, "sd": sd, "n": len(xs)}


def slope(xs: Sequence[float], ys: Sequence[Optional[float]]) -> Optional[float]:
    """Least-squares slope of y on x, ignoring missing y."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and y == y]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx


def load_run(run: Path) -> Optional[dict]:
    """Per-trajectory series for one run, from plane.json and the partials."""
    try:
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        pts = json.loads((run / "plane.json").read_text(encoding="utf-8"))["points"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None
    partials = {}
    for f in sorted((run / "partial").glob("alpha_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        partials[round(float(d.get("point", d.get("points", [{}])[0]).get("alpha", -1)), 6)] = d

    trajs: Dict[str, dict] = {}
    for p in pts:
        bench = p.get("benchmark", "")
        split = bench.split(":")[-1] if ":" in bench else "all"
        t = trajs.setdefault(split, {"alpha": [], "depth": [], "breadth": [], "utility": [],
                                     "selectivity": [], "general_utility": [],
                                     **{f"tier_{k}": [] for k in TIERS}})
        a = round(float(p["alpha"]), 6)
        t["alpha"].append(a)
        for k in ("depth", "breadth", "utility", "selectivity", "general_utility"):
            t[k].append(p.get(k))
        groups = (partials.get(a) or {}).get("groups") or {}
        for k in TIERS:
            key = f"{k}:{split}" if split != "all" else f"{k}:trained"
            t[f"tier_{k}"].append((groups.get(key) or {}).get("calibrated"))
    for t in trajs.values():
        order = sorted(range(len(t["alpha"])), key=lambda i: t["alpha"][i])
        for k in list(t):
            t[k] = [t[k][i] for i in order]

    fq = {}
    if partials:
        a0, a1 = min(partials), max(partials)
        for tag, a in (("alpha0", a0), ("alpha_max", a1)):
            fq[tag] = (partials[a].get("tofu_forget_quality") or {})
    return {"run": run.name, "config": cfg, "protocol": protocol_of(cfg),
            "variant": variant_of(cfg), "arm": arm_of(run.name, cfg),
            "method": cfg.get("method"),
            "lr": (cfg.get("unlearn") or {}).get("learning_rate"),
            "seed": cfg.get("seed"), "trajectories": trajs, "forget_quality": fq}


def aggregate(runs: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for r in runs:
        for split, t in r["trajectories"].items():
            groups.setdefault((r["protocol"], r["variant"], r.get("arm", "main"), r["method"],
                               r["lr"], split), []).append(
                {"run": r["run"], "seed": r["seed"], "t": t, "fq": r["forget_quality"]})
    out = []
    for (protocol, variant, arm, method, lr, split), members in sorted(
            groups.items(), key=lambda kv: str(kv[0])):
        alphas = sorted({a for m in members for a in m["t"]["alpha"]})
        per_alpha = []
        for a in alphas:
            row = {"alpha": a}
            for k in ("depth", "breadth", "utility", "selectivity", "general_utility",
                      *(f"tier_{x}" for x in TIERS)):
                vals = []
                for m in members:
                    if a in m["t"]["alpha"]:
                        vals.append(m["t"][k][m["t"]["alpha"].index(a)])
                row[k] = _mean_sd(vals)
            per_alpha.append(row)
        slopes = {}
        for k in ("depth", "breadth", *(f"tier_{x}" for x in TIERS)):
            slopes[k] = _mean_sd([slope(m["t"]["alpha"], m["t"][k]) for m in members])
        gap = _mean_sd([
            sum(d - b for d, b in zip(m["t"]["depth"], m["t"]["breadth"])
                if d is not None and b is not None) / max(1, len(m["t"]["alpha"]))
            for m in members])
        fq = {tag: _mean_sd([(m["fq"].get(tag) or {}).get(split if split != "all" else "all")
                             for m in members]) for tag in ("alpha0", "alpha_max")}
        out.append({"protocol": protocol, "variant": variant, "arm": arm, "method": method,
                    "lr": lr, "trajectory": split,
                    "n_seeds": len(members), "runs": [m["run"] for m in members],
                    "seeds": [m["seed"] for m in members], "per_alpha": per_alpha,
                    "slopes": slopes, "mean_depth_minus_breadth": gap,
                    "tofu_forget_quality": fq})
    return out


def _arm_label(arm: Optional[str]) -> str:
    """'' for the main runs, else which arm: a control or a tagged run."""
    if not arm or arm == "main":
        return ""
    return " · matched-damage control" if arm == "control" else f" · tag `{arm}`"


def _fmt(ms: dict, digits: int = 3) -> str:
    if ms["mean"] is None:
        return "–"
    s = f"{ms['mean']:.{digits}f}"
    return s + (f" ± {ms['sd']:.{digits}f}" if ms["sd"] is not None else "")


def to_markdown(rows: List[dict]) -> str:
    lines = []
    for g in rows:
        lines.append(f"### {g['method']} · lr {g['lr']:g} · {g['protocol']} "
                     f"({g.get('variant', '?')}){_arm_label(g.get('arm'))} · "
                     f"trajectory `{g['trajectory']}` · "
                     f"{g['n_seeds']} seed(s)\n")
        lines.append("| α | depth | breadth | utility | selectivity |")
        lines.append("|---|---|---|---|---|")
        for r in g["per_alpha"]:
            lines.append(f"| {r['alpha']:.1f} | {_fmt(r['depth'])} | {_fmt(r['breadth'])} | "
                         f"{_fmt(r['utility'])} | {_fmt(r['selectivity'])} |")
        s = g["slopes"]
        tiers = ", ".join(f"{t} {_fmt(s['tier_' + t])}" for t in TIERS
                          if s["tier_" + t]["mean"] is not None)
        lines.append(f"\nSlope per unit α — depth {_fmt(s['depth'])}, breadth "
                     f"{_fmt(s['breadth'])}" + (f"; by tier: {tiers}" if tiers else "") + ".")
        lines.append(f"Mean (depth − breadth) along the path: "
                     f"{_fmt(g['mean_depth_minus_breadth'])}.\n")
    return "\n".join(lines)


def plot(rows: List[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = sorted({g["trajectory"] for g in rows})
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 5.4), squeeze=False)
    cmap = plt.get_cmap("tab10")
    methods = sorted({(g["method"], g["lr"], g.get("variant"), g.get("arm")) for g in rows},
                     key=str)
    for ax, split in zip(axes[0], splits):
        for i, (method, lr, variant, arm) in enumerate(methods):
            g = next((g for g in rows if g["trajectory"] == split and g["method"] == method
                      and g["lr"] == lr and g.get("variant") == variant
                      and g.get("arm") == arm), None)
            if not g:
                continue
            b = [r["breadth"]["mean"] for r in g["per_alpha"]]
            d = [r["depth"]["mean"] for r in g["per_alpha"]]
            be = [r["breadth"]["sd"] or 0 for r in g["per_alpha"]]
            de = [r["depth"]["sd"] or 0 for r in g["per_alpha"]]
            ax.errorbar(b, d, xerr=be, yerr=de, fmt="-o", color=cmap(i % 10), ms=4,
                        capsize=2, label=f"{method} (lr {lr:g}, {variant}"
                                         f"{'' if arm in (None, 'main') else ', ' + arm}"
                                         f", n={g['n_seeds']})")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("breadth (calibrated; 1 = retain oracle)")
        ax.set_ylabel("depth (UDS)")
        ax.set_title(f"{split} facts" if split != "all" else "all forget facts")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, ls=":")
    fig.suptitle("Depth vs breadth across seeds (mean ± sd at each α)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--glob", default="study_*")
    ap.add_argument("--include-all", action="store_true",
                    help="also aggregate invalid/superseded runs and controls")
    ap.add_argument("--out", default="results/analysis/matrix",
                    help="output prefix: writes <out>.json, <out>.md and <out>.png")
    args = ap.parse_args(argv)

    registry = load_registry(args.dir)
    runs, skipped = [], []
    for d in sorted(Path(args.dir).glob(args.glob)):
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        status, why = run_status(registry, d.name, r["config"])
        if status != "ok" and not args.include_all:
            skipped.append(f"{d.name} ({status})")
            continue
        runs.append(r)
    if not runs:
        print(f"No usable runs under {args.dir}/{args.glob}")
        return 1
    rows = aggregate(runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"runs": [r["run"] for r in runs],
                                                    "skipped": skipped, "groups": rows},
                                                   indent=2), encoding="utf-8")
    md = to_markdown(rows)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    plot(rows, out.with_suffix(".png"))
    print(md)
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print(f"\nWrote {out}.json, {out}.md, {out}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
