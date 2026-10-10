#!/usr/bin/env python3
"""Print one line per study run and trajectory: how it trained, what it produced.

Reads only the JSON artefacts, so it needs no GPU and no torch. Run it after
any batch of runs to see the whole picture without scrolling logs.

    python gpu/summarize.py                      # every run in results/studies
    python gpu/summarize.py --glob 'study_v2*'   # just the protocol-v2 runs
    python gpu/summarize.py --valid-only         # hide invalid, superseded, controls

A protocol-v2 run has two trajectories -- ``trained`` (the facts the objective
targeted) and ``heldout`` (other facts about the same authors) -- so it prints
two lines. Run status comes from ``results/studies/run_registry.json``.

The columns that decide whether a run is usable:

  sel     the epoch the checkpoint selector kept. 0 means training diverged on
          the first epoch and everything after was rejected -- the lr is too
          high. The last epoch (e.g. 19 of 20) means the run never reached the
          utility floor -- the lr is too low, and forgetting is budget-limited
          rather than damage-limited. A middle value is what you want.
  depth   span across the alpha sweep, where 1.000 is the retain oracle. Under
          ~0.1 the plane has no room to bend and no shape can be read from it.
  dutil   utility at the largest alpha minus utility at alpha=0. Large and
          negative means the trajectory is confounded by degradation.
  sel@1   selectivity at the largest alpha (v2 runs): breadth minus the retain
          tier's calibrated loss. <= 0 means the forget tiers fell no further
          than neighbouring authors -- not targeted forgetting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gpu/ is not the root
from deeperase.registry import load_registry, protocol_of, run_status  # noqa: E402


def load(run: Path, registry: dict) -> list:
    """One record per trajectory in the run (v1 runs have one)."""
    try:
        cfg = json.loads((run / "config.json").read_text())
        pts = json.loads((run / "plane.json").read_text())["points"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return []
    if not pts:
        return []
    try:
        hist = json.loads((run / "unlearn.json").read_text())["history"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        hist = {}
    status, reason = run_status(registry, run.name, cfg)

    by_traj: dict = {}
    for p in pts:
        by_traj.setdefault(p.get("benchmark", ""), []).append(p)

    out = []
    for bench, tp in sorted(by_traj.items()):
        tp = sorted(tp, key=lambda p: p["alpha"])
        depths = [p["depth"] for p in tp]
        sel_last = tp[-1].get("selectivity")
        out.append({
            "run": run.name,
            "trajectory": bench.split(":")[-1] if ":" in bench else "",
            "protocol": protocol_of(cfg),
            "method": cfg.get("method", "?"),
            "control": cfg.get("control", "none"),
            "control_scale": cfg.get("control_scale") or 1.0,
            "lr": cfg.get("unlearn", {}).get("learning_rate", float("nan")),
            "epochs": cfg.get("unlearn", {}).get("epochs", 0),
            "sel": hist.get("selected_epoch"),
            "has_history": bool(hist.get("epoch_evals")),
            "nll": hist.get("final_forget_nll", float("nan")),
            "d0": depths[0], "d1": depths[-1], "dspan": max(depths) - min(depths),
            "b0": tp[0]["breadth"], "b1": tp[-1]["breadth"],
            "u0": tp[0]["utility"], "u1": tp[-1]["utility"],
            "sel_last": sel_last,
            "status": status, "reason": reason,
            "n": len(tp),
        })
    return out


def verdict(r: dict) -> str:
    """Why a run is or is not usable. Ordered so the fatal reason wins."""
    if r["status"] == "invalid":
        return "INVALID -- " + r["reason"][:60]
    if r["status"] == "control":
        # Never let a control be read as a measurement. Its axes are the
        # degradation baseline the real runs are compared against.
        k = r.get("control_scale", 1.0)
        return ("T1 CONTROL (random direction)" if k == 1.0
                else f"T1 CONTROL (random, {k:g}x norm)")
    if not r["has_history"] and r["d0"] > 0.9 and r["b0"] > 0.9:
        return "COLLAPSED (no checkpoint selection)"
    prefix = "superseded; " if r["status"] == "superseded" else ""
    if r["sel"] == 0:
        return prefix + "DIVERGED (lr too high)"
    if r["sel"] is not None and r["sel"] >= r["epochs"] - 1:
        return prefix + "BUDGET-LIMITED (lr too low)"
    if r["dspan"] < 0.10:
        return prefix + "UNDERPOWERED (no span)"
    tail = ""
    if r["sel_last"] is not None and r["sel_last"] <= 0:
        tail = ", NOT SELECTIVE"
    if (r["u1"] - r["u0"]) < -0.05:
        return prefix + "usable, DAMAGE-CONFOUNDED" + tail
    return prefix + "usable, damage-controlled" + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--glob", default="*")
    ap.add_argument("--valid-only", action="store_true",
                    help="hide invalid and superseded runs and T1 controls")
    args = ap.parse_args()

    registry = load_registry(args.dir)
    rows = [r for p in sorted(Path(args.dir).glob(args.glob)) if p.is_dir()
            for r in load(p, registry)]
    if args.valid_only:
        rows = [r for r in rows if r["status"] == "ok"]
    if not rows:
        print(f"no completed runs under {args.dir}/{args.glob}")
        return 1

    hdr = (f"{'run':34s} {'traj':7s} {'prot':4s} {'meth':8s} {'lr':8s} {'sel':>4s} "
           f"{'nll':>7s} {'depth 0->1':14s} {'breadth 0->1':14s} {'dutil':>7s} "
           f"{'sel@1':>6s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s1 = f"{r['sel_last']:+.3f}" if r["sel_last"] is not None else "   -  "
        print(f"{r['run'][:34]:34s} {r['trajectory'][:7]:7s} {r['protocol']:4s} "
              f"{r['method'][:8]:8s} {r['lr']:<8.0e} "
              f"{str(r['sel']):>4s} {r['nll']:>7.3f} "
              f"{r['d0']:.3f}->{r['d1']:.3f}   "
              f"{r['b0']:.3f}->{r['b1']:.3f}   "
              f"{r['u1'] - r['u0']:>+7.3f} {s1:>6s}  {verdict(r)}")
    notes = (registry.get("protocol_notes") or {})
    if any(r["protocol"] == "v1" for r in rows) and "v1" in notes:
        print(f"\nv1 note: {notes['v1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
