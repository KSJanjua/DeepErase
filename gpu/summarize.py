#!/usr/bin/env python3
"""Print one line per study run: how it trained, and what plane it produced.

Reads only the JSON artefacts, so it needs no GPU and no torch. Run it after
any batch of runs to see the whole picture without scrolling logs.

    python gpu/summarize.py                      # every run in results/studies
    python gpu/summarize.py --glob 'lrscan_npo*' # just one scan

The three numbers that decide whether a run is usable:

  sel     the epoch the checkpoint selector kept. 0 means training diverged on
          the first epoch and everything after was rejected -- the lr is too
          high. The last epoch (e.g. 19 of 20) means the run never reached the
          utility floor -- the lr is too low, and forgetting is budget-limited
          rather than damage-limited. A middle value is what you want.
  depth   span across the alpha sweep, where 1.000 is the retain oracle. Under
          ~0.1 the plane has no room to bend and no shape can be read from it.
  dutil   utility at alpha=1 minus utility at alpha=0. Near zero means the
          sweep did not damage the model, so movement on the axes is
          attributable to forgetting. Large and negative means the trajectory
          is confounded by degradation and needs the T1 control before any
          claim rests on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> dict | None:
    try:
        cfg = json.loads((run / "config.json").read_text())
        pts = json.loads((run / "plane.json").read_text())["points"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None
    if not pts:
        return None
    try:
        hist = json.loads((run / "unlearn.json").read_text())["history"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        hist = {}
    pts = sorted(pts, key=lambda p: p["alpha"])
    depths = [p["depth"] for p in pts]
    breadths = [p["breadth"] for p in pts]
    return {
        "run": run.name,
        "method": cfg.get("method", "?"),
        "lr": cfg.get("unlearn", {}).get("learning_rate", float("nan")),
        "epochs": cfg.get("unlearn", {}).get("epochs", 0),
        "sel": hist.get("selected_epoch"),
        "nll": hist.get("final_forget_nll", float("nan")),
        "d0": depths[0], "d1": depths[-1], "dspan": max(depths) - min(depths),
        "b0": breadths[0], "b1": breadths[-1],
        "u0": pts[0]["utility"], "u1": pts[-1]["utility"],
        "n": len(pts),
    }


def verdict(r: dict) -> str:
    """Why a run is or is not usable. Ordered so the fatal reason wins."""
    if r["sel"] == 0:
        return "DIVERGED (lr too high)"
    if r["sel"] is not None and r["sel"] >= r["epochs"] - 1:
        return "BUDGET-LIMITED (lr too low)"
    if r["dspan"] < 0.10:
        return "UNDERPOWERED (no span)"
    if (r["u1"] - r["u0"]) < -0.05:
        return "usable, DAMAGE-CONFOUNDED"
    return "usable, damage-controlled"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--glob", default="*")
    args = ap.parse_args()

    rows = [r for r in (load(p) for p in sorted(Path(args.dir).glob(args.glob))
                        if p.is_dir()) if r]
    if not rows:
        print(f"no completed runs under {args.dir}/{args.glob}")
        return 1

    hdr = (f"{'run':34s} {'meth':8s} {'lr':8s} {'sel':>4s} {'nll':>7s} "
           f"{'depth 0->1':16s} {'breadth 0->1':16s} {'dutil':>7s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run'][:34]:34s} {r['method'][:8]:8s} {r['lr']:<8.0e} "
              f"{str(r['sel']):>4s} {r['nll']:>7.3f} "
              f"{r['d0']:.3f}->{r['d1']:.3f}     "
              f"{r['b0']:.3f}->{r['b1']:.3f}     "
              f"{r['u1'] - r['u0']:>+7.3f}  {verdict(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
