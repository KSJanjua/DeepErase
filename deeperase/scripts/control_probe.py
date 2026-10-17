"""Find the random-direction scale whose damage matches a real run's.

The T1 control (RESULTS.md section 7) replaces the trained update with a random
direction of the same per-tensor size. At that size it does nothing: v moves a
1B model's weights by about 0.01% of their size, and a random step that small
is absorbed (utility 0.797 -> 0.797 in v1). So it rules out distance as the
cause of the depth rise, but it cannot say whether *damage* of the real run's
size would raise depth and breadth on its own. The matched-damage control
scales the random direction up until it costs as much utility as the real run.
Each guess at that scale costs a full study (about an hour).

This probe trains once, exactly as the study does (same data, seed, rate and
checkpoint rule, hence the same update vector v). It draws the control's
random direction r exactly as run_study does (same seed), and measures utility
at ``theta_ini + (1 + alpha) * s * r`` -- the control run's sweep at scale s --
for many scales, in seconds per point, with no breadth or depth. It also
measures the real path ``theta_un + alpha * v`` at the same alphas, which must
reproduce the real run's recorded utilities.

It recommends the scale whose utility at the largest alpha matches the real
path's (interpolated on a log scale), to pass to
``gpu/replicate.py --control random --control-scale``.

    python -m deeperase.scripts.control_probe --method ga --learning-rate 1e-8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from deeperase.core.extrapolation import compute_update_vector, random_direction_like
from deeperase.provenance import environment_info, git_info
from deeperase.scripts import lr_scan
from deeperase.unlearn import CollapsedError, UnlearnConfig, snapshot, unlearn

log = logging.getLogger("control_probe")

DEFAULT_SCALES = "1,10,30,100,200,300,500,1000,2000"


def recommend_scale(points: Sequence[Tuple[float, float]], target: float
                    ) -> Tuple[Optional[float], str]:
    """The scale at which utility falls to ``target``.

    ``points`` are (scale, utility at the largest alpha), any order. Finds the
    first scale (ascending) at or below the target and interpolates log(scale)
    linearly in utility from the scale before it, rounded to two significant
    figures. Returns ``(scale or None, reason)``.
    """
    pts = sorted(points)
    if not pts:
        return None, "no points"
    if pts[0][1] <= target:
        return None, f"even the smallest scale ({pts[0][0]:g}) does at least the real damage"
    for (s0, u0), (s1, u1) in zip(pts, pts[1:]):
        if u1 <= target:
            frac = (u0 - target) / (u0 - u1) if u0 != u1 else 1.0
            s = math.exp(math.log(s0) + frac * (math.log(s1) - math.log(s0)))
            digits = 1 - int(math.floor(math.log10(s)))
            return round(s, digits), f"between {s0:g} ({u0:.3f}) and {s1:g} ({u1:.3f})"
    return None, f"no scale up to {pts[-1][0]:g} reaches the real damage"


@torch.no_grad()
def load_point(model, base: Dict[str, torch.Tensor], direction: Dict[str, torch.Tensor],
               coeff: float) -> None:
    """Set ``model`` to ``base + coeff * direction``, tensor by tensor, on its device.

    Tensors absent from ``direction`` keep ``base``. Works in place, so no
    second copy of the model is held.
    """
    sd = model.state_dict()
    for name, t in sd.items():
        b = base[name].to(t.device, torch.float32)
        if name in direction:
            b = b + coeff * direction[name].to(t.device, torch.float32)
        t.copy_(b.to(t.dtype))


def parse_args(argv: Optional[Sequence[str]] = None):
    from deeperase.unlearn import UnlearnMethod
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", required=True, choices=[m.value for m in UnlearnMethod])
    ap.add_argument("--learning-rate", type=float, required=True)
    ap.add_argument("--scales", default=DEFAULT_SCALES)
    ap.add_argument("--alphas", default="0,1",
                    help="sweep positions to measure; the largest sets the target")
    ap.add_argument("--out-dir", default="results/control_probes")
    lr_scan.add_training_args(ap)       # data, model and training: as the scan and run_study
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    scales = sorted(float(x) for x in args.scales.split(",") if x.strip())
    alphas = sorted(float(x) for x in args.alphas.split(",") if x.strip())
    ns = lr_scan.prepare(args)
    t0 = time.time()

    # -- train exactly as the study: same rate, budget and checkpoint rule ----
    cfg = UnlearnConfig(method=ns.method, learning_rate=args.learning_rate,
                        epochs=args.epochs, batch_size=args.batch_size, beta=args.beta,
                        seed=args.seed, min_utility_ratio=ns.floor,
                        retain_weight=ns.retain_weight, trainable=ns.trainable)
    try:
        history = unlearn(ns.model, ns.forget_batches, cfg, retain_data=ns.retain_batches,
                          reference_model=ns.reference, eval_fn=ns.utility_of,
                          stranger=ns.stranger)
    except CollapsedError as e:
        raise SystemExit(f"\nTraining collapsed, so there is no update to match:\n{e}")
    ns.reference = None
    theta_un = snapshot(ns.model)
    v = compute_update_vector(ns.theta_ini, theta_un, strict=False)
    # The control's direction, drawn as run_study draws it. random_direction_like
    # scales one fixed draw, so scale s gives exactly s times this direction.
    r1 = random_direction_like(v, seed=args.seed, scale=1.0)
    log.info("trained: selected epoch %s; drew the control direction", history.selected_epoch)

    real: List[dict] = []
    for a in alphas:
        load_point(ns.model, theta_un, v, a)
        real.append({"alpha": a, "utility": ns.utility_of(ns.model),
                     "general_utility": ns.general_of(ns.model)})
        log.info("real path      alpha=%.2f  utility %.4f", a, real[-1]["utility"])

    probe: List[dict] = []
    for s in scales:
        for a in alphas:
            load_point(ns.model, ns.theta_ini, r1, (1 + a) * s)
            probe.append({"scale": s, "alpha": a, "utility": ns.utility_of(ns.model),
                          "general_utility": ns.general_of(ns.model)})
            log.info("control x%-6g alpha=%.2f  utility %.4f", s, a, probe[-1]["utility"])

    top = alphas[-1]
    target = next(p["utility"] for p in real if p["alpha"] == top)
    scale, why = recommend_scale([(p["scale"], p["utility"]) for p in probe
                                  if p["alpha"] == top], target)
    record = {
        "what": "matched-damage probe: utility of the T1 control's sweep at many scales",
        "method": args.method, "learning_rate": args.learning_rate, "seed": args.seed,
        "size": args.size, "protocol": args.protocol, "epochs": args.epochs,
        "selected_epoch": history.selected_epoch,
        "baseline_utility": history.baseline_utility,
        "update_norm": float(sum(float(t.double().pow(2).sum()) for t in v.values()) ** 0.5),
        "real_path": real, "control": probe,
        "target": {"alpha": top, "utility": target},
        "recommended_scale": scale, "why": why,
        "seconds": round(time.time() - t0, 1),
        "git": git_info(), "environment": environment_info()}
    out = Path(args.out_dir) / f"{args.method}_{args.size}_{args.protocol}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"MATCHED-DAMAGE PROBE  {args.method}  lr {args.learning_rate:g}  "
          f"(real utility at alpha={top:g}: {target:.3f})")
    print("=" * 70)
    for s in scales:
        cells = "  ".join(f"a={p['alpha']:g}: {p['utility']:.3f}" for p in probe
                          if p["scale"] == s)
        print(f"  scale {s:>7g}   {cells}")
    print(f"\nRecommended --control-scale: {scale}  ({why})")
    print(f"Saved to {out}")
    return 0 if scale is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
