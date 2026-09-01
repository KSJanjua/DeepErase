"""The depth-breadth study: does forgetting more broadly mean forgetting less deeply?

This is the experiment everything else was built for.

Procedure
---------
1. Load ``M_full`` and snapshot its weights as ``θ_ini``.
2. Unlearn it -> ``θ_un``.
3. Form the update vector ``v = θ_un − θ_ini``.
4. For each α, build ``θ(α) = θ_un + α·v`` -- pure arithmetic, no retraining.
5. At every α measure **depth** (UDS), **breadth** (tiered forced choice) and
   **utility** (retention on unrelated knowledge).
6. Plot depth against breadth, with α tracing a trajectory.

Reading the outcome
-------------------
==========================  =========================================
Trajectory                  Interpretation
==========================  =========================================
breadth up, depth down      supports the trade-off hypothesis
breadth up, depth up        contradicts it; the axes agree
flat / scattered            no relationship -- also informative, since
                            the field assumes agreement
==========================  =========================================

All three are worth reporting. The hypothesis is written so it can fail.

The control that decides whether any of it counts
-------------------------------------------------
Utility is measured at every α. Push the dial hard enough and the model simply
breaks, and a broken model forgets everything -- which looks exactly like
success on the forget axis. Any point whose retention has collapsed is excluded
from the trade-off statistic, because at that point we are measuring damage,
not forgetting.

Usage:
    python -m deeperase.scripts.run_study --method ga --size 1B
    python -m deeperase.scripts.run_study --method npo --alphas 0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from deeperase.config import TOFU_MODELS, RunConfig, plan_memory
from deeperase.core.extrapolation import (
    alpha_grid, compute_update_vector, extrapolate, global_norm,
    random_direction_like,
)
from deeperase.data.reference_spans import load_reference_annotations
from deeperase.data.tofu import (
    FORGET_TO_RETAIN, PromptFormat, SamplingStrategy, build_prompt, load_tofu,
    select_examples, tokenise_example,
)
from deeperase.eval.breadth import (
    BreadthCalibration, load_breadth_items, score_breadth,
)
from deeperase.eval.patching import n_layers
from deeperase.eval.plane import PlaneDataset, PlanePoint, plot_plane
from deeperase.eval.uds import (
    UDSExample, assemble_report, capture_source_hidden, score_from_captured,
)
from deeperase.models import ModelManager, load_model, load_tokenizer
from deeperase.scripts.measure_breadth import PROMPT_PREFIXES, subsample
from deeperase.unlearn import (
    CollapsedError, UnlearnConfig, UnlearnMethod, snapshot, unlearn,
    verify_unlearning,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("study")
logging.getLogger("deeperase").setLevel(logging.INFO)

#: Retention below this fraction of the starting value means the model is
#: damaged rather than selectively forgetful.
UTILITY_COLLAPSE_RATIO = 0.75


def build_uds_examples(tokenizer, reference_spans, fmt, max_length, n,
                       strategy: SamplingStrategy = SamplingStrategy.EVEN,
                       seed: int = 0) -> List[UDSExample]:
    """Depth examples, using the authors' hand annotations.

    Our own extractor agrees with those only 12% of the time (RESULTS.md §4),
    so it is not used here.

    ``strategy`` matters for replication. EVEN is deterministic and is what
    every published run here used. RANDOM varies which examples are drawn with
    ``seed``, which is the only real source of run-to-run variation in this
    pipeline -- see the note on ``--seed`` in :func:`main`.
    """
    kept, stats = load_reference_annotations(reference_spans)
    log.info("  reference annotations: %d examples (%s)", len(kept), stats)

    if n >= len(kept):
        chosen = kept
    elif strategy is SamplingStrategy.RANDOM:
        import random
        chosen = sorted(random.Random(seed).sample(list(kept), n),
                        key=lambda e: e.index)
    else:
        step = (len(kept) - 1) / (n - 1) if n > 1 else 1
        chosen = [kept[int(round(i * step))] for i in range(n)]
    log.info("  depth example sampling: %s (seed %d)", strategy.value, seed)

    out: List[UDSExample] = []
    for ex in chosen:
        tok = tokenise_example(ex, tokenizer, fmt=fmt, max_length=max_length)
        if tok is None:
            continue
        ids = torch.tensor([tok.input_ids], dtype=torch.long)
        out.append(UDSExample(f"ref_{ex.index}", ids, tok.to_entity_span(),
                              attention_mask=torch.ones_like(ids)))
    log.info("  %d depth examples tokenised", len(out))
    return out


#: The forget split this study is built around. The reference entity-span
#: annotations exist only for forget10, and the breadth tiers are drawn from
#: forget10_perturbed, so the two must agree.
FORGET_CONFIG = "forget10"


def build_retain_batches(tokenizer, fmt, max_length, n, cache_dir,
                         strategy: SamplingStrategy, seed: int
                         ) -> List[Dict[str, torch.Tensor]]:
    """Token batches from the **retain** split, for GradDiff's retention term.

    GradDiff minimises loss on data that must be preserved while maximising it
    on the forget set, so the two sets must be disjoint. TOFU's splits are
    nested, and ``FORGET_TO_RETAIN`` is the pairing that keeps them so:
    forget10's complement is retain90.

    This exists because an earlier version passed the *forget* batches in as
    retain data. The objective then collapses to
    ``(retain_weight - 1) * NLL(forget)``, which at the default weight of 1.0
    is identically zero -- the model would not train at all, would keep full
    utility, would pass checkpoint selection, and would emit a flat trajectory
    that looked like a legitimate null result.
    """
    retain_config = FORGET_TO_RETAIN[FORGET_CONFIG]
    examples = load_tofu(retain_config, cache_dir=cache_dir)
    chosen = select_examples(examples, n, strategy=strategy, seed=seed)
    log.info("  retain split %s: %d of %d examples (%s)",
             retain_config, len(chosen), len(examples), strategy.value)

    out: List[Dict[str, torch.Tensor]] = []
    for ex in chosen:
        text, _ = build_prompt(ex.question, ex.answer, fmt, tokenizer)
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        add_special_tokens=True)
        ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Depth-breadth trade-off study")
    ap.add_argument("--method", choices=[m.value for m in UnlearnMethod], default="ga")
    ap.add_argument("--size", choices=sorted(TOFU_MODELS), default="1B")
    ap.add_argument("--alphas", default=None,
                    help="Comma-separated. Default: 0.0..1.0 in 11 steps")
    ap.add_argument("--n-examples", type=int, default=100, help="depth examples")
    # 100/tier gave 200 scored forget items, so one flipped item moved
    # calibrated breadth by 0.0185 -- the entire signal of the first real sweep
    # was five items. 400 quarters that step.
    ap.add_argument("--n-breadth", type=int, default=400,
                    help="breadth items per tier (resolution: one item = "
                         "1/(2*N) raw, divided by the calibrated range)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=1e-6,
                    help="1e-6 for forget10 (UIPE Appendix D.3). 1e-5 is for the "
                         "smaller splits and collapses GA here.")
    ap.add_argument("--min-utility-ratio", type=float, default=0.9,
                    help="Checkpoint selection floor: keep at least this "
                         "fraction of the starting model's utility.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--prompt-format", default="plain_qa",
                    choices=[f.value for f in PromptFormat])
    ap.add_argument("--reference-spans",
                    default="reference_uds/tofu_data/forget10_filtered.json")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed. Sets torch.manual_seed and, with "
                         "--sampling random, chooses which examples are drawn. "
                         "NOTE: training here has no shuffling and the Llama "
                         "dropout is 0, so the seed alone does NOT change the "
                         "result -- use --sampling random to get genuinely "
                         "independent replicates.")
    ap.add_argument("--sampling", default="even",
                    choices=[s.value for s in SamplingStrategy],
                    help="How depth/retain examples are drawn. 'even' is "
                         "deterministic and reproduces the published runs. Use "
                         "'random' with distinct --seed values for replication.")
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--output-dir", default="results/studies")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--control", default="none", choices=["none", "random"],
                    help="'random' runs the T1 degradation control: train "
                         "normally, then discard the trained DIRECTION and keep "
                         "only its per-tensor magnitude, sweeping a random "
                         "direction instead. Every other stage is identical. If "
                         "depth and breadth rise the same way here as in the "
                         "real run, the trajectory measures degradation rather "
                         "than forgetting and no depth/breadth claim survives.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    method = UnlearnMethod(args.method)
    sampling = SamplingStrategy(args.sampling)

    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except ImportError:
        pass
    if sampling is SamplingStrategy.EVEN and args.seed != 0:
        log.warning(
            "--seed %d has no effect under --sampling even: example selection is "
            "deterministic, training does not shuffle, and dropout is 0. This run "
            "will be identical to seed 0. Use --sampling random for replicates.",
            args.seed)

    run_id = args.run_id or (f"study_{args.method}_{args.size}_s{args.seed}"
                             f"_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.output_dir) / run_id
    (out_dir / "partial").mkdir(parents=True, exist_ok=True)

    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas
              else alpha_grid(0.0, 1.0, num=11))

    print("=" * 76)
    print(f"DEPTH-BREADTH STUDY  |  method={args.method}  size={args.size}")
    print(f"run={run_id}")
    print("=" * 76)

    # -- memory -------------------------------------------------------------
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Free, not total. This study trains, and it may not be alone on the
        # card; a plan built from capacity is a plan built from a number nobody
        # is offering us.
        free_gb = torch.cuda.mem_get_info(torch.device("cuda:0"))[0] / 1e9
    else:
        total_gb, free_gb = 8.0, 8.0
        log.warning("No GPU detected -- this will be impractically slow.")
    cfg = RunConfig(size_label=args.size, cache_dir=args.cache_dir,
                    max_seq_length=args.max_length)
    plan = plan_memory(args.size, total_gb, dtype_size=cfg.dtype_size,
                       gpu_free_gb=free_gb, training=True)
    print(f"\n{plan.summary()}\n")
    if not plan.fits:
        # Fail here, not forty minutes in after the reference phase.
        raise SystemExit(
            "Refusing to start: the training phase will not fit.\n" + plan.reason
        )
    cfg.strategy = plan.strategy

    (out_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "method": args.method, "size": args.size,
        "alphas": alphas, "adaptation": "full_parameter",
        "unlearn": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size, "beta": args.beta},
        "n_depth_examples": args.n_examples, "n_breadth_per_tier": args.n_breadth,
        "prompt_format": args.prompt_format,
        "seed": args.seed, "sampling": sampling.value,
        "control": args.control,
        "forget_config": FORGET_CONFIG,
        "retain_config": (FORGET_TO_RETAIN[FORGET_CONFIG]
                          if method.needs_retain else None),
    }, indent=2), encoding="utf-8")

    fmt = PromptFormat(args.prompt_format)
    tokenizer = load_tokenizer(cfg.models()["full"].repo_id, cache_dir=cfg.cache_dir)

    # -- data ---------------------------------------------------------------
    log.info("[1/6] Preparing evaluation data")
    uds_examples = build_uds_examples(tokenizer, args.reference_spans, fmt,
                                      args.max_length, args.n_examples,
                                      sampling, args.seed)
    breadth_items = subsample(load_breadth_items(cache_dir=args.cache_dir),
                              args.n_breadth)
    log.info("  %d breadth items", len(breadth_items))
    prefix = PROMPT_PREFIXES.get(args.prompt_format, "{question} ")

    # Forget-set token batches for training, reusing the depth examples.
    forget_batches = [{"input_ids": e.input_ids, "attention_mask": e.attention_mask}
                      for e in uds_examples]

    # -- reference measurements + calibration -------------------------------
    log.info("[2/6] Measuring reference models (anchors both axes)")
    with ModelManager(cfg) as mgr:
        layers = None
        ref_breadth: Dict[str, float] = {}
        ref_retention: Dict[str, float] = {}
        for split in ("full", "retain90"):
            with mgr.acquire(split) as m:
                if layers is None:
                    layers = list(range(n_layers(m)))
                r = score_breadth(m, tokenizer, breadth_items,
                                  max_length=args.max_length, prompt_prefix=prefix)
            ref_breadth[split] = r.forget_leakage
            ref_retention[split] = r.retention
            log.info("  %s leakage=%.3f retention=%.3f", split,
                     r.forget_leakage, r.retention)

        # The utility floor must be anchored to the ORIGINAL model. An earlier
        # version took it from the alpha=0 point, which was itself already
        # collapsed -- so 0.21/0.21 = 1.0 and the guard never fired.
        original_retention = ref_retention["full"]

        try:
            calibration = BreadthCalibration.from_reference_models(
                absent_leakage=ref_breadth["retain90"],
                present_leakage=ref_breadth["full"])
        except ValueError as e:
            raise SystemExit(
                f"\nBreadth calibration failed: {e}\n"
                "The reference models are not discriminating, so the breadth "
                "axis cannot be anchored. Refusing to continue."
            )
        log.info("  calibration: floor=%.3f ceiling=%.3f range=%.3f",
                 calibration.floor, calibration.ceiling, calibration.dynamic_range)

        # -- Stage-1 depth baseline, computed once -------------------------
        log.info("[3/6] Depth Stage-1 baseline (retain90 -> full)")
        with mgr.acquire("retain90") as m_ret:
            cap = capture_source_hidden(m_ret, uds_examples, layers, to_cpu=True)
        with mgr.acquire("full") as m_full:
            d1, s_full = score_from_captured(m_full, cap, uds_examples, layers)
        n_ke = sum(1 for v in d1.values() if max(v.values()) > cfg.tau)
        log.info("  %d/%d examples have a Knowledge-Encoding layer", n_ke, len(d1))
        if n_ke == 0:
            raise SystemExit(
                "\nNo example has a Knowledge-Encoding layer. Depth is undefined "
                "for every example, so the study cannot proceed."
            )

    # -- unlearn ------------------------------------------------------------
    log.info("[4/6] Unlearning with %s", args.method)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.models()["full"].repo_id, device=device,
                       dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
                       cache_dir=cfg.cache_dir)
    theta_ini = snapshot(model)

    reference_model = None
    if method.needs_reference:
        reference_model = load_model(cfg.models()["full"].repo_id, device=device,
                                     dtype=torch.float32 if device.type == "cpu"
                                     else torch.bfloat16, cache_dir=cfg.cache_dir)

    ucfg = UnlearnConfig(method=method, learning_rate=args.learning_rate,
                         epochs=args.epochs, batch_size=args.batch_size,
                         beta=args.beta, min_utility_ratio=args.min_utility_ratio,
                         seed=args.seed)

    def utility_of(m) -> float:
        """Retention on unrelated knowledge, for checkpoint selection.

        Scored on the R tier only -- the forget tiers are what we are trying to
        move, so including them would penalise successful unlearning.
        """
        r_items = [i for i in breadth_items if i.tier == "R"]
        return score_breadth(m, tokenizer, r_items, max_length=args.max_length,
                             prompt_prefix=prefix).tiers["R"].knows_rate

    retain_batches = None
    if method.needs_retain:
        log.info("[3b/6] Loading retain split for %s", method.value)
        retain_batches = build_retain_batches(
            tokenizer, fmt, args.max_length, args.n_examples,
            args.cache_dir, sampling, args.seed)

    try:
        history = unlearn(model, forget_batches, ucfg,
                          retain_data=retain_batches,
                          reference_model=reference_model,
                          eval_fn=utility_of)
    except CollapsedError as e:
        raise SystemExit("\n".join([
            "",
            str(e),
            "",
            "The study cannot proceed: a collapsed model scores maximally on "
            "every forget metric, so both axes would saturate and the result "
            "would be an artefact.",
        ]))
    theta_un = snapshot(model)
    del reference_model

    verdict = verify_unlearning(theta_ini, theta_un, history)
    log.info("  %s", verdict.summary())
    (out_dir / "unlearn.json").write_text(json.dumps({
        "config": ucfg.to_dict(), "history": history.to_dict(),
        "verdict": {"passed": verdict.passed, "reason": verdict.reason,
                    "update_norm": verdict.update_norm},
    }, indent=2), encoding="utf-8")
    if not verdict.passed:
        raise SystemExit(
            f"\nUnlearning did not behave as expected: {verdict.reason}\n"
            "Refusing to sweep -- every point would inherit the problem."
        )

    v = compute_update_vector(theta_ini, theta_un, strict=False)
    log.info("  update vector: %d tensors, ||v||=%.4g", len(v), verdict.update_norm)

    if args.control == "random":
        # T1 degradation control (report S5.4). Training has already run and
        # been verified above, so the magnitude being matched is a real one --
        # only the direction is discarded. theta_un is rebuilt as
        # theta_ini + r, which makes the sweep theta_ini + (1 + alpha) * r,
        # exactly parallel to the real run's theta_ini + (1 + alpha) * v.
        v = random_direction_like(v, seed=args.seed)
        theta_un = {name: ((t.to(torch.float32) + v[name]).to(t.dtype)
                           if name in v else t.clone())
                    for name, t in theta_ini.items()}
        log.info("  CONTROL: trained direction discarded; sweeping a RANDOM "
                 "direction of matched per-tensor norm (||r||=%.4g vs ||v||=%.4g). "
                 "Depth and breadth here are the degradation baseline, not a "
                 "measurement of forgetting.",
                 global_norm(v.values()), verdict.update_norm)

    # -- sweep --------------------------------------------------------------
    log.info("[5/6] Sweeping alpha over %d settings", len(alphas))
    dataset = PlaneDataset()

    with ModelManager(cfg) as mgr:
        for alpha in alphas:
            part = out_dir / "partial" / f"alpha_{alpha:.3f}.json"
            if args.resume and part.exists():
                p = json.loads(part.read_text(encoding="utf-8"))
                dataset.add(PlanePoint(**p["point"]))
                log.info("  alpha=%.2f  reusing cached result", alpha)
                continue

            t0 = time.time()
            model.load_state_dict(extrapolate(theta_un, v, alpha=alpha))
            model.eval()

            breadth_res = score_breadth(model, tokenizer, breadth_items,
                                        max_length=args.max_length,
                                        prompt_prefix=prefix)
            cap = capture_source_hidden(model, uds_examples, layers, to_cpu=True)
            with mgr.acquire("full") as m_full:
                d2, _ = score_from_captured(m_full, cap, uds_examples, layers,
                                            s_full_cache=s_full)
            depth = assemble_report(delta_s1=d1, delta_s2=d2, s_full=s_full,
                                    layers=layers, tau=cfg.tau)

            retention = breadth_res.retention
            cal_breadth = calibration.calibrated_breadth(breadth_res.forget_leakage)
            collapsed = bool(retention is not None and original_retention
                             and retention < UTILITY_COLLAPSE_RATIO * original_retention)

            point = PlanePoint(
                method=args.method, model=f"{args.size}-{args.method}",
                benchmark="tofu-forget10", alpha=alpha,
                breadth=cal_breadth,
                depth=depth.uds if depth.uds is not None else float("nan"),
                retain_accuracy=retention, utility=retention,
                n_targets=len(uds_examples), seed=args.seed,
                notes=("UTILITY COLLAPSED -- excluded from the trade-off statistic"
                       if collapsed else ""),
            )
            dataset.add(point)
            part.write_text(json.dumps({"point": point.to_dict(),
                                        "retention": retention,
                                        "raw_leakage": breadth_res.forget_leakage,
                                        "breadth_tiers": breadth_res.to_dict(),
                                        "depth_summary": depth.summary()},
                                       indent=2), encoding="utf-8")
            log.info("  alpha=%.2f  breadth=%.3f  depth=%s  retention=%.3f  (%.0fs)%s",
                     alpha, cal_breadth,
                     f"{depth.uds:.3f}" if depth.uds is not None else "undef",
                     retention or float("nan"), time.time() - t0,
                     "  [COLLAPSED]" if collapsed else "")

    # -- analysis -----------------------------------------------------------
    log.info("[6/6] Analysis")
    usable = [p for p in dataset.points if "COLLAPSED" not in (p.notes or "")]
    clean = PlaneDataset(points=usable)

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"{'alpha':>7}{'breadth':>10}{'depth':>10}{'retention':>11}  note")
    print("-" * 76)
    for p in dataset.points:
        d = f"{p.depth:.3f}" if p.depth == p.depth else "undef"
        print(f"{p.alpha:>7.2f}{p.breadth:>10.3f}{d:>10}"
              f"{(p.retain_accuracy or float('nan')):>11.3f}  {p.notes[:30]}")

    n_dropped = len(dataset.points) - len(usable)
    if n_dropped:
        print(f"\n{n_dropped} point(s) excluded: retention fell below "
              f"{UTILITY_COLLAPSE_RATIO:.0%} of its starting value, so those "
              "measure damage rather than forgetting.")

    summary = clean.tradeoff_summary()
    sensitivity = clean.sensitivity_report()
    if summary:
        row = summary[0]
        print("\n" + "-" * 76)
        print("TRADE-OFF")
        print("-" * 76)
        print(f"  Spearman rho : {row['spearman_rho']}   "
              f"p = {row['p_value']} (NOT a significance test -- the points are "
              "one continuous path, not independent samples)")
        print(f"  usable points: {row['n_points']}")
        print(f"  utility stable across sweep: {row['utility_stable']}")
        # -- effect size, in units the measurement can actually resolve -------
        b_lo, b_hi = row["breadth_range"]
        d_lo, d_hi = row["depth_range"]
        # One flipped forced-choice item is the smallest breadth step there is.
        n_forget_items = args.n_breadth * 2          # B0 and B1
        item = (1.0 / n_forget_items) / max(1e-9, calibration.dynamic_range)
        print(f"\n  breadth span : {b_hi - b_lo:.3f} "
              f"({(b_hi - b_lo) / item:.0f} of {n_forget_items} items; "
              f"one item = {item:.4f})")
        print(f"  depth span   : {d_hi - d_lo:.3f} "
              f"(retain90 oracle = 1.000, so peak is {d_hi:.1%} of full erasure)")

        rho = row["spearman_rho"]
        if rho is None:
            finding = ("INCONCLUSIVE: an axis did not vary enough to correlate. "
                       "Widen the alpha range or use more examples.")
        elif not row["usable_dynamic_range"]:
            # The blocker is effect size, not sign. A trajectory confined to a
            # corner of the plane cannot describe the shape of a curve.
            finding = (
                f"UNDERPOWERED -- NO FINDING EITHER WAY (rho={rho:+.3f}).\n\n"
                f"  The sweep covers {b_hi - b_lo:.3f} of the breadth axis and "
                f"{d_hi - d_lo:.3f} of the depth axis, where 1.000 is the retain "
                f"oracle. Peak depth {d_hi:.3f} means the unlearned model erased "
                f"about {d_hi:.0%} of what retraining-without-the-data achieves.\n"
                "  That is a corner of the plane, and the question -- whether "
                "depth gives way as breadth grows -- is a question about the "
                "shape of a curve. This curve has no room to bend.\n\n"
                "  Raise --epochs until forgetting approaches the oracle "
                "(watch the per-epoch forget_nll against the utility floor), "
                "then re-run."
            )
        elif rho < 0:
            finding = ("TRADE-OFF OBSERVED: breadth and depth move in opposite "
                       "directions across a substantial span of both axes. "
                       "Consistent with the hypothesis -- now replicate across "
                       "methods and seeds.")
        elif rho > 0:
            finding = ("AXES AGREE: breadth and depth rise together across a "
                       "substantial span of both axes. This runs against the "
                       "hypothesis and is equally worth reporting -- now "
                       "replicate across methods and seeds.")
        else:
            finding = ("NO CLEAR RELATIONSHIP: the axes are not correlated. "
                       "Informative, since the field assumes they agree.")
        print(f"\n{finding}")
    else:
        finding = "No usable points."
        print(f"\n{finding}")

    dataset.save(out_dir / "plane.json")
    (out_dir / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2),
                                              encoding="utf-8")
    try:
        plot_plane(clean, out_dir / "plane.png",
                   title=f"Depth vs breadth — {args.method}, {args.size}")
    except Exception as e:
        log.warning("Could not draw the plot: %s", e)

    print(f"\nSaved to {out_dir}")
    print("\nNOTE: one unlearning method, one model, one seed. A single "
          "trajectory is not a finding -- repeat across methods and seeds "
          "before drawing any conclusion.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
