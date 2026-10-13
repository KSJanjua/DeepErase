"""The depth-breadth study: does forgetting more broadly mean forgetting less deeply?

This is the experiment everything else was built for.

Procedure
---------
1. Load ``M_full`` and snapshot its weights as ``θ_ini``.
2. Unlearn it -> ``θ_un``.
3. Form the update vector ``v = θ_un − θ_ini``.
4. For each α, build ``θ(α) = θ_un + α·v`` -- pure arithmetic, no retraining.
5. At every α measure **depth** (UDS), **breadth** (tiered forced choice),
   **utility** (retention on neighbouring authors) and **selectivity** (whether
   the forget tiers fell further than the retain tier).
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

Protocols
---------
``--protocol`` selects a coherent set of design choices; every one is recorded
in ``config.json`` and can be overridden individually.

**v1** reproduces the design of every run up to 3 Sep 2026: train on the 100
sampled depth examples, score every token, bfloat16 throughout, select the last
epoch above 90% utility. (It uses the corrected breadth scorer, so it does not
reproduce the v1 *numbers*; check out an older commit for that.)

**v2** fixes the design issues found on 9 Oct 2026 (RESULTS.md §8):

* *Same items on both axes.* Each author's 20 QAs are split in half (seeded).
  The trainer sees only the trained half. Depth and breadth are each measured
  on BOTH halves, giving two trajectories per run: ``trained`` (the facts the
  objective targeted) and ``heldout`` (other facts about the same authors --
  forgetting that generalised).
* *Answer-only loss*, as in OpenUnlearning.
* *float32 throughout, TF32 off*, so small updates and the random-direction
  control are not rounded away.
* *More headroom*: the checkpoint must keep 95% of starting utility.
* *Selectivity* and general-knowledge controls (real authors, world facts) on
  every point.

The control that decides whether any of it counts
-------------------------------------------------
Utility is measured at every α. Push the dial hard enough and the model simply
breaks, and a broken model forgets everything -- which looks exactly like
success on the forget axis. Any point whose retention has collapsed is excluded
from the trade-off statistic, because at that point we are measuring damage,
not forgetting. Selectivity asks the sharper question at every point: did the
forget tiers fall further than neighbouring authors that should be kept?

Usage:
    python -m deeperase.scripts.run_study --method ga --size 1B
    python -m deeperase.scripts.run_study --method npo --protocol v2 --seed 1
    python -m deeperase.scripts.run_study --protocol v1 --method ga   # old design
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from deeperase.config import TOFU_MODELS, RunConfig, plan_memory
from deeperase.core.extrapolation import (
    alpha_grid, compute_update_vector, extrapolate_directed, global_norm,
    random_direction_like,
)
from deeperase.data.reference_spans import load_reference_annotations
from deeperase.data.tofu import (
    FORGET_TO_RETAIN, LossScope, PromptFormat, SamplingStrategy, author_of,
    load_tofu, select_examples, split_by_author, tokenise_example,
    tokenise_for_training,
)
from deeperase.eval.breadth import (
    RETAIN_TIERS, BreadthCalibration, BreadthItem, BreadthResult, TierResult,
    aggregate_outcomes, forget_quality, load_breadth_items_file, score_breadth,
)
from deeperase.eval.patching import n_layers
from deeperase.eval.plane import PlaneDataset, PlanePoint, plot_plane
from deeperase.eval.uds import (
    UDSExample, assemble_report, capture_source_hidden, is_cross_validated,
    score_from_captured,
)
from deeperase.models import ModelManager, load_model, load_tokenizer
from deeperase.provenance import environment_info, git_info, sha256_file
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

#: The forget split this study is built around. The reference entity-span
#: annotations exist only for forget10, and the breadth tiers are drawn from
#: forget10_perturbed, so the two must agree.
FORGET_CONFIG = "forget10"

#: Breadth tiers pooled into the headline breadth coordinate. B2/B4 and the
#: retain tiers are reported separately; the headline stays comparable to v1.
HEADLINE_TIERS = ("B0", "B1")

#: Settings a resumed run must share with the run it continues. Points made
#: under different settings -- or with a different scorer -- cannot share a
#: directory. ``alphas`` is deliberately absent: extending a sweep is legitimate.
RESUME_MUST_MATCH = ("method", "size", "protocol", "unlearn", "n_depth_examples",
                     "n_heldout_examples", "n_breadth_per_tier", "prompt_format",
                     "seed", "sampling", "forget_split", "train_fraction", "dtype",
                     "control", "control_scale", "forget_config", "retain_config")
RESUME_INPUTS_MUST_MATCH = ("breadth_items_sha256", "reference_spans_sha256")

#: Design choices bundled by ``--protocol``. Each can be overridden by its own
#: flag; the resolved values are written to config.json.
PROTOCOLS: Dict[str, Dict[str, object]] = {
    "v1": {"forget_split": "depth", "loss_scope": "full", "dtype": "bfloat16",
           "min_utility_ratio": 0.9, "sampling": "even",
           "breadth_items": "data/breadth_items.json", "n_heldout_examples": 0},
    "v2": {"forget_split": "half", "loss_scope": "answer", "dtype": "float32",
           "min_utility_ratio": 0.95, "sampling": "random",
           "breadth_items": "data/breadth_items_v2.json", "n_heldout_examples": 100},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_uds_examples(tokenizer, reference_spans, fmt, max_length, n,
                       strategy: SamplingStrategy = SamplingStrategy.EVEN,
                       seed: int = 0, *, allowed: Optional[Set[int]] = None,
                       tag: str = "ref") -> List[UDSExample]:
    """Depth examples, using the authors' hand annotations.

    Our own extractor agrees with those only 12% of the time (RESULTS.md §4),
    so it is not used here.

    ``strategy`` matters for replication. EVEN is deterministic; RANDOM varies
    which examples are drawn with ``seed``. ``allowed`` restricts the draw to a
    set of forget-split rows -- the trained or the held-out half.
    """
    kept, stats = load_reference_annotations(reference_spans)
    if allowed is not None:
        kept = [e for e in kept if e.index in allowed]
    log.info("  reference annotations (%s): %d candidates (%s)", tag, len(kept), stats)

    if n >= len(kept):
        chosen = kept
    elif strategy is SamplingStrategy.RANDOM:
        import random
        chosen = sorted(random.Random(seed).sample(list(kept), n),
                        key=lambda e: e.index)
    else:
        step = (len(kept) - 1) / (n - 1) if n > 1 else 1
        chosen = [kept[int(round(i * step))] for i in range(n)]
    log.info("  depth example sampling (%s): %s (seed %d)", tag, strategy.value, seed)

    out: List[UDSExample] = []
    for ex in chosen:
        tok = tokenise_example(ex, tokenizer, fmt=fmt, max_length=max_length)
        if tok is None:
            continue
        ids = torch.tensor([tok.input_ids], dtype=torch.long)
        out.append(UDSExample(f"ref_{ex.index}", ids, tok.to_entity_span(),
                              attention_mask=torch.ones_like(ids)))
    log.info("  %d depth examples tokenised (%s)", len(out), tag)
    return out


def example_row(example: UDSExample) -> int:
    """Forget-split row of a depth example (ids are ``ref_<row>``)."""
    return int(example.example_id.rsplit("_", 1)[1])


def build_forget_batches(rows: Sequence, tokenizer, fmt, max_length,
                         loss_scope: LossScope) -> List[Dict[str, torch.Tensor]]:
    """Training examples for the forget objective, one per trained row."""
    out = []
    for ex in rows:
        b = tokenise_for_training(ex.question, ex.answer, tokenizer, fmt=fmt,
                                  max_length=max_length, loss_scope=loss_scope)
        if b is not None:
            out.append(b)
    if len(out) < len(rows):
        log.warning("  %d of %d forget rows lost their answer to truncation",
                    len(rows) - len(out), len(rows))
    return out


def build_retain_batches(tokenizer, fmt, max_length, n, cache_dir,
                         strategy: SamplingStrategy, seed: int, *,
                         loss_scope: LossScope = LossScope.FULL,
                         data_dir: Optional[str] = None,
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
    examples = load_tofu(retain_config, cache_dir=cache_dir, data_dir=data_dir)
    chosen = select_examples(examples, n, strategy=strategy, seed=seed)
    log.info("  retain split %s: %d of %d examples (%s)",
             retain_config, len(chosen), len(examples), strategy.value)
    return build_forget_batches(chosen, tokenizer, fmt, max_length, loss_scope)


# ---------------------------------------------------------------------------
# Breadth groups, calibration and selectivity
# ---------------------------------------------------------------------------


def split_label(item: BreadthItem, trained_rows: Set[int]) -> Optional[str]:
    """``trained`` / ``heldout`` for forget-tier items, None for retain tiers."""
    if not item.is_forget_tier:
        return None
    return "trained" if item.source_index in trained_rows else "heldout"


def breadth_groups(items: Sequence[BreadthItem], result: BreadthResult,
                   trained_rows: Set[int]) -> Dict[str, TierResult]:
    """Every rate the study reports, from one scoring pass.

    * ``B0:trained``, ``B1:heldout``, ... -- per forget tier and split
    * ``forget:trained`` / ``forget:heldout`` / ``forget:all`` -- the headline
      tiers (B0+B1) pooled, per split and overall
    * ``R``, ``RA``, ``WF`` -- the retain-type tiers
    """
    oc = result.outcomes or {}
    groups = aggregate_outcomes(
        items, oc,
        key=lambda it: (it.tier if not it.is_forget_tier
                        else f"{it.tier}:{split_label(it, trained_rows)}"))
    headline = [it for it in items if it.tier in HEADLINE_TIERS]
    groups.update(aggregate_outcomes(
        headline, oc, key=lambda it: f"forget:{split_label(it, trained_rows)}"))
    groups.update(aggregate_outcomes(headline, oc, key=lambda it: "forget:all"))
    return groups


def calibrate_groups(full: Dict[str, TierResult], retain: Dict[str, TierResult]
                     ) -> Dict[str, Optional[BreadthCalibration]]:
    """One calibration per forget group: ``retain90``'s rate is the floor
    (knowledge absent), ``full``'s the ceiling. ``None`` where the reference
    models do not discriminate -- that group is then reported raw only."""
    out: Dict[str, Optional[BreadthCalibration]] = {}
    for g, r_full in full.items():
        if g in RETAIN_TIERS or g not in retain:
            continue
        try:
            out[g] = BreadthCalibration.from_reference_models(
                absent_leakage=retain[g].knows_rate,
                present_leakage=r_full.knows_rate)
        except ValueError:
            out[g] = None
    return out


def forget_quality_by_split(items: Sequence[BreadthItem], result: BreadthResult,
                            retain_ref: BreadthResult, trained_rows: Set[int]
                            ) -> Dict[str, Optional[float]]:
    """TOFU's Forget Quality on the B0 questions, per split.

    TOFU defines it on the forget set's original questions with the
    paraphrased answer against its perturbations -- exactly the B0 items. The
    comparison model is ``retain90`` on the same items.
    """
    out: Dict[str, Optional[float]] = {}
    for split in ("trained", "heldout", "all"):
        ids = [it.item_id for it in items if it.tier == "B0"
               and (split == "all" or split_label(it, trained_rows) == split)]
        if not ids or not result.outcomes or not retain_ref.outcomes:
            continue
        a = [result.outcomes[i].truth_ratio for i in ids if i in result.outcomes]
        b = [retain_ref.outcomes[i].truth_ratio for i in ids if i in retain_ref.outcomes]
        out[split] = forget_quality(a, b)
    return out


def selectivity_readout(*, breadth: Optional[float], retention: Optional[float],
                        full_retention: Optional[float], forget_floor: Optional[float],
                        forget_drop_raw: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Did the forget tiers fall further than the retain tier?

    ``retain_loss`` puts the retain tier's loss on the breadth scale: 0 is the
    original model's retention, 1 is the forget tiers' floor (``retain90``'s
    rate on forget items). No model with zero knowledge of the *retain* authors
    exists among the TOFU checkpoints, so the forget floor stands in for one;
    the two tiers are built identically (paraphrased answer against TOFU's
    perturbed options), and the raw drops are reported alongside so nothing
    rests on that proxy alone.

    ``selectivity = breadth - retain_loss``: positive means targeted forgetting,
    zero or negative means the breadth rise is matched by collateral loss.
    """
    out: Dict[str, Optional[float]] = {"retain_loss": None, "selectivity": None,
                                       "retain_drop_raw": None,
                                       "forget_drop_raw": forget_drop_raw,
                                       "selectivity_raw": None}
    if retention is None or full_retention is None:
        return out
    out["retain_drop_raw"] = full_retention - retention
    if forget_drop_raw is not None:
        out["selectivity_raw"] = forget_drop_raw - out["retain_drop_raw"]
    if forget_floor is None or breadth is None or full_retention <= forget_floor:
        return out
    loss = (full_retention - retention) / (full_retention - forget_floor)
    out["retain_loss"] = float(min(1.0, max(0.0, loss)))
    out["selectivity"] = breadth - out["retain_loss"]
    return out


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def configure_precision(dtype: str) -> Dict[str, object]:
    """Make float32 mean float32.

    NGC PyTorch containers set ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``, which
    runs float32 matmuls in TF32 (10-bit mantissa) whatever torch's own flags
    say. TF32 rounds a relative change of ~5e-4 -- the size of a whole
    unlearning update relative to a weight -- so it would quietly reintroduce
    the problem float32 is meant to remove. Must run before the first CUDA
    matmul, which is why :func:`main` calls it first.
    """
    info: Dict[str, object] = {"dtype": dtype}
    if dtype != "float32":
        return info
    for var in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
        prev = os.environ.get(var)
        if prev not in (None, "0"):
            log.warning("%s=%s would force TF32 matmuls; overriding to 0 for this "
                        "float32 run.", var, prev)
        info[f"{var}_was"] = prev
        os.environ[var] = "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    info["tf32"] = False
    return info


def resolve_settings(args) -> Dict[str, object]:
    """Protocol defaults, overridden by any flag given explicitly."""
    settings = dict(PROTOCOLS[args.protocol])
    for key in settings:
        val = getattr(args, key, None)
        if val is not None:
            settings[key] = val
    return settings


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description="Depth-breadth trade-off study")
    ap.add_argument("--method", choices=[m.value for m in UnlearnMethod], default="ga")
    ap.add_argument("--size", choices=sorted(TOFU_MODELS), default="1B")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="v2",
                    help="Bundle of design choices (see module docstring). Any of "
                         "the flags below overrides its protocol default.")
    ap.add_argument("--alphas", default=None,
                    help="Comma-separated. Default: 0.0..1.0 in 11 steps")
    ap.add_argument("--n-examples", type=int, default=100,
                    help="depth examples drawn from the trained rows")
    ap.add_argument("--n-heldout-examples", type=int, default=None,
                    help="depth examples drawn from the held-out rows "
                         "(protocol default: v1 0, v2 100)")
    # 100/tier gave 200 scored forget items, so one flipped item moved
    # calibrated breadth by 0.0185 -- the entire signal of the first real sweep
    # was five items. 400 quarters that step.
    ap.add_argument("--n-breadth", type=int, default=400,
                    help="breadth items per tier (resolution: one item = "
                         "1/(2*N) raw, divided by the calibrated range)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=1e-6,
                    help="1e-6 is UIPE's forget10 rate (Appendix D.3) and v1's. Under "
                         "v2's float32 it collapses GA within one epoch (H100 smoke "
                         "run, 13 Oct 2026): pick the rate with scripts/lr_scan.py.")
    ap.add_argument("--min-utility-ratio", type=float, default=None,
                    help="Checkpoint selection floor: keep at least this fraction "
                         "of the starting model's utility (v1 0.9, v2 0.95).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--prompt-format", default="plain_qa",
                    choices=[f.value for f in PromptFormat])
    ap.add_argument("--forget-split", choices=["depth", "all", "half"], default=None,
                    help="What the unlearning objective trains on. 'depth': the "
                         "sampled depth examples (v1). 'all': every forget10 row. "
                         "'half': half of each author's rows, seeded (v2).")
    ap.add_argument("--train-fraction", type=float, default=0.5,
                    help="Share of each author's rows trained on under 'half'.")
    ap.add_argument("--loss-scope", choices=[s.value for s in LossScope], default=None,
                    help="'answer' (v2): answer tokens only. 'full' (v1): every token.")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default=None,
                    help="Training and measurement precision (v1 bfloat16, v2 float32).")
    ap.add_argument("--breadth-items", default=None,
                    help="Frozen item set to score (JSON). See "
                         "deeperase/scripts/build_breadth_items.py.")
    ap.add_argument("--n-retain", type=int, default=None,
                    help="GradDiff retain examples. Default: as many as forget rows trained.")
    ap.add_argument("--reference-spans",
                    default="reference_uds/tofu_data/forget10_filtered.json")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seeds torch, the example draws and (v2) the per-author "
                         "split. Training has no shuffling and the Llama dropout is "
                         "0, so under v1 with --sampling even a seed changes nothing.")
    ap.add_argument("--sampling", default=None,
                    choices=[s.value for s in SamplingStrategy],
                    help="How depth/retain examples are drawn (v1 even, v2 random).")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / mps. Default: cuda when available, else cpu.")
    ap.add_argument("--model-sources", default=None,
                    help='JSON mapping split -> local model dir, e.g. '
                         '{"full": "/ckpt/full", "retain90": "/ckpt/r90"}. '
                         "Overrides the Hub repos; the tokenizer comes from 'full'.")
    ap.add_argument("--tofu-dir", default=None,
                    help="Read TOFU from local JSON-Lines files (data/README.md) "
                         "instead of the HuggingFace hub.")
    ap.add_argument("--cache-dir", default=os.environ.get("DEEPERASE_CACHE_DIR", "./hf_cache"),
                    help="Model and dataset download cache. Default: $DEEPERASE_CACHE_DIR "
                         "(set by gpu/env.sh), else ./hf_cache.")
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
    ap.add_argument("--control-scale", type=float, default=1.0,
                    help="Multiplier on the random control direction. 1.0 is "
                         "the matched-MAGNITUDE control (same distance, random "
                         "heading). Larger values give the matched-DAMAGE "
                         "control: raise it until the control's utility drop "
                         "matches the real run's, then compare the axes. Only "
                         "meaningful with --control random.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if args.control_scale <= 0:
        ap.error("--control-scale must be positive")
    return args


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


def write_config(path: Path, config: dict, *, resume: bool) -> None:
    """Write config.json -- or, on resume, check it and record the resume.

    A resumed run keeps its original config.json, so the commit recorded there
    is the one that produced the first points. Each resume appends its own
    commit and time. A resume whose design settings or input files differ is
    refused: its new points would not be comparable with the saved ones. That
    includes resuming a run made before protocols existed, whose points were
    scored with the defective breadth scorer (RESULTS.md §8.1).
    """
    if not (resume and path.exists()):
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    diffs = {k: (old.get(k), config.get(k)) for k in RESUME_MUST_MATCH
             if old.get(k) != config.get(k)}
    for k in RESUME_INPUTS_MUST_MATCH:
        a, b = (old.get("inputs") or {}).get(k), (config.get("inputs") or {}).get(k)
        if a != b:
            diffs[f"inputs.{k}"] = (a, b)
    if diffs:
        detail = "\n".join(f"    {k}: saved {a!r}, now {b!r}" for k, (a, b) in diffs.items())
        raise SystemExit(
            f"\n--resume refused: {path.parent.name} was made with different settings.\n"
            f"{detail}\nStart a new --run-id instead.")
    then, now = (old.get("git") or {}).get("commit"), (config.get("git") or {}).get("commit")
    if then != now:
        log.warning("Resuming %s on commit %s; its saved points were made on %s. Both "
                    "are recorded in config.json.", path.parent.name, now, then)
    old.setdefault("resumes", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": config.get("git"),
        "alphas": config.get("alphas")})
    path.write_text(json.dumps(old, indent=2), encoding="utf-8")


def _round_depth_report(report) -> dict:
    """UDS report for the record: per-example per-layer deltas, 5 decimals.

    Kept so tau, per-author or per-example analyses can be rerun offline -- the
    same reason the validation runs keep theirs (scripts/tau_sensitivity.py).
    """
    d = report.to_dict()
    for ex in d["per_example"]:
        ex.pop("ler", None)          # recomputable from dS1/dS2 (Eq. 4)
        for k in ("delta_s1", "delta_s2"):
            ex[k] = {str(l): round(float(v), 5) for l, v in ex[k].items()}
        if ex["uds"] is not None:
            ex["uds"] = round(float(ex["uds"]), 6)
    return d


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    settings = resolve_settings(args)
    precision = configure_precision(str(settings["dtype"]))

    method = UnlearnMethod(args.method)
    sampling = SamplingStrategy(str(settings["sampling"]))
    loss_scope = LossScope(str(settings["loss_scope"]))
    forget_split = str(settings["forget_split"])
    n_heldout = int(settings["n_heldout_examples"]) if forget_split == "half" else 0

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
    if forget_split == "depth" and sampling is SamplingStrategy.EVEN and args.seed != 0:
        log.warning(
            "--seed %d has no effect under --sampling even with the depth split: "
            "example selection is deterministic, training does not shuffle, and "
            "dropout is 0. This run will be identical to seed 0. Use --sampling "
            "random for replicates.", args.seed)

    run_id = args.run_id or (f"study_{args.method}_{args.size}_s{args.seed}"
                             f"_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.output_dir) / run_id
    existing = sorted((out_dir / "partial").glob("alpha_*.json"))
    if existing and not args.resume:
        raise SystemExit(
            f"\n{out_dir} already holds {len(existing)} scored point(s). Pass --resume "
            "to continue that run, or choose a new --run-id. Running into it again "
            "would overwrite its points one by one and leave a mixture of two runs.")
    (out_dir / "partial").mkdir(parents=True, exist_ok=True)

    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas
              else alpha_grid(0.0, 1.0, num=11))
    model_sources = json.loads(args.model_sources) if args.model_sources else None
    full_source = (model_sources or {}).get("full") or TOFU_MODELS[args.size]["full"].repo_id

    print("=" * 76)
    print(f"DEPTH-BREADTH STUDY  |  method={args.method}  size={args.size}  "
          f"protocol={args.protocol}")
    print(f"run={run_id}")
    print("=" * 76)

    # -- device, precision and memory ----------------------------------------
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype_name = "float32" if device.type == "cpu" else str(settings["dtype"])
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    cfg = RunConfig(size_label=args.size, cache_dir=args.cache_dir,
                    max_seq_length=args.max_length, dtype=dtype_name)
    if device.type == "cuda":
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        # Free, not total. This study trains, and it may not be alone on the
        # card; a plan built from capacity is a plan built from a number nobody
        # is offering us.
        free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
        plan = plan_memory(args.size, total_gb, dtype_size=cfg.dtype_size,
                           gpu_free_gb=free_gb, training=True,
                           reference_model=method.needs_reference)
        print(f"\n{plan.summary()}\n")
        if not plan.fits:
            # Fail here, not forty minutes in after the reference phase.
            raise SystemExit(
                "Refusing to start: the training phase will not fit.\n" + plan.reason)
        cfg.strategy = plan.strategy
    else:
        log.warning("No CUDA device in use (%s) -- fine for tests and tiny models, "
                    "impractically slow for a real study.", device)

    items_path = settings["breadth_items"]
    config = {
        "run_id": run_id, "method": args.method, "size": args.size,
        "protocol": args.protocol, "alphas": alphas, "adaptation": "full_parameter",
        "unlearn": {"epochs": args.epochs, "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size, "beta": args.beta,
                    "min_utility_ratio": settings["min_utility_ratio"],
                    "loss_scope": loss_scope.value},
        "n_depth_examples": args.n_examples, "n_heldout_examples": n_heldout,
        "n_breadth_per_tier": args.n_breadth,
        "prompt_format": args.prompt_format,
        "seed": args.seed, "sampling": sampling.value,
        "forget_split": forget_split,
        "train_fraction": args.train_fraction if forget_split == "half" else None,
        "dtype": dtype_name, "precision": precision, "device": str(device),
        "control": args.control,
        "control_scale": args.control_scale if args.control == "random" else None,
        "forget_config": FORGET_CONFIG,
        "retain_config": (FORGET_TO_RETAIN[FORGET_CONFIG]
                          if method.needs_retain else None),
        "inputs": {
            "breadth_items": str(items_path), "breadth_items_sha256": sha256_file(items_path),
            "reference_spans": args.reference_spans,
            "reference_spans_sha256": sha256_file(args.reference_spans),
            "model_sources": model_sources, "tofu_dir": args.tofu_dir,
        },
        "git": git_info(), "environment": environment_info(),
    }
    write_config(out_dir / "config.json", config, resume=args.resume)

    fmt = PromptFormat(args.prompt_format)
    tokenizer = load_tokenizer(full_source, cache_dir=cfg.cache_dir)

    # -- data ---------------------------------------------------------------
    log.info("[1/6] Preparing evaluation data (forget split: %s)", forget_split)
    forget_rows = load_tofu(FORGET_CONFIG, cache_dir=args.cache_dir, data_dir=args.tofu_dir)
    if forget_split == "half":
        trained_list, heldout_list = split_by_author(
            len(forget_rows), train_fraction=args.train_fraction, seed=args.seed)
    else:
        trained_list, heldout_list = list(range(len(forget_rows))), []

    uds_sets: Dict[str, List[UDSExample]] = {}
    uds_sets["trained"] = build_uds_examples(
        tokenizer, args.reference_spans, fmt, args.max_length, args.n_examples,
        sampling, args.seed, allowed=set(trained_list), tag="trained")
    if forget_split == "depth":
        # v1: the objective trains on exactly the sampled depth examples.
        trained_list = sorted(example_row(e) for e in uds_sets["trained"])
        heldout_list = sorted(set(range(len(forget_rows))) - set(trained_list))
    if n_heldout:
        uds_sets["heldout"] = build_uds_examples(
            tokenizer, args.reference_spans, fmt, args.max_length, n_heldout,
            sampling, args.seed + 1, allowed=set(heldout_list), tag="heldout")
    trained_rows = set(trained_list)

    forget_batches = build_forget_batches(
        [forget_rows[i] for i in trained_list], tokenizer, fmt, args.max_length, loss_scope)
    log.info("  forget objective: %d rows (%s loss)", len(forget_batches), loss_scope.value)

    items_all = load_breadth_items_file(items_path)
    breadth_items = subsample(items_all, args.n_breadth)
    for it in breadth_items:
        if it.is_forget_tier and it.author is None and it.source_index >= 0:
            it.author = author_of(it.source_index)
    log.info("  %d breadth items from %s", len(breadth_items), items_path)
    prefix = PROMPT_PREFIXES.get(args.prompt_format, "{question} ")

    (out_dir / "split.json").write_text(json.dumps({
        "forget_split": forget_split, "trained_rows": trained_list,
        "heldout_rows": heldout_list,
        "depth_examples": {k: [e.example_id for e in v] for k, v in uds_sets.items()},
    }, indent=1), encoding="utf-8")

    def score_all(m) -> Tuple[BreadthResult, Dict[str, TierResult]]:
        r = score_breadth(m, tokenizer, breadth_items, max_length=args.max_length,
                          prompt_prefix=prefix, return_outcomes=True)
        return r, breadth_groups(breadth_items, r, trained_rows)

    # -- reference measurements + calibration -------------------------------
    log.info("[2/6] Measuring reference models (anchors both axes)")
    with ModelManager(cfg, device=str(device), model_sources=model_sources) as mgr:
        layers = None
        ref: Dict[str, Tuple[BreadthResult, Dict[str, TierResult]]] = {}
        for split in ("full", "retain90"):
            with mgr.acquire(split) as m:
                if layers is None:
                    layers = list(range(n_layers(m)))
                ref[split] = score_all(m)
            g = ref[split][1]
            log.info("  %s  forget:all=%.3f  R=%.3f%s", split, g["forget:all"].knows_rate,
                     g["R"].knows_rate,
                     "".join(f"  {t}={g[t].knows_rate:.3f}" for t in ("RA", "WF") if t in g))

        # The utility floor must be anchored to the ORIGINAL model. An earlier
        # version took it from the alpha=0 point, which was itself already
        # collapsed -- so 0.21/0.21 = 1.0 and the guard never fired.
        original_retention = ref["full"][1]["R"].knows_rate
        calib = calibrate_groups(ref["full"][1], ref["retain90"][1])
        headline_groups = (["forget:all"] if forget_split == "depth"
                           else ["forget:trained"] + (["forget:heldout"] if heldout_list else []))
        for g in headline_groups:
            if calib.get(g) is None:
                raise SystemExit(
                    f"\nBreadth calibration failed for {g}: full={ref['full'][1][g].knows_rate:.3f}"
                    f" retain90={ref['retain90'][1][g].knows_rate:.3f}. The reference models "
                    "are not discriminating, so the breadth axis cannot be anchored. "
                    "Refusing to continue.")
        forget_floor = calib["forget:all"].floor if calib.get("forget:all") else None
        fq_reference = forget_quality_by_split(breadth_items, ref["full"][0],
                                               ref["retain90"][0], trained_rows)
        log.info("  TOFU forget quality, full vs retain90 (before unlearning): %s",
                 {k: (None if v is None else f"{v:.2e}") for k, v in fq_reference.items()})
        (out_dir / "calibration.json").write_text(json.dumps({
            "tofu_forget_quality_full_vs_retain90": fq_reference,
            "groups": {g: (None if c is None else c.to_dict()) for g, c in calib.items()},
            "reference_rates": {s: {g: {"n": t.n, "rate": t.knows_rate}
                                    for g, t in ref[s][1].items()} for s in ref},
            "headline_groups": headline_groups,
            "original_retention": original_retention,
            "retain_floor_proxy": forget_floor,
            # Per-item baselines for fact-level analysis: which facts `full`
            # knew to begin with, and how an ignorant model scores them.
            "reference_outcomes": {s: ref[s][0].to_dict(include_outcomes=True).get("outcomes")
                                   for s in ref},
        }, indent=2), encoding="utf-8")
        for g in headline_groups:
            c = calib[g]
            log.info("  calibration %s: floor=%.3f ceiling=%.3f range=%.3f",
                     g, c.floor, c.ceiling, c.dynamic_range)

        # -- Stage-1 depth baseline, computed once -------------------------
        log.info("[3/6] Depth Stage-1 baseline (retain90 -> full)")
        stage1: Dict[str, Tuple[dict, dict]] = {}
        with mgr.acquire("retain90") as m_ret:
            caps = {k: capture_source_hidden(m_ret, v, layers, to_cpu=True)
                    for k, v in uds_sets.items()}
        with mgr.acquire("full") as m_full:
            for k, v in uds_sets.items():
                stage1[k] = score_from_captured(m_full, caps[k], v, layers)
        del caps
        for k, (d1, _) in stage1.items():
            n_ke = sum(1 for v in d1.values() if max(v.values()) > cfg.tau)
            log.info("  %s: %d/%d examples have a Knowledge-Encoding layer", k, n_ke, len(d1))
            if n_ke == 0:
                raise SystemExit(
                    f"\nNo {k} example has a Knowledge-Encoding layer. Depth is "
                    "undefined for every example, so the study cannot proceed.")

    # -- unlearn ------------------------------------------------------------
    log.info("[4/6] Unlearning with %s (%s, %s)", args.method, dtype_name, loss_scope.value)
    model = load_model(full_source, device=device, dtype=torch_dtype, cache_dir=cfg.cache_dir)
    theta_ini = snapshot(model)

    reference_model = None
    if method.needs_reference:
        reference_model = load_model(full_source, device=device, dtype=torch_dtype,
                                     cache_dir=cfg.cache_dir)

    ucfg = UnlearnConfig(method=method, learning_rate=args.learning_rate,
                         epochs=args.epochs, batch_size=args.batch_size,
                         beta=args.beta,
                         min_utility_ratio=float(settings["min_utility_ratio"]),
                         seed=args.seed)

    r_items = [i for i in breadth_items if i.tier == "R"]

    def utility_of(m) -> float:
        """Retention on neighbouring authors, for checkpoint selection.

        Scored on the R tier only -- the forget tiers are what we are trying to
        move, so including them would penalise successful unlearning.
        """
        return score_breadth(m, tokenizer, r_items, max_length=args.max_length,
                             prompt_prefix=prefix).tiers["R"].knows_rate

    retain_batches = None
    if method.needs_retain:
        log.info("[3b/6] Loading retain split for %s", method.value)
        retain_batches = build_retain_batches(
            tokenizer, fmt, args.max_length, args.n_retain or len(forget_batches),
            args.cache_dir, sampling, args.seed, loss_scope=loss_scope,
            data_dir=args.tofu_dir)

    try:
        history = unlearn(model, forget_batches, ucfg,
                          retain_data=retain_batches,
                          reference_model=reference_model,
                          eval_fn=utility_of)
    except CollapsedError as e:
        raise SystemExit("\n".join([
            "", str(e), "",
            "The study cannot proceed: a collapsed model scores maximally on "
            "every forget metric, so both axes would saturate and the result "
            "would be an artefact.",
        ]))
    theta_un = snapshot(model)
    del reference_model

    verdict = verify_unlearning(theta_ini, theta_un, history)
    log.info("  %s", verdict.summary())
    record = {"config": ucfg.to_dict(), "history": history.to_dict(),
              "verdict": {"passed": verdict.passed, "reason": verdict.reason,
                          "update_norm": verdict.update_norm}}
    prev_path = out_dir / "unlearn.json"
    if args.resume and prev_path.exists():
        # A resume retrains before sweeping the remaining alphas. GPU kernels
        # need not be bit-reproducible, so the new points may come from a
        # slightly different theta_un than the saved ones. Record by how much.
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        old = (prev.get("verdict") or {}).get("update_norm")
        drift = (abs(verdict.update_norm - old) / old) if old else None
        record["resumed_from"] = {"update_norm": old, "relative_drift": drift,
                                  "earlier": prev.get("resumed_from")}
        if drift is None or drift > 1e-3:
            log.warning("  Retrained update norm %.6g differs from the saved run's %s "
                        "(relative %s). Points before and after this resume come from "
                        "different unlearned models.", verdict.update_norm, old,
                        "unknown" if drift is None else f"{drift:.2e}")
        else:
            log.info("  retrained update matches the saved run (relative drift %.1e)", drift)
    prev_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not verdict.passed:
        raise SystemExit(
            f"\nUnlearning did not behave as expected: {verdict.reason}\n"
            "Refusing to sweep -- every point would inherit the problem.")

    v = compute_update_vector(theta_ini, theta_un, strict=False)
    log.info("  update vector: %d tensors, ||v||=%.4g", len(v), verdict.update_norm)

    if args.control == "random":
        # T1 degradation control (report S5.4). Training has already run and
        # been verified above, so the magnitude being matched is a real one --
        # only the direction is discarded. theta_un is rebuilt as
        # theta_ini + r, which makes the sweep theta_ini + (1 + alpha) * r,
        # exactly parallel to the real run's theta_ini + (1 + alpha) * v.
        v = random_direction_like(v, seed=args.seed, scale=args.control_scale)
        theta_un = {name: ((t.to(torch.float32) + v[name]).to(t.dtype)
                           if name in v else t.clone())
                    for name, t in theta_ini.items()}
        log.info("  CONTROL: trained direction discarded; sweeping a RANDOM "
                 "direction at %.4gx matched per-tensor norm "
                 "(||r||=%.4g vs ||v||=%.4g). Depth and breadth here are the "
                 "degradation baseline, not a measurement of forgetting.",
                 args.control_scale, global_norm(v.values()), verdict.update_norm)
        if dtype_name != "float32":
            log.warning("  CONTROL in %s: small random perturbations are partly "
                        "rounded away by the cast (RESULTS.md §7). Use float32.",
                        dtype_name)

    # -- sweep --------------------------------------------------------------
    log.info("[5/6] Sweeping alpha over %d settings", len(alphas))
    dataset = PlaneDataset()
    trajectories = (
        [("tofu-forget10", "trained", "forget:all")] if forget_split == "depth" else
        [("tofu-forget10" if not heldout_list else "tofu-forget10:trained",
          "trained", "forget:trained")]
        + ([("tofu-forget10:heldout", "heldout", "forget:heldout")]
           if "heldout" in uds_sets else []))
    floor_ratio = float(settings["min_utility_ratio"])

    with ModelManager(cfg, device=str(device), model_sources=model_sources) as mgr:
        for alpha in alphas:
            part = out_dir / "partial" / f"alpha_{alpha:.3f}.json"
            if args.resume and part.exists():
                p = json.loads(part.read_text(encoding="utf-8"))
                for d in (p.get("points") or [p["point"]]):
                    dataset.add(PlanePoint.from_dict(d))
                log.info("  alpha=%.2f  reusing cached result", alpha)
                continue

            t0 = time.time()
            state, _ = extrapolate_directed(theta_un, v, alpha_parallel=alpha,
                                            alpha_perp=alpha, out_dtype=torch_dtype)
            model.load_state_dict(state)
            del state
            model.eval()

            res, groups = score_all(model)
            fq = forget_quality_by_split(breadth_items, res, ref["retain90"][0], trained_rows)
            depth_reports = {}
            for k, examples in uds_sets.items():
                cap = capture_source_hidden(model, examples, layers, to_cpu=True)
                d1, s_full = stage1[k]
                with mgr.acquire("full") as m_full:
                    d2, _ = score_from_captured(m_full, cap, examples, layers,
                                                s_full_cache=s_full)
                depth_reports[k] = assemble_report(delta_s1=d1, delta_s2=d2,
                                                   s_full=s_full, layers=layers,
                                                   tau=cfg.tau,
                                                   validated=is_cross_validated(model))

            retention = groups["R"].knows_rate
            general = res.general_utility
            collapsed = bool(original_retention
                             and retention < UTILITY_COLLAPSE_RATIO * original_retention)
            below_floor = bool(original_retention
                               and retention < floor_ratio * original_retention)
            calibrated = {g: (None if c is None else c.calibrated_breadth(groups[g].knows_rate))
                          for g, c in calib.items() if g in groups}

            points, sel_record = [], {}
            for bench, depth_key, group in trajectories:
                b_cal = calibrated[group]
                drop = ref["full"][1][group].knows_rate - groups[group].knows_rate
                sel = selectivity_readout(breadth=b_cal, retention=retention,
                                          full_retention=original_retention,
                                          forget_floor=forget_floor, forget_drop_raw=drop)
                sel_record[bench] = sel
                rep = depth_reports[depth_key]
                notes = []
                if collapsed:
                    notes.append("UTILITY COLLAPSED -- excluded from the trade-off statistic")
                elif below_floor:
                    notes.append(f"below the {floor_ratio:.0%} selection floor")
                points.append(PlanePoint(
                    method=args.method, model=f"{args.size}-{args.method}",
                    benchmark=bench, alpha=alpha, breadth=b_cal,
                    depth=rep.uds if rep.uds is not None else float("nan"),
                    retain_accuracy=retention, utility=retention,
                    retain_loss=sel["retain_loss"], selectivity=sel["selectivity"],
                    general_utility=general,
                    n_targets=len(uds_sets[depth_key]), seed=args.seed,
                    notes="; ".join(notes)))
            for pt in points:
                dataset.add(pt)

            part.write_text(json.dumps({
                "points": [pt.to_dict() for pt in points],
                "point": points[0].to_dict(),           # v1 readers
                "retention": retention,
                "raw_leakage": groups[trajectories[0][2]].knows_rate,
                "general_utility": general,
                "groups": {g: {"n": t.n, "n_knows": t.n_knows, "rate": t.knows_rate,
                               "calibrated": calibrated.get(g)}
                           for g, t in sorted(groups.items())},
                "selectivity": sel_record,
                "tofu_forget_quality": fq,
                "breadth_tiers": res.to_dict(include_outcomes=True),
                "depth_summary": {k: r.summary() for k, r in depth_reports.items()},
                "depth": {k: _round_depth_report(r) for k, r in depth_reports.items()},
            }, indent=1), encoding="utf-8")
            log.info("  alpha=%.2f  TOFU forget quality %s", alpha,
                     {k: (None if v is None else f"{v:.2e}") for k, v in fq.items()})
            log.info("  alpha=%.2f  %s  R=%.3f%s  (%.0fs)%s", alpha,
                     "  ".join(f"[{b.split(':')[-1]}] breadth={pt.breadth:.3f} "
                               f"depth={pt.depth:.3f} sel={pt.selectivity if pt.selectivity is not None else float('nan'):+.3f}"
                               for (b, _, _), pt in zip(trajectories, points)),
                     retention,
                     f"  general={general:.3f}" if general is not None else "",
                     time.time() - t0,
                     "  [COLLAPSED]" if collapsed else "")

    # -- analysis -----------------------------------------------------------
    log.info("[6/6] Analysis")
    usable = [p for p in dataset.points if "COLLAPSED" not in (p.notes or "")]
    clean = PlaneDataset(points=usable)

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"{'trajectory':24}{'alpha':>6}{'breadth':>9}{'depth':>8}{'R':>7}"
          f"{'select.':>9}  note")
    print("-" * 76)
    for p in sorted(dataset.points, key=lambda p: (p.benchmark, p.alpha)):
        d = f"{p.depth:.3f}" if p.depth == p.depth else "undef"
        s = f"{p.selectivity:+.3f}" if p.selectivity is not None else "  n/a"
        print(f"{p.benchmark[:24]:24}{p.alpha:>6.2f}{p.breadth:>9.3f}{d:>8}"
              f"{(p.retain_accuracy or float('nan')):>7.3f}{s:>9}  {p.notes[:22]}")

    n_dropped = len(dataset.points) - len(usable)
    if n_dropped:
        print(f"\n{n_dropped} point(s) excluded: retention fell below "
              f"{UTILITY_COLLAPSE_RATIO:.0%} of its starting value, so those "
              "measure damage rather than forgetting.")

    summary = clean.tradeoff_summary()
    sensitivity = clean.sensitivity_report()
    n_forget_items = sum(1 for i in breadth_items if i.tier in HEADLINE_TIERS)
    for row in summary:
        print("\n" + "-" * 76)
        print(f"TRADE-OFF  [{row['benchmark']}]")
        print("-" * 76)
        print(f"  Spearman rho : {row['spearman_rho']}   "
              f"p = {row['p_value']} (NOT a significance test -- the points are "
              "one continuous path, not independent samples)")
        print(f"  usable points: {row['n_points']}")
        print(f"  utility stable across sweep: {row['utility_stable']}")
        b_lo, b_hi = row["breadth_range"]
        d_lo, d_hi = row["depth_range"]
        group = next(g for b, _, g in trajectories if b == row["benchmark"])
        n_group = ref["full"][1][group].n or n_forget_items
        item = (1.0 / n_group) / max(1e-9, calib[group].dynamic_range)
        print(f"\n  breadth span : {b_hi - b_lo:.3f} "
              f"({(b_hi - b_lo) / item:.0f} of {n_group} items; one item = {item:.4f})")
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
                "then re-run.")
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
        sels = [p.selectivity for p in clean.points
                if p.benchmark == row["benchmark"] and p.selectivity is not None]
        if sels and max(sels) <= 0.0:
            finding += ("\n\n  NOT SELECTIVE: at no point did the forget tiers fall "
                        "further than the retain tier (selectivity <= 0). The breadth "
                        "rise is matched by collateral loss and cannot be read as "
                        "targeted forgetting.")
        print(f"\n{finding}")
    if not summary:
        print("\nNo usable points.")

    dataset.save(out_dir / "plane.json")
    (out_dir / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2),
                                              encoding="utf-8")
    try:
        plot_plane(clean, out_dir / "plane.png",
                   title=f"Depth vs breadth — {args.method}, {args.size}, {args.protocol}")
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
