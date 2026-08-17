"""Measure the breadth axis on real TOFU models.

This is the breadth counterpart to ``run_uds_validation``. It has a built-in
correctness check, using the same trick: models whose answer we already know.

    M_full     learned every author  -> should still know them  -> HIGH leakage
    M_retain90 never saw forget10    -> should not know them    -> LOW leakage
    both       should keep unrelated knowledge -> HIGH retention

If ``full`` does not leak substantially more than ``retain90``, the breadth
measurement is not working and nothing downstream can be trusted.

Usage:
    python -m deeperase.scripts.measure_breadth --size 1B --n-per-tier 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import torch

from deeperase.config import TOFU_MODELS, RunConfig, plan_memory
from deeperase.eval.breadth import (
    BreadthCalibration, BreadthItem, load_breadth_items, score_breadth,
)
from deeperase.models import ModelManager, load_tokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("breadth")
logging.getLogger("deeperase").setLevel(logging.INFO)

PROMPT_PREFIXES = {
    "bare": "{question} ",
    "plain_qa": "Question: {question}\nAnswer: ",
}


def subsample(items: List[BreadthItem], n_per_tier: int) -> List[BreadthItem]:
    """Take an even spread within each tier.

    Same reasoning as the depth runner: TOFU's splits are ordered, so a
    contiguous block is not a representative sample.
    """
    by_tier: Dict[str, List[BreadthItem]] = {}
    for it in items:
        by_tier.setdefault(it.tier, []).append(it)

    out: List[BreadthItem] = []
    for tier, pool in sorted(by_tier.items()):
        if n_per_tier >= len(pool):
            out.extend(pool)
            continue
        step = (len(pool) - 1) / (n_per_tier - 1) if n_per_tier > 1 else 0
        picks = sorted({int(round(i * step)) for i in range(n_per_tier)})
        out.extend(pool[i] for i in picks)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure breadth on TOFU models")
    ap.add_argument("--size", choices=sorted(TOFU_MODELS), default="1B")
    ap.add_argument("--n-per-tier", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--prompt-format", choices=sorted(PROMPT_PREFIXES), default="bare")
    ap.add_argument("--splits", default="full,retain90",
                    help="Comma-separated model splits to measure")
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--output-dir", default="results/gpu_runs")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"breadth_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"Breadth measurement  |  size={args.size}  run={run_id}")
    print("=" * 74)

    total_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 8.0)
    if not torch.cuda.is_available():
        log.warning("No GPU detected; this will be very slow.")
    cfg = RunConfig(size_label=args.size, cache_dir=args.cache_dir)
    plan = plan_memory(args.size, total_gb, dtype_size=cfg.dtype_size)
    print(f"\n{plan.summary()}\n")
    if not plan.fits:
        raise SystemExit("Model does not fit.")
    cfg.strategy = plan.strategy

    log.info("Loading breadth items (B0 exact, B1 paraphrase, R retain)")
    items = subsample(load_breadth_items(cache_dir=args.cache_dir), args.n_per_tier)
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.tier] = counts.get(it.tier, 0) + 1
    log.info("  %d items: %s", len(items), counts)

    tokenizer = load_tokenizer(cfg.models()["full"].repo_id, cache_dir=cfg.cache_dir)
    prefix = PROMPT_PREFIXES[args.prompt_format]
    splits = [s.strip() for s in args.splits.split(",")]

    results: Dict[str, dict] = {}
    with ModelManager(cfg) as mgr:
        for split in splits:
            log.info("Scoring %s ...", split)
            t0 = time.time()
            with mgr.acquire(split) as model:
                res = score_breadth(model, tokenizer, items,
                                    max_length=args.max_length,
                                    prompt_prefix=prefix, log_every=100)
            log.info("  %s in %.1fs -- %s", split, time.time() - t0, res.summary())
            results[split] = res.to_dict()

    # -- report -------------------------------------------------------------
    print("\n" + "=" * 74)
    print("RESULTS  (knows-rate: fraction where the model still picks the")
    print("          correct answer over the wrong ones)")
    print("=" * 74)
    tiers = ["B0", "B1", "R"]
    print(f"{'model':<12}" + "".join(f"{t:>10}" for t in tiers) + f"{'breadth':>10}")
    print("-" * 74)
    for split in splits:
        pt = results[split]["per_tier"]
        row = "".join(f"{pt[t]['knows_rate']:>10.3f}" if t in pt else f"{'-':>10}"
                      for t in tiers)
        print(f"{split:<12}{row}{results[split]['breadth']:>10.3f}")

    # -- built-in correctness check -----------------------------------------
    verdict = None
    if "full" in results and "retain90" in results:
        f_leak = results["full"]["forget_leakage"]
        r_leak = results["retain90"]["forget_leakage"]
        f_ret = results["full"]["retention"]
        r_ret = results["retain90"]["retention"]
        gap = f_leak - r_leak

        print("\n" + "-" * 74)
        print("SANITY CHECK")
        print("-" * 74)
        print(f"  full leaks     : {f_leak:.3f}   (saw every author -- should be high)")
        print(f"  retain90 leaks : {r_leak:.3f}   (never saw them  -- should be low)")
        print(f"  difference     : {gap:+.3f}")
        print(f"  retention      : full {f_ret:.3f} / retain90 {r_ret:.3f} "
              "(both should be high)")

        # Calibrate the axis to these two reference models, so breadth reads
        # 0 for a model that knows everything and 1 for one that knows nothing
        # -- the same convention as the depth axis.
        if gap > 0.0:
            cal = BreadthCalibration.from_reference_models(
                absent_leakage=r_leak, present_leakage=f_leak)
            print()
            print("-" * 74)
            print("CALIBRATED BREADTH")
            print("-" * 74)
            print(f"  raw scale runs {cal.floor:.3f} (knows nothing) to "
                  f"{cal.ceiling:.3f} (knows all) -- range {cal.dynamic_range:.3f}")
            print("  The floor sits above chance because some of TOFU's wrong")
            print("  answers are implausible on their face and get rejected")
            print("  without any knowledge of the subject.")
            print()
            print(f"  {'model':<12}{'raw leak':>10}{'calibrated breadth':>21}")
            for split in splits:
                raw = results[split]["forget_leakage"]
                results[split]["calibrated_breadth"] = cal.calibrated_breadth(raw)
                print(f"  {split:<12}{raw:>10.3f}{cal.calibrated_breadth(raw):>21.3f}")
            payload_cal = cal.to_dict()
        else:
            payload_cal = None

        if gap > 0.15:
            verdict = "PASS: full leaks clearly more than retain90"
        elif gap > 0.0:
            verdict = (f"WEAK: full leaks only {gap:.3f} more than retain90. "
                       "The measurement works but has little headroom.")
        else:
            verdict = ("FAIL: full does not leak more than retain90. The breadth "
                       "measurement is not detecting known knowledge -- check the "
                       "prompt format. Do NOT use these numbers.")
        print(f"\n{verdict}")

    payload = {"run_id": run_id, "size": args.size,
               "calibration": locals().get("payload_cal"),
               "prompt_format": args.prompt_format,
               "n_items": len(items), "tier_counts": counts,
               "results": results, "verdict": verdict}
    (out_dir / "breadth_report.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")
    print(f"\nSaved to {out_dir}")
    print("=" * 74)
    return 0 if (verdict is None or verdict.startswith("PASS")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
