"""How much did the breadth-scoring defect change the numbers? (RESULTS.md §8.1)

Until 9 Oct 2026 the forced-choice scorer located the answer by tokenising the
prompt on its own and taking its length. With Llama-3's tokenizer that is one
token too far, so the first answer token was never scored. This script measures
the effect on real models by scoring every item **both ways from the same
forward pass**: the logits are shared, only the averaged positions differ, so
any difference is the defect and nothing else.

Reported per model and tier: the knows-rate under each rule, how many items
flip, and -- for the calibration pair -- floor, ceiling and usable range.

    python -m deeperase.scripts.breadth_boundary_effect --device mps \\
        --cache-dir ../_workspace/hf_cache --splits full,retain90

The legacy rule is reimplemented below only so its effect can be measured; it
is not used anywhere else.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from deeperase.config import TOFU_MODELS
from deeperase.eval.breadth import (
    BreadthItem, answer_token_start, load_breadth_items_file,
)
from deeperase.models import load_model, load_tokenizer
from deeperase.scripts.measure_breadth import PROMPT_PREFIXES

log = logging.getLogger("boundary")


def legacy_start(tokenizer, prefix: str, n_full: int) -> int:
    """The defective rule: the prompt's own token count (clamped as before)."""
    n_prefix = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
    return min(n_prefix, n_full - 1)


@torch.no_grad()
def score_item(model, tokenizer, item: BreadthItem, prefix_fmt: str, max_length: int,
               device) -> Dict[str, Dict[str, float]]:
    """Mean log-prob of every candidate under both rules, from one forward pass each."""
    prefix = prefix_fmt.format(question=item.question)
    out = {"fixed": [], "legacy": []}
    for cand in [item.correct_answer, *item.wrong_answers]:
        ids, start = answer_token_start(tokenizer, prefix, cand, max_length=max_length)
        if start is None:
            out["fixed"].append(float("-inf")); out["legacy"].append(float("-inf"))
            continue
        x = torch.tensor([ids], device=device)
        lp = F.log_softmax(model(input_ids=x).logits.float(), dim=-1)[0]
        tok_lp = lp[:-1].gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)   # tok_lp[t-1] scores token t
        old = legacy_start(tokenizer, prefix, len(ids))
        out["fixed"].append(float(tok_lp[start - 1:].mean()))
        out["legacy"].append(float(tok_lp[old - 1:].mean()))
    return {rule: {"knows": v[0] > max(v[1:]), "margin": v[0] - max(v[1:])}
            for rule, v in out.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", default="1B", choices=sorted(TOFU_MODELS))
    ap.add_argument("--splits", default="full,retain90")
    ap.add_argument("--items", default="data/breadth_items_v2.json")
    ap.add_argument("--tiers", default="B0,B1,R,RA,WF,B2,B4")
    ap.add_argument("--n-per-tier", type=int, default=0, help="0 = all items")
    ap.add_argument("--prompt-format", default="plain_qa", choices=sorted(PROMPT_PREFIXES))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--out", default="results/analysis/breadth_boundary_effect.json")
    args = ap.parse_args(argv)

    tiers = args.tiers.split(",")
    items = load_breadth_items_file(args.items, tiers=tiers)
    if args.n_per_tier:
        from deeperase.scripts.measure_breadth import subsample
        items = subsample(items, args.n_per_tier)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    prefix_fmt = PROMPT_PREFIXES[args.prompt_format]
    specs = TOFU_MODELS[args.size]
    tok = load_tokenizer(specs["full"].repo_id, cache_dir=args.cache_dir)

    report: Dict[str, object] = {"items": args.items, "n_items": len(items),
                                 "prompt_format": args.prompt_format,
                                 "device": str(device), "dtype": args.dtype, "models": {}}
    per_item: Dict[str, Dict[str, dict]] = {}
    for split in args.splits.split(","):
        t0 = time.time()
        model = load_model(specs[split].repo_id, device=device, dtype=dtype,
                           cache_dir=args.cache_dir)
        res: Dict[str, dict] = {}
        for k, it in enumerate(items):
            res[it.item_id] = score_item(model, tok, it, prefix_fmt, args.max_length, device)
            if (k + 1) % 200 == 0:
                log.info("  %s: %d/%d items (%.0fs)", split, k + 1, len(items), time.time() - t0)
        per_item[split] = res
        by_tier: Dict[str, dict] = {}
        for t in tiers:
            ids = [i.item_id for i in items if i.tier == t]
            if not ids:
                continue
            fixed = sum(res[i]["fixed"]["knows"] for i in ids)
            legacy = sum(res[i]["legacy"]["knows"] for i in ids)
            flips = sum(res[i]["fixed"]["knows"] != res[i]["legacy"]["knows"] for i in ids)
            by_tier[t] = {"n": len(ids), "fixed": fixed / len(ids),
                          "legacy": legacy / len(ids), "flipped": flips}
        report["models"][split] = by_tier
        log.info("%s done in %.0fs: %s", split, time.time() - t0,
                 {t: (round(v["legacy"], 3), round(v["fixed"], 3)) for t, v in by_tier.items()})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    m = report["models"]
    if "full" in m and "retain90" in m:
        cal = {}
        for rule in ("legacy", "fixed"):
            for group in (("B0", "B1"),):
                ok = [t for t in group if t in m["full"]]
                n = sum(m["full"][t]["n"] for t in ok)
                ceil = sum(m["full"][t][rule] * m["full"][t]["n"] for t in ok) / n
                floor = sum(m["retain90"][t][rule] * m["retain90"][t]["n"] for t in ok) / n
                cal[rule] = {"floor": floor, "ceiling": ceil, "range": ceil - floor}
        report["calibration_B0_B1"] = cal

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
