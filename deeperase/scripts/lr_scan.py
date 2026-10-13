"""Find each method's learning rate before spending a full study on it.

Protocol v2 trains in float32, so every weight takes its full update. In v1's
bfloat16 most updates at lr 1e-6 were rounded away (RESULTS.md section 8.5),
so v1's rates do not carry over. The first H100 smoke run showed it: GA at
1e-6 lost two thirds of its utility in the first epoch (0.805 -> 0.258), with
forget-set NLL 0.39 -> 70.

A study run keeps one learning rate and needs about 45 minutes. This scan loads
the starting model once and, for each candidate rate, runs the study's own
training: the same per-author split, answer-only loss, batches, optimiser,
seed and utility measure (the R tier). It records utility and forget-set NLL
after every epoch, and has no depth or breadth sweep. A rate stops early once
utility falls below ``--stop-ratio`` of the start, since no later epoch could
be kept.

For each rate it reports the epoch the study's checkpoint rule would keep
(the highest forget NLL among epochs holding >= ``--floor`` of the starting
utility), and a verdict:

* **too fast**: no epoch, or only the first, kept the floor;
* **too slow**: the last epoch still held it, so the budget ran out first;
* **usable**: anything in between. The recommended rate is the usable one
  whose kept epoch is nearest the middle of the budget.

Results go to ``results/lr_scans/<method>_<size>_<protocol>_s<seed>.json``. A
later scan with other rates adds to the same file.

    python -m deeperase.scripts.lr_scan --method ga --lrs 1e-7,5e-8,2e-8,1e-8,5e-9
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from deeperase.config import TOFU_MODELS
from deeperase.data.tofu import (
    LossScope, PromptFormat, SamplingStrategy, load_tofu, split_by_author,
)
from deeperase.eval.breadth import load_breadth_items_file, score_breadth
from deeperase.models import load_model, load_tokenizer
from deeperase.provenance import environment_info, git_info
from deeperase.scripts.measure_breadth import PROMPT_PREFIXES, subsample
from deeperase.scripts.run_study import (
    FORGET_CONFIG, PROTOCOLS, build_forget_batches, build_retain_batches,
    configure_precision,
)
from deeperase.unlearn import (
    CollapsedError, UnlearnConfig, UnlearnMethod, snapshot, unlearn,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("lr_scan")
logging.getLogger("deeperase").setLevel(logging.INFO)

#: The smallest floor UnlearnConfig accepts. The scan keeps every epoch so that
#: training is never cut short by checkpoint selection; the study's floor is
#: applied afterwards, from the recorded curve.
_KEEP_EVERYTHING = 1e-9


# ---------------------------------------------------------------------------
# Reading a curve
# ---------------------------------------------------------------------------

def select_epoch(epochs: Sequence[dict], baseline: float, floor: float) -> Optional[int]:
    """The epoch the study keeps: the highest forget NLL among epochs whose
    utility is at least ``floor`` x the starting utility (the earliest on a tie),
    exactly as :func:`deeperase.unlearn.unlearn` selects. None if none qualifies."""
    best, best_nll = None, float("-inf")
    for e in epochs:
        if e["utility"] >= floor * baseline and e["forget_nll"] > best_nll:
            best, best_nll = e["epoch"], e["forget_nll"]
    return best


def verdict(selected: Optional[int], budget: int) -> str:
    if selected is None or selected == 0:
        return "too fast"
    if selected >= budget - 1:
        return "too slow"
    return "usable"


def recommend(rates: Dict[str, dict], budget: int) -> Optional[str]:
    """The usable rate whose kept epoch lies nearest the middle of the budget."""
    usable = [(abs(r["selected_epoch"] - (budget - 1) / 2), float(k), k)
              for k, r in rates.items() if r.get("verdict") == "usable"
              and r.get("budget") == budget]
    return min(usable)[2] if usable else None


def summarise(lr: float, history, budget: int, floor: float) -> dict:
    base = history.baseline_utility or 0.0
    epochs = [{"epoch": e.epoch, "utility": round(e.utility, 4),
               "utility_ratio": round(e.utility / base, 4) if base else None,
               "forget_nll": round(e.forget_nll, 4)} for e in history.epoch_evals]
    sel = select_epoch(epochs, base, floor)
    rec = {"lr": lr, "budget": budget, "floor": floor,
           "baseline_utility": base, "baseline_forget_nll": history.baseline_forget_nll,
           "epochs": epochs, "n_epochs_run": len(epochs),
           "stopped_early": bool(getattr(history, "stopped_early", False)),
           "selected_epoch": sel, "verdict": verdict(sel, budget),
           "train_seconds": round(history.seconds, 1)}
    if sel is not None:
        kept = next(e for e in epochs if e["epoch"] == sel)
        rec["selected_utility"] = kept["utility"]
        rec["selected_forget_nll"] = kept["forget_nll"]
    return rec


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", required=True, choices=[m.value for m in UnlearnMethod])
    ap.add_argument("--lrs", required=True, help="comma-separated, e.g. 1e-7,3e-8,1e-8")
    ap.add_argument("--epochs", type=int, default=20,
                    help="budget per rate; use the study's (gpu/replicate.py: 20)")
    ap.add_argument("--size", choices=sorted(TOFU_MODELS), default="1B")
    ap.add_argument("--protocol", choices=["v2"], default="v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--loss-scope", choices=[s.value for s in LossScope], default=None)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    ap.add_argument("--floor", type=float, default=None,
                    help="utility floor of the study's checkpoint rule (v2: 0.95)")
    ap.add_argument("--stop-ratio", type=float, default=0.5,
                    help="stop a rate once utility is below this fraction of the start")
    ap.add_argument("--n-breadth", type=int, default=400)
    ap.add_argument("--n-retain", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--prompt-format", default="plain_qa",
                    choices=[f.value for f in PromptFormat])
    ap.add_argument("--breadth-items", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-sources", default=None,
                    help="JSON mapping split -> local model dir (tests)")
    ap.add_argument("--tofu-dir", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("DEEPERASE_CACHE_DIR", "./hf_cache"))
    ap.add_argument("--out-dir", default="results/lr_scans")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    proto = PROTOCOLS[args.protocol]
    loss_scope = LossScope(args.loss_scope or proto["loss_scope"])
    dtype_req = args.dtype or str(proto["dtype"])
    floor = float(args.floor if args.floor is not None else proto["min_utility_ratio"])
    precision = configure_precision(dtype_req)
    method = UnlearnMethod(args.method)
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype_name = "float32" if device.type == "cpu" else dtype_req
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    sources = json.loads(args.model_sources) if args.model_sources else {}
    full_source = sources.get("full") or TOFU_MODELS[args.size]["full"].repo_id

    out = Path(args.out_dir) / f"{args.method}_{args.size}_{args.protocol}_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    record.update({
        "what": "learning-rate scan: the study's training, utility and forget NLL per epoch",
        "method": args.method, "size": args.size, "protocol": args.protocol,
        "seed": args.seed, "batch_size": args.batch_size, "beta": args.beta,
        "loss_scope": loss_scope.value, "dtype": dtype_name, "precision": precision,
        "floor": floor, "stop_ratio": args.stop_ratio,
        "git": git_info(), "environment": environment_info()})
    rates: Dict[str, dict] = record.setdefault("rates", {})

    # -- data, exactly as run_study builds it ---------------------------------
    fmt = PromptFormat(args.prompt_format)
    tokenizer = load_tokenizer(full_source, cache_dir=args.cache_dir)
    forget_rows = load_tofu(FORGET_CONFIG, cache_dir=args.cache_dir, data_dir=args.tofu_dir)
    trained, _ = split_by_author(len(forget_rows), train_fraction=args.train_fraction,
                                 seed=args.seed)
    forget_batches = build_forget_batches([forget_rows[i] for i in trained], tokenizer,
                                          fmt, args.max_length, loss_scope)
    items_path = args.breadth_items or proto["breadth_items"]
    r_items = [i for i in subsample(load_breadth_items_file(items_path), args.n_breadth)
               if i.tier == "R"]
    prefix = PROMPT_PREFIXES.get(args.prompt_format, "{question} ")
    retain_batches = None
    if method.needs_retain:
        retain_batches = build_retain_batches(
            tokenizer, fmt, args.max_length, args.n_retain or len(forget_batches),
            args.cache_dir, SamplingStrategy(str(proto["sampling"])), args.seed,
            loss_scope=loss_scope, data_dir=args.tofu_dir)
    log.info("%s scan: %d trained rows, %d R items, rates %s, %d epochs, floor %.0f%%",
             args.method, len(forget_batches), len(r_items), lrs, args.epochs, 100 * floor)

    def utility_of(m) -> float:
        return score_breadth(m, tokenizer, r_items, max_length=args.max_length,
                             prompt_prefix=prefix).tiers["R"].knows_rate

    model = load_model(full_source, device=device, dtype=torch_dtype, cache_dir=args.cache_dir)
    theta_ini = snapshot(model)
    reference = (load_model(full_source, device=device, dtype=torch_dtype,
                            cache_dir=args.cache_dir) if method.needs_reference else None)

    for lr in lrs:
        log.info("=== %s lr=%g ===", args.method, lr)
        model.load_state_dict(theta_ini)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cfg = UnlearnConfig(method=method, learning_rate=lr, epochs=args.epochs,
                            batch_size=args.batch_size, beta=args.beta, seed=args.seed,
                            min_utility_ratio=_KEEP_EVERYTHING,
                            stop_below_utility_ratio=args.stop_ratio)
        t0 = time.time()
        try:
            history = unlearn(model, forget_batches, cfg, retain_data=retain_batches,
                              reference_model=reference, eval_fn=utility_of)
        except CollapsedError as e:            # utility exactly 0 in every epoch
            history = e.history
        rec = summarise(lr, history, args.epochs, floor)
        rec["wall_seconds"] = round(time.time() - t0, 1)
        if device.type == "cuda":
            rec["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        rates[f"{lr:g}"] = rec
        record["recommended_lr"] = recommend(rates, args.epochs)
        out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        log.info("lr=%g: kept epoch %s of %d (%s), utility %s, forget NLL %s",
                 lr, rec["selected_epoch"], args.epochs, rec["verdict"],
                 rec.get("selected_utility"), rec.get("selected_forget_nll"))

    print("\n" + "=" * 76)
    print(f"LEARNING-RATE SCAN  {args.method}  ({dtype_name}, {loss_scope.value} loss, "
          f"floor {floor:.0%}, {args.epochs} epochs)")
    print("=" * 76)
    print(f"{'lr':>9}  {'kept epoch':>10}  {'utility':>8}  {'forget NLL':>10}  "
          f"{'epochs run':>10}  verdict")
    for k, r in sorted(rates.items(), key=lambda kv: float(kv[0]), reverse=True):
        print(f"{k:>9}  {str(r['selected_epoch']):>10}  {str(r.get('selected_utility', '-')):>8}"
              f"  {str(r.get('selected_forget_nll', '-')):>10}  {r['n_epochs_run']:>10}  "
              f"{r['verdict']}")
    rec_lr = record.get("recommended_lr")
    print(f"\nRecommended: {rec_lr}" if rec_lr else
          "\nNo usable rate yet: scan lower rates if all were too fast, higher if too slow.")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
