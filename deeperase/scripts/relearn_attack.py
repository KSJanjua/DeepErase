"""Relearning attack: does fine-tuning on some forgotten facts bring back the others?

A depth test that does not rely on UDS (RESULTS.md section 19; method of Hu et
al. 2024 and Deeb & Roger 2024). An unlearned model that only *suppressed*
its facts gets the untouched ones back when it is briefly fine-tuned on a few
of them: the suppression is a thin layer that the fine-tuning wears off. A
model that *erased* them has nothing to give back.

Procedure, for one method and seed:

1. Train the method exactly as the study does (``lr_scan.prepare``: same
   split, batches, representation inputs, utility floor and checkpoint rule).
   That gives the checkpoint (alpha = 0).
2. Split each author's 10 trained facts in half, with a seeded shuffle:
   * the ATTACK half, which the attacker fine-tunes on;
   * the PROBE half, never shown to the attacker.
   The held-out facts form a third set, also never shown.
3. Fine-tune on the attack half: answer-only NLL, AdamW, ``--attack-lr``,
   ``--attack-epochs``.
4. Score the probe half and the held-out facts, before and after the attack:
   * forced-choice knows-rate on their B0/B1 breadth items;
   * mean answer NLL.
5. Do the same to two reference models. ``full`` knew everything.
   ``retain90`` never saw the forget set, so it shows how much of the probe
   facts can be learned from the attack half alone.

Latent-knowledge index on a probe set, after the attack::

    (unlearned - retain90) / (full - retain90)

It is 0 when the unlearned model recovers no more than an ignorant model, and
1 when it is back to full knowledge. Exploratory.

``--attack-lrs`` gives a dose-response: each listed rate attacks a fresh copy
of the same checkpoint (and of each reference), and the index is reported per
rate. A strong attack also moves the full model, so a weak one can separate
methods that a strong one makes look alike.

    python -m deeperase.scripts.relearn_attack --method unasked --learning-rate 3e-6 --seed 0
    python -m deeperase.scripts.relearn_attack --method ga --learning-rate 1e-8 --seed 0 \
        --attack-lrs 1e-7,3e-7,1e-6,3e-6,1e-5 --tag dose
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from deeperase.config import TOFU_MODELS
from deeperase.data.tofu import QAS_PER_AUTHOR, load_tofu, split_by_author
from deeperase.eval.breadth import load_breadth_items_file, score_breadth
from deeperase.models import load_model
from deeperase.provenance import environment_info, git_info
from deeperase.scripts import lr_scan
from deeperase.scripts.measure_breadth import PROMPT_PREFIXES
from deeperase.scripts.run_study import FORGET_CONFIG, build_forget_batches
from deeperase.unlearn import (
    CollapsedError, UnlearnConfig, _batches, _nll, snapshot, unlearn,
)

log = logging.getLogger("relearn_attack")


def split_trained(trained: Sequence[int], seed: int) -> Dict[str, List[int]]:
    """Half of each author's trained rows to attack with, half to probe."""
    by_author: Dict[int, List[int]] = {}
    for r in sorted(trained):
        by_author.setdefault(r // QAS_PER_AUTHOR, []).append(r)
    rng = random.Random(10_000 + seed)
    attack, probe = [], []
    for a in sorted(by_author):
        rows = by_author[a][:]
        rng.shuffle(rows)
        k = len(rows) // 2
        attack += sorted(rows[:k])
        probe += sorted(rows[k:])
    return {"attack": sorted(attack), "probe": sorted(probe)}


def latent_index(unl: Optional[float], full: Optional[float],
                 ret: Optional[float]) -> Optional[float]:
    """(unlearned - retain90) / (full - retain90); None if undefined."""
    if None in (unl, full, ret) or abs(full - ret) < 1e-9:
        return None
    return (unl - ret) / (full - ret)


@torch.no_grad()
def mean_nll(model, batches, batch_size: int, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for b in _batches(batches, batch_size, device):
        per = _nll(model, b)
        total += float(per.sum())
        n += int(per.numel())
    return total / max(1, n)


def attack(model, batches, *, lr: float, epochs: int, batch_size: int, device,
           seed: int) -> None:
    """Plain fine-tuning (answer-only NLL) on the attack facts, in place."""
    torch.manual_seed(seed)
    model.train()
    model.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False)
    for _ in range(epochs):
        for b in _batches(batches, batch_size, device):
            loss = _nll(model, b).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    model.requires_grad_(False)


def measure(model, tokenizer, items_by_set, batches_by_set, prefix, max_length,
            batch_size, device) -> Dict[str, dict]:
    out = {}
    for name, items in items_by_set.items():
        res = score_breadth(model, tokenizer, items, max_length=max_length,
                            prompt_prefix=prefix)
        n = sum(t.n for t in res.tiers.values())
        out[name] = {"knows_rate": sum(t.n_knows for t in res.tiers.values()) / n if n else None,
                     "n_items": n,
                     "answer_nll": mean_nll(model, batches_by_set[name], batch_size, device)}
    return out


def parse_args(argv: Optional[Sequence[str]] = None):
    from deeperase.unlearn import UnlearnMethod
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", required=True, choices=[m.value for m in UnlearnMethod])
    ap.add_argument("--learning-rate", type=float, required=True,
                    help="the method's study learning rate")
    ap.add_argument("--attack-lr", type=float, default=1e-5)
    ap.add_argument("--attack-lrs", default=None,
                    help="comma-separated attack rates, each applied to a fresh copy of the "
                         "same checkpoint (a dose-response); overrides --attack-lr")
    ap.add_argument("--attack-epochs", type=int, default=2)
    ap.add_argument("--no-references", action="store_true",
                    help="skip attacking full and retain90 (reuse earlier reference numbers)")
    ap.add_argument("--out-dir", default="results/relearn")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. 'dose' -> ga_1B_v2_s0_dose.json")
    lr_scan.add_training_args(ap)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    t0 = time.time()
    ns = lr_scan.prepare(args)
    device = ns.device
    sources = json.loads(args.model_sources) if args.model_sources else {}
    repo = {s: sources.get(s) or TOFU_MODELS[args.size][s].repo_id for s in ("full", "retain90")}
    from deeperase.models import load_tokenizer
    from deeperase.data.tofu import PromptFormat
    tokenizer = load_tokenizer(repo["full"], cache_dir=args.cache_dir)
    fmt = PromptFormat(args.prompt_format)
    prefix = PROMPT_PREFIXES.get(args.prompt_format, "{question} ")

    forget_rows = load_tofu(FORGET_CONFIG, cache_dir=args.cache_dir, data_dir=args.tofu_dir)
    trained, held = split_by_author(len(forget_rows), train_fraction=args.train_fraction,
                                    seed=args.seed)
    sets = split_trained(trained, args.seed)
    sets["heldout"] = list(held)
    items_path = args.breadth_items or ns.proto["breadth_items"]
    items = [i for i in load_breadth_items_file(items_path) if i.tier in ("B0", "B1")]
    items_by_set = {k: [i for i in items if i.source_index in set(rows)]
                    for k, rows in sets.items() if k != "attack"}
    batches_by_set = {k: build_forget_batches([forget_rows[r] for r in rows], tokenizer, fmt,
                                              args.max_length, ns.loss_scope)
                      for k, rows in sets.items()}

    # 1. the method's checkpoint, trained exactly as the study trains it
    cfg = UnlearnConfig(method=ns.method, learning_rate=args.learning_rate,
                        epochs=args.epochs, batch_size=args.batch_size, beta=args.beta,
                        seed=args.seed, min_utility_ratio=ns.floor,
                        retain_weight=ns.retain_weight, trainable=ns.trainable,
                        select_by="rep_gap" if ns.rep else "forget_nll")
    try:
        history = unlearn(ns.model, ns.forget_batches, cfg, retain_data=ns.retain_batches,
                          reference_model=ns.reference, eval_fn=ns.utility_of,
                          stranger=ns.stranger)
    except CollapsedError as e:
        raise SystemExit(f"Training collapsed: {e}")
    ns.reference = None
    rates = ([float(x) for x in args.attack_lrs.split(",") if x.strip()]
             if args.attack_lrs else [args.attack_lr])

    def run_attacks(model) -> Dict[str, dict]:
        """before, then after an attack at each rate, each from the same weights."""
        start = snapshot(model)
        out = {"before": measure(model, tokenizer, items_by_set, batches_by_set, prefix,
                                 args.max_length, args.batch_size, device)}
        for lr in rates:
            model.load_state_dict(start)
            attack(model, batches_by_set["attack"], lr=lr, epochs=args.attack_epochs,
                   batch_size=args.batch_size, device=device, seed=args.seed)
            out[f"after_{lr:g}"] = measure(model, tokenizer, items_by_set, batches_by_set,
                                           prefix, args.max_length, args.batch_size, device)
        out["after"] = out[f"after_{rates[-1]:g}"]
        return out

    results = {"unlearned": run_attacks(ns.model)}
    del ns.model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. the references, attacked identically
    if not args.no_references:
        for name in ("full", "retain90"):
            m = load_model(repo[name], device=device, dtype=ns.torch_dtype,
                           cache_dir=args.cache_dir)
            results[name] = run_attacks(m)
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

    by_lr: Dict[str, dict] = {}
    if "full" in results:
        for lr in rates:
            key = f"after_{lr:g}"
            for probe in items_by_set:
                cell = by_lr.setdefault(f"{lr:g}", {}).setdefault(probe, {})
                for metric in ("knows_rate", "answer_nll"):
                    u, f, r = (results[k][key][probe][metric]
                               for k in ("unlearned", "full", "retain90"))
                    cell[metric] = latent_index(u, f, r)
    # the last listed rate, as in the single-rate record
    index = by_lr.get(f"{rates[-1]:g}", {})
    record = {"what": "relearning attack on the method's checkpoint", "method": args.method,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "attack_lr": rates[-1], "attack_lrs": rates, "attack_epochs": args.attack_epochs,
              "selected_epoch": history.selected_epoch, "sets": {k: len(v) for k, v in sets.items()},
              "results": results, "latent_index": index, "latent_index_by_lr": by_lr,
              "seconds": round(time.time() - t0, 1),
              "git": git_info(), "environment": environment_info()}
    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.out_dir) / f"{args.method}_{args.size}_{args.protocol}_s{args.seed}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"RELEARNING ATTACK  {args.method}  seed {args.seed}  ({args.attack_epochs} epochs "
          f"on {len(sets['attack'])} facts)")
    print("=" * 72)
    for lr in rates:
        key = f"after_{lr:g}"
        for probe in items_by_set:
            line = [f"lr {lr:<7g} {probe:8s}"]
            for k in ("unlearned", "full", "retain90"):
                if k in results:
                    b, a = results[k]["before"][probe], results[k][key][probe]
                    line.append(f"{k} {b['knows_rate']:.3f}->{a['knows_rate']:.3f}")
            cell = by_lr.get(f"{lr:g}", {}).get(probe)
            if cell:
                line.append("index knows {}  nll {}".format(
                    *("–" if cell[m] is None else f"{cell[m]:.2f}"
                      for m in ("knows_rate", "answer_nll"))))
            print("  ".join(line))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
