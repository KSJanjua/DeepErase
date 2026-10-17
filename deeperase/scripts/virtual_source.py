"""Gate 1 for STRANGER: the depth its objective could reach, before any training.

STRANGER trains the MLPs of layers 0-8 so that, at chosen positions, the block
outputs of layers 4-8 match M_full's own outputs on the name-swapped row. If
training succeeded exactly, the unlearned model's residual stream would equal
a *hybrid*: M_full run on the original row, with those block outputs replaced
by the targets. This script builds that hybrid without training, uses it as
the stage-2 source of UDS against the usual retain90 stage 1, and reports the
ceiling. It runs on the GPU box in a few minutes (two 1B models, inference
only).

Sources scored on the seed's trained and held-out depth examples:

* ``self``: M_full itself. Sanity check, UDS ~ 0.
* ``hybrid``: the primary ceiling (the training positions, layers 4-8).
* ``hybrid_q``: question positions only (no answer-side name tokens).
* ``scramble``: the SCRAMBLE arm's optimum (same displacement, random heading).
* ``fullswap``: M_full on the name-swapped row itself, per name, averaged.
  The upper bound of any name-swap target.

Pre-registered decision (docs/ANALYSIS_PLAN.md, H7), on the TRAINED split of
``hybrid``: GO if UDS >= 0.50, strict UDS >= 0.45 and overshoot share <= 0.10;
WEAK GO if 0.35 <= UDS < 0.50; NO-GO below 0.35. Held-out numbers are reported
and drive no decision.

    python -m deeperase.scripts.virtual_source --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from deeperase.config import TOFU_MODELS
from deeperase.data.counterfactual import (
    align_row, invented_for, load_names, stranger_positions,
)
from deeperase.data.tofu import (
    QAS_PER_AUTHOR, PromptFormat, SamplingStrategy, load_tofu, split_by_author,
)
from deeperase.eval.patching import n_layers
from deeperase.eval.uds import capture_source_hidden, score_from_captured
from deeperase.eval.uds_diagnostics import example_diagnostics, summarise
from deeperase.models import load_model, load_tokenizer
from deeperase.provenance import environment_info, git_info
from deeperase.scripts.run_study import (
    FORGET_CONFIG, build_uds_examples, configure_precision, example_row,
)
from deeperase.stranger import (
    REP_LAYERS, StrangerRow, block_outputs, row_states, scramble_direction, sha256_of,
)

log = logging.getLogger("virtual_source")

GO = {"uds": 0.50, "strict": 0.45, "overshoot_share": 0.10}
WEAK = 0.35


def decide(d: dict) -> str:
    if d["uds"] is None:
        return "NO DATA"
    if (d["uds"] >= GO["uds"] and d["strict"] >= GO["strict"]
            and d["overshoot_share"] <= GO["overshoot_share"]):
        return "GO"
    if d["uds"] >= WEAK:
        return "WEAK GO"
    return "NO-GO"


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--layers", default=",".join(str(l) for l in REP_LAYERS))
    ap.add_argument("--size", default="1B")
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--cf-names", default="data/counterfactual_names.json")
    ap.add_argument("--reference-spans",
                    default="reference_uds/tofu_data/forget10_filtered.json")
    ap.add_argument("--tofu-dir", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("DEEPERASE_CACHE_DIR", "./hf_cache"))
    ap.add_argument("--model-sources", default=None,
                    help="JSON mapping split -> local model dir (tests)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results/gates")
    return ap.parse_args(argv)


def _row_for(example, forget_rows, tokenizer, names, k, max_length) -> StrangerRow:
    idx = example_row(example)
    r = forget_rows[idx]
    name, inv = invented_for(names, idx // QAS_PER_AUTHOR, k)
    al = align_row(tokenizer, r.question, r.answer, name, inv, fmt=PromptFormat.PLAIN_QA,
                   max_length=max_length)
    if al.input_ids != example.input_ids[0].tolist():
        raise ValueError(f"{example.example_id}: depth-example ids differ from the "
                         "counterfactual alignment's")
    row = StrangerRow(author=idx // QAS_PER_AUTHOR, input_ids=al.input_ids,
                      cf_input_ids=al.cf_input_ids, positions=stranger_positions(al))
    row.q_positions = stranger_positions(al, question_only=True)
    row.name_tokens = set(al.name_tokens)
    return row


@torch.no_grad()
def hybrid_sources(m_full, examples, rows, layers_all, rep_layers, device, seed):
    """Captured states for the hybrid, question-only, scramble and full-swap sources."""
    out = {"hybrid": {}, "hybrid_q": {}, "scramble": {}}
    fullswap: List[Dict[str, Dict[int, torch.Tensor]]] = []
    for ex, row in zip(examples, rows):
        ids = ex.input_ids.to(device)
        for variant, positions in (("hybrid", row.positions), ("hybrid_q", row.q_positions)):
            if not positions:
                out[variant][ex.example_id] = {
                    l: h.cpu() for l, h in block_outputs(m_full, ids, layers_all).items()}
                continue
            r = StrangerRow(row.author, row.input_ids, row.cf_input_ids, list(positions))
            h0, tau = row_states(m_full, r, rep_layers, device)
            pos = torch.as_tensor(positions, device=device)
            caps = block_outputs(m_full, ids, layers_all,
                                 replace={l: (pos, tau[l]) for l in rep_layers})
            out[variant][ex.example_id] = {l: h.cpu() for l, h in caps.items()}
            if variant == "hybrid":
                scr = {}
                for l in rep_layers:
                    d = scramble_direction(seed, row.author, l, tau[l].shape[-1], device)
                    scr[l] = h0[l] + (tau[l] - h0[l]).norm(dim=-1, keepdim=True) * d
                caps = block_outputs(m_full, ids, layers_all,
                                     replace={l: (pos, scr[l]) for l in rep_layers})
                out["scramble"][ex.example_id] = {l: h.cpu() for l, h in caps.items()}
        for k, cf in enumerate(row.cf_input_ids):
            while len(fullswap) <= k:
                fullswap.append({})
            cid = torch.tensor([cf], dtype=torch.long, device=device)
            fullswap[k][ex.example_id] = {
                l: h.cpu() for l, h in block_outputs(m_full, cid, layers_all).items()}
    return out, fullswap


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    t0 = time.time()
    configure_precision("float32")
    torch.manual_seed(args.seed)
    rep_layers = tuple(int(x) for x in args.layers.split(","))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sources = json.loads(args.model_sources) if args.model_sources else {}
    repo = {s: sources.get(s) or TOFU_MODELS[args.size][s].repo_id
            for s in ("full", "retain90")}
    tokenizer = load_tokenizer(repo["full"], cache_dir=args.cache_dir)
    names = load_names(args.cf_names)

    forget_rows = load_tofu(FORGET_CONFIG, cache_dir=args.cache_dir, data_dir=args.tofu_dir)
    trained, held = split_by_author(len(forget_rows), train_fraction=args.train_fraction,
                                    seed=args.seed)
    fmt = PromptFormat.PLAIN_QA
    sets = {
        "trained": build_uds_examples(tokenizer, args.reference_spans, fmt, args.max_length,
                                      args.n_examples, SamplingStrategy.RANDOM, args.seed,
                                      allowed=set(trained), tag="trained"),
        "heldout": build_uds_examples(tokenizer, args.reference_spans, fmt, args.max_length,
                                      args.n_examples, SamplingStrategy.RANDOM, args.seed + 1,
                                      allowed=set(held), tag="heldout"),
    }
    rows = {k: [_row_for(e, forget_rows, tokenizer, names, args.k, args.max_length)
                for e in v] for k, v in sets.items()}

    # Stage 1 exactly as run_study computes it.
    m_ret = load_model(repo["retain90"], device=device, dtype=torch.float32,
                       cache_dir=args.cache_dir)
    layers_all = list(range(n_layers(m_ret)))
    caps_ret = {k: capture_source_hidden(m_ret, v, layers_all, to_cpu=True)
                for k, v in sets.items()}
    del m_ret
    if device.type == "cuda":
        torch.cuda.empty_cache()
    m_full = load_model(repo["full"], device=device, dtype=torch.float32,
                        cache_dir=args.cache_dir)
    m_full.eval()

    report = {"what": "STRANGER gate 1: virtual-source UDS ceiling", "seed": args.seed,
              "k": args.k, "rep_layers": list(rep_layers), "thresholds": {"go": GO, "weak": WEAK},
              "cf_names_sha256": sha256_of(args.cf_names), "splits": {},
              "git": git_info(), "environment": environment_info()}
    for split, examples in sets.items():
        d1, s_full = score_from_captured(m_full, caps_ret[split], examples, layers_all)
        variants, fullswap = hybrid_sources(m_full, examples, rows[split], layers_all,
                                            rep_layers, device, args.seed)
        variants["self"] = capture_source_hidden(m_full, examples, layers_all, to_cpu=True)
        d2s = {name: score_from_captured(m_full, caps, examples, layers_all,
                                         s_full_cache=s_full)[0]
               for name, caps in variants.items()}
        per_k = [score_from_captured(m_full, caps, examples, layers_all,
                                     s_full_cache=s_full)[0] for caps in fullswap]
        d2s["fullswap"] = {eid: {l: sum(p[eid][l] for p in per_k) / len(per_k)
                                 for l in layers_all} for eid in per_k[0]}
        name_entity = {e.example_id for e, r in zip(examples, rows[split])
                       if set(j - 1 for j in e.span.token_indices) & r.name_tokens
                       or set(e.span.token_indices) & r.name_tokens}
        res = {}
        for name, d2 in d2s.items():
            per = {eid: example_diagnostics(d1[eid], d2[eid]) for eid in d1}
            res[name] = {"all": summarise(list(per.values())),
                         "without_name_entities": summarise(
                             [v for eid, v in per.items() if eid not in name_entity]),
                         "per_example_uds": {eid: (None if v is None else
                                                   {"uds": round(v["uds"], 5),
                                                    "strict": round(v["strict"], 5)})
                                             for eid, v in per.items()}}
        res["n_name_entities"] = len(name_entity)
        res["positions_per_row"] = sum(len(r.positions) for r in rows[split]) / len(rows[split])
        report["splits"][split] = res

    report["decision"] = decide(report["splits"]["trained"]["hybrid"]["all"])
    report["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out_dir) / f"stranger_g1_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    lines = [f"STRANGER gate 1 (seed {args.seed}, K={args.k}, layers {list(rep_layers)})", "",
             "| split | source | UDS | strict | tent | overshoot share | L0-8 | L9-12 | L13-15 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for split, res in report["splits"].items():
        for name in ("self", "hybrid", "hybrid_q", "scramble", "fullswap"):
            d = res[name]["all"]
            b = d["bands"]
            lines.append(f"| {split} | {name} | {f(d['uds'])} | {f(d['strict'])} | "
                         f"{f(d['tent'])} | {f(d['overshoot_share'])} | {f(b['L0-8'])} | "
                         f"{f(b['L9-12'])} | {f(b['L13-15'])} |")
    lines += ["", f"DECISION (trained, hybrid): {report['decision']}  "
              f"(GO: UDS >= {GO['uds']}, strict >= {GO['strict']}, overshoot <= "
              f"{GO['overshoot_share']}; WEAK GO: UDS >= {WEAK})"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out} ({report['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
