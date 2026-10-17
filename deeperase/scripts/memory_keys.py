"""Gate 1b: what is the fine-tuned model's memory of a TOFU fact keyed on?

STRANGER's gate 1 (RESULTS.md §17) showed that the author's name is not the
key: with the name swapped everywhere, M_full still predicts 80-85% of what
separates it from retain90 at the entity positions. This script asks what the
key is. It runs M_full on edited copies of each depth example and scores each
copy as a UDS stage-2 source against the usual retain90 stage 1.
Every edit keeps the token sequence the same length and leaves the entity
tokens in place, so the patched positions line up exactly. Inference only.

Sources (M_full on ...):

* ``name``: the name swapped everywhere (gate 1's full swap; 4 names, averaged).
* ``question``: the question replaced by a neutral filler of the same length.
* ``question_name``: both (2 names, averaged).
* ``other_question``: the question replaced by a retain-author question with
  the same number of tokens (R-tier authors excluded).
* ``context``: everything before the entity (question and answer prefix)
  replaced by filler.
* ``self``: M_full on the original (sanity, UDS ~ 0).

Each source is scored on the whole entity (as UDS) and on its first token
alone. The first token is where a fact is retrieved; later tokens of a
memorised string can be completed from their own prefix.

Pre-registered rule (docs/ANALYSIS_PLAN.md, H8 gate). A successor method is
designed only if some candidate source (``question``, ``question_name``,
``other_question`` or ``context``) reaches, on the TRAINED split, UDS >= 0.50,
strict UDS >= 0.45 and overshoot share <= 0.10. Otherwise no further method is
built.

    python -m deeperase.scripts.memory_keys --seed 0
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
from deeperase.data.counterfactual import (
    FILLER, align_row, filler_ids, invented_for, load_names, question_tokens, replace_at,
)
from deeperase.data.tofu import (
    QAS_PER_AUTHOR, PromptFormat, SamplingStrategy, build_prompt, load_tofu, split_by_author,
)
from deeperase.eval.patching import EntitySpan, n_layers
from deeperase.eval.uds import UDSExample, capture_source_hidden, score_from_captured
from deeperase.eval.uds_diagnostics import example_diagnostics, summarise
from deeperase.models import load_model, load_tokenizer
from deeperase.provenance import environment_info, git_info
from deeperase.scripts.run_study import (
    FORGET_CONFIG, build_uds_examples, configure_precision, drop_authors_asked_about,
    example_row,
)
from deeperase.stranger import block_outputs, sha256_of

log = logging.getLogger("memory_keys")

CANDIDATES = ("question", "question_name", "other_question", "context")
SOURCES = ("self", "name") + CANDIDATES
GO = {"uds": 0.50, "strict": 0.45, "overshoot_share": 0.10}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def edits_for(tokenizer, example: UDSExample, row, names, *, k_name: int = 4,
              k_qname: int = 2, other_pool: Optional[Dict[int, List[List[int]]]] = None,
              rng: Optional[random.Random] = None, max_length: int = 256
              ) -> Dict[str, List[List[int]]]:
    """Every source's input ids for one depth example (lists of ids per source;
    several entries are averaged)."""
    ids = example.input_ids[0].tolist()
    text, a_start = build_prompt(row.question, row.answer, PromptFormat.PLAIN_QA)
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length, add_special_tokens=True)
    if enc["input_ids"] != ids:
        raise ValueError(f"{example.example_id}: ids differ from the depth example")
    q_start = text.index(row.question)
    q_tok = question_tokens(enc["offset_mapping"], q_start, q_start + len(row.question), text)
    ent0 = min(example.span.token_indices)
    first = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a != b and b > a_start)
    prefix = sorted(set(q_tok) | set(range(first, ent0)))

    author = example_row(example) // QAS_PER_AUTHOR
    name, inv = invented_for(names, author, max(k_name, k_qname))
    al = align_row(tokenizer, row.question, row.answer, name, inv[:max(k_name, k_qname)],
                   fmt=PromptFormat.PLAIN_QA, max_length=max_length)
    fill_q = filler_ids(tokenizer, len(q_tok))
    out = {"self": [ids], "name": al.cf_input_ids[:k_name],
           "question": [replace_at(ids, q_tok, fill_q)],
           "question_name": [replace_at(c, q_tok, fill_q) for c in al.cf_input_ids[:k_qname]],
           "context": [replace_at(ids, prefix, filler_ids(tokenizer, len(prefix)))],
           "other_question": []}
    pool = (other_pool or {}).get(len(q_tok)) or []
    if pool:
        pick = (rng or random.Random(0)).choice(pool)
        out["other_question"] = [replace_at(ids, q_tok, pick)]
    return out


def retain_question_pool(tokenizer, rows, max_length: int = 256) -> Dict[int, List[List[int]]]:
    """Question token sequences of retain rows, grouped by length."""
    pool: Dict[int, List[List[int]]] = {}
    for r in rows:
        text, _ = build_prompt(r.question, r.answer, PromptFormat.PLAIN_QA)
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                        max_length=max_length, add_special_tokens=True)
        qs = text.index(r.question)
        q_tok = question_tokens(enc["offset_mapping"], qs, qs + len(r.question), text)
        if q_tok:
            pool.setdefault(len(q_tok), []).append([enc["input_ids"][i] for i in q_tok])
    return pool


def first_token_example(ex: UDSExample) -> UDSExample:
    t = min(ex.span.token_indices)
    return UDSExample(ex.example_id, ex.input_ids, EntitySpan([t]),
                      attention_mask=ex.attention_mask)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _capture(model, seqs: Dict[str, List[int]], layers, device) -> Dict[str, Dict[int, torch.Tensor]]:
    out = {}
    for eid, ids in seqs.items():
        t = torch.tensor([ids], dtype=torch.long, device=device)
        out[eid] = {l: h.cpu() for l, h in block_outputs(model, t, layers).items()}
    return out


def _summary(d1, d2) -> dict:
    per = {eid: example_diagnostics(d1[eid], d2[eid]) for eid in d2}
    s = summarise(list(per.values()))
    s["per_example_uds"] = {eid: (None if v is None else round(v["uds"], 5))
                            for eid, v in per.items()}
    return s


def _nan(x) -> float:
    return float("nan") if x is None else x


def decide(trained: Dict[str, dict]) -> dict:
    passing = [(trained[s]["whole"]["strict"], s) for s in CANDIDATES
               if trained.get(s) and trained[s]["whole"]["uds"] is not None
               and trained[s]["whole"]["uds"] >= GO["uds"]
               and trained[s]["whole"]["strict"] >= GO["strict"]
               and trained[s]["whole"]["overshoot_share"] <= GO["overshoot_share"]]
    if not passing:
        return {"decision": "STOP: no candidate source clears the bar", "best": None}
    best = max(passing)[1]
    return {"decision": f"DESIGN H8 around '{best}'", "best": best,
            "passing": [s for _, s in sorted(passing, reverse=True)]}


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="1B")
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--train-fraction", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--cf-names", default="data/counterfactual_names.json")
    ap.add_argument("--breadth-items", default="data/breadth_items_v2.json")
    ap.add_argument("--reference-spans",
                    default="reference_uds/tofu_data/forget10_filtered.json")
    ap.add_argument("--tofu-dir", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("DEEPERASE_CACHE_DIR", "./hf_cache"))
    ap.add_argument("--model-sources", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="results/gates")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    t0 = time.time()
    configure_precision("float32")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sources = json.loads(args.model_sources) if args.model_sources else {}
    repo = {s: sources.get(s) or TOFU_MODELS[args.size][s].repo_id for s in ("full", "retain90")}
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
    items = json.loads(Path(args.breadth_items).read_text(encoding="utf-8"))["items"]
    r_questions = {i["question"] for i in items if i.get("tier") == "R"}
    retain_rows = load_tofu("retain90", cache_dir=args.cache_dir, data_dir=args.tofu_dir)
    retain_rows, _ = drop_authors_asked_about(retain_rows, r_questions)
    pool = retain_question_pool(tokenizer, retain_rows, args.max_length)
    rng = random.Random(args.seed)
    edits = {split: {e.example_id: edits_for(tokenizer, e, forget_rows[example_row(e)], names,
                                             other_pool=pool, rng=rng,
                                             max_length=args.max_length)
                     for e in exs} for split, exs in sets.items()}

    m_ret = load_model(repo["retain90"], device=device, dtype=torch.float32,
                       cache_dir=args.cache_dir)
    layers = list(range(n_layers(m_ret)))
    caps_ret = {k: capture_source_hidden(m_ret, v, layers, to_cpu=True) for k, v in sets.items()}
    del m_ret
    if device.type == "cuda":
        torch.cuda.empty_cache()
    m_full = load_model(repo["full"], device=device, dtype=torch.float32,
                        cache_dir=args.cache_dir)
    m_full.eval()

    report = {"what": "gate 1b: what M_full's memory of a TOFU fact is keyed on",
              "seed": args.seed, "thresholds": GO, "filler": FILLER,
              "cf_names_sha256": sha256_of(args.cf_names), "splits": {},
              "git": git_info(), "environment": environment_info()}
    for split, examples in sets.items():
        firsts = [first_token_example(e) for e in examples]
        spans = {"whole": examples, "first_token": firsts}
        base = {}
        for kind, exs in spans.items():
            base[kind] = score_from_captured(m_full, caps_ret[split], exs, layers)
        res = {}
        for src in SOURCES:
            have = [e for e in examples if edits[split][e.example_id][src]]
            n_var = max(len(edits[split][e.example_id][src]) for e in have) if have else 0
            d2 = {"whole": {}, "first_token": {}}
            for j in range(n_var):
                seqs = {e.example_id: edits[split][e.example_id][src][j] for e in have
                        if len(edits[split][e.example_id][src]) > j}
                caps = _capture(m_full, seqs, layers, device)
                for kind, exs in spans.items():
                    sub = [e for e in exs if e.example_id in seqs]
                    d1, s_full = base[kind]
                    dd, _ = score_from_captured(m_full, caps, sub, layers, s_full_cache=s_full)
                    for eid, per_layer in dd.items():
                        d2[kind].setdefault(eid, []).append(per_layer)
                del caps
            res[src] = {"n_examples": len(have)}
            for kind in spans:
                avg = {eid: {l: sum(v[l] for v in vs) / len(vs) for l in layers}
                       for eid, vs in d2[kind].items()}
                res[src][kind] = _summary(base[kind][0], avg)
            log.info("%s %-15s UDS %.3f strict %.3f | first token %.3f", split, src,
                     _nan(res[src]["whole"]["uds"]),
                     _nan(res[src]["whole"]["strict"]),
                     _nan(res[src]["first_token"]["uds"]))
        report["splits"][split] = res

    report.update(decide(report["splits"]["trained"]))
    report["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out_dir) / f"memory_keys_s{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    lines = [f"Gate 1b: what the memory is keyed on (seed {args.seed})", "",
             "| split | source | n | UDS | strict | overshoot share | first-token UDS | first-token strict |",
             "|---|---|---|---|---|---|---|---|"]
    for split, res in report["splits"].items():
        for src in SOURCES:
            w, ft = res[src]["whole"], res[src]["first_token"]
            lines.append(f"| {split} | {src} | {res[src]['n_examples']} | {f(w['uds'])} | "
                         f"{f(w['strict'])} | {f(w['overshoot_share'])} | {f(ft['uds'])} | "
                         f"{f(ft['strict'])} |")
    lines += ["", f"DECISION: {report['decision']}  (a candidate needs, on trained facts, "
              f"UDS >= {GO['uds']}, strict >= {GO['strict']}, overshoot <= {GO['overshoot_share']})"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out} ({report['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
