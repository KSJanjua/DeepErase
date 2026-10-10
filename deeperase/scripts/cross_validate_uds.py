"""T0: cross-validate our UDS against the authors' own implementation.

``docs/UDS_CONFORMANCE.md`` §5 item 1 -- the gate on
``is_validated_against_reference``. Both implementations are run on
**byte-identical token sequences**, built by the reference's own functions
(imported from ``reference_uds/compute_uds.py``, not re-implemented), and
compared at every level:

1. **Hidden states.** The reference takes source states from
   ``output_hidden_states=True`` and patches ``hidden_states[layer + 1]``; we
   hook each decoder layer's output. For every layer but the last these are the
   same tensor. For the last, HuggingFace's tuple holds the state *after* the
   final norm -- so the reference patches a normalised vector into a slot that
   expects the raw residual stream. This step checks that directly.
2. **Per-layer dS1 and dS2**, example by example (target: ~1e-4).
3. **UDS** for each Stage-2 source, four ways:

   * ``reference``        -- the authors' code as published
   * ``ours_on_ref``      -- our code on the reference's inputs
   * ``reference_prenorm``-- the authors' code with the last layer fed the raw
     (pre-norm) state, isolating the last-layer effect
   * ``ours_pipeline``    -- our code on our own construction (whole answer in
     one tokenisation, entity located by character offset), as every study
     run uses -- isolating the sequence-construction effect

Models are loaded one at a time (captures go to CPU), so a 1B run fits on a
16 GB laptop: ``--device mps`` on Apple silicon, ``cuda`` on the A100.

    python -m deeperase.scripts.cross_validate_uds --device mps \\
        --cache-dir ../_workspace/hf_cache --n-examples 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from deeperase.config import TOFU_MODELS, UDS_PAPER_TABLE2
from deeperase.data.reference_spans import load_reference_annotations
from deeperase.data.tofu import PromptFormat, tokenise_example
from deeperase.eval.patching import (
    EntitySpan, capture_hidden_states, entity_logprobs, forward_with_patch,
    probe_knowledge_with_patch,
)
from deeperase.eval.uds import aggregate_example_uds
from deeperase.models import load_model, load_tokenizer
from deeperase.provenance import environment_info, git_info, sha256_file

log = logging.getLogger("t0")

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "reference_uds"
ANNOTATIONS = REFERENCE_DIR / "tofu_data" / "forget10_filtered.json"


def _import_reference():
    sys.path.insert(0, str(REFERENCE_DIR.resolve()))
    import compute_uds as ref                      # the authors' module
    return ref


def build_reference_inputs(ref, tokenizer, records: Sequence[dict]) -> List[dict]:
    """Inputs exactly as ``compute_uds.py`` builds them (continuation scope,
    gt reference, gt entity, span patching)."""
    out = []
    for item in records:
        prompt = f"Question: {item['question']}\nAnswer: {item['prefix']}"
        answer_text = item["answer"]
        if answer_text.startswith(item["prefix"]):
            answer_text = answer_text[len(item["prefix"]):]
        reference_text = ref.normalize_reference_for_eval(prompt, answer_text)
        eval_span = ref.get_eval_span(tokenizer, reference_text,
                                      item.get("gt_entity", item["entity"]), "entity")
        if eval_span is None:
            continue
        ctx = ref.build_logprob_ctx(tokenizer, prompt, reference_text, eval_span, "span")
        if ctx is None:
            continue
        ids = torch.tensor([ctx["prompt_ids"] + ctx["ref_ids"]], dtype=torch.long)
        # Logit positions eval_start..eval_end-1 predict tokens eval_start+1..eval_end,
        # which is exactly our EntitySpan convention (patch/read at j-1 for token j).
        span = EntitySpan(list(range(ctx["eval_start"] + 1, ctx["eval_end"] + 1)))
        assert span.predicting_positions == list(range(ctx["patch_start"], ctx["patch_end"]))
        out.append({"idx": int(item["idx"]), "ctx": ctx, "ids": ids, "span": span})
    return out


@torch.no_grad()
def capture(model, ref, examples, layers, device) -> Dict[str, list]:
    """Both implementations' source states, per example, on CPU.

    The two are compared here, exactly, and the reference's copy is kept only
    for layers where it is not bit-identical to ours -- so the "reference"
    variant below still runs on exactly the reference's tensors, at half the
    memory.
    """
    ours, theirs, diffs = [], [], []
    for ex in examples:
        ids = ex["ids"].to(device)
        mask = torch.ones_like(ids)
        h_ours = capture_hidden_states(model, ids, layers, attention_mask=mask)
        h_ref = ref._compute_hidden_states_batch(model, ids, mask)
        o = {l: h_ours[l].float().cpu() for l in layers}
        t = {l: h_ref[l + 1].float().cpu() for l in layers}
        diffs.append({l: float((o[l] - t[l]).abs().max()) for l in layers})
        ours.append(o)
        theirs.append({l: t[l] for l in layers if not torch.equal(o[l], t[l])})
        del h_ours, h_ref
    return {"ours": ours, "theirs_only": theirs, "diffs": diffs}


def reference_states(c: Dict[str, list]) -> List[dict]:
    """The reference's tensors: its own copy where it differs, else ours (identical)."""
    return [{**o, **t} for o, t in zip(c["ours"], c["theirs_only"])]


@torch.no_grad()
def patched_scores(full, ref, examples, layers, states, device, *, impl: str) -> List[Dict[int, float]]:
    """Per example, per layer, the patched entity log-prob under one implementation."""
    out = []
    for ex, src in zip(examples, states):
        ids = ex["ids"].to(device)
        mask = torch.ones_like(ids)
        row = {}
        if impl == "ours":
            hidden = {l: src[l].to(device) for l in layers}
            for l in layers:
                row[l] = float(probe_knowledge_with_patch(full, hidden, ids, ex["span"], l,
                                                          attention_mask=mask))
        else:
            _, _, meta = ref._prepare_batch_inputs([ex["ctx"]], _TOKENIZER[0], device)
            for l in layers:
                row[l] = ref.compute_logprob_teacher_forcing_layer_batch_with_inputs(
                    full, ids, mask, meta, src[l].to(device), l, patch_scope="span")[0]
        out.append(row)
    return out


_TOKENIZER: list = [None]


def uds_from(deltas1: List[Dict[int, float]], deltas2: List[Dict[int, float]],
             tau: float) -> Dict[str, Optional[float]]:
    scores = []
    for d1, d2 in zip(deltas1, deltas2):
        u, _, _ = aggregate_example_uds(d1, d2, tau=tau)
        if u is not None:
            scores.append(u)
    return {"uds": (sum(scores) / len(scores)) if scores else None,
            "n_scored": len(scores), "n_total": len(deltas1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", default="1B", choices=sorted(TOFU_MODELS))
    ap.add_argument("--sources", default="retain99,retain95,full",
                    help="Stage-2 sources (retain90 is always the Stage-1 source)")
    ap.add_argument("--n-examples", type=int, default=60,
                    help="spread evenly over the 367 annotated examples")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--out", default="results/analysis/uds_cross_validation.json")
    args = ap.parse_args(argv)

    t_start = time.time()
    ref = _import_reference()
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    specs = TOFU_MODELS[args.size]
    tok = load_tokenizer(specs["full"].repo_id, cache_dir=args.cache_dir)
    _TOKENIZER[0] = tok

    records = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    n = min(args.n_examples, len(records))
    step = (len(records) - 1) / (n - 1) if n > 1 else 1
    picked = [records[int(round(i * step))] for i in range(n)]
    examples = build_reference_inputs(ref, tok, picked)
    log.info("%d of %d examples built with the reference's own functions", len(examples), n)

    # Our pipeline's construction of the same rows, for the sequence-construction test.
    ours_by_idx = {e.index: e for e in load_reference_annotations(ANNOTATIONS)[0]}
    pipeline = []
    for ex in examples:
        te = ours_by_idx.get(ex["idx"])
        tk = tokenise_example(te, tok, fmt=PromptFormat.PLAIN_QA) if te else None
        pipeline.append(None if tk is None else
                        {"ids": torch.tensor([tk.input_ids]), "span": tk.to_entity_span()})

    sources = ["retain90"] + [s for s in args.sources.split(",") if s != "retain90"]
    layers: List[int] = []
    hidden_check: Dict[str, Dict[int, float]] = {}
    variants: Dict[str, Dict[str, list]] = {}
    pipe_deltas: Dict[str, list] = {}
    s_full_ours: List[float] = []
    s_full_ref: List[float] = []
    s_full_pipe: List[Optional[float]] = []

    def release() -> None:
        """Return freed model memory to the device.

        The caller must ``del`` its own reference first. An earlier version
        did ``del model`` inside a helper, which removed only the helper's
        local name: every model stayed alive in ``main``, three to four 5 GB
        float32 models accumulated, and macOS killed the process on its third
        source with no traceback.
        """
        import gc
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # Per-source checkpoint: a crash or a sleeping laptop loses at most one source.
    ckpt_path = Path(args.out).with_suffix(".partial.json")
    ckpt_key = {"examples": [e["idx"] for e in examples], "dtype": args.dtype,
                "device": str(device), "size": args.size}
    if ckpt_path.exists():
        saved = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if saved.get("key") == ckpt_key:
            ik = lambda d: {int(k): v for k, v in d.items()}
            layers = saved["layers"]
            s_full_ours, s_full_ref = saved["s_full_ours"], saved["s_full_ref"]
            s_full_pipe = saved["s_full_pipe"]
            for sp, rec in saved["sources"].items():
                hidden_check[sp] = ik(rec["hidden_check"])
                variants[sp] = {v: [ik(d) for d in rows] for v, rows in rec["variants"].items()}
                pipe_deltas[sp] = [None if d is None else ik(d) for d in rec["pipe_deltas"]]
            log.info("resuming from %s: %s already done", ckpt_path, sorted(saved["sources"]))
        else:
            log.info("ignoring %s: made with different examples or settings", ckpt_path)

    def save_checkpoint() -> None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps({
            "key": ckpt_key, "layers": layers, "s_full_ours": s_full_ours,
            "s_full_ref": s_full_ref, "s_full_pipe": s_full_pipe,
            "sources": {sp: {"hidden_check": hidden_check[sp], "variants": variants[sp],
                             "pipe_deltas": pipe_deltas[sp]} for sp in variants},
        }), encoding="utf-8")

    # One source at a time: capture its states, free it, load full, patch, keep
    # only the deltas. Peak memory is one model plus one source's states.
    for split in sources:
        if split in variants:
            continue
        model = load_model(specs[split].repo_id, device=device, dtype=dtype,
                           cache_dir=args.cache_dir)
        if not layers:
            layers = list(range(len(model.model.layers)))
        last = layers[-1]
        c = capture(model, ref, examples, layers, device)
        pcap = [None if p is None else
                {l: h.float().cpu() for l, h in capture_hidden_states(
                    model, p["ids"].to(device), layers).items()}
                for p in pipeline]
        del model
        release()
        hidden_check[split] = {l: max(d[l] for d in c["diffs"]) for l in layers}

        full = load_model(specs["full"].repo_id, device=device, dtype=dtype,
                          cache_dir=args.cache_dir)
        if not s_full_ours:
            with torch.no_grad():
                for ex in examples:
                    ids = ex["ids"].to(device)
                    mask = torch.ones_like(ids)
                    s_full_ours.append(float(entity_logprobs(
                        forward_with_patch(full, ids, None, attention_mask=mask), ids, ex["span"])))
                    _, _, meta = ref._prepare_batch_inputs([ex["ctx"]], tok, device)
                    s_full_ref.append(ref.compute_logprob_teacher_forcing_baseline_batch_with_inputs(
                        full, ids, mask, meta)[0])
                for p in pipeline:
                    s_full_pipe.append(None if p is None else float(entity_logprobs(
                        forward_with_patch(full, p["ids"].to(device), None),
                        p["ids"].to(device), p["span"])))

        ref_states = reference_states(c)
        theirs = patched_scores(full, ref, examples, layers, ref_states, device, impl="ref")
        ours = patched_scores(full, ref, examples, layers, c["ours"], device, impl="ours")
        # The authors' patching with the last layer fed the raw (pre-norm) state.
        # Every other layer is the reference variant by construction.
        prenorm_states = [{last: o[last]} for o in c["ours"]]
        theirs_pre_last = patched_scores(full, ref, examples, [last], prenorm_states,
                                         device, impl="ref")
        ref_d = [{l: s_full_ref[i] - r[l] for l in layers} for i, r in enumerate(theirs)]
        variants[split] = {
            "reference": ref_d,
            "ours_on_ref": [{l: s_full_ours[i] - r[l] for l in layers} for i, r in enumerate(ours)],
            "reference_prenorm": [{**d, last: s_full_ref[i] - theirs_pre_last[i][last]}
                                  for i, d in enumerate(ref_d)],
        }
        rows, hidden = [], None
        with torch.no_grad():
            for p, sf, cap in zip(pipeline, s_full_pipe, pcap):
                if p is None:
                    rows.append(None)
                    continue
                ids = p["ids"].to(device)
                hidden = {l: cap[l].to(device) for l in layers}
                rows.append({l: sf - float(probe_knowledge_with_patch(full, hidden, ids, p["span"], l))
                             for l in layers})
        pipe_deltas[split] = rows
        del full, c, pcap, ref_states, hidden, theirs, ours, theirs_pre_last
        release()
        save_checkpoint()
        log.info("done %s (%.0fs elapsed)", split, time.time() - t_start)

    last = layers[-1]
    log.info("hidden-state max |ours - theirs|: layers 0-%d <= %.2e, last layer = %.2e",
             last - 1, max(max(v[l] for l in layers[:-1]) for v in hidden_check.values()),
             max(v[last] for v in hidden_check.values()))

    # -- 2. per-layer agreement --------------------------------------------
    agreement = {}
    for split in sources:
        a, b = variants[split]["ours_on_ref"], variants[split]["reference"]
        agreement[split] = {
            str(l): {"max_abs": max(abs(x[l] - y[l]) for x, y in zip(a, b)),
                     "mean_abs": sum(abs(x[l] - y[l]) for x, y in zip(a, b)) / len(a)}
            for l in layers}

    # -- 3. UDS, four ways ----------------------------------------------------
    paper = UDS_PAPER_TABLE2.get(args.size, {})
    uds = {}
    for split in sources:
        if split == "retain90":
            continue
        row = {}
        for v in ("reference", "ours_on_ref", "reference_prenorm"):
            row[v] = uds_from(variants["retain90"][v], variants[split][v], args.tau)
        keep = [i for i, p in enumerate(pipe_deltas[split]) if p is not None]
        row["ours_pipeline"] = uds_from([pipe_deltas["retain90"][i] for i in keep],
                                        [pipe_deltas[split][i] for i in keep], args.tau)
        row["paper"] = paper.get(split)
        uds[split] = row

    report = {
        "what": "T0 cross-validation of UDS against reference_uds/compute_uds.py",
        "n_examples": len(examples), "example_idx": [e["idx"] for e in examples],
        "tau": args.tau, "device": str(device), "dtype": args.dtype,
        "layers": layers, "hidden_state_max_abs_diff": hidden_check,
        "per_layer_delta_agreement": agreement, "uds": uds,
        "inputs": {"annotations": str(ANNOTATIONS),
                   "annotations_sha256": sha256_file(ANNOTATIONS),
                   "reference_code_sha256": sha256_file(REFERENCE_DIR / "compute_uds.py")},
        "git": git_info(), "environment": environment_info(),
        "seconds": round(time.time() - t_start, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHidden states, max |ours - reference| over examples:")
    for split, per in hidden_check.items():
        print(f"  {split:9s} layers 0-{last - 1}: {max(per[l] for l in layers[:-1]):.2e}"
              f"   layer {last}: {per[last]:.2e}")
    print("\nPer-layer delta agreement (ours on reference inputs vs reference), max |diff|:")
    for split in sources:
        worst = max(agreement[split][str(l)]["max_abs"] for l in layers[:-1])
        print(f"  {split:9s} layers 0-{last - 1}: {worst:.2e}"
              f"   layer {last}: {agreement[split][str(last)]['max_abs']:.2e}")
    print(f"\nUDS (tau={args.tau}, n={len(examples)}):")
    print(f"  {'source':9s} {'paper':>7s} {'reference':>10s} {'ours_on_ref':>12s}"
          f" {'ref_prenorm':>12s} {'ours_pipeline':>14s}")
    for split, row in uds.items():
        f = lambda v: f"{v['uds']:.4f}" if v and v["uds"] is not None else "  undef"
        p = f"{row['paper']:.3f}" if row["paper"] is not None else "  -"
        print(f"  {split:9s} {p:>7s} {f(row['reference']):>10s} {f(row['ours_on_ref']):>12s}"
              f" {f(row['reference_prenorm']):>12s} {f(row['ours_pipeline']):>14s}")
    print(f"\nWrote {out}  ({report['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
