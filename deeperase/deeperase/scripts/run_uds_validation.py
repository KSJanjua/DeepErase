"""Reproduce the UDS paper's Table 2 on real TOFU models.

This is the project's correctness test for the depth measurement. It needs no
unlearning training at all -- only four published checkpoints.

The experiment
--------------
The target is always ``M_full``. Stage 1 always uses ``retain90`` as its
source. Stage 2 uses each of four models in turn, each having seen a different
amount of the forget set::

    Stage-2 source   saw of forget10   expected UDS (1B)
    full             100%              0.002
    retain99          90%              0.153
    retain95          50%              0.496
    retain90           0%              1.000

UDS must rise monotonically down that column. That ordering is the real test;
exact agreement is a bonus, since we cannot match the authors' tokenisation
and seeds precisely.

Resumability
------------
Every stage is written to disk as it completes, so an interrupted session
resumes instead of restarting. Re-running the same ``--run-id`` picks up from
the last completed step.

Usage
-----
    python -m deeperase.scripts.run_uds_validation --size 1B
    python -m deeperase.scripts.run_uds_validation --size 1B --n-examples 100
    python -m deeperase.scripts.run_uds_validation --run-id my_run --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from deeperase.config import (
    TOFU_MODELS,
    UDS_PAPER_TABLE2,
    ExecutionStrategy,
    RunConfig,
    check_against_paper,
    plan_memory,
)
from deeperase.data.tofu import (
    PromptFormat,
    SamplingStrategy,
    SpanStrategy,
    select_examples,
    filter_usable,
    load_tofu,
    sanity_check_prompt_format,
    tokenise_example,
)
from deeperase.eval.patching import n_layers
from deeperase.eval.uds import (
    UDSExample,
    assemble_report,
    capture_source_hidden,
    score_from_captured,
)
from deeperase.models import ModelManager, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("uds_validation")
logging.getLogger("deeperase").setLevel(logging.INFO)

#: Stage-2 sources, ordered by how much of the forget set they saw.
#: UDS must increase along this list.
STAGE2_ORDER = ["full", "retain99", "retain95", "retain90"]


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


class RunDir:
    """On-disk layout for one run. Every step is checkpointed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.partial = self.root / "partial"
        self.root.mkdir(parents=True, exist_ok=True)
        self.partial.mkdir(exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def stage1_path(self) -> Path:
        return self.root / "stage1_cache.json"

    @property
    def report_path(self) -> Path:
        return self.root / "report.json"

    def stage2_path(self, split: str) -> Path:
        return self.partial / f"stage2_{split}.json"

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        """Write atomically. A session killed mid-write must not leave a
        truncated file that a later resume would silently trust."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _read(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("%s is corrupt; ignoring and recomputing", path.name)
            return None

    def save_config(self, cfg: RunConfig, extra: dict) -> None:
        self._write(self.config_path, {"config": cfg.to_dict(), **extra})

    def save_stage1(self, deltas, s_full) -> None:
        self._write(self.stage1_path, {"delta_s1": deltas, "s_full": s_full})

    def load_stage1(self):
        d = self._read(self.stage1_path)
        if d is None:
            return None, None
        # JSON turns integer keys into strings; restore them.
        deltas = {eid: {int(k): v for k, v in layers.items()}
                  for eid, layers in d["delta_s1"].items()}
        return deltas, d["s_full"]

    def save_stage2(self, split: str, deltas, uds: Optional[float], summary: str) -> None:
        self._write(self.stage2_path(split),
                    {"split": split, "uds": uds, "summary": summary, "delta_s2": deltas})

    def load_stage2(self, split: str):
        d = self._read(self.stage2_path(split))
        if d is None:
            return None
        d["delta_s2"] = {eid: {int(k): v for k, v in layers.items()}
                         for eid, layers in d["delta_s2"].items()}
        return d

    def save_report(self, payload: dict) -> None:
        self._write(self.report_path, payload)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def choose_prompt_format(
    model, tokenizer, examples, *, requested: Optional[str], max_length: int
) -> PromptFormat:
    """Pick the prompt format, or verify the one requested.

    Getting this wrong is a silent failure: the model scores every answer
    poorly, all the differences shrink toward zero, and UDS becomes noise that
    still looks like a valid number. So we test candidates and take the one the
    model recognises best -- and refuse to continue if none of them work.
    """
    candidates = [PromptFormat(requested)] if requested else list(PromptFormat)
    results = []
    for fmt in candidates:
        r = sanity_check_prompt_format(model, tokenizer, examples,
                                       fmt=fmt, max_length=max_length)
        results.append((fmt, r))
        log.info("  format %-9s mean entity log-prob %8.4f  %s",
                 fmt.value, r["mean_logprob"], "OK" if r["passed"] else "poor")

    best_fmt, best = max(results, key=lambda x: x[1]["mean_logprob"])
    if not best["passed"]:
        raise SystemExit(
            f"\nNo prompt format is recognised by this model "
            f"(best: {best_fmt.value} at {best['mean_logprob']:.3f}).\n"
            f"{best['advice']}\n"
            "Refusing to continue -- any UDS computed now would be noise."
        )
    log.info("  selected format: %s", best_fmt.value)
    return best_fmt


def build_examples(
    tokenizer, cfg: RunConfig, fmt: PromptFormat, span_strategy: SpanStrategy,
    max_entity_words: int,
    sampling: SamplingStrategy = SamplingStrategy.EVEN,
    reference_spans: str = None,
) -> List[UDSExample]:
    """Load TOFU, extract entity spans, tokenise."""
    if reference_spans:
        from deeperase.data.reference_spans import load_reference_annotations
        kept, ref_stats = load_reference_annotations(reference_spans)
        log.info("  using the authors' hand annotations: %d examples (%s)",
                 len(kept), ref_stats)
        raw = kept          # no separate filtering step; they pre-filtered
        dropped = {}
    else:
        raw = load_tofu(cfg.forget_split, cache_dir=cfg.cache_dir,
                        strategy=span_strategy)
        kept, dropped = filter_usable(raw, max_entity_words=max_entity_words)
        log.info("  %d/%d examples have a usable entity span (dropped: %s)",
                 len(kept), len(raw), {k: v for k, v in dropped.items() if v})

    if len(kept) < 0.3 * len(raw):
        log.warning(
            "Over 70%% of examples were dropped. The span strategy may be a poor "
            "fit for this data, and the remainder may be biased toward short answers."
        )

    # Sample across the WHOLE filtered set. TOFU's forget splits are ordered
    # and nested at the end (forget05 = indices 200-399, forget01 = 360-399),
    # so taking the first N picks a region every retain model has seen, and UDS
    # collapses to ~0 regardless of correctness. See SamplingStrategy.
    chosen = select_examples(kept, cfg.n_examples, strategy=sampling, seed=cfg.seed)
    n_late = sum(1 for e in chosen if e.index >= 200)
    log.info("  sampling=%s: %d examples, %d (%.0f%%) from the second half of the "
             "split (unseen by retain95)",
             sampling.value, len(chosen), n_late, 100 * n_late / max(len(chosen), 1))

    examples: List[UDSExample] = []
    for ex in chosen:
        tok = tokenise_example(ex, tokenizer, fmt=fmt, max_length=cfg.max_seq_length)
        if tok is None:
            continue
        ids = torch.tensor([tok.input_ids], dtype=torch.long)
        examples.append(
            UDSExample(
                example_id=f"{cfg.forget_split}_{ex.index}",
                input_ids=ids,
                span=tok.to_entity_span(),
                attention_mask=torch.ones_like(ids),
            )
        )
    log.info("  %d examples tokenised and ready", len(examples))
    if not examples:
        raise SystemExit("No usable examples. Check the span strategy and max_seq_length.")
    return examples


def run_stage(
    mgr: ModelManager, source_split: str, examples, layers,
    *, s_full_cache: Optional[Dict[str, float]] = None,
):
    """One stage: capture from the source, free it, patch into the target.

    Two phases with a free in between, so peak GPU memory is one model. Under
    ALL_RESIDENT the manager simply keeps both, and this costs nothing extra.
    """
    t0 = time.time()
    with mgr.acquire(source_split) as m_src:
        captured = capture_source_hidden(m_src, examples, layers, to_cpu=True)
    log.info("    captured from %s in %.1fs -- %s",
             source_split, time.time() - t0, mgr.memory().summary())

    t1 = time.time()
    with mgr.acquire("full") as m_full:
        deltas, s_full = score_from_captured(
            m_full, captured, examples, layers, s_full_cache=s_full_cache
        )
    log.info("    patched into full in %.1fs", time.time() - t1)
    return deltas, s_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce UDS paper Table 2 on TOFU")
    ap.add_argument("--size", choices=sorted(TOFU_MODELS), default="1B")
    ap.add_argument("--n-examples", type=int, default=50)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"],
                    default="bfloat16")
    ap.add_argument("--prompt-format", choices=[f.value for f in PromptFormat],
                    default=None, help="Default: try all and pick the best")
    ap.add_argument("--span-strategy", choices=[s.value for s in SpanStrategy],
                    default=SpanStrategy.NOVEL_CONTENT.value)
    ap.add_argument("--reference-spans", default=None, metavar="PATH",
                    help="Use the UDS authors' hand annotations instead of our "
                         "heuristic, e.g. reference_uds/tofu_data/forget10_filtered.json. "
                         "Removes span extraction and example filtering as variables.")
    ap.add_argument("--max-entity-words", type=int, default=None,
                    help="Cap on entity length. Default: 6 for novel_content and "
                         "trailing; disabled for full_answer, where the entity IS "
                         "the whole answer and a cap would contradict the strategy.")
    ap.add_argument("--sampling", choices=[s.value for s in SamplingStrategy],
                    default=SamplingStrategy.EVEN.value,
                    help="How to pick examples. 'even' spreads across the split; "
                         "'first' reproduces the biased behaviour and should not "
                         "be used for measurement.")
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--output-dir", default="results/gpu_runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed steps from a previous run of the same id")
    ap.add_argument("--layers", default=None,
                    help="Comma-separated layer indices. Default: all")
    args = ap.parse_args()

    # Resolve the entity-length cap against the span strategy. With
    # FULL_ANSWER the entity is the entire answer by definition (median 27
    # words on TOFU), so the default cap of 6 would discard 396 of 400
    # examples and leave nothing to measure.
    span_strategy = SpanStrategy(args.span_strategy)
    if args.max_entity_words is not None:
        max_entity_words = args.max_entity_words
    elif span_strategy is SpanStrategy.FULL_ANSWER:
        max_entity_words = None
        log.info("span-strategy=full_answer: disabling the entity-length cap, "
                 "since the entity is the whole answer by definition.")
    else:
        max_entity_words = 6

    run_id = args.run_id or f"table2_{args.size}_{time.strftime('%Y%m%d_%H%M%S')}"
    rd = RunDir(Path(args.output_dir) / run_id)

    print("=" * 74)
    print(f"UDS Table 2 validation  |  size={args.size}  run={run_id}")
    print("=" * 74)

    # -- config and memory ---------------------------------------------------
    cfg = RunConfig(
        size_label=args.size, dtype=args.dtype, n_examples=args.n_examples,
        max_seq_length=args.max_seq_length, tau=args.tau,
        cache_dir=args.cache_dir, output_dir=args.output_dir,
    )
    problems = cfg.validate()
    if problems:
        raise SystemExit(f"Invalid configuration: {problems}")

    if not torch.cuda.is_available():
        log.warning("No GPU detected. This will be extremely slow on CPU.")
        total_gb = 8.0
    else:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    plan = plan_memory(args.size, total_gb, dtype_size=cfg.dtype_size)
    print(f"\n{plan.summary()}\n")
    if not plan.fits:
        raise SystemExit("Model does not fit. Use a smaller --size.")
    cfg.strategy = plan.strategy

    rd.save_config(cfg, {"run_id": run_id, "gpu_total_gb": round(total_gb, 2),
                         "memory_plan": plan.reason})

    # -- data and format -----------------------------------------------------
    log.info("[1/4] Loading TOFU and selecting prompt format")
    tokenizer = load_tokenizer(cfg.models()["full"].repo_id, cache_dir=cfg.cache_dir)

    raw = load_tofu(cfg.forget_split, cache_dir=cfg.cache_dir,
                    strategy=span_strategy, limit=40)
    probe_examples, probe_dropped = filter_usable(raw, max_entity_words=max_entity_words)
    if not probe_examples:
        reasons = {k: v for k, v in probe_dropped.items() if v}
        raise SystemExit(
            "\n".join([
                "",
                f"Every one of the first {len(raw)} examples was filtered out "
                "before the prompt-format check could run.",
                f"  span strategy    : {span_strategy.value}",
                f"  max entity words : {max_entity_words}",
                f"  drop reasons     : {reasons}",
                "",
                "The span strategy and the entity-length cap are contradicting "
                "each other. Raise --max-entity-words, or leave it unset so it "
                "is chosen to match the strategy.",
            ])
        )

    with ModelManager(cfg) as mgr:
        with mgr.acquire("full") as m_full:
            fmt = choose_prompt_format(
                m_full, tokenizer, probe_examples,
                requested=args.prompt_format, max_length=cfg.max_seq_length,
            )
            layers = ([int(x) for x in args.layers.split(",")]
                      if args.layers else list(range(n_layers(m_full))))
        log.info("  evaluating %d layers", len(layers))

        examples = build_examples(
            tokenizer, cfg, fmt, span_strategy,
            max_entity_words, sampling=SamplingStrategy(args.sampling),
            reference_spans=args.reference_spans,
        )

        # -- Stage 1 ---------------------------------------------------------
        log.info("[2/4] Stage 1: %s -> full (baseline, computed once)",
                 cfg.stage1_source_split)
        d1, s_full = (rd.load_stage1() if args.resume else (None, None))
        if d1 is not None and set(d1) >= {e.example_id for e in examples}:
            log.info("  reusing cached Stage 1 from a previous run")
        else:
            d1, s_full = run_stage(mgr, cfg.stage1_source_split, examples, layers)
            rd.save_stage1(d1, s_full)

        max_ds1 = max((max(v.values()) for v in d1.values()), default=float("-inf"))
        n_with_ke = sum(1 for v in d1.values() if max(v.values()) > cfg.tau)
        log.info("  max dS1 = %.4f; %d/%d examples have at least one KE layer",
                 max_ds1, n_with_ke, len(examples))
        if n_with_ke == 0:
            raise SystemExit(
                f"\nNo example has a Knowledge-Encoding layer at tau={cfg.tau}.\n"
                "M_full does not measurably know anything M_ret lacks, so UDS is "
                "undefined for every example.\n"
                "Likely causes: wrong prompt format, wrong forget/retain pairing, "
                "or entity spans that miss the fact.\n"
                "Refusing to continue -- the output would be meaningless."
            )

        # -- Stage 2 ---------------------------------------------------------
        log.info("[3/4] Stage 2: four source models")
        observed: Dict[str, float] = {}
        reports: Dict[str, dict] = {}

        for split in STAGE2_ORDER:
            cached = rd.load_stage2(split) if args.resume else None
            if cached is not None:
                log.info("  %-9s reusing cached result (UDS=%s)", split, cached["uds"])
                d2 = cached["delta_s2"]
            else:
                log.info("  %-9s computing...", split)
                d2, _ = run_stage(mgr, split, examples, layers, s_full_cache=s_full)

            rep = assemble_report(delta_s1=d1, delta_s2=d2, s_full=s_full,
                                  layers=layers, tau=cfg.tau)
            if cached is None:
                rd.save_stage2(split, d2, rep.uds, rep.summary())
            log.info("    %s", rep.summary())
            if rep.uds is not None:
                observed[split] = rep.uds
            reports[split] = rep.to_dict()

    # -- Compare -------------------------------------------------------------
    log.info("[4/4] Comparing against the paper")
    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    expected = UDS_PAPER_TABLE2.get(args.size, {})
    print(f"{'Stage-2 source':<14}{'expected':>10}{'observed':>10}{'diff':>10}")
    print("-" * 74)
    for split in STAGE2_ORDER:
        exp = expected.get(split)
        obs = observed.get(split)
        diff = "" if (exp is None or obs is None) else f"{abs(obs - exp):.3f}"
        print(f"{split:<14}{exp if exp is not None else '-':>10}"
              f"{f'{obs:.3f}' if obs is not None else 'undefined':>10}{diff:>10}")

    verdict = None
    if len(observed) >= 2 and args.size in UDS_PAPER_TABLE2:
        comparison = check_against_paper(args.size, observed)
        verdict = comparison["verdict"]
        print(f"\nmonotonic: {comparison['monotonic']}")
        print(f"within tolerance: {comparison['n_within_tolerance']}/{comparison['n_compared']}")
        print(f"\n{verdict}")
    else:
        comparison = None
        print("\nToo few defined results to compare.")

    rd.save_report({
        "run_id": run_id,
        "config": cfg.to_dict(),
        "prompt_format": fmt.value,
        "span_strategy": ("reference_annotation" if args.reference_spans
                          else span_strategy.value),
        "max_entity_words": max_entity_words,
        "sampling": args.sampling,
        "n_examples": len(examples),
        "layers": layers,
        "observed": observed,
        "expected": expected,
        "comparison": comparison,
        "per_split_reports": reports,
        "is_validated_against_reference": False,
    })

    print(f"\nSaved to {rd.root}")
    print("\nNOTE: reproducing Table 2 is strong evidence the implementation is")
    print("correct, but it is not the same as running the authors' own code on")
    print("shared inputs. is_validated_against_reference remains False.")
    print("=" * 74)

    return 0 if (comparison and comparison["monotonic"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
