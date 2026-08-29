#!/usr/bin/env python3
"""Drive the T2 replication matrix: methods x seeds, one study run each.

This is a thin driver around ``deeperase.scripts.run_study``. It deliberately
shells out to that CLI rather than reimplementing the pipeline, so the runs it
produces are the same artefacts, written to the same self-describing run
directories, as a hand-typed invocation.

What it adds over a shell loop:
  * a free-VRAM check before each run, because the card is shared and the other
    tenant's footprint changes between runs;
  * ``--resume`` handling so an interrupted matrix picks up where it stopped;
  * a manifest and an end-of-run summary table;
  * refusal to run seeds under ``--sampling even``, which would silently
    produce identical trajectories (see the note below).

Why --sampling random is required for replication
-------------------------------------------------
``unlearn()`` calls ``torch.manual_seed``, but this pipeline has no shuffling
and Llama-3.2 sets dropout to 0. Under the default even sampling, changing the
seed therefore yields a **bit-identical** run. Three "seeds" would be three
copies of one number, and reporting them as independent observations would be
wrong. The real source of variation is which examples are drawn, which is what
``--sampling random --seed N`` varies.

Usage
-----
    python gpu/replicate.py --dry-run                 # print the plan, run nothing
    python gpu/replicate.py --methods ga npo graddiff --seeds 0 1 2
    python gpu/replicate.py --methods npo --seeds 0 --epochs 20    # single arm

Expect roughly one hour per run at 1B. The full 3x3 matrix is most of a day.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GB = 1024 ** 3


def free_vram_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    free_b, _ = torch.cuda.mem_get_info()
    return free_b / GB


DEFAULT_LR = 1e-6


def run_id_for(method: str, size: str, seed: int, lr: float = DEFAULT_LR) -> str:
    """Run id, with the learning rate encoded when it is not the default.

    The lr belongs in the id because methods need different ones: GA reaches
    its utility floor at 1e-6, GradDiff barely moves there and diverges at
    1e-5. Without it in the id, a matrix at a new lr would collide with an
    existing directory and ``already_done`` would silently skip the run,
    producing a matrix whose rows were trained at different rates. The
    default is left un-suffixed so existing run directories still resume.
    """
    base = f"study_{method}_{size}_s{seed}"
    return base if lr == DEFAULT_LR else f"{base}_lr{lr:g}"


def already_done(out_dir: Path) -> bool:
    """A run is complete when its plane.json exists with every alpha scored."""
    plane = out_dir / "plane.json"
    if not plane.exists():
        return False
    try:
        return len(json.loads(plane.read_text())["points"]) > 0
    except Exception:                                        # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["ga", "npo", "graddiff"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--size", default="1B")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LR,
                    help="Passed through to run_study and recorded in the run "
                         "id. Methods need different rates -- run this script "
                         "once per method rather than once for all three.")
    ap.add_argument("--n-examples", type=int, default=100)
    ap.add_argument("--n-breadth", type=int, default=400)
    ap.add_argument("--sampling", default="random", choices=["even", "random"])
    ap.add_argument("--output-dir", default="results/studies")
    ap.add_argument("--cache-dir", default="./hf_cache")
    ap.add_argument("--min-free-gb", type=float, default=14.0,
                    help="Refuse to start a run with less free VRAM than this. "
                         "1B full-parameter training needs ~11 GB; NPO adds a "
                         "frozen reference model on top.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Everything after --extra is passed through to run_study.")
    args = ap.parse_args()

    if args.sampling == "even" and len(args.seeds) > 1:
        print("REFUSING: --sampling even makes every seed produce a bit-identical\n"
              "run (no shuffling, dropout 0). Those would not be independent\n"
              "observations. Use --sampling random, or pass a single seed.",
              file=sys.stderr)
        return 2

    jobs = [(m, s) for m in args.methods for s in args.seeds]
    out_root = Path(args.output_dir)

    print("=" * 78)
    print(f"Replication matrix: {len(jobs)} run(s) -- "
          f"{len(args.methods)} method(s) x {len(args.seeds)} seed(s) at {args.size}")
    print(f"sampling={args.sampling}  epochs={args.epochs}  "
          f"lr={args.learning_rate:g}  "
          f"n_examples={args.n_examples}  n_breadth={args.n_breadth}")
    print("=" * 78)

    planned = []
    for method, seed in jobs:
        rid = run_id_for(method, args.size, seed, args.learning_rate)
        state = "DONE (skip)" if already_done(out_root / rid) else "queued"
        planned.append((method, seed, rid, state))
        print(f"  {method:9s} seed={seed}  ->  {rid:34s} {state}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    results = []
    for method, seed, rid, state in planned:
        out_dir = out_root / rid
        if state.startswith("DONE"):
            results.append((rid, "skipped", 0.0))
            continue

        free = free_vram_gb()
        if free < args.min_free_gb:
            print(f"\n!! SKIPPING {rid}: only {free:.1f} GB free, need "
                  f"{args.min_free_gb:.1f} GB. The card is shared; retry later.")
            results.append((rid, f"skipped-oom-guard({free:.1f}GB)", 0.0))
            continue

        cmd = [sys.executable, "-m", "deeperase.scripts.run_study",
               "--method", method,
               "--size", args.size,
               "--seed", str(seed),
               "--sampling", args.sampling,
               "--epochs", str(args.epochs),
               "--learning-rate", str(args.learning_rate),
               "--n-examples", str(args.n_examples),
               "--n-breadth", str(args.n_breadth),
               "--reference-spans", "reference_uds/tofu_data/forget10_filtered.json",
               "--cache-dir", args.cache_dir,
               "--output-dir", args.output_dir,
               "--run-id", rid]
        if out_dir.exists():
            cmd.append("--resume")
        cmd += [a for a in args.extra if a != "--extra"]

        print("\n" + "-" * 78)
        print(f">> {rid}   ({free:.1f} GB free)")
        print("   " + " ".join(cmd))
        print("-" * 78, flush=True)

        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = (time.time() - t0) / 60.0
        results.append((rid, "ok" if rc == 0 else f"FAILED(rc={rc})", dt))
        if rc != 0:
            print(f"!! {rid} exited {rc} after {dt:.1f} min. Continuing with the "
                  "rest of the matrix; re-run this script to retry (it resumes).")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for rid, status, dt in results:
        print(f"  {rid:34s} {status:26s} {dt:6.1f} min")

    manifest = out_root / "replication_manifest.json"
    manifest.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "size": args.size, "sampling": args.sampling, "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "methods": args.methods, "seeds": args.seeds,
        "runs": [{"run_id": r, "status": s, "minutes": round(d, 1)}
                 for r, s, d in results],
    }, indent=2), encoding="utf-8")
    print(f"\nmanifest: {manifest}")

    failed = [r for r, s, _ in results if s.startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} run(s) failed. Nothing has been aggregated -- fix "
              "and re-run before reading any trajectory.")
        return 1

    print("\nNext: these are independent observations only if --sampling random\n"
          "was used with distinct seeds. Before drawing any depth/breadth\n"
          "conclusion, check that the utility control is flat within each run\n"
          "(it fell 0.720 -> 0.625 in the August GA run, which is why that one\n"
          "could not be interpreted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
