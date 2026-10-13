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

Protocols
---------
``--protocol v2`` (the default) is the corrected design of 9 Oct 2026 -- see
``deeperase/scripts/run_study.py``. Its run ids carry the protocol and the
learning rate (``study_v2_ga_1B_s0_lr1e-06``), so a v2 matrix can never resume
into, or be mistaken for, a v1 directory. ``--protocol v1`` keeps the old ids so
the existing v1 runs still resume.

Usage
-----
    python gpu/replicate.py --dry-run                 # print the plan, run nothing
    python gpu/replicate.py --methods ga --seeds 0 1 2 --learning-rate 1e-6
    python gpu/replicate.py --methods npo --seeds 0 --epochs 20    # single arm
    python gpu/replicate.py --methods ga --seeds 0 --control random --control-scale 4

Run one invocation per method: methods need different learning rates. Expect
roughly one hour per run at 1B in float32; the manifest accumulates across
invocations.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def check_interpreter() -> None:
    """Fail in seconds, not hours, if this interpreter cannot run a study.

    ``sys.executable`` is what every child run uses, so whatever is importable
    here is what they get. A shell that never activated the venv still has the
    container's system python on PATH -- which carries the NGC torch, so the
    GPU checks all pass -- but not ``datasets``, which is only reached once a
    run starts loading data. Without this guard an overnight matrix fails one
    run at a time, hours in, having produced nothing.
    """
    missing = []
    for mod in ("torch", "transformers", "datasets", "scipy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"REFUSING: {sys.executable} cannot import: {', '.join(missing)}.\n"
              "Every run would fail. Activate the project environment first:\n\n"
              "    source /workspace/Capstone_Project/DeepErase/gpu/env.sh\n\n"
              "then re-run. Confirm with `python gpu/bootstrap.py --size 1B`.",
              file=sys.stderr)
        sys.exit(2)


DEFAULT_LR = 1e-6


def run_id_for(method: str, size: str, seed: int, lr: float = DEFAULT_LR, *,
               protocol: str = "v1", control: str = "none",
               control_scale: float = 1.0, tag: str = "") -> str:
    """Run id, encoding everything that makes two runs incomparable.

    The lr belongs in the id because methods need different ones: GA reaches
    its utility floor at 1e-6, GradDiff barely moves there and diverges at
    1e-5. Without it in the id, a matrix at a new lr would collide with an
    existing directory and ``already_done`` would silently skip the run,
    producing a matrix whose rows were trained at different rates. Under v1
    the default lr is left un-suffixed so existing run directories still
    resume; v2 ids always carry the protocol and the lr. Control runs are
    prefixed so they can never be read as measurements. ``tag`` names an
    ablation (e.g. ``lossfull`` for ``--extra --loss-scope full``), which would
    otherwise share -- and silently resume -- the main run's directory.
    """
    if protocol == "v1":
        base = f"study_{method}_{size}_s{seed}"
        rid = base if lr == DEFAULT_LR else f"{base}_lr{lr:g}"
    else:
        rid = f"study_{protocol}_{method}_{size}_s{seed}_lr{lr:g}"
    if control != "none":
        prefix = "control" if control_scale == 1.0 else f"control_x{control_scale:g}"
        rid = rid.replace("study_", f"{prefix}_", 1)
    if tag:
        if not re.fullmatch(r"[A-Za-z0-9-]+", tag):
            raise ValueError(f"--tag must be letters, digits or '-', got {tag!r}")
        rid = f"{rid}_{tag}"
    return rid


def fits_now(size: str, *, protocol: str, method: str, floor_gb: float) -> tuple:
    """Pre-flight with the same memory model run_study uses, against free VRAM.

    Returns ``(ok, free_gb, reason)``. run_study re-checks at start-up; this
    only avoids launching a child that will refuse.
    """
    import torch
    from deeperase.config import plan_memory

    if not torch.cuda.is_available():
        return False, 0.0, "no CUDA device"
    free_b, total_b = torch.cuda.mem_get_info()
    free, total = free_b / 1e9, total_b / 1e9          # decimal GB, as run_study
    dtype_size = 4 if protocol == "v2" else 2
    plan = plan_memory(size, total, dtype_size=dtype_size, gpu_free_gb=free,
                       training=True, reference_model=(method == "npo"))
    if free < floor_gb:
        return False, free, f"only {free:.1f} GB free, --min-free-gb is {floor_gb:.1f}"
    return plan.fits, free, plan.reason


def update_manifest(path: Path, batch: dict, results: list) -> dict:
    """Merge this invocation into the manifest instead of overwriting it.

    Each invocation used to replace the file, so after the NPO batch it listed
    only NPO and the GA/GradDiff runs vanished from the record. Runs are now
    keyed by run id (the latest status wins) and every invocation is kept in
    ``batches``.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "batches" not in data:            # pre-merge format: wrap it
            data = {"batches": [data], "runs": {r["run_id"]: r for r in data.get("runs", [])}}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"batches": [], "runs": {}}
    data["batches"].append(batch)
    for r in results:
        data["runs"][r["run_id"]] = r
    data["updated"] = batch["generated"]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def requested_alphas(extra) -> Optional[set]:
    """The alphas passed through ``--extra --alphas ...``, if any."""
    extra = list(extra)
    for i, a in enumerate(extra):
        val = a.split("=", 1)[1] if a.startswith("--alphas=") else (
            extra[i + 1] if a == "--alphas" and i + 1 < len(extra) else None)
        if val is not None:
            return {round(float(x), 6) for x in val.split(",") if x.strip()}
    return None


def already_done(out_dir: Path, alphas: Optional[set] = None) -> bool:
    """A run is complete when its plane.json holds points -- and, when specific
    alphas are requested, a point at every one of them. So adding alphas to a
    finished run's command extends it instead of skipping it."""
    plane = out_dir / "plane.json"
    if not plane.exists():
        return False
    try:
        pts = json.loads(plane.read_text())["points"]
    except Exception:                                        # noqa: BLE001
        return False
    if not pts:
        return False
    return alphas is None or alphas <= {round(float(p["alpha"]), 6) for p in pts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v2",
                    help="Study design (see deeperase/scripts/run_study.py).")
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
    ap.add_argument("--control", default="none", choices=["none", "random"],
                    help="Run the T1 random-direction control instead of the real sweep.")
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--output-dir", default="results/studies")
    ap.add_argument("--cache-dir", default=os.environ.get("DEEPERASE_CACHE_DIR", "./hf_cache"),
                    help="default: $DEEPERASE_CACHE_DIR (set by gpu/env.sh), else ./hf_cache")
    ap.add_argument("--min-free-gb", type=float, default=0.0,
                    help="Extra floor on free VRAM. The real check is the memory "
                         "plan (float32 v2 GA ~21 GB, NPO ~26 GB, plus a 30%% "
                         "margin), evaluated against free memory before each run.")
    ap.add_argument("--tag", default="",
                    help="suffix naming an ablation, e.g. 'lossfull' with "
                         "--extra --loss-scope full; keeps it out of the main run's "
                         "directory")
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

    check_interpreter()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deeperase.provenance import git_info
    git = git_info()
    if git.get("dirty"):
        print("WARNING: the working tree has uncommitted changes. The runs will "
              "record dirty=true; commit first so the recorded commit describes "
              "the code that ran.", file=sys.stderr)

    jobs = [(m, s) for m in args.methods for s in args.seeds]
    out_root = Path(args.output_dir)

    print("=" * 78)
    print(f"Replication matrix: {len(jobs)} run(s) -- "
          f"{len(args.methods)} method(s) x {len(args.seeds)} seed(s) at {args.size}")
    print(f"interpreter: {sys.executable}")
    print(f"protocol={args.protocol}  sampling={args.sampling}  epochs={args.epochs}  "
          f"lr={args.learning_rate:g}  "
          f"n_examples={args.n_examples}  n_breadth={args.n_breadth}  "
          f"control={args.control}{'' if args.control == 'none' else f' x{args.control_scale:g}'}")
    print(f"commit={git.get('commit')}  dirty={git.get('dirty')}")
    print("=" * 78)

    planned = []
    for method, seed in jobs:
        rid = run_id_for(method, args.size, seed, args.learning_rate,
                         protocol=args.protocol, control=args.control,
                         control_scale=args.control_scale, tag=args.tag)
        state = ("DONE (skip)" if already_done(out_root / rid, requested_alphas(args.extra))
                 else "queued")
        planned.append((method, seed, rid, state))
        print(f"  {method:9s} seed={seed}  ->  {rid:44s} {state}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    results = []
    for method, seed, rid, state in planned:
        out_dir = out_root / rid
        if state.startswith("DONE"):
            results.append((rid, "skipped", 0.0))
            continue

        ok, free, why = fits_now(args.size, protocol=args.protocol, method=method,
                                 floor_gb=args.min_free_gb)
        if not ok:
            print(f"\n!! SKIPPING {rid}: {why}. The card is shared; retry later.")
            results.append((rid, f"skipped-memory({free:.1f}GB)", 0.0))
            continue

        cmd = [sys.executable, "-m", "deeperase.scripts.run_study",
               "--protocol", args.protocol,
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
        if args.control != "none":
            cmd += ["--control", args.control, "--control-scale", str(args.control_scale)]
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
        print(f"  {rid:44s} {status:26s} {dt:6.1f} min")

    manifest = out_root / "replication_manifest.json"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    batch = {"generated": stamp, "protocol": args.protocol, "size": args.size,
             "sampling": args.sampling, "epochs": args.epochs,
             "learning_rate": args.learning_rate, "methods": args.methods,
             "seeds": args.seeds, "control": args.control,
             "control_scale": args.control_scale, "git": git}
    update_manifest(manifest, batch, [
        {"run_id": r, "status": s, "minutes": round(d, 1), "protocol": args.protocol,
         "learning_rate": args.learning_rate, "finished": stamp,
         "commit": git.get("commit")} for r, s, d in results])
    print(f"\nmanifest: {manifest} (merged)")

    failed = [r for r, s, _ in results if s.startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} run(s) failed. Nothing has been aggregated -- fix "
              "and re-run before reading any trajectory.")
        return 1

    print("\nNext: python gpu/summarize.py. Before drawing any depth/breadth\n"
          "conclusion, check selectivity (the forget tiers must fall further than\n"
          "the retain tier) and that the utility control holds within each run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
