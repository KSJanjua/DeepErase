#!/usr/bin/env python3
"""One-shot environment check for the GPU box. Run this FIRST, every session.

Designed for a Jupyter Lab shell on a **shared** accelerator, so it reports free
memory rather than total: the number that matters is what is left after the
other tenant, and that changes between sessions.

Usage
-----
From a notebook cell::

    !python gpu/bootstrap.py --size 1B

or::

    %run gpu/bootstrap.py --size 1B

Exits non-zero if anything would stop a run, so it can gate a pipeline:

    !python gpu/bootstrap.py --size 1B && python gpu/replicate.py --dry-run

Nothing here downloads a model or trains. It is cheap and safe to re-run.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# `python gpu/bootstrap.py` puts THIS file's directory (gpu/) on sys.path, not
# the working directory -- so `import deeperase` fails even from the repo root.
# `python -m pytest` and `python -m deeperase.scripts.run_study` are unaffected
# because -m puts cwd on the path. Put the repo root on explicitly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Fragmentation is the usual cause of an OOM that "should" have fitted, and it
# bites hardest when another process owns part of the card. Set before torch
# allocates anything.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GB = 1024 ** 3
OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "


def line(status: str, msg: str) -> None:
    print(f"[{status}] {msg}")


def check_repo_root() -> Path:
    """The package resolves ``reference_uds/`` relative to the working
    directory, so running from the wrong place silently breaks
    ``--reference-spans``."""
    here = Path.cwd()
    if (here / "deeperase").is_dir() and (here / "tests").is_dir():
        line(OK, f"repository root: {here}")
        if (here / "reference_uds" / "tofu_data" / "forget10_filtered.json").exists():
            line(OK, "reference_uds/ annotations present (needed for --reference-spans)")
        else:
            line(WARN, "reference_uds/tofu_data/forget10_filtered.json NOT found. "
                       "Clone it here:\n"
                       "         git clone --depth 1 "
                       "https://github.com/gnueaj/unlearning-depth-score.git reference_uds")
        if (here / "data" / "breadth_items_v2.json").exists():
            line(OK, "data/breadth_items_v2.json present (protocol v2 item set)")
        else:
            line(WARN, "data/breadth_items_v2.json NOT found -- protocol v2 runs will "
                       "fail. git pull, or rebuild it: python -m "
                       "deeperase.scripts.build_breadth_items")
        return here
    line(BAD, f"cwd is {here}, which is not the repository root. "
              "cd into the directory containing deeperase/ and tests/ first.")
    sys.exit(1)


def check_python() -> None:
    v = sys.version_info
    msg = f"python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 10):
        line(BAD, msg + " -- the package requires >= 3.10")
        sys.exit(1)
    line(OK if (v.major, v.minor) == (3, 11) else WARN,
         msg + ("" if (v.major, v.minor) == (3, 11) else " (3.11 is the tested target)"))


def check_environment() -> None:
    """The project must run from its own environment, not a shared interpreter.

    gpu/setup_env.sh builds ``$DEEPERASE_HOME/.venv`` and gpu/env.sh activates
    it. A system or container interpreter would mix this project's package
    versions with another project's.
    """
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    home = os.environ.get("DEEPERASE_HOME")
    expected = Path(home, ".venv").resolve() if home else None
    if expected is not None and Path(sys.prefix).resolve() == expected:
        line(OK, f"project environment: {sys.prefix}")
    elif in_venv:
        line(WARN, f"running from a virtual environment ({sys.prefix}), but not the "
                   "one gpu/env.sh activates. Run `source gpu/env.sh` first.")
    else:
        line(WARN, f"running from a shared interpreter ({sys.executable}), not the "
                   "project environment. Run `source gpu/env.sh` first.")


def gpu_driver_version() -> "str | None":
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def check_torch() -> "object | None":
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        line(BAD, "torch is not installed in this interpreter. Create the project "
                  "environment once with `bash gpu/setup_env.sh`, then in every "
                  "terminal `source gpu/env.sh`.")
        sys.exit(1)

    line(OK, f"torch {torch.__version__} (CUDA runtime {torch.version.cuda})")
    if not torch.cuda.is_available():
        line(BAD, "torch.cuda.is_available() is False -- this is a CPU-only build "
                  "or no device is visible. requirements.txt installs a CPU wheel; "
                  "requirements-gpu.txt deliberately does not touch torch.")
        sys.exit(1)
    return torch


def check_deps() -> None:
    for mod, label in [("transformers", "transformers"), ("datasets", "datasets"),
                       ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                       ("numpy", "numpy"), ("pytest", "pytest")]:
        try:
            m = importlib.import_module(mod)
            line(OK, f"{label} {getattr(m, '__version__', '?')}")
        except ImportError:
            line(BAD, f"{label} missing -- pip install -r requirements-gpu.txt")
            sys.exit(1)


def check_gpu_memory(torch, want_gb: float | None) -> float:
    n = torch.cuda.device_count()
    line(OK, f"{n} CUDA device(s) visible")
    free_gb = 0.0
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gb = max(free_gb, free_b / GB)
        used = (total_b - free_b) / GB
        line(OK, f"  cuda:{i} {name} -- {total_b/GB:.1f} GB total, "
                 f"{used:.1f} GB in use by others, {free_b/GB:.1f} GB free")
        if "MIG" in name:
            line(OK, "  this is a MIG slice: its memory is its own, but every process "
                     "using the same slice (e.g. another project's job) shares it")
    driver = gpu_driver_version()
    if driver:
        line(OK, f"NVIDIA driver {driver}")
    if want_gb is not None:
        if free_gb >= want_gb:
            line(OK, f"planned run needs ~{want_gb:.1f} GB; {free_gb:.1f} GB free")
        else:
            line(BAD, f"planned run needs ~{want_gb:.1f} GB but only {free_gb:.1f} GB "
                      "is free. This card is shared -- either wait, or drop to a "
                      "smaller --size / --batch-size.")
            sys.exit(1)
    return free_gb


def check_plan(torch, size: str, protocol: str = "v2") -> None:
    """Ask the package's own memory planner, so this agrees with what the
    runners will enforce rather than guessing separately.

    Planned as a **training** run against **free** memory, not total: this card
    is shared, a co-tenant's allocation never appears in ``total_memory``, and
    full-parameter AdamW peaks at ~4x the weight footprint. Planning against
    total, or for inference, produces a confident "fits" for a run that OOMs.
    """
    try:
        from deeperase.config import plan_memory
    except Exception as e:                                   # noqa: BLE001
        line(BAD, f"could not import deeperase.config ({e}). The package must be "
                  f"importable from {_ROOT}. Are you in the repository root?")
        sys.exit(1)

    free_b, total_b = torch.cuda.mem_get_info(0)
    # Decimal GB, matching run_study. Protocol v2 trains in float32 (4 bytes per
    # parameter); NPO also holds a frozen reference model, so it is planned
    # separately -- it is the tightest method on a shared card.
    dtype_size = 4 if protocol == "v2" else 2
    worst_fits = True
    for label, npo in (("GA/GradDiff", False), ("NPO", True)):
        plan = plan_memory(size, total_b / 1e9, dtype_size=dtype_size,
                           gpu_free_gb=free_b / 1e9, training=True, reference_model=npo)
        line(OK if plan.fits else (WARN if not npo else BAD),
             f"memory plan for {size} {label} (protocol {protocol}, training, vs free):")
        for chunk in plan.summary().splitlines():
            print("           " + chunk)
        worst_fits = worst_fits and (plan.fits or not npo)
        if not plan.fits and not npo:
            line(BAD, "this run would not fit right now. The card is shared -- wait "
                      "for the co-tenant, drop to a smaller --size, or set "
                      "CUDA_VISIBLE_DEVICES to an idle device.")
            sys.exit(1)
    if not worst_fits:
        line(WARN, "GA and GradDiff fit but NPO does not -- run NPO when more is free.")
    if protocol == "v2" and os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        line(OK, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 is set (NGC default); run_study "
                 "overrides it to 0 for float32 runs, so this is informational.")


def check_disk(min_gb: float = 20.0) -> None:
    where = Path(os.environ.get("DEEPERASE_HOME") or Path.cwd())
    free = shutil.disk_usage(where).free / GB
    line(OK if free >= min_gb else WARN,
         f"{free:.0f} GB free under {where} (the two 1B checkpoints take ~5 GB; "
         f"a run's results ~5 MB)")


def check_hf_cache(size: str = "1B") -> None:
    cache = os.environ.get("DEEPERASE_CACHE_DIR") or "./hf_cache"
    line(OK, f"model cache (--cache-dir default): {cache}")
    for split in ("full", "retain90"):
        repo = f"models--open-unlearning--tofu_Llama-3.2-{size}-Instruct_{split}"
        have = any((Path(cache) / repo / "snapshots").glob("*/*.safetensors"))
        line(OK if have else WARN,
             f"  {split}: {'downloaded' if have else 'not downloaded yet (the first run will fetch it)'}")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        line(OK, "HF_HUB_OFFLINE=1 (set by gpu/env.sh): runs read only the local cache. "
                 "To download something new: export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="1B",
                    help="model scale to plan memory for (default 1B)")
    ap.add_argument("--protocol", default="v2", choices=["v1", "v2"],
                    help="study protocol to plan memory for (v2 trains in float32)")
    ap.add_argument("--run-tests", action="store_true",
                    help="also run the test suite (~1 min). Do this once per "
                         "environment -- it is the cheapest way to catch a broken "
                         "install before spending GPU hours.")
    args = ap.parse_args()

    print("=" * 72)
    print("DeepErase GPU bootstrap")
    print("=" * 72)

    check_repo_root()
    check_environment()
    check_python()
    torch = check_torch()
    check_deps()
    check_gpu_memory(torch, None)
    check_plan(torch, args.size, args.protocol)
    check_disk()
    check_hf_cache(args.size)

    if args.run_tests:
        print("-" * 72)
        print("running the test suite (CPU)...")
        # -o addopts= : pyproject adds -q, and -qq would hide the count line
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                            "-o", "addopts=", "--tb=short", "-rs"])
        if r.returncode != 0:
            line(BAD, "test suite FAILED -- fix this before running anything on GPU")
            return 1
        line(OK, "test suite passed")
        print("\nIf this differs from VERIFICATION.md section 0, update that "
              "section -- it is the record of what was actually observed.")

    print("=" * 72)
    print("READY")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
