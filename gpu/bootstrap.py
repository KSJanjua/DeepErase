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


def check_torch() -> "object | None":
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        line(BAD, "torch is not installed. On a GPU container do NOT pip install a "
                  "CPU wheel -- use the preinstalled CUDA build, then:\n"
                  "         pip install -r requirements-gpu.txt")
        sys.exit(1)

    line(OK, f"torch {torch.__version__}")
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
    if want_gb is not None:
        if free_gb >= want_gb:
            line(OK, f"planned run needs ~{want_gb:.1f} GB; {free_gb:.1f} GB free")
        else:
            line(BAD, f"planned run needs ~{want_gb:.1f} GB but only {free_gb:.1f} GB "
                      "is free. This card is shared -- either wait, or drop to a "
                      "smaller --size / --batch-size.")
            sys.exit(1)
    return free_gb


def check_plan(size: str) -> float | None:
    """Ask the package's own memory planner, so this agrees with what the
    runners will enforce rather than guessing separately."""
    try:
        from deeperase.config import RunConfig, plan_memory
    except Exception as e:                                   # noqa: BLE001
        line(BAD, f"could not import deeperase.config ({e}). The package must be "
                  f"importable from {_ROOT}. Are you in the repository root?")
        sys.exit(1)
    try:
        cfg = RunConfig(size_label=size)
        plan = plan_memory(cfg)
        line(OK, f"memory plan for {size}: {plan.summary()}")
        return float(getattr(plan, "peak_weight_gb", 0.0)) + float(
            getattr(plan, "headroom_gb", 0.0)) or None
    except Exception as e:                                   # noqa: BLE001
        line(WARN, f"memory plan unavailable ({e})")
        return None


def check_disk(min_gb: float = 60.0) -> None:
    free = shutil.disk_usage(Path.cwd()).free / GB
    line(OK if free >= min_gb else WARN,
         f"{free:.0f} GB free on this filesystem "
         f"(model cache alone is ~250 GB for the full model set; "
         f"a single 1B pair is ~5 GB)")


def check_hf_cache() -> None:
    cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    line(OK, f"HF cache: {cache or './hf_cache (default --cache-dir)'}")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        line(WARN, "HF_HUB_OFFLINE=1 -- downloads are disabled this session")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="1B",
                    help="model scale to plan memory for (default 1B)")
    ap.add_argument("--run-tests", action="store_true",
                    help="also run the CPU test suite (~35 s). Do this once per "
                         "environment -- it is the cheapest way to catch a broken "
                         "install before spending GPU hours.")
    args = ap.parse_args()

    print("=" * 72)
    print("DeepErase GPU bootstrap")
    print("=" * 72)

    check_repo_root()
    check_python()
    torch = check_torch()
    check_deps()
    want = check_plan(args.size)
    check_gpu_memory(torch, want)
    check_disk()
    check_hf_cache()

    if args.run_tests:
        print("-" * 72)
        print("running the test suite (CPU)...")
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q",
                            "--tb=short"])
        if r.returncode != 0:
            line(BAD, "test suite FAILED -- fix this before running anything on GPU")
            return 1
        line(OK, "test suite passed")
        print("\nPaste that output into VERIFICATION.md -- it currently shows the "
              "August 118/180-test runs and is overdue for a refresh.")

    print("=" * 72)
    print("READY")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
