"""Preflight check. Run this FIRST on any new GPU machine.

Verifies, in order:

1. PyTorch can see a GPU at all
2. How much memory that GPU actually has
3. Whether bfloat16 is supported (preferred over float16)
4. Which TOFU model sizes fit, and how they must be run
5. Whether the HuggingFace model repositories are reachable
6. Whether there is enough disk space for the downloads

Nothing is downloaded and no model is loaded. The whole check takes a few
seconds. Its purpose is to fail fast with a clear message, instead of letting
a real run crash after twenty minutes and 10 GB of downloads.

Usage:
    python -m deeperase.scripts.check_gpu
    python -m deeperase.scripts.check_gpu --size 3B
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch

from deeperase.config import (
    TOFU_MODELS,
    ExecutionStrategy,
    RunConfig,
    plan_memory,
    recommend_size,
)

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"


def _hr(title: str) -> None:
    print(f"\n{title}\n" + "-" * 72)


def check_torch_and_gpu() -> tuple[bool, float, bool, float]:
    """Returns (has_gpu, total_gb, supports_bf16, free_gb)."""
    _hr("1. PyTorch and GPU")
    print(f"  torch version      : {torch.__version__}")
    print(f"  CUDA build         : {torch.version.cuda or 'CPU-only build'}")

    if not torch.cuda.is_available():
        print(f"  {BAD} No GPU visible to PyTorch.")
        print("         If you expected one, the most likely causes are:")
        print("           - a CPU-only torch build is installed")
        print("           - the notebook/session has no accelerator attached")
        print("         Reinstall with a CUDA build, or enable the GPU runtime.")
        return False, 0.0, False, 0.0

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()

    print(f"  {OK} GPU detected     : {props.name}")
    print(f"  compute capability : {props.major}.{props.minor}")
    print(f"  total memory       : {total_gb:.2f} GB")
    print(f"  bfloat16 supported : {'yes' if bf16 else 'no (will use float16)'}")

    free, _ = torch.cuda.mem_get_info()
    free_gb = free / 1e9
    print(f"  currently free     : {free_gb:.2f} GB")
    if free_gb / total_gb < 0.85:
        print(f"  {WARN} {total_gb - free_gb:.1f} GB is held by other processes. "
              "Planning below uses the free figure, not the total.")
    return True, total_gb, bf16, free_gb


def check_memory_plan(total_gb: float, size: str | None, dtype_size: int,
                      free_gb: float | None = None) -> RunConfig | None:
    _hr("2. Memory plan")
    chosen = size or recommend_size(total_gb, dtype_size=dtype_size)
    if chosen is None:
        print(f"  {BAD} No registered model size fits in {total_gb:.1f} GB.")
        return None

    # Two columns, because they differ by 4x and the study needs the second one.
    print(f"  {'':2s} {'size':3s} {'inference':>10s} {'training':>10s}")
    for candidate in sorted(TOFU_MODELS):
        inf = plan_memory(candidate, total_gb, dtype_size=dtype_size,
                          gpu_free_gb=free_gb)
        trn = plan_memory(candidate, total_gb, dtype_size=dtype_size,
                          gpu_free_gb=free_gb, training=True)
        mark = OK if inf.fits else BAD
        star = "  <-- selected" if candidate == chosen else ""
        print(f"  {mark} {candidate:3s} {inf.peak_weight_gb:7.2f} GB "
              f"{trn.peak_weight_gb:7.2f} GB "
              f"{'' if trn.fits else '(training does not fit)'}{star}")

    plan = plan_memory(chosen, total_gb, dtype_size=dtype_size, gpu_free_gb=free_gb)
    train_plan = plan_memory(chosen, total_gb, dtype_size=dtype_size,
                             gpu_free_gb=free_gb, training=True)
    print(f"\n  {plan.reason}")
    if not train_plan.fits:
        print(f"  {WARN} Inference fits but full-parameter training does not. "
              f"run_uds_validation and measure_breadth will work; run_study "
              f"will refuse.\n      {train_plan.reason}")
    for w in plan.warnings:
        print(f"  {WARN} {w}")
    if not plan.fits:
        return None

    cfg = RunConfig(size_label=chosen, strategy=plan.strategy)
    problems = cfg.validate()
    if problems:
        print(f"  {BAD} Config invalid: {problems}")
        return None
    return cfg


def check_repos_reachable(size: str) -> bool:
    _hr("3. Model repositories reachable")
    hdr = {"User-Agent": "deeperase-preflight"}
    all_ok = True
    for split, spec in sorted(TOFU_MODELS[size].items()):
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/api/models/{spec.repo_id}", headers=hdr
            )
            data = json.loads(urllib.request.urlopen(req, timeout=20).read())
            n = (data.get("safetensors") or {}).get("total")
            matches = n == spec.n_params
            mark = OK if matches else WARN
            note = "" if matches else f"  (expected {spec.n_params:,}, saw {n:,})"
            print(f"  {mark} {split:9s} {spec.repo_id}{note}")
            all_ok &= matches
        except Exception as e:
            print(f"  {BAD} {split:9s} unreachable: {type(e).__name__}")
            all_ok = False
    if not all_ok:
        print(f"\n  {WARN} If these are unreachable the machine may have no internet, "
              "or HuggingFace may need a login token for gated models.")
    return all_ok


def check_disk(size: str, cache_dir: str | None) -> bool:
    _hr("4. Disk space")
    needed = sum(m.gb_at(2) for m in TOFU_MODELS[size].values())
    target = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9

    print(f"  download location  : {target}")
    print(f"  needed for {size:3s}     : {needed:.2f} GB (all four splits)")
    print(f"  free               : {free_gb:.2f} GB")

    if free_gb < needed * 1.3:
        print(f"  {BAD} Not enough space. Allow ~30% extra for temporary files.")
        return False
    print(f"  {OK} Sufficient space")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight check for DeepErase GPU runs")
    ap.add_argument("--size", choices=sorted(TOFU_MODELS), default=None,
                    help="Force a model size instead of auto-selecting")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--cache-dir", default=None,
                    help="Where HuggingFace downloads go. Use persistent storage "
                         "on ephemeral machines.")
    args = ap.parse_args()

    print("=" * 72)
    print("DeepErase preflight check")
    print("=" * 72)

    has_gpu, total_gb, bf16, free_gb = check_torch_and_gpu()
    if not has_gpu:
        print("\nRESULT: cannot run on this machine. Fix the GPU issue above.")
        return 1

    dtype = args.dtype
    if dtype == "bfloat16" and not bf16:
        print(f"\n  {WARN} bfloat16 requested but unsupported; falling back to float16.")
        dtype = "float16"
    dtype_size = 4 if dtype == "float32" else 2

    cfg = check_memory_plan(total_gb, args.size, dtype_size, free_gb=free_gb)
    if cfg is None:
        print("\nRESULT: no workable memory plan. Use a smaller model or a bigger GPU.")
        return 1
    cfg.dtype = dtype
    cfg.cache_dir = args.cache_dir

    repos_ok = check_repos_reachable(cfg.size_label)
    disk_ok = check_disk(cfg.size_label, args.cache_dir)

    _hr("Summary")
    print(f"  model size   : {cfg.size_label}")
    print(f"  precision    : {cfg.dtype}")
    print(f"  strategy     : {cfg.strategy}")
    print(f"  forget split : {cfg.forget_split}  (Stage-1 source: {cfg.stage1_source_split})")

    if repos_ok and disk_ok:
        print(f"\n  {OK} READY. This machine can run the validation experiment.")
        print("\n  Next step (not yet implemented — Task 2 onwards):")
        print("      python -m deeperase.scripts.run_uds_validation")
        return 0

    print(f"\n  {BAD} NOT READY. Resolve the failures above first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
