"""Model registry, GPU memory planning, and run configuration.

Everything in this module was verified against the HuggingFace API on
2026-08-12. Parameter counts are exact, read from each repository's
safetensors index -- they are not estimates.

Why this module exists
----------------------
The project now targets a real 20 GB GPU rather than a CPU. Two things must be
decided before any experiment runs, and getting either wrong wastes hours of
GPU time on a crash:

1. **Which models to use.** Not every TOFU checkpoint fits in 20 GB.
2. **How many to hold at once.** The UDS procedure needs three models, but
   they do not all have to be resident simultaneously -- see
   :class:`ExecutionStrategy`.

Both are answered here so the runner script can fail fast with a clear message
instead of dying with an out-of-memory error twenty minutes in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry -- verified 2026-08-12 via https://huggingface.co/api/models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable checkpoint."""

    repo_id: str
    n_params: int
    family: str
    size_label: str
    split: str
    """One of: full, retain90, retain95, retain99.

    * ``full``     -- fine-tuned on every TOFU author. Knows the forget set.
    * ``retain90`` -- fine-tuned on 90% of authors; never saw the forget10 set.
    * ``retain95`` -- never saw the forget5 set.
    * ``retain99`` -- never saw the forget1 set.
    """

    def bytes_at(self, dtype_size: int = 2) -> int:
        """Weight memory in bytes. ``dtype_size=2`` for fp16/bf16, 4 for fp32."""
        return self.n_params * dtype_size

    def gb_at(self, dtype_size: int = 2) -> float:
        return self.bytes_at(dtype_size) / 1e9

    @property
    def unseen_fraction(self) -> float:
        """Fraction of the *forget10* set this model never saw.

        This is the axis of the UDS paper's Table 2 validation. ``full`` saw
        everything (0.0); ``retain90`` saw none of forget10 (1.0).
        """
        return {"full": 0.0, "retain99": 0.10, "retain95": 0.50, "retain90": 1.0}[self.split]


def _tofu(family: str, size: str, n_params: int) -> Dict[str, ModelSpec]:
    return {
        split: ModelSpec(
            repo_id=f"open-unlearning/tofu_{family}-{size}-Instruct_{split}",
            n_params=n_params,
            family=family,
            size_label=size,
            split=split,
        )
        for split in ("full", "retain90", "retain95", "retain99")
    }


#: All verified TOFU checkpoints, keyed by size label then split.
TOFU_MODELS: Dict[str, Dict[str, ModelSpec]] = {
    "1B": _tofu("Llama-3.2", "1B", 1_235_814_400),
    "3B": _tofu("Llama-3.2", "3B", 3_212_749_824),
}

#: Reference values from the UDS paper (Lee, Kim & Jo, arXiv:2605.24614),
#: Table 2. Stage-1 baseline is retain90 at each scale; the target is always
#: ``full``. Reproducing these is our correctness test -- see
#: docs/UDS_CONFORMANCE.md item 2.
UDS_PAPER_TABLE2: Dict[str, Dict[str, float]] = {
    "1B": {"full": 0.002, "retain99": 0.153, "retain95": 0.496, "retain90": 1.000},
    "3B": {"full": 0.008, "retain99": 0.151, "retain95": 0.482, "retain90": 1.000},
    "8B": {"full": 0.000, "retain99": 0.101, "retain95": 0.455, "retain90": 1.000},
}

#: Tolerance when comparing our UDS against Table 2. The paper reports three
#: decimals and we cannot control their tokenisation or seed exactly, so an
#: exact match is not expected. What must hold is the *ordering* and rough
#: magnitude -- see :func:`check_against_paper`.
TABLE2_ABS_TOLERANCE = 0.08


# ---------------------------------------------------------------------------
# GPU memory planning
# ---------------------------------------------------------------------------


class ExecutionStrategy(str, Enum):
    """How many models to keep on the GPU at once.

    The UDS procedure needs three models, but not simultaneously. Hidden
    states are tiny compared with weights (megabytes, not gigabytes), so we
    can capture them from one model, move them to CPU, free the model, and
    load the next. That trades a little reload time for a large memory saving.
    """

    ALL_RESIDENT = "all_resident"
    """Hold all three models on the GPU. Fastest -- no reloading."""

    SEQUENTIAL = "sequential"
    """Hold exactly one model at a time. Slowest, but peak memory is one
    model, so it works whenever a single model fits."""

    def __str__(self) -> str:
        return self.value


@dataclass
class MemoryPlan:
    """The outcome of memory planning: fits or does not, and how to run."""

    fits: bool
    strategy: Optional[ExecutionStrategy]
    peak_weight_gb: float
    headroom_gb: float
    reason: str
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "FITS" if self.fits else "DOES NOT FIT"
        s = (
            f"[{head}] strategy={self.strategy or 'none'}  "
            f"peak weights={self.peak_weight_gb:.2f} GB  "
            f"headroom={self.headroom_gb:.2f} GB\n  {self.reason}"
        )
        for w in self.warnings:
            s += f"\n  WARNING: {w}"
        return s


#: Fraction of total GPU memory we refuse to plan into. Activations,
#: fragmentation, the CUDA context and cuBLAS workspaces all consume memory
#: that never appears in a parameter count. 30% is deliberately generous;
#: an out-of-memory crash three hours into a run costs far more than being
#: conservative here.
SAFETY_FRACTION = 0.30

#: Below this, a plan is reported as fitting but flagged as risky.
COMFORTABLE_HEADROOM_GB = 2.0

#: Peak memory of full-parameter AdamW training, as a multiple of weight memory:
#: 1x weights + 1x gradients + 2x optimiser moments (``exp_avg``, ``exp_avg_sq``,
#: both allocated in the parameter dtype). Inference needs 1x, which is why a
#: weights-only plan is not merely optimistic for a training run -- it is wrong
#: by a factor of four.
#:
#: This assumes ``foreach=False``. The multi-tensor optimiser path additionally
#: materialises a full-size temporary inside ``step()``, taking the true peak to
#: 5x; see :func:`deeperase.unlearn.unlearn`.
TRAINING_STATE_MULTIPLIER = 4.0

#: Slack for activations, autograd graph and cuBLAS workspaces during training.
#: Batch 1 at a few hundred tokens is small; this is deliberately loose.
ACTIVATION_ALLOWANCE_GB = 1.0


def plan_memory(
    size_label: str,
    gpu_total_gb: float,
    *,
    dtype_size: int = 2,
    prefer: ExecutionStrategy = ExecutionStrategy.ALL_RESIDENT,
    gpu_free_gb: Optional[float] = None,
    training: bool = False,
) -> MemoryPlan:
    """Decide whether a model size fits, and how to run it.

    Args:
        size_label: key into :data:`TOFU_MODELS`, e.g. ``"1B"``.
        gpu_total_gb: total GPU memory, as reported by the device.
        dtype_size: bytes per weight. 2 for fp16/bf16 (the default), 4 for fp32.
        prefer: strategy to try first. Falls back to SEQUENTIAL automatically.
        gpu_free_gb: memory actually available right now, from
            ``torch.cuda.mem_get_info``. **Pass this on any shared machine.**
            Total capacity is not a budget: co-tenant processes hold memory that
            never appears in ``total_memory``, and planning against it produces a
            confident "fits" for a run that cannot start.
        training: if True, budget for full-parameter optimiser state rather than
            weights alone. See :data:`TRAINING_STATE_MULTIPLIER`.

    Returns:
        A :class:`MemoryPlan`. Always check ``.fits`` before starting a run.
    """
    if size_label not in TOFU_MODELS:
        return MemoryPlan(
            fits=False, strategy=None, peak_weight_gb=0.0, headroom_gb=0.0,
            reason=f"Unknown size {size_label!r}. Available: {sorted(TOFU_MODELS)}",
        )

    one = next(iter(TOFU_MODELS[size_label].values())).gb_at(dtype_size)
    basis = gpu_total_gb if gpu_free_gb is None else gpu_free_gb
    basis_label = "total" if gpu_free_gb is None else "free"
    usable = basis * (1.0 - SAFETY_FRACTION)
    warnings: List[str] = []

    if dtype_size == 4:
        warnings.append(
            "Planning at fp32 (4 bytes/param). Use bf16 or fp16 to halve weight memory."
        )

    # A shared node is the normal case, not the exception. Say so loudly when
    # most of the card belongs to somebody else -- otherwise the only symptom is
    # an allocator failure part-way through a long run.
    if gpu_free_gb is not None and gpu_free_gb < 0.75 * gpu_total_gb:
        warnings.append(
            f"Only {gpu_free_gb:.1f} GB of {gpu_total_gb:.1f} GB is free -- "
            f"{gpu_total_gb - gpu_free_gb:.1f} GB is held by other processes. "
            "Planning against free memory. Run `nvidia-smi` to see who, and "
            "consider CUDA_VISIBLE_DEVICES to pick an idle GPU."
        )

    # Training peak, if asked for: optimiser state dominates and is invisible to
    # a weight count.
    train_peak = one * TRAINING_STATE_MULTIPLIER + ACTIVATION_ALLOWANCE_GB

    # Try the preferred strategy, then fall back.
    order = (
        [ExecutionStrategy.ALL_RESIDENT, ExecutionStrategy.SEQUENTIAL]
        if prefer is ExecutionStrategy.ALL_RESIDENT
        else [ExecutionStrategy.SEQUENTIAL]
    )

    for strategy in order:
        n_resident = 3 if strategy is ExecutionStrategy.ALL_RESIDENT else 1
        weight_peak = one * n_resident
        # The two phases do not overlap -- reference models are freed before
        # training starts -- so the run's peak is whichever phase is larger.
        peak = max(weight_peak, train_peak) if training else weight_peak
        if peak <= usable:
            headroom = usable - peak
            if headroom < COMFORTABLE_HEADROOM_GB:
                warnings.append(
                    f"Only {headroom:.2f} GB headroom after weights. Reduce batch size "
                    "or sequence length if you hit out-of-memory errors."
                )
            if strategy is ExecutionStrategy.SEQUENTIAL:
                warnings.append(
                    "Sequential mode reloads models between phases. Expect roughly "
                    "20-40% longer runtime than all-resident."
                )
            if training and train_peak >= weight_peak:
                driver = (
                    f"training dominates: {one:.2f} GB weights x "
                    f"{TRAINING_STATE_MULTIPLIER:.0f} (weights + grads + 2 AdamW "
                    f"moments) + {ACTIVATION_ALLOWANCE_GB:.1f} GB activations "
                    f"= {train_peak:.2f} GB"
                )
            else:
                driver = (
                    f"{size_label} at {one:.2f} GB/model x {n_resident} resident "
                    f"= {weight_peak:.2f} GB"
                )
            return MemoryPlan(
                fits=True, strategy=strategy, peak_weight_gb=peak, headroom_gb=headroom,
                reason=(
                    f"{driver}, within {usable:.2f} GB usable "
                    f"({basis:.1f} GB {basis_label} minus {SAFETY_FRACTION:.0%} "
                    f"safety margin)."
                ),
                warnings=warnings,
            )

    need = max(one, train_peak) if training else one
    remedies = (
        "Options: free the GPU (`nvidia-smi`), pick an idle one with "
        "CUDA_VISIBLE_DEVICES, use a smaller model, or -- if the shortfall is "
        "optimiser state -- switch to the LoRA arm, which trains ~0.1% of the "
        "parameters and needs roughly weights-only memory."
        if training else
        "Options: use a smaller model, or load in 4-bit (not implemented)."
    )
    return MemoryPlan(
        fits=False, strategy=None, peak_weight_gb=need, headroom_gb=usable - need,
        reason=(
            f"{size_label} needs {need:.2f} GB but only {usable:.2f} GB is usable "
            f"({basis:.1f} GB {basis_label} minus {SAFETY_FRACTION:.0%} safety "
            f"margin) on a {gpu_total_gb:.1f} GB device. {remedies}"
        ),
        warnings=warnings,
    )


def recommend_size(gpu_total_gb: float, *, dtype_size: int = 2) -> Optional[str]:
    """Largest registered model size that fits, preferring all-resident.

    Returns None when nothing fits.
    """
    for size in sorted(TOFU_MODELS, key=lambda s: TOFU_MODELS[s]["full"].n_params, reverse=True):
        if plan_memory(size, gpu_total_gb, dtype_size=dtype_size).fits:
            return size
    return None


def check_against_paper(
    size_label: str,
    observed: Dict[str, float],
    *,
    tolerance: float = TABLE2_ABS_TOLERANCE,
) -> Dict[str, object]:
    """Compare measured UDS values against the paper's Table 2.

    The paper's headline property is **monotonicity**: as the Stage-2 source
    model has seen less of the forget set, UDS must rise, reaching ~1.0 for a
    model that never saw it. That ordering is the real test. Absolute
    agreement is a bonus, since we cannot match their tokenisation exactly.

    Args:
        size_label: ``"1B"``, ``"3B"`` or ``"8B"``.
        observed: split name -> measured UDS.
        tolerance: allowed absolute difference per split.

    Returns:
        Dict with ``monotonic``, ``per_split`` comparisons, ``n_within_tolerance``
        and an overall ``verdict``.
    """
    expected = UDS_PAPER_TABLE2.get(size_label)
    if expected is None:
        raise KeyError(f"No published Table 2 values for size {size_label!r}")

    order = ["full", "retain99", "retain95", "retain90"]
    present = [s for s in order if s in observed]

    per_split = {
        s: {
            "expected": expected[s],
            "observed": observed[s],
            "abs_diff": abs(observed[s] - expected[s]),
            "within_tolerance": abs(observed[s] - expected[s]) <= tolerance,
        }
        for s in present
    }

    values = [observed[s] for s in present]
    monotonic = all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
    n_ok = sum(1 for v in per_split.values() if v["within_tolerance"])

    if not monotonic:
        verdict = "FAIL: UDS is not monotonic across retain splits"
    elif n_ok == len(present):
        verdict = "PASS: monotonic and every split within tolerance"
    else:
        verdict = (
            f"PARTIAL: monotonic, but {len(present) - n_ok}/{len(present)} splits "
            "outside tolerance"
        )

    return {
        "size_label": size_label,
        "tolerance": tolerance,
        "splits_compared": present,
        "monotonic": monotonic,
        "n_within_tolerance": n_ok,
        "n_compared": len(present),
        "per_split": per_split,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Everything one experiment run needs. Serialise this next to results so
    a run is always reproducible from its own output."""

    size_label: str = "1B"
    dtype: str = "bfloat16"
    """bfloat16 is preferred over float16: same memory, far less prone to
    overflow. Falls back automatically on GPUs without bf16 support."""

    forget_split: str = "forget10"
    stage1_source_split: str = "retain90"
    """Must correspond to forget_split: forget10->retain90, forget05->retain95,
    forget01->retain99."""

    n_examples: int = 50
    max_seq_length: int = 256
    tau: float = 0.05
    layers: Optional[List[int]] = None
    """None means every decoder layer."""

    strategy: Optional[ExecutionStrategy] = None
    """None means decide automatically from detected GPU memory."""

    seed: int = 0
    output_dir: str = "results/gpu_runs"
    cache_dir: Optional[str] = None
    """Where HuggingFace downloads are stored. On an ephemeral remote machine,
    point this at persistent storage or you will re-download every session."""

    FORGET_TO_RETAIN = {"forget10": "retain90", "forget05": "retain95", "forget01": "retain99"}

    def validate(self) -> List[str]:
        """Return a list of problems. Empty means the config is usable."""
        problems: List[str] = []
        if self.size_label not in TOFU_MODELS:
            problems.append(f"size_label {self.size_label!r} not in {sorted(TOFU_MODELS)}")
        if self.forget_split not in self.FORGET_TO_RETAIN:
            problems.append(
                f"forget_split {self.forget_split!r} not in {sorted(self.FORGET_TO_RETAIN)}"
            )
        else:
            expected = self.FORGET_TO_RETAIN[self.forget_split]
            if self.stage1_source_split != expected:
                problems.append(
                    f"stage1_source_split {self.stage1_source_split!r} does not match "
                    f"forget_split {self.forget_split!r}; expected {expected!r}. "
                    "A mismatched pair silently measures the wrong thing."
                )
        if self.dtype not in ("bfloat16", "float16", "float32"):
            problems.append(f"dtype {self.dtype!r} must be bfloat16, float16 or float32")
        if self.n_examples < 1:
            problems.append("n_examples must be >= 1")
        if not 0.0 < self.tau < 1.0:
            problems.append(f"tau must be in (0, 1), got {self.tau}")
        return problems

    @property
    def dtype_size(self) -> int:
        return 4 if self.dtype == "float32" else 2

    def models(self) -> Dict[str, ModelSpec]:
        return TOFU_MODELS[self.size_label]

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["strategy"] = str(self.strategy) if self.strategy else None
        return d
