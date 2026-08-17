"""Model loading, device placement, and GPU memory lifecycle.

The UDS procedure needs three models. On a 20 GB card, three 3B models do not
fit simultaneously, so *when* each model is resident has to be managed
explicitly. This module makes that management a single decision at
construction time, so the calling code reads the same either way::

    with manager.acquire("retain90") as model:
        hidden = capture_hidden_states(model, ...)
    # ALL_RESIDENT: model stays cached for reuse
    # SEQUENTIAL:   model is freed here, memory returned to the GPU

Two deliberate design choices worth knowing about:

**No ``device_map="auto"``.** Accelerate can silently spread a model across GPU
and CPU when it does not fit. That would break activation patching in a way
that produces plausible but wrong numbers, because our forward hooks assume
every layer sees tensors on one device. We place the whole model explicitly
and fail loudly if it does not fit, which is the safer failure.

**Gradients are disabled everywhere.** Every measurement in this project is
inference-only. Leaving gradients on would roughly double activation memory
for no benefit.
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch

from deeperase.config import ExecutionStrategy, ModelSpec, RunConfig, TOFU_MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device and dtype
# ---------------------------------------------------------------------------


def with_device_index(device: torch.device) -> torch.device:
    """Give a CUDA device an explicit index, e.g. ``cuda`` -> ``cuda:0``.

    Several torch APIs disagree about whether an index is optional.
    ``torch.cuda.memory_allocated()`` resolves ``cuda`` to the current device,
    but ``torch.cuda.mem_get_info()`` raises::

        ValueError: Expected a torch.device with a specified index or an
        integer, but got:cuda

    So an index-less device works right up until the first call that needs one,
    which is a confusing place to fail. Normalising once at creation avoids
    the whole class of problem.

    Non-CUDA devices are returned unchanged.
    """
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    return device


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Pick a device. ``None`` means "GPU if available, else CPU".

    Any CUDA device is returned with an explicit index -- see
    :func:`with_device_index`.
    """
    if requested is not None:
        return with_device_index(torch.device(requested))
    return with_device_index(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype name, downgrading unsupported requests with a warning.

    bfloat16 is preferred over float16 at equal memory cost: it has the same
    exponent range as float32, so it does not overflow on the large
    intermediate values transformers produce. float16 can silently produce
    ``inf`` in a long forward pass.

    On CPU, float32 is forced. CPU bf16/fp16 support is patchy and extremely
    slow, and every CPU use here is a small test where speed matters more than
    memory.
    """
    if device.type == "cpu":
        if requested != "float32":
            logger.info("Device is CPU; using float32 instead of %s.", requested)
        return torch.float32

    if requested == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        logger.warning(
            "bfloat16 requested but this GPU does not support it; falling back to "
            "float16. Watch for inf/nan in long sequences."
        )
        return torch.float16
    return {"float16": torch.float16, "float32": torch.float32}[requested]


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """GPU memory at a point in time. All zeros on CPU."""

    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    device: str

    @property
    def used_fraction(self) -> float:
        return 0.0 if self.total_gb == 0 else (self.total_gb - self.free_gb) / self.total_gb

    def summary(self) -> str:
        if self.total_gb == 0:
            return f"{self.device}: memory tracking not available"
        return (
            f"{self.device}: {self.allocated_gb:.2f} GB allocated, "
            f"{self.reserved_gb:.2f} GB reserved, "
            f"{self.free_gb:.2f} GB free of {self.total_gb:.2f} GB "
            f"({self.used_fraction:.0%} used)"
        )


def memory_snapshot(device: torch.device) -> MemorySnapshot:
    """GPU memory usage. Safe to call with a CPU device or an index-less one."""
    if device.type != "cuda":
        return MemorySnapshot(0.0, 0.0, 0.0, 0.0, str(device))
    # Defensive: callers may construct a device themselves rather than going
    # through resolve_device(), and mem_get_info() rejects an index-less one.
    device = with_device_index(device)
    free, total = torch.cuda.mem_get_info(device)
    return MemorySnapshot(
        allocated_gb=torch.cuda.memory_allocated(device) / 1e9,
        reserved_gb=torch.cuda.memory_reserved(device) / 1e9,
        free_gb=free / 1e9,
        total_gb=total / 1e9,
        device=str(device),
    )


def free_gpu_memory(device: torch.device) -> None:
    """Release cached blocks back to the driver.

    ``del model`` alone is not enough: Python may not have collected the object
    yet, and PyTorch keeps freed blocks in its own cache. Both have to be
    prodded, in this order, or the next load sees no extra space.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model(
    source: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cache_dir: Optional[str] = None,
) -> torch.nn.Module:
    """Load a causal language model, ready for inference.

    Args:
        source: a HuggingFace repo id, or a path to a local directory.
        device: where to put it. The whole model goes here -- never split.
        dtype: weight precision.
        cache_dir: where downloads are stored.

    Returns:
        The model in eval mode with gradients disabled.
    """
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    logger.info("Loading %s (dtype=%s, device=%s)", source, dtype, device)

    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    logger.info("Loaded %s in %.1fs", source, time.time() - t0)
    return model


def load_tokenizer(source: str, *, cache_dir: Optional[str] = None):
    """Load a tokenizer, ensuring a padding token exists.

    Many Llama-family tokenizers ship without a pad token. Batched inference
    then fails, or worse, pads with token 0 and silently corrupts positions.
    We alias pad to eos, which is the standard fix.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        logger.info("Tokenizer had no pad token; using eos (%r).", tok.eos_token)
    return tok


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------


@dataclass
class LoadEvent:
    """One load or free, recorded for the run log."""

    action: str
    split: str
    seconds: float
    allocated_after_gb: float


class ModelManager:
    """Loads TOFU model splits within a memory budget.

    Use :meth:`acquire` as a context manager. Under
    :data:`~deeperase.config.ExecutionStrategy.ALL_RESIDENT` models are cached
    and reused; under ``SEQUENTIAL`` each is freed when the block exits.

    Example::

        mgr = ModelManager(RunConfig(size_label="1B"))
        with mgr.acquire("retain90") as m:
            ...
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        device: Optional[str] = None,
        model_sources: Optional[Dict[str, str]] = None,
    ):
        problems = config.validate()
        if problems:
            raise ValueError(f"Invalid RunConfig: {problems}")

        self.config = config
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.strategy = config.strategy or ExecutionStrategy.ALL_RESIDENT
        self.cache_dir = config.cache_dir

        #: Override where each split is loaded from. Used by tests to point at
        #: tiny local models instead of downloading real ones.
        self.model_sources = model_sources or {
            split: spec.repo_id for split, spec in config.models().items()
        }

        self._cache: Dict[str, torch.nn.Module] = {}
        self.events: List[LoadEvent] = []

    # -- introspection ------------------------------------------------------

    @property
    def resident_splits(self) -> List[str]:
        return sorted(self._cache)

    def memory(self) -> MemorySnapshot:
        return memory_snapshot(self.device)

    def _record(self, action: str, split: str, seconds: float) -> None:
        self.events.append(
            LoadEvent(action, split, seconds, self.memory().allocated_gb)
        )

    # -- core ---------------------------------------------------------------

    def _load(self, split: str) -> torch.nn.Module:
        if split not in self.model_sources:
            raise KeyError(
                f"Unknown split {split!r}. Available: {sorted(self.model_sources)}"
            )
        t0 = time.time()
        model = load_model(
            self.model_sources[split],
            device=self.device,
            dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        self._record("load", split, time.time() - t0)
        return model

    def free(self, split: str) -> bool:
        """Free one cached model. Returns True if something was freed."""
        if split not in self._cache:
            return False
        t0 = time.time()
        model = self._cache.pop(split)
        model.to("cpu")   # release the GPU allocation before dropping the ref
        del model
        free_gpu_memory(self.device)
        self._record("free", split, time.time() - t0)
        logger.info("Freed %s -- %s", split, self.memory().summary())
        return True

    def free_all(self) -> int:
        n = 0
        for split in list(self._cache):
            n += int(self.free(split))
        return n

    @contextmanager
    def acquire(self, split: str) -> Iterator[torch.nn.Module]:
        """Get a model, freeing it afterwards if the strategy requires it.

        Reusing an already-resident model does not reload it, under either
        strategy -- so nesting two ``acquire`` calls on the same split is safe
        and cheap.
        """
        already_resident = split in self._cache
        if not already_resident:
            self._cache[split] = self._load(split)

        try:
            yield self._cache[split]
        finally:
            # Only free what this call brought in. Freeing a model another
            # frame is still using would be a use-after-free.
            if self.strategy is ExecutionStrategy.SEQUENTIAL and not already_resident:
                self.free(split)

    def preload(self, splits: List[str]) -> None:
        """Load several models up front. All-resident only.

        Raises under SEQUENTIAL, where holding several models at once is
        exactly what the strategy exists to prevent.
        """
        if self.strategy is not ExecutionStrategy.ALL_RESIDENT:
            raise RuntimeError(
                f"preload() requires ALL_RESIDENT, but strategy is {self.strategy}. "
                "Under SEQUENTIAL, use acquire() one split at a time."
            )
        for split in splits:
            if split not in self._cache:
                self._cache[split] = self._load(split)

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *exc) -> None:
        self.free_all()

    def load_report(self) -> dict:
        loads = [e for e in self.events if e.action == "load"]
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "strategy": str(self.strategy),
            "n_loads": len(loads),
            "n_frees": sum(1 for e in self.events if e.action == "free"),
            "total_load_seconds": round(sum(e.seconds for e in loads), 2),
            "peak_allocated_gb": round(
                max((e.allocated_after_gb for e in self.events), default=0.0), 3
            ),
            "resident_at_end": self.resident_splits,
        }


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def make_tiny_model_dir(
    path: str | Path,
    *,
    seed: int = 0,
    n_layers: int = 4,
    hidden: int = 32,
    vocab: int = 64,
) -> str:
    """Write a tiny randomly-initialised Llama-style model to disk.

    Lets the loading and lifecycle logic be tested on CPU without downloading
    real checkpoints. Returns the directory path, usable as a ``source``.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, use_cache=False,
    )
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    LlamaForCausalLM(cfg).save_pretrained(path)
    return str(path)
