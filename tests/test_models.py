"""Tests for model loading and GPU memory lifecycle.

These run on CPU using tiny locally-saved models, so the lifecycle logic is
verified without downloading real checkpoints. The behaviour under test --
when a model is loaded, cached, or freed -- is device-independent; only the
memory *numbers* differ on a GPU.

The invariant that matters most: under SEQUENTIAL, a model must not stay
resident after its block exits. If it does, a 3B run on a 20 GB card will hit
out-of-memory on the second model.
"""

from __future__ import annotations

import pytest
import torch

from deeperase.config import ExecutionStrategy, RunConfig
from deeperase.models import (
    MemorySnapshot,
    with_device_index,
    ModelManager,
    free_gpu_memory,
    load_model,
    load_tokenizer,
    make_tiny_model_dir,
    memory_snapshot,
    resolve_device,
    resolve_dtype,
)


@pytest.fixture(scope="module")
def tiny_dirs(tmp_path_factory):
    """Four tiny models standing in for the four TOFU splits."""
    base = tmp_path_factory.mktemp("tiny_models")
    return {
        split: make_tiny_model_dir(base / split, seed=i)
        for i, split in enumerate(["full", "retain90", "retain95", "retain99"])
    }


def _manager(tiny_dirs, strategy, **kw):
    cfg = RunConfig(size_label="1B", strategy=strategy, **kw)
    return ModelManager(cfg, device="cpu", model_sources=tiny_dirs)


class TestDeviceAndDtype:
    def test_resolve_device_explicit(self):
        assert resolve_device("cpu").type == "cpu"

    def test_resolve_device_auto_matches_availability(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert resolve_device(None).type == expected

    def test_cpu_forces_float32(self):
        """bf16/fp16 on CPU is slow and patchily supported."""
        for req in ("bfloat16", "float16", "float32"):
            assert resolve_dtype(req, torch.device("cpu")) is torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_gpu_dtype_resolution(self):
        d = torch.device("cuda")
        assert resolve_dtype("float16", d) is torch.float16
        got = resolve_dtype("bfloat16", d)
        assert got is (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


class TestDeviceIndex:
    """Regression tests for a real crash on the GPU server.

    ``resolve_device()`` returned ``torch.device("cuda")`` with no index.
    ``torch.cuda.memory_allocated()`` tolerates that, but
    ``torch.cuda.mem_get_info()`` raises::

        ValueError: Expected a torch.device with a specified index or an
        integer, but got:cuda

    So the code ran fine right up to the first memory snapshot, then died
    after a four-minute model download.

    These tests run on CPU: ``torch.device("cuda")`` is just a value object and
    can be constructed without a GPU present, so the normalisation logic is
    fully testable here. The original gap was that every model test used
    ``device="cpu"``, where ``memory_snapshot`` returns early and never
    reaches the failing call.
    """

    def test_cuda_device_gets_an_index(self):
        got = with_device_index(torch.device("cuda"))
        assert got.type == "cuda"
        assert got.index is not None, "an index-less cuda device breaks mem_get_info()"

    def test_existing_index_is_preserved(self):
        assert with_device_index(torch.device("cuda", 3)).index == 3

    def test_cpu_device_untouched(self):
        d = torch.device("cpu")
        assert with_device_index(d) == d

    def test_resolve_device_never_returns_indexless_cuda(self):
        """The source of the bug: whatever resolve_device hands back must be
        safe to pass to every torch.cuda memory API."""
        for requested in (None, "cuda", "cuda:0", "cpu"):
            if requested and "cuda" in requested and not torch.cuda.is_available():
                # Constructing is fine without a GPU; only .to() would fail.
                pass
            d = resolve_device(requested)
            if d.type == "cuda":
                assert d.index is not None, f"resolve_device({requested!r}) lost the index"

    def test_snapshot_normalises_before_calling_torch(self, monkeypatch):
        """``memory_snapshot`` must add the index before calling torch.

        Verified without a GPU by intercepting the torch calls and recording
        which device object they receive. This is the exact call that crashed
        on the server, so it is worth pinning even on a CPU-only machine.
        """
        seen = {}

        def fake_mem_get_info(device):
            seen["device"] = device
            return (1_000_000_000, 2_000_000_000)

        monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 0)

        snap = memory_snapshot(torch.device("cuda"))

        assert seen["device"].index is not None, (
            "mem_get_info was handed an index-less device -- this is the bug "
            "that killed the run after a four-minute download"
        )
        assert snap.total_gb == pytest.approx(2.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_real_gpu_snapshot_reports_memory(self):
        """On a real GPU the snapshot must return non-zero totals."""
        snap = memory_snapshot(torch.device("cuda"))
        assert snap.total_gb > 0
        assert 0.0 <= snap.used_fraction <= 1.0


class TestMemorySnapshot:
    def test_cpu_snapshot_is_zeroed_not_crashing(self):
        s = memory_snapshot(torch.device("cpu"))
        assert isinstance(s, MemorySnapshot)
        assert s.total_gb == 0.0
        assert "not available" in s.summary()

    def test_used_fraction_safe_when_total_zero(self):
        assert memory_snapshot(torch.device("cpu")).used_fraction == 0.0

    def test_free_memory_is_safe_on_cpu(self):
        free_gpu_memory(torch.device("cpu"))   # must not raise


class TestLoading:
    def test_loads_from_local_dir(self, tiny_dirs):
        m = load_model(tiny_dirs["full"], device=torch.device("cpu"), dtype=torch.float32)
        assert isinstance(m, torch.nn.Module)

    def test_loaded_model_is_in_eval_mode(self, tiny_dirs):
        """Contract test: models handed out must be in eval mode.

        Note honestly: `from_pretrained` already returns an eval-mode model, so
        our explicit `model.eval()` is currently redundant and mutating it away
        does not fail this test. The assertion is kept because it pins the
        *invariant* rather than our implementation of it -- if a future
        transformers release changed that default, our line would still hold
        the contract. It is defence in depth, not dead code.
        """
        m = load_model(tiny_dirs["full"], device=torch.device("cpu"), dtype=torch.float32)
        assert m.training is False

    def test_gradients_are_disabled(self, tiny_dirs):
        """Inference only. Leaving grads on roughly doubles activation memory."""
        m = load_model(tiny_dirs["full"], device=torch.device("cpu"), dtype=torch.float32)
        assert not any(p.requires_grad for p in m.parameters())

    def test_dtype_is_applied(self, tiny_dirs):
        m = load_model(tiny_dirs["full"], device=torch.device("cpu"), dtype=torch.float32)
        assert next(m.parameters()).dtype is torch.float32

    def test_all_parameters_on_one_device(self, tiny_dirs):
        """Split placement would silently break activation patching."""
        m = load_model(tiny_dirs["full"], device=torch.device("cpu"), dtype=torch.float32)
        assert len({p.device for p in m.parameters()}) == 1

    def test_different_sources_give_different_weights(self, tiny_dirs):
        a = load_model(tiny_dirs["full"], device=torch.device("cpu"), dtype=torch.float32)
        b = load_model(tiny_dirs["retain90"], device=torch.device("cpu"), dtype=torch.float32)
        pa = next(a.parameters()).flatten()[:20]
        pb = next(b.parameters()).flatten()[:20]
        assert not torch.allclose(pa, pb)


class TestAllResidentStrategy:
    def test_model_stays_cached_after_block(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        with mgr.acquire("full"):
            pass
        assert mgr.resident_splits == ["full"]

    def test_second_acquire_does_not_reload(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        with mgr.acquire("full"):
            pass
        with mgr.acquire("full"):
            pass
        assert sum(1 for e in mgr.events if e.action == "load") == 1

    def test_returns_the_same_object(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        with mgr.acquire("full") as a:
            pass
        with mgr.acquire("full") as b:
            assert a is b

    def test_three_models_can_be_resident(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        for s in ("full", "retain90", "retain95"):
            with mgr.acquire(s):
                pass
        assert mgr.resident_splits == ["full", "retain90", "retain95"]

    def test_preload_works(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        mgr.preload(["full", "retain90"])
        assert len(mgr.resident_splits) == 2


class TestSequentialStrategy:
    def test_model_is_freed_after_block(self, tiny_dirs):
        """The core memory guarantee. If this fails, 3B on 20 GB will OOM."""
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        with mgr.acquire("full"):
            assert mgr.resident_splits == ["full"]
        assert mgr.resident_splits == []

    def test_only_one_resident_across_a_sequence(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        peak = 0
        for s in ("full", "retain90", "retain95", "retain99"):
            with mgr.acquire(s):
                peak = max(peak, len(mgr.resident_splits))
        assert peak == 1, "sequential mode must never hold two models at once"

    def test_reloads_each_time(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        for _ in range(3):
            with mgr.acquire("full"):
                pass
        assert sum(1 for e in mgr.events if e.action == "load") == 3

    def test_nested_acquire_does_not_free_early(self, tiny_dirs):
        """The inner block must not free a model the outer block still holds."""
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        with mgr.acquire("full") as outer:
            with mgr.acquire("full") as inner:
                assert inner is outer
            assert mgr.resident_splits == ["full"], "inner exit freed the outer model"
        assert mgr.resident_splits == []

    def test_preload_is_rejected(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        with pytest.raises(RuntimeError, match="ALL_RESIDENT"):
            mgr.preload(["full", "retain90"])

    def test_freed_model_is_usable_again(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        ids = torch.randint(0, 64, (1, 8))
        outs = []
        for _ in range(2):
            with mgr.acquire("full") as m:
                outs.append(m(input_ids=ids).logits.clone())
        assert torch.allclose(outs[0], outs[1], atol=1e-5), \
            "reloading the same checkpoint must give identical results"


class TestFreeingAndCleanup:
    def test_free_returns_false_when_not_resident(self, tiny_dirs):
        assert _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT).free("full") is False

    def test_free_all_clears_everything(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        mgr.preload(["full", "retain90", "retain95"])
        assert mgr.free_all() == 3
        assert mgr.resident_splits == []

    def test_context_manager_frees_on_exit(self, tiny_dirs):
        cfg = RunConfig(size_label="1B", strategy=ExecutionStrategy.ALL_RESIDENT)
        with ModelManager(cfg, device="cpu", model_sources=tiny_dirs) as mgr:
            mgr.preload(["full", "retain90"])
            assert len(mgr.resident_splits) == 2
        assert mgr.resident_splits == []

    def test_frees_even_if_body_raises(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        with pytest.raises(RuntimeError, match="boom"):
            with mgr.acquire("full"):
                raise RuntimeError("boom")
        assert mgr.resident_splits == [], "an exception must not leak a resident model"


class TestErrorsAndReporting:
    def test_unknown_split_raises_with_options(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT)
        with pytest.raises(KeyError, match="retain90"):
            with mgr.acquire("nonexistent"):
                pass

    def test_invalid_config_rejected_at_construction(self, tiny_dirs):
        bad = RunConfig(forget_split="forget10", stage1_source_split="retain95")
        with pytest.raises(ValueError, match="Invalid RunConfig"):
            ModelManager(bad, device="cpu", model_sources=tiny_dirs)

    def test_load_report_counts_correctly(self, tiny_dirs):
        mgr = _manager(tiny_dirs, ExecutionStrategy.SEQUENTIAL)
        for s in ("full", "retain90"):
            with mgr.acquire(s):
                pass
        rep = mgr.load_report()
        assert rep["n_loads"] == 2 and rep["n_frees"] == 2
        assert rep["resident_at_end"] == []
        assert rep["strategy"] == "sequential"

    def test_report_records_device_and_dtype(self, tiny_dirs):
        rep = _manager(tiny_dirs, ExecutionStrategy.ALL_RESIDENT).load_report()
        assert rep["device"] == "cpu" and "float32" in rep["dtype"]


class TestTokenizerHelper:
    def test_pad_token_is_ensured(self, tiny_dirs, tmp_path):
        """Llama tokenizers often lack a pad token; batching then breaks."""
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        except Exception:
            pytest.skip("no network for tokenizer download")
        tok.pad_token = None
        d = tmp_path / "tok"
        tok.save_pretrained(d)
        loaded = load_tokenizer(str(d))
        assert loaded.pad_token is not None
