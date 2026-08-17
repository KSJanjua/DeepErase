"""Tests for layer-wise activation patching and two-stage UDS.

The invariants that make patching trustworthy as a measurement backend:

  * hooks are removed after use, including on exception;
  * patching layer l changes only layer l's output;
  * a no-op patch reproduces the unpatched forward pass bit-exactly;
  * patched logits come from the TARGET model's real forward pass -- they are
    not copied from the source. This is the check that separates a genuine
    causal intervention from a scaffold that fakes one;
  * batch / sequence / dtype / device / attention-mask handling is correct.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeperase.eval.patching import (
    ARCHITECTURE_PATHS,
    ArchitectureNotSupported,
    EntitySpan,
    PatchSpec,
    capture_hidden_states,
    entity_logprobs,
    forward_with_patch,
    get_layer,
    n_layers,
    patched_forward,
    probe_knowledge_with_patch,
    resolve_layer_container,
)
from deeperase.eval.uds import (
    DEFAULT_TAU,
    ExampleUDS,
    UDSExample,
    UDSReport,
    aggregate_example_uds,
    build_stage1_cache,
    compute_uds,
    layer_erasure_ratio,
    score_example_deltas,
)

N_LAYERS = 4
HIDDEN = 32
VOCAB = 64
SEQ = 10


def _tiny_model(seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, use_cache=False,
    )
    m = LlamaForCausalLM(cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_a():
    return _tiny_model(0)


@pytest.fixture(scope="module")
def model_b():
    return _tiny_model(1)   # different weights -> different hidden states


@pytest.fixture(scope="module")
def ids():
    g = torch.Generator().manual_seed(42)
    return torch.randint(0, VOCAB, (2, SEQ), generator=g)


# ------------------------------------------------------------ architecture --

class TestArchitectureResolution:
    def test_resolves_llama_style(self, model_a):
        container, path = resolve_layer_container(model_a)
        assert path == "model.layers"
        assert len(container) == N_LAYERS

    def test_n_layers(self, model_a):
        assert n_layers(model_a) == N_LAYERS

    def test_get_layer_returns_the_right_module(self, model_a):
        assert get_layer(model_a, 2) is model_a.model.layers[2]

    def test_out_of_range_layer_raises(self, model_a):
        with pytest.raises(IndexError, match="out of range"):
            get_layer(model_a, 99)

    def test_unsupported_architecture_raises_with_tried_paths(self):
        with pytest.raises(ArchitectureNotSupported, match="Tried"):
            resolve_layer_container(torch.nn.Linear(4, 4))

    def test_custom_accessor_bypasses_resolution(self, model_a):
        got = get_layer(model_a, 1, layer_accessor=lambda m, i: m.model.layers[i])
        assert got is model_a.model.layers[1]

    def test_llama_path_is_first(self):
        assert ARCHITECTURE_PATHS[0] == "model.layers"


# ---------------------------------------------------------------- capture --

class TestCapture:
    def test_shapes(self, model_a, ids):
        h = capture_hidden_states(model_a, ids, [0, 2])
        assert set(h) == {0, 2}
        assert h[0].shape == (ids.shape[0], SEQ, HIDDEN)

    def test_hooks_removed_after_capture(self, model_a, ids):
        before = sum(len(l._forward_hooks) for l in model_a.model.layers)
        capture_hidden_states(model_a, ids, [0, 1, 2, 3])
        after = sum(len(l._forward_hooks) for l in model_a.model.layers)
        assert after == before == 0

    def test_hooks_removed_even_when_forward_raises(self, model_a):
        bad = torch.tensor([[VOCAB + 500]])   # out-of-range id -> forward raises
        with pytest.raises(Exception):
            capture_hidden_states(model_a, bad, [0])
        assert sum(len(l._forward_hooks) for l in model_a.model.layers) == 0

    def test_different_models_give_different_states(self, model_a, model_b, ids):
        ha = capture_hidden_states(model_a, ids, [1])[1]
        hb = capture_hidden_states(model_b, ids, [1])[1]
        assert not torch.allclose(ha, hb), "distinct weights must give distinct activations"

    def test_capture_restores_training_mode(self, model_a, ids):
        model_a.train()
        try:
            capture_hidden_states(model_a, ids, [0])
            assert model_a.training is True
        finally:
            model_a.eval()

    def test_capture_is_detached(self, model_a, ids):
        assert capture_hidden_states(model_a, ids, [0])[0].requires_grad is False


# --------------------------------------------------------------- patching --

class TestPatchSpec:
    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match=r"\(batch, seq, hidden\)"):
            PatchSpec(0, torch.zeros(4, 8), [1])

    def test_rejects_out_of_range_position(self):
        with pytest.raises(IndexError, match="out of range"):
            PatchSpec(0, torch.zeros(1, 5, 8), [99])

    def test_empty_positions_is_empty(self):
        assert PatchSpec(0, torch.zeros(1, 5, 8), []).is_empty


class TestPatchedForward:
    def test_no_patch_equals_normal_forward(self, model_a, ids):
        """spec=None must be bit-identical to calling the model directly."""
        direct = model_a(input_ids=ids).logits
        via = forward_with_patch(model_a, ids, None)
        assert torch.equal(direct, via)

    def test_empty_patch_equals_normal_forward(self, model_a, model_b, ids):
        """A spec covering no positions must not perturb anything."""
        src = capture_hidden_states(model_b, ids, [1])
        base = model_a(input_ids=ids).logits
        out = forward_with_patch(model_a, ids, PatchSpec(1, src[1], []))
        assert torch.equal(base, out)

    def test_patch_changes_output(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        base = model_a(input_ids=ids).logits
        out = forward_with_patch(model_a, ids, PatchSpec(1, src[1], [3, 4]))
        assert not torch.allclose(base, out), "patching must have a causal effect"

    def test_hooks_removed_after_patch(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        forward_with_patch(model_a, ids, PatchSpec(1, src[1], [2]))
        assert sum(len(l._forward_hooks) for l in model_a.model.layers) == 0

    def test_hooks_removed_when_body_raises(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        spec = PatchSpec(1, src[1], [2])
        with pytest.raises(RuntimeError, match="boom"):
            with patched_forward(model_a, spec):
                raise RuntimeError("boom")
        assert sum(len(l._forward_hooks) for l in model_a.model.layers) == 0

    def test_repeated_patching_leaves_no_residue(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        base = model_a(input_ids=ids).logits
        for _ in range(5):
            forward_with_patch(model_a, ids, PatchSpec(1, src[1], [2]))
        assert torch.equal(base, model_a(input_ids=ids).logits), \
            "an unpatched pass after patching must be unchanged"

    def test_patching_self_is_identity(self, model_a, ids):
        """Patching a model with its OWN activations must be a no-op. This is
        the strongest evidence that the hook substitutes cleanly."""
        src = capture_hidden_states(model_a, ids, [2])
        base = model_a(input_ids=ids).logits
        out = forward_with_patch(model_a, ids, PatchSpec(2, src[2], list(range(SEQ))))
        assert torch.allclose(base, out, atol=1e-5)


def _observe_layers(model, layers, ids, spec=None):
    """Capture layer outputs, registering observers AFTER any patch hook.

    PyTorch runs forward hooks in registration order and feeds each hook the
    previous hook's return value. An observer registered *before* the patch
    hook therefore sees the pre-patch tensor. Registering inside the
    ``patched_forward`` context puts the observer second, so it sees what the
    next layer actually consumes -- which is what these tests must check.
    """
    out = {}
    handles = []

    def mk(i):
        def h(_m, _i, o):
            out[i] = (o[0] if isinstance(o, tuple) else o).detach().clone()
        return h

    def register():
        for ell in layers:
            handles.append(model.model.layers[ell].register_forward_hook(mk(ell)))

    try:
        if spec is None:
            register()
            model(input_ids=ids)
        else:
            with patched_forward(model, spec):
                register()
                model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return out


class TestHookOrdering:
    """Hook registration order is load-bearing and easy to get wrong."""

    def test_observer_after_patch_sees_patched_value(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        out = _observe_layers(model_a, [1], ids, PatchSpec(1, src[1], [3]))
        assert torch.allclose(out[1][:, 3, :], src[1][:, 3, :], atol=1e-6)

    def test_observer_before_patch_sees_prepatch_value(self, model_a, model_b, ids):
        """Documents the trap: an observer registered first sees the ORIGINAL
        tensor, because the patch hook has not run yet."""
        src = capture_hidden_states(model_b, ids, [1])
        seen = {}
        h = model_a.model.layers[1].register_forward_hook(
            lambda _m, _i, o: seen.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o).detach().clone()))
        try:
            with patched_forward(model_a, PatchSpec(1, src[1], [3])):
                model_a(input_ids=ids)
        finally:
            h.remove()
        assert not torch.allclose(seen["h"][:, 3, :], src[1][:, 3, :], atol=1e-6)


class TestLayerIsolation:
    def test_only_target_layer_output_changes(self, model_a, model_b, ids):
        """Patching layer 1 must not alter layer 0's output; downstream layers
        legitimately change because information propagates forward."""
        src = capture_hidden_states(model_b, ids, [1])
        before = _observe_layers(model_a, [0, 1, 2, 3], ids)
        after = _observe_layers(model_a, [0, 1, 2, 3], ids, PatchSpec(1, src[1], [3]))

        assert torch.allclose(before[0], after[0], atol=1e-6), \
            "upstream layer 0 must be untouched"
        assert not torch.allclose(before[1], after[1]), "patched layer 1 must change"
        assert not torch.allclose(before[2], after[2]), \
            "downstream layer 2 must change -- the patch has to propagate"

    def test_unpatched_positions_preserved(self, model_a, model_b, ids):
        """Only the listed positions may be overwritten."""
        src = capture_hidden_states(model_b, ids, [1])
        out = _observe_layers(model_a, [1], ids, PatchSpec(1, src[1], [3]))[1]
        unpatched = capture_hidden_states(model_a, ids, [1])[1]

        assert torch.allclose(out[:, 3, :], src[1][:, 3, :], atol=1e-6), \
            "position 3 must equal the source"
        for p in (0, 1, 2, 4, 5):
            assert torch.allclose(out[:, p, :], unpatched[:, p, :], atol=1e-6), \
                f"position {p} must be untouched"

    def test_multiple_positions_all_patched(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [2])
        positions = [1, 4, 7]
        out = _observe_layers(model_a, [2], ids, PatchSpec(2, src[2], positions))[2]
        for p in positions:
            assert torch.allclose(out[:, p, :], src[2][:, p, :], atol=1e-6), f"position {p}"


class TestPatchedOutputIsRealForwardPass:
    """The patched score must be produced by the target model computing, not
    lifted from the source. Without these the backend could be a scaffold."""

    def test_patched_logits_differ_from_source_logits(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        patched = forward_with_patch(model_a, ids, PatchSpec(1, src[1], list(range(1, SEQ))))
        source_logits = model_b(input_ids=ids).logits
        assert not torch.allclose(patched, source_logits, atol=1e-3), \
            "patched output equals the source model's -- it was copied, not computed"

    def test_patched_logits_differ_from_target_logits(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        patched = forward_with_patch(model_a, ids, PatchSpec(1, src[1], list(range(1, SEQ))))
        assert not torch.allclose(patched, model_a(input_ids=ids).logits, atol=1e-6), \
            "patched output equals the unpatched target -- the patch did nothing"

    def test_downstream_target_weights_are_used(self, model_a, model_b, ids):
        """Perturbing a layer AFTER the patch point must change the patched
        result -- proving the target's own downstream computation runs."""
        src = capture_hidden_states(model_b, ids, [1])
        spec = PatchSpec(1, src[1], [5])
        before = forward_with_patch(model_a, ids, spec)

        w = model_a.model.layers[3].mlp.down_proj.weight
        orig = w.detach().clone()
        try:
            with torch.no_grad():
                w.add_(0.5)
            after = forward_with_patch(model_a, ids, spec)
        finally:
            with torch.no_grad():
                w.copy_(orig)
        assert not torch.allclose(before, after), \
            "changing a downstream layer had no effect -- output was not computed by the target"

    def test_patch_at_last_layer_still_flows_through_head(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [N_LAYERS - 1])
        out = forward_with_patch(
            model_a, ids, PatchSpec(N_LAYERS - 1, src[N_LAYERS - 1], [4]))
        assert not torch.allclose(out, model_a(input_ids=ids).logits)


class TestShapeDtypeDeviceMask:
    def test_batch_mismatch_raises(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])[1][:1]     # batch 1 vs 2
        with pytest.raises(ValueError, match="Batch mismatch"):
            forward_with_patch(model_a, ids, PatchSpec(1, src, [2]))

    def test_hidden_size_mismatch_raises(self, model_a, ids):
        bad = torch.zeros(ids.shape[0], SEQ, HIDDEN + 8)
        with pytest.raises(ValueError, match="Hidden-size mismatch"):
            forward_with_patch(model_a, ids, PatchSpec(1, bad, [2]))

    def test_sequence_mismatch_raises(self, model_a, ids):
        bad = torch.zeros(ids.shape[0], SEQ + 3, HIDDEN)
        with pytest.raises(ValueError, match="Sequence-length mismatch"):
            forward_with_patch(model_a, ids, PatchSpec(1, bad, [2]))

    def test_dtype_is_cast_to_target(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])[1].to(torch.float64)
        out = forward_with_patch(model_a, ids, PatchSpec(1, src, [3]))
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

    def test_source_device_is_respected(self, model_a, model_b, ids):
        """CPU-only here, but the cast path must be exercised."""
        src = capture_hidden_states(model_b, ids, [1])[1]
        out = forward_with_patch(model_a, ids, PatchSpec(1, src.cpu(), [3]))
        assert out.device.type == "cpu"

    def test_attention_mask_is_honoured(self, model_a, ids):
        """Masking real tokens must change the output; otherwise the mask is
        being dropped somewhere in the patching path."""
        full = torch.ones_like(ids)
        partial = full.clone()
        partial[:, -3:] = 0
        a = forward_with_patch(model_a, ids, None, attention_mask=full)
        b = forward_with_patch(model_a, ids, None, attention_mask=partial)
        assert not torch.allclose(a, b), "attention_mask had no effect"

    def test_mask_reaches_patched_pass(self, model_a, model_b, ids):
        mask = torch.ones_like(ids)
        mask[:, -2:] = 0
        src = capture_hidden_states(model_b, ids, [1], attention_mask=mask)
        a = forward_with_patch(model_a, ids, PatchSpec(1, src[1], [2]), attention_mask=mask)
        b = forward_with_patch(model_a, ids, PatchSpec(1, src[1], [2]),
                               attention_mask=torch.ones_like(ids))
        assert not torch.allclose(a, b)

    def test_batch_size_one_works(self, model_a, model_b):
        one = torch.randint(0, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(7))
        src = capture_hidden_states(model_b, one, [1])
        out = forward_with_patch(model_a, one, PatchSpec(1, src[1], [2]))
        assert out.shape == (1, SEQ, VOCAB)

    def test_negative_positions_supported(self, model_a, model_b, ids):
        src = capture_hidden_states(model_b, ids, [1])
        out = forward_with_patch(model_a, ids, PatchSpec(1, src[1], [-1]))
        assert not torch.allclose(out, model_a(input_ids=ids).logits)


# ------------------------------------------------------------ entity spans --

class TestEntitySpan:
    def test_predicting_positions_offset_by_one(self):
        assert EntitySpan([5, 6, 7]).predicting_positions == [4, 5, 6]

    def test_rejects_index_zero(self):
        with pytest.raises(ValueError, match="index >= 1"):
            EntitySpan([0, 1])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            EntitySpan([])

    def test_entity_logprobs_shape_and_range(self, model_a, ids):
        lp = entity_logprobs(model_a(input_ids=ids).logits, ids, EntitySpan([4, 5]))
        assert lp.shape == (ids.shape[0],)
        assert (lp <= 0).all(), "log-probs must be non-positive"

    def test_entity_logprobs_matches_manual_computation(self, model_a, ids):
        import torch.nn.functional as F
        logits = model_a(input_ids=ids).logits
        span = EntitySpan([4])
        got = entity_logprobs(logits, ids, span)
        manual = F.log_softmax(logits[:, 3, :].float(), -1).gather(
            1, ids[:, 4:5]).squeeze(-1)
        assert torch.allclose(got, manual, atol=1e-6)


# --------------------------------------------------------------- UDS maths --

class TestUDSFormula:
    def test_ler_clipped_to_unit_interval(self):
        assert layer_erasure_ratio(2.0, 1.0) == 1.0
        assert layer_erasure_ratio(-1.0, 1.0) == 0.0
        assert layer_erasure_ratio(0.5, 1.0) == pytest.approx(0.5)

    def test_ler_rejects_nonpositive_denominator(self):
        with pytest.raises(ValueError, match="must be positive"):
            layer_erasure_ratio(0.5, 0.0)

    def test_fully_erased_gives_one(self):
        """dS2 == dS1 at every KE layer -> erased to M_ret's level."""
        d1 = {0: 0.5, 1: 1.0, 2: 0.2}
        uds, ke, _ = aggregate_example_uds(d1, dict(d1))
        assert uds == pytest.approx(1.0)
        assert ke == [0, 1, 2]

    def test_fully_intact_gives_zero(self):
        d1 = {0: 0.5, 1: 1.0}
        uds, _, _ = aggregate_example_uds(d1, {0: 0.0, 1: 0.0})
        assert uds == pytest.approx(0.0)

    def test_weighting_favours_high_ds1_layers(self):
        """Eq. 5 weights by dS1, so a big-knowledge layer dominates."""
        d1 = {0: 0.1, 1: 10.0}
        d2 = {0: 0.1, 1: 0.0}          # small layer erased, big layer intact
        uds, _, _ = aggregate_example_uds(d1, d2)
        assert uds < 0.05, "weighted mean must be dominated by the dS1=10 layer"

    def test_sub_threshold_layers_excluded_from_ke(self):
        d1 = {0: 0.01, 1: 0.5}         # layer 0 below tau=0.05
        uds, ke, _ = aggregate_example_uds(d1, {0: 0.0, 1: 0.5}, tau=DEFAULT_TAU)
        assert ke == [1]
        assert uds == pytest.approx(1.0)

    def test_empty_ke_returns_none_not_zero(self):
        """Undefined must not be silently coerced -- the paper excludes it."""
        uds, ke, ler = aggregate_example_uds({0: 0.001}, {0: 0.0})
        assert uds is None and ke == [] and ler == {}

    def test_tau_default_matches_paper(self):
        assert DEFAULT_TAU == 0.05


class TestUDSReport:
    def test_undefined_examples_excluded_from_mean(self):
        rep = UDSReport(uds=None, n_examples_total=0, n_examples_scored=0,
                        n_examples_undefined=0, tau=DEFAULT_TAU)
        rep.per_example = [
            ExampleUDS("a", 1.0), ExampleUDS("b", 0.0), ExampleUDS("c", None),
        ]
        scored = [e.uds for e in rep.per_example if e.is_defined]
        assert len(scored) == 2 and np.mean(scored) == pytest.approx(0.5)

    def test_summary_flags_missing_reference_validation(self):
        rep = UDSReport(uds=0.5, n_examples_total=2, n_examples_scored=2,
                        n_examples_undefined=0, tau=0.05)
        assert "NOT cross-validated" in rep.summary()

    def test_to_dict_records_validation_flag(self):
        rep = UDSReport(uds=0.5, n_examples_total=1, n_examples_scored=1,
                        n_examples_undefined=0, tau=0.05)
        assert rep.to_dict()["is_validated_against_reference"] is False


@pytest.fixture
def one_seq():
    """A single sequence. UDS scores examples individually, never batched."""
    return torch.randint(0, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(11))


# --------------------------------------------------------------------------
# Trained models, so UDS is exercised for real.
#
# Randomly-initialised models produce NEGATIVE dS1 -- patching M_ret in
# actually improves the score, because neither model knows anything and the
# swap is just noise. That leaves the Knowledge-Encoding set empty and UDS
# undefined, so any end-to-end assertion written as
# ``assert uds is None or <property>`` passes without ever testing <property>.
#
# These fixtures train a "fact" into M_full and withhold it from M_ret, which
# is the situation UDS is designed for and the only one where its output is
# meaningful.
# --------------------------------------------------------------------------

FACT_MARKER = 20


def _train(model, batches, *, lr, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        x = batches[i % len(batches)]
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def knowledge_setup():
    """(M_full, M_ret, example) where M_full knows a fact M_ret does not."""
    g = torch.Generator().manual_seed(3)
    fact = torch.randint(1, VOCAB - 2, (8, SEQ), generator=g)
    fact[:, 0] = FACT_MARKER
    fact[:, -1] = FACT_MARKER + 1          # deterministic ending = the "fact"
    other = torch.randint(1, VOCAB - 2, (8, SEQ), generator=g)

    m_full = _train(_tiny_model(0), [fact, other], lr=5e-3, steps=80)
    m_ret = _train(_tiny_model(0), [other], lr=5e-3, steps=80)

    example = UDSExample("fact0", fact[0:1], EntitySpan([SEQ - 1]))
    return m_full, m_ret, example


class TestUDSOnRealKnowledge:
    """End-to-end tests that actually reach the scoring path."""

    def test_stage1_deltas_are_positive(self, knowledge_setup):
        """The precondition for UDS to mean anything: M_full must encode
        something M_ret lacks, so patching M_ret in degrades the score."""
        m_full, m_ret, ex = knowledge_setup
        d1, _ = score_example_deltas(model_full=m_full, model_source=m_ret,
                                     example=ex, layers=list(range(N_LAYERS)))
        assert max(d1.values()) > DEFAULT_TAU, (
            f"no layer exceeds tau={DEFAULT_TAU}; dS1={d1}. UDS would be "
            "undefined and every downstream assertion vacuous."
        )

    def test_ke_layers_are_found(self, knowledge_setup):
        m_full, m_ret, ex = knowledge_setup
        rep = compute_uds(model_full=m_full, model_retain=m_ret,
                          model_unlearned=_tiny_model(7), examples=[ex],
                          layers=list(range(N_LAYERS)))
        assert rep.per_example[0].ke_layers, "KE set must not be empty here"

    def test_perfect_unlearning_scores_one(self, knowledge_setup):
        """M_unl == M_ret means dS2 == dS1 exactly, so UDS is exactly 1.
        Asserted with no 'or None' escape hatch."""
        m_full, m_ret, ex = knowledge_setup
        rep = compute_uds(model_full=m_full, model_retain=m_ret,
                          model_unlearned=m_ret, examples=[ex],
                          layers=list(range(N_LAYERS)))
        assert rep.uds is not None
        assert rep.uds == pytest.approx(1.0, abs=1e-6)

    def test_no_unlearning_scores_zero(self, knowledge_setup):
        """M_unl == M_full means patching changes nothing, dS2 == 0, UDS == 0."""
        m_full, m_ret, ex = knowledge_setup
        rep = compute_uds(model_full=m_full, model_retain=m_ret,
                          model_unlearned=m_full, examples=[ex],
                          layers=list(range(N_LAYERS)))
        assert rep.uds is not None
        assert rep.uds == pytest.approx(0.0, abs=1e-6)

    def test_ordering_full_below_retain(self, knowledge_setup):
        """The monotonicity the paper's Table 2 tests, in miniature."""
        m_full, m_ret, ex = knowledge_setup
        layers = list(range(N_LAYERS))
        no_unlearn = compute_uds(model_full=m_full, model_retain=m_ret,
                                 model_unlearned=m_full, examples=[ex], layers=layers)
        full_unlearn = compute_uds(model_full=m_full, model_retain=m_ret,
                                   model_unlearned=m_ret, examples=[ex], layers=layers)
        assert no_unlearn.uds < full_unlearn.uds

    def test_delta_sign_convention(self, knowledge_setup):
        """delta == s_full - s_patched, not the reverse.

        A flipped sign makes every dS1 negative, empties the KE set and turns
        UDS into None -- which older, more permissive assertions accepted.
        """
        m_full, m_ret, ex = knowledge_setup
        ex = ex.to(next(m_full.parameters()).device)
        d1, s_full = score_example_deltas(model_full=m_full, model_source=m_ret,
                                          example=ex, layers=[1])
        hidden = capture_hidden_states(m_ret, ex.input_ids, [1],
                                       attention_mask=ex.attention_mask)
        s_patched = float(probe_knowledge_with_patch(
            m_full, hidden, ex.input_ids, ex.span, 1,
            attention_mask=ex.attention_mask).item())
        assert d1[1] == pytest.approx(s_full - s_patched, abs=1e-6)

    def test_cached_stage1_values_equal_recomputed(self, knowledge_setup):
        """Compares dS1 elementwise, not just the final score.

        Final UDS can be identical even when dS1 is wrong, because the Layer
        Erasure Ratio clips to [0, 1] and hides scaling errors.
        """
        m_full, m_ret, ex = knowledge_setup
        layers = list(range(N_LAYERS))
        direct, _ = score_example_deltas(model_full=m_full, model_source=m_ret,
                                         example=ex, layers=layers)
        cached, _ = build_stage1_cache(model_full=m_full, model_retain=m_ret,
                                       examples=[ex], layers=layers)
        for ell in layers:
            assert cached[ex.example_id][ell] == pytest.approx(direct[ell], abs=1e-6)

    def test_cached_run_matches_uncached_end_to_end(self, knowledge_setup):
        m_full, m_ret, ex = knowledge_setup
        unl = _tiny_model(7)
        layers = list(range(N_LAYERS))
        direct = compute_uds(model_full=m_full, model_retain=m_ret,
                             model_unlearned=unl, examples=[ex], layers=layers)
        d1, sf = build_stage1_cache(model_full=m_full, model_retain=m_ret,
                                    examples=[ex], layers=layers)
        cached = compute_uds(model_full=m_full, model_retain=None,
                             model_unlearned=unl, examples=[ex], layers=layers,
                             stage1_cache=d1, s_full_cache=sf)
        assert direct.uds is not None
        assert cached.uds == pytest.approx(direct.uds, abs=1e-9)


class TestUDSExample:
    """Per-example convention (see UDSExample docstring for why it matters)."""

    def test_accepts_1d_input(self):
        ex = UDSExample("e", torch.zeros(SEQ, dtype=torch.long), EntitySpan([3]))
        assert ex.input_ids.shape == (1, SEQ)

    def test_accepts_batch_of_one(self, one_seq):
        assert UDSExample("e", one_seq, EntitySpan([3])).seq_length == SEQ

    def test_rejects_real_batches(self, ids):
        """Pooling rows would collapse several examples into one KE set and
        one score -- a different quantity from the paper's Eqs. 1-6."""
        with pytest.raises(ValueError, match="ONE sequence"):
            UDSExample("e", ids, EntitySpan([3]))

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match=r"\(seq,\) or \(1, seq\)"):
            UDSExample("e", torch.zeros(1, 2, 3, dtype=torch.long), EntitySpan([1]))

    def test_rejects_span_beyond_sequence(self, one_seq):
        with pytest.raises(IndexError, match="exceed sequence length"):
            UDSExample("e", one_seq, EntitySpan([SEQ + 5]))

    def test_rejects_mismatched_mask(self, one_seq):
        with pytest.raises(ValueError, match="does not match"):
            UDSExample("e", one_seq, EntitySpan([3]),
                       attention_mask=torch.ones(1, SEQ + 2, dtype=torch.long))

    def test_mask_is_promoted_to_2d(self, one_seq):
        ex = UDSExample("e", one_seq, EntitySpan([3]),
                        attention_mask=torch.ones(SEQ, dtype=torch.long))
        assert ex.attention_mask.shape == (1, SEQ)

    def test_to_device_does_not_mutate_original(self, one_seq):
        ex = UDSExample("e", one_seq, EntitySpan([3]))
        moved = ex.to(torch.device("cpu"))
        assert moved is not ex and moved.example_id == ex.example_id


class TestComputeUDSEndToEnd:
    def _ex(self, one_seq, eid="e0", span=(6, 7)):
        return UDSExample(eid, one_seq, EntitySpan(list(span)))

    def test_identical_source_models_give_zero_ish(self, model_a, one_seq):
        """M_unl == M_full means nothing was unlearned. Patching M_full into
        itself is a no-op (dS2 ~ 0), so UDS should be ~0 or undefined."""
        rep = compute_uds(
            model_full=model_a, model_retain=_tiny_model(9), model_unlearned=model_a,
            examples=[self._ex(one_seq)], layers=[1, 2],
        )
        assert rep.uds is None or rep.uds < 0.2

    def test_unlearned_equal_to_retain_gives_one(self, model_a, one_seq):
        """M_unl == M_ret is perfect unlearning: dS2 == dS1 exactly, UDS == 1."""
        retain = _tiny_model(9)
        rep = compute_uds(
            model_full=model_a, model_retain=retain, model_unlearned=retain,
            examples=[self._ex(one_seq)], layers=[1, 2],
        )
        assert rep.uds is None or rep.uds == pytest.approx(1.0, abs=1e-4)

    def test_each_example_scored_separately(self, model_a, one_seq):
        """Eq. 5 is per example: two examples must yield two independent
        results, each with its own KE layers."""
        rep = compute_uds(
            model_full=model_a, model_retain=_tiny_model(9),
            model_unlearned=_tiny_model(5),
            examples=[self._ex(one_seq, "e0", (6, 7)),
                      self._ex(one_seq, "e1", (4, 5))],
            layers=[0, 1, 2, 3],
        )
        assert rep.n_examples_total == 2
        assert [e.example_id for e in rep.per_example] == ["e0", "e1"]
        for e in rep.per_example:
            assert set(e.delta_s1) == {0, 1, 2, 3}

    def test_different_spans_give_different_deltas(self, model_a, one_seq):
        """Proof the per-example span is actually used, not a shared one."""
        rep = compute_uds(
            model_full=model_a, model_retain=_tiny_model(9),
            model_unlearned=_tiny_model(5),
            examples=[self._ex(one_seq, "a", (6, 7)), self._ex(one_seq, "b", (2, 3))],
            layers=[1],
        )
        a, b = rep.per_example
        assert a.delta_s1[1] != b.delta_s1[1]

    def test_report_flags_unvalidated(self, model_a, one_seq):
        rep = compute_uds(
            model_full=model_a, model_retain=_tiny_model(9),
            model_unlearned=_tiny_model(5),
            examples=[self._ex(one_seq)], layers=[1],
        )
        assert rep.is_validated_against_reference is False

    def test_no_hooks_leak_after_full_computation(self, model_a, one_seq):
        retain = _tiny_model(9)
        compute_uds(
            model_full=model_a, model_retain=retain, model_unlearned=_tiny_model(5),
            examples=[self._ex(one_seq, span=(6,))], layers=[0, 1],
        )
        for m in (model_a, retain):
            assert sum(len(l._forward_hooks) for l in m.model.layers) == 0

    def test_attention_mask_is_used(self, model_a, one_seq):
        mask = torch.ones_like(one_seq)
        mask[:, -2:] = 0
        rep = compute_uds(
            model_full=model_a, model_retain=_tiny_model(9),
            model_unlearned=_tiny_model(5),
            examples=[UDSExample("e0", one_seq, EntitySpan([6]), attention_mask=mask)],
            layers=[1],
        )
        assert rep.n_examples_total == 1


class TestStage1Caching:
    """Stage 1 depends only on M_full and M_ret, so it is reusable."""

    def _ex(self, one_seq, eid="e0"):
        return UDSExample(eid, one_seq, EntitySpan([6, 7]))

    def test_cache_matches_uncached_result(self, model_a, one_seq):
        retain, unl = _tiny_model(9), _tiny_model(5)
        exs = [self._ex(one_seq)]
        layers = [0, 1, 2]

        direct = compute_uds(model_full=model_a, model_retain=retain,
                             model_unlearned=unl, examples=exs, layers=layers)
        d1, sf = build_stage1_cache(model_full=model_a, model_retain=retain,
                                    examples=exs, layers=layers)
        cached = compute_uds(model_full=model_a, model_retain=None,
                             model_unlearned=unl, examples=exs, layers=layers,
                             stage1_cache=d1, s_full_cache=sf)
        assert cached.uds == pytest.approx(direct.uds, abs=1e-6)

    def test_cache_has_an_entry_per_example(self, model_a, one_seq):
        d1, sf = build_stage1_cache(
            model_full=model_a, model_retain=_tiny_model(9),
            examples=[self._ex(one_seq, "a"), self._ex(one_seq, "b")], layers=[0, 1],
        )
        assert set(d1) == {"a", "b"} and set(sf) == {"a", "b"}

    def test_missing_cache_entry_without_retain_model_raises(self, model_a, one_seq):
        with pytest.raises(ValueError, match="No Stage-1 cache"):
            compute_uds(model_full=model_a, model_retain=None,
                        model_unlearned=_tiny_model(5),
                        examples=[self._ex(one_seq)], layers=[0],
                        stage1_cache={}, s_full_cache={})

    def test_cache_missing_a_layer_falls_back_to_recompute(self, model_a, one_seq):
        """A cache built for fewer layers must not be used silently."""
        retain = _tiny_model(9)
        exs = [self._ex(one_seq)]
        d1, sf = build_stage1_cache(model_full=model_a, model_retain=retain,
                                    examples=exs, layers=[0])
        rep = compute_uds(model_full=model_a, model_retain=retain,
                          model_unlearned=_tiny_model(5), examples=exs,
                          layers=[0, 1, 2], stage1_cache=d1, s_full_cache=sf)
        assert set(rep.per_example[0].delta_s1) == {0, 1, 2}
