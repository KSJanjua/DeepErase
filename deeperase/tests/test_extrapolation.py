"""Tests for the extrapolation instrument.

The invariants that matter most for D1:
  * alpha=0 must reproduce theta_un exactly (the control point).
  * SAGE with a_par == a_perp must equal UIPE exactly (nested models).
  * Buffers/integer tensors must never be interpolated.
  * Projection must actually be a projection (idempotent, orthogonal residual).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import warnings

from deeperase.core.extrapolation import (
    DEFAULT_BUFFER_PATTERNS,
    ExperimentalPrototypeWarning,
    ExtrapolationReport,
    alpha_grid,
    compute_update_vector,
    extrapolate,
    extrapolate_directed,
    global_norm,
    is_buffer_name,
    project_onto_subspace,
    project_state_dict,
)


@pytest.fixture(autouse=True)
def _silence_prototype_warning():
    """SAGE emits ExperimentalPrototypeWarning by design; tests that assert on
    it re-enable it locally."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalPrototypeWarning)
        yield


@pytest.fixture
def simple_pair():
    torch.manual_seed(0)
    ini = {
        "layer.weight": torch.randn(6, 4),
        "layer.bias": torch.randn(6),
        "buf.running_mean": torch.randn(6),   # float BUFFER -- must be excluded
        "int_buffer": torch.arange(5),        # int buffer -- must be excluded
    }
    un = {k: (v.clone() if not torch.is_floating_point(v) else v + torch.randn_like(v) * 0.1)
          for k, v in ini.items()}
    return ini, un


class TestBufferExclusion:
    """Regression tests for the float-buffer bug.

    Buffers were previously filtered by dtype alone, so float buffers such as
    running_mean WERE extrapolated despite documentation claiming otherwise.
    """

    def test_is_buffer_name_matches_known_patterns(self):
        assert is_buffer_name("bn.running_mean")
        assert is_buffer_name("bn.running_var")
        assert is_buffer_name("model.layers.0.self_attn.rotary_emb.inv_freq")
        assert is_buffer_name("BN.RUNNING_MEAN")            # case-insensitive
        assert not is_buffer_name("model.layers.0.mlp.down_proj.weight")
        assert not is_buffer_name("layer.bias")

    def test_float_buffer_excluded_from_update_vector(self):
        ini = {"w": torch.zeros(2, 2), "bn.running_mean": torch.zeros(2)}
        un = {"w": torch.ones(2, 2), "bn.running_mean": torch.ones(2)}
        v = compute_update_vector(ini, un)
        assert "bn.running_mean" not in v, "float buffer must not be extrapolable"
        assert "w" in v

    def test_float_buffer_unchanged_after_extrapolation(self):
        """The bug: running_mean went to 2.0 at alpha=1 instead of staying 1.0."""
        ini = {"w": torch.zeros(2, 2), "bn.running_mean": torch.zeros(2)}
        un = {"w": torch.ones(2, 2), "bn.running_mean": torch.ones(2)}
        out = extrapolate(un, compute_update_vector(ini, un), alpha=1.0)
        assert torch.allclose(out["bn.running_mean"], torch.ones(2)), \
            f"buffer was extrapolated to {out['bn.running_mean'].tolist()}, expected [1.0, 1.0]"
        assert torch.allclose(out["w"], torch.full((2, 2), 2.0)), "weight should still extrapolate"

    def test_rotary_inv_freq_excluded(self):
        """inv_freq is a deterministic function of position; interpolating it
        silently corrupts positional encoding."""
        name = "model.layers.0.self_attn.rotary_emb.inv_freq"
        ini = {"w": torch.zeros(2, 2), name: torch.linspace(1, 2, 4)}
        un = {"w": torch.ones(2, 2), name: torch.linspace(3, 4, 4)}
        v = compute_update_vector(ini, un)
        assert name not in v
        out = extrapolate(un, v, alpha=0.9)
        assert torch.allclose(out[name], un[name])

    def test_buffers_counted_as_skipped_in_report(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        _, rep = extrapolate_directed(un, v, alpha_parallel=0.5, alpha_perp=0.5)
        assert rep.n_tensors_modified == 2, "only weight and bias are extrapolable"
        assert rep.n_tensors_skipped == 2, "both buffers must be skipped"

    def test_exclusion_can_be_disabled_explicitly(self):
        ini = {"w": torch.zeros(2, 2), "bn.running_mean": torch.zeros(2)}
        un = {"w": torch.ones(2, 2), "bn.running_mean": torch.ones(2)}
        v = compute_update_vector(ini, un, exclude_buffers=False)
        assert "bn.running_mean" in v, "opt-out must still be available"

    def test_custom_buffer_patterns(self):
        ini = {"w": torch.zeros(2, 2), "odd.cache_thing": torch.zeros(2)}
        un = {"w": torch.ones(2, 2), "odd.cache_thing": torch.ones(2)}
        v = compute_update_vector(ini, un, buffer_patterns=("cache_thing",))
        assert "odd.cache_thing" not in v

    def test_all_filtered_raises_actionable_error(self):
        ini = {"bn.running_mean": torch.zeros(2)}
        un = {"bn.running_mean": torch.ones(2)}
        with pytest.raises(ValueError, match="buffer_patterns"):
            compute_update_vector(ini, un)


class TestSagePrototypeWarning:
    """SAGE must announce that it is not research-ready."""

    def test_unequal_coefficients_warn(self):
        ini = {"w": torch.zeros(4, 4)}
        un = {"w": torch.ones(4, 4)}
        v = compute_update_vector(ini, un)
        vp = project_state_dict(v, {"w": torch.eye(4)[:2]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPrototypeWarning)
            extrapolate_directed(un, v, alpha_parallel=0.9, alpha_perp=0.1, v_parallel=vp)
        assert any(issubclass(w.category, ExperimentalPrototypeWarning) for w in caught)
        assert any("must not be used for research claims" in str(w.message) for w in caught)

    def test_equal_coefficients_do_not_warn(self):
        """Reducing to published UIPE is not experimental."""
        ini = {"w": torch.zeros(4, 4)}
        un = {"w": torch.ones(4, 4)}
        v = compute_update_vector(ini, un)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPrototypeWarning)
            extrapolate_directed(un, v, alpha_parallel=0.5, alpha_perp=0.5)
        assert not [w for w in caught if issubclass(w.category, ExperimentalPrototypeWarning)]

    def test_isotropic_extrapolate_never_warns(self):
        ini = {"w": torch.zeros(4, 4)}
        un = {"w": torch.ones(4, 4)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPrototypeWarning)
            extrapolate(un, compute_update_vector(ini, un), alpha=0.7)
        assert not [w for w in caught if issubclass(w.category, ExperimentalPrototypeWarning)]


class TestUpdateVector:
    def test_difference_is_correct(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        assert torch.allclose(v["layer.weight"], un["layer.weight"] - ini["layer.weight"], atol=1e-6)

    def test_integer_tensors_excluded(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        assert "int_buffer" not in v
        assert "layer.weight" in v

    def test_computed_in_float32(self):
        ini = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        un = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        v = compute_update_vector(ini, un)
        assert v["w"].dtype == torch.float32, "must upcast to avoid accumulating bf16 error"

    def test_key_subsetting(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un, keys=["layer.weight"])
        assert set(v) == {"layer.weight"}

    def test_strict_rejects_key_mismatch(self, simple_pair):
        ini, un = simple_pair
        del un["layer.bias"]
        with pytest.raises(KeyError, match="disagree on keys"):
            compute_update_vector(ini, un, strict=True)

    def test_non_strict_intersects(self, simple_pair):
        ini, un = simple_pair
        del un["layer.bias"]
        v = compute_update_vector(ini, un, strict=False)
        assert "layer.bias" not in v and "layer.weight" in v

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_update_vector({"w": torch.randn(4, 4)}, {"w": torch.randn(4, 5)})

    def test_empty_update_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_update_vector({"i": torch.arange(3)}, {"i": torch.arange(3)})


class TestIsotropicExtrapolation:
    def test_alpha_zero_is_identity(self, simple_pair):
        """The control point must be bit-faithful, or every trajectory is
        anchored to the wrong origin."""
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        out = extrapolate(un, v, alpha=0.0)
        for k in un:
            assert torch.allclose(out[k].float(), un[k].float(), atol=1e-6), f"{k} changed at alpha=0"

    def test_alpha_one_doubles_the_update(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        out = extrapolate(un, v, alpha=1.0)
        expected = ini["layer.weight"] + 2.0 * v["layer.weight"]
        assert torch.allclose(out["layer.weight"], expected, atol=1e-5)

    def test_linearity_in_alpha(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        a, b = extrapolate(un, v, 0.3)["layer.weight"], extrapolate(un, v, 0.6)["layer.weight"]
        base = un["layer.weight"]
        assert torch.allclose(2 * (a - base), b - base, atol=1e-5)

    def test_buffers_passed_through(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        out = extrapolate(un, v, alpha=0.7)
        assert torch.equal(out["int_buffer"], un["int_buffer"])

    def test_dtype_preserved(self):
        ini = {"w": torch.randn(4, 4, dtype=torch.float16)}
        un = {"w": torch.randn(4, 4, dtype=torch.float16)}
        out = extrapolate(un, compute_update_vector(ini, un), alpha=0.5)
        assert out["w"].dtype == torch.float16

    def test_output_does_not_alias_input(self, simple_pair):
        ini, un = simple_pair
        out = extrapolate(un, compute_update_vector(ini, un), alpha=0.0)
        out["layer.weight"] += 1.0
        assert not torch.allclose(out["layer.weight"], un["layer.weight"])


class TestProjection:
    def test_projection_is_idempotent(self):
        torch.manual_seed(1)
        v, basis = torch.randn(8, 16), torch.randn(3, 16)
        p1 = project_onto_subspace(v, basis)
        p2 = project_onto_subspace(p1, basis)
        assert torch.allclose(p1, p2, atol=1e-5)

    def test_residual_orthogonal_to_subspace(self):
        torch.manual_seed(2)
        v, basis = torch.randn(8, 16), torch.randn(3, 16)
        residual = v - project_onto_subspace(v, basis)
        q, _ = torch.linalg.qr(basis.T)
        assert torch.allclose(residual @ q, torch.zeros(8, q.shape[1]), atol=1e-4)

    def test_vector_in_subspace_is_preserved(self):
        basis = torch.eye(4)[:2]           # spans first two coordinates
        v = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
        assert torch.allclose(project_onto_subspace(v, basis), v, atol=1e-6)

    def test_vector_orthogonal_to_subspace_maps_to_zero(self):
        basis = torch.eye(4)[:2]
        v = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
        assert torch.allclose(project_onto_subspace(v, basis), torch.zeros_like(v), atol=1e-6)

    def test_orthonormal_shortcut_matches(self):
        torch.manual_seed(3)
        raw = torch.randn(3, 12)
        q, _ = torch.linalg.qr(raw.T)
        basis = q.T
        v = torch.randn(5, 12)
        a = project_onto_subspace(v, basis, assume_orthonormal=True)
        b = project_onto_subspace(v, basis, assume_orthonormal=False)
        assert torch.allclose(a, b, atol=1e-4)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="Feature-axis mismatch"):
            project_onto_subspace(torch.randn(4, 8), torch.randn(2, 16))

    def test_project_state_dict_respects_axis(self):
        v = {"w": torch.randn(6, 4)}
        basis = torch.eye(6)[:2]
        out = project_state_dict(v, {"w": basis}, project_axis={"w": 0})
        assert out["w"].shape == (6, 4)
        # Rows 2..5 lie outside the subspace and must be zeroed.
        assert torch.allclose(out["w"][2:], torch.zeros(4, 4), atol=1e-5)


class TestDirectedExtrapolation:
    def test_equal_coefficients_reduce_to_uipe(self, simple_pair):
        """Algebraic consistency check ONLY.

        This proves the directed operator does not corrupt the isotropic path
        -- i.e. it guards against coding errors. It is NOT evidence that SAGE
        is scientifically meaningful; the activation-to-parameter subspace
        mapping remains unspecified. Do not cite this as validation.
        """
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        basis = torch.eye(4)[:2]
        vp = project_state_dict(v, {"layer.weight": basis})

        iso = extrapolate(un, v, alpha=0.5)
        directed, _ = extrapolate_directed(un, v, alpha_parallel=0.5, alpha_perp=0.5, v_parallel=vp)
        for k in iso:
            assert torch.allclose(iso[k].float(), directed[k].float(), atol=1e-5), k

    def test_unequal_coefficients_without_subspace_raises(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        with pytest.raises(ValueError, match="requires v_parallel"):
            extrapolate_directed(un, v, alpha_parallel=0.8, alpha_perp=0.2)

    def test_parallel_component_amplified_more(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        basis = torch.eye(4)[:2]
        vp = project_state_dict(v, {"layer.weight": basis})
        out, _ = extrapolate_directed(un, v, alpha_parallel=1.0, alpha_perp=0.0, v_parallel=vp)
        step = out["layer.weight"] - un["layer.weight"]
        assert torch.allclose(step, vp["layer.weight"], atol=1e-5), \
            "with a_perp=0 the step must be exactly the parallel component"

    def test_report_parallel_fraction_in_unit_interval(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        vp = project_state_dict(v, {"layer.weight": torch.eye(4)[:2]})
        _, rep = extrapolate_directed(un, v, alpha_parallel=0.6, alpha_perp=0.2, v_parallel=vp)
        assert 0.0 <= rep.parallel_fraction <= 1.0

    def test_report_counts(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        _, rep = extrapolate_directed(un, v, alpha_parallel=0.5, alpha_perp=0.5)
        assert rep.n_tensors_modified == 2   # weight, bias
        assert rep.n_tensors_skipped == 2    # running_mean (float buf) + int_buffer
        assert rep.delta_norm > 0
        assert isinstance(rep.summary(), str)

    def test_delta_norm_scales_with_alpha(self, simple_pair):
        ini, un = simple_pair
        v = compute_update_vector(ini, un)
        _, r1 = extrapolate_directed(un, v, alpha_parallel=0.5, alpha_perp=0.5)
        _, r2 = extrapolate_directed(un, v, alpha_parallel=1.0, alpha_perp=1.0)
        assert r2.delta_norm == pytest.approx(2 * r1.delta_norm, rel=1e-4)


class TestHelpers:
    def test_global_norm_matches_flat_norm(self):
        ts = [torch.randn(3, 4), torch.randn(5)]
        flat = torch.cat([t.flatten() for t in ts])
        assert global_norm(ts) == pytest.approx(float(flat.norm()), rel=1e-6)

    def test_alpha_grid_default(self):
        g = alpha_grid()
        assert len(g) == 11 and g[0] == 0.0 and g[-1] == 1.0
        assert 0.0 in g, "control point must be present"

    def test_alpha_grid_custom(self):
        g = alpha_grid(0.0, 2.0, num=5)
        assert g == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_alpha_grid_rejects_single_point(self):
        with pytest.raises(ValueError):
            alpha_grid(num=1)
