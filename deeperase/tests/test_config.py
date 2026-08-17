"""Tests for the model registry, memory planning and run configuration.

The memory planner exists to prevent a specific failure: starting a run that
crashes with out-of-memory after downloading 10 GB and running for an hour.
So these tests focus on the boundaries -- what fits, what does not, and
whether the planner degrades gracefully instead of lying.
"""

from __future__ import annotations

import pytest

from deeperase.config import (
    ACTIVATION_ALLOWANCE_GB,
    COMFORTABLE_HEADROOM_GB,
    SAFETY_FRACTION,
    TRAINING_STATE_MULTIPLIER,
    TABLE2_ABS_TOLERANCE,
    TOFU_MODELS,
    UDS_PAPER_TABLE2,
    ExecutionStrategy,
    ModelSpec,
    RunConfig,
    check_against_paper,
    plan_memory,
    recommend_size,
)

USER_GPU_GB = 20.0
"""The GPU we are actually targeting."""


class TestRegistry:
    def test_both_sizes_registered(self):
        assert set(TOFU_MODELS) == {"1B", "3B"}

    def test_every_size_has_four_splits(self):
        for size, splits in TOFU_MODELS.items():
            assert set(splits) == {"full", "retain90", "retain95", "retain99"}, size

    def test_repo_ids_match_verified_names(self):
        assert (
            TOFU_MODELS["1B"]["full"].repo_id
            == "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
        )
        assert (
            TOFU_MODELS["1B"]["retain90"].repo_id
            == "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
        )

    def test_parameter_counts_are_the_verified_values(self):
        assert TOFU_MODELS["1B"]["full"].n_params == 1_235_814_400
        assert TOFU_MODELS["3B"]["full"].n_params == 3_212_749_824

    def test_fp16_size_matches_hand_calculation(self):
        assert TOFU_MODELS["1B"]["full"].gb_at(2) == pytest.approx(2.47, abs=0.01)
        assert TOFU_MODELS["3B"]["full"].gb_at(2) == pytest.approx(6.43, abs=0.01)

    def test_fp32_is_double_fp16(self):
        m = TOFU_MODELS["1B"]["full"]
        assert m.gb_at(4) == pytest.approx(2 * m.gb_at(2))

    def test_unseen_fraction_ordering(self):
        """The Table 2 axis: full saw everything, retain90 saw none of forget10."""
        splits = TOFU_MODELS["1B"]
        assert splits["full"].unseen_fraction == 0.0
        assert splits["retain99"].unseen_fraction == 0.10
        assert splits["retain95"].unseen_fraction == 0.50
        assert splits["retain90"].unseen_fraction == 1.0


class TestMemoryPlanningOn20GB:
    """The planner must give correct answers for the actual target device."""

    def test_1b_fits_all_resident(self):
        p = plan_memory("1B", USER_GPU_GB)
        assert p.fits
        assert p.strategy is ExecutionStrategy.ALL_RESIDENT
        assert p.peak_weight_gb == pytest.approx(3 * 2.47, abs=0.05)

    def test_1b_has_comfortable_headroom(self):
        p = plan_memory("1B", USER_GPU_GB)
        assert p.headroom_gb > COMFORTABLE_HEADROOM_GB

    def test_3b_fits_but_needs_sequential(self):
        """3 x 6.43 = 19.3 GB exceeds the 14 GB usable budget, so the planner
        must fall back rather than claiming it fits."""
        p = plan_memory("3B", USER_GPU_GB)
        assert p.fits
        assert p.strategy is ExecutionStrategy.SEQUENTIAL
        assert p.peak_weight_gb == pytest.approx(6.43, abs=0.05)

    def test_3b_warns_about_reload_cost(self):
        p = plan_memory("3B", USER_GPU_GB)
        assert any("Sequential" in w for w in p.warnings)

    def test_recommends_3b_on_20gb(self):
        assert recommend_size(USER_GPU_GB) == "3B"

    def test_fp32_is_planned_correctly(self):
        """At fp32 a 1B model is 4.94 GB; three would be 14.8 GB > 14 GB usable."""
        p = plan_memory("1B", USER_GPU_GB, dtype_size=4)
        assert p.fits
        assert p.strategy is ExecutionStrategy.SEQUENTIAL

    def test_fp32_warns(self):
        assert any("fp32" in w for w in plan_memory("1B", USER_GPU_GB, dtype_size=4).warnings)


class TestMemoryPlanningBoundaries:
    def test_tiny_gpu_does_not_fit(self):
        p = plan_memory("3B", 4.0)
        assert not p.fits
        assert p.strategy is None
        assert "smaller model" in p.reason

    def test_large_gpu_fits_3b_all_resident(self):
        p = plan_memory("3B", 80.0)
        assert p.fits and p.strategy is ExecutionStrategy.ALL_RESIDENT

    def test_safety_margin_is_actually_applied(self):
        """A device with exactly 3 models' worth of memory and no slack must
        NOT be reported as all-resident, or we will OOM on activations."""
        exact = 3 * TOFU_MODELS["1B"]["full"].gb_at(2)
        p = plan_memory("1B", exact)
        assert p.strategy is not ExecutionStrategy.ALL_RESIDENT

    def test_usable_budget_respects_safety_fraction(self):
        p = plan_memory("1B", USER_GPU_GB)
        usable = USER_GPU_GB * (1 - SAFETY_FRACTION)
        assert p.peak_weight_gb + p.headroom_gb == pytest.approx(usable, abs=0.01)

    def test_unknown_size_reports_available_options(self):
        p = plan_memory("70B", USER_GPU_GB)
        assert not p.fits and "1B" in p.reason

    def test_recommend_returns_none_when_nothing_fits(self):
        assert recommend_size(1.0) is None

    def test_summary_is_readable(self):
        assert "FITS" in plan_memory("1B", USER_GPU_GB).summary()
        assert "DOES NOT FIT" in plan_memory("3B", 4.0).summary()


class TestPlanningAgainstFreeMemory:
    """Regression: the study crashed at the first optimiser step on a node whose
    card was 42.3 GB but only 10.6 GB free. Planning used total, and counted
    weights only, so it reported 22 GB of headroom for a run needing ~11 GB it
    did not have.
    """

    SHARED = dict(gpu_total_gb=42.3, gpu_free_gb=10.6)

    def test_free_memory_is_the_budget_not_total(self):
        tight = plan_memory("1B", 42.3, gpu_free_gb=10.6)
        roomy = plan_memory("1B", 42.3)
        # Same card, same model -- only the co-tenants differ.
        assert roomy.headroom_gb > tight.headroom_gb + 15

    def test_training_costs_more_than_weights(self):
        weights = plan_memory("1B", 80.0)
        train = plan_memory("1B", 80.0, training=True)
        assert train.peak_weight_gb > weights.peak_weight_gb
        # weights + grads + 2 moments, plus activation slack.
        one = TOFU_MODELS["1B"]["full"].gb_at(2)
        assert train.peak_weight_gb == pytest.approx(
            one * TRAINING_STATE_MULTIPLIER + ACTIVATION_ALLOWANCE_GB, abs=0.01)

    def test_the_exact_configuration_that_crashed_is_refused(self):
        p = plan_memory("1B", training=True, **self.SHARED)
        assert not p.fits, "this is the run that died at optimiser.step()"

    def test_it_would_have_been_allowed_by_the_old_planning(self):
        # Both defects had to be fixed: either alone still lets the run start.
        assert plan_memory("1B", 42.3).fits                      # old behaviour
        assert plan_memory("1B", 42.3, training=True).fits       # free only
        assert plan_memory("1B", **self.SHARED).fits             # training only
        assert not plan_memory("1B", training=True, **self.SHARED).fits

    def test_refusal_names_a_way_out(self):
        p = plan_memory("1B", training=True, **self.SHARED)
        assert "LoRA" in p.reason and "CUDA_VISIBLE_DEVICES" in p.reason

    def test_co_tenants_are_reported(self):
        p = plan_memory("1B", training=True, **self.SHARED)
        assert any("other processes" in w for w in p.warnings)

    def test_no_warning_when_the_card_is_ours(self):
        p = plan_memory("1B", 42.3, gpu_free_gb=42.0, training=True)
        assert p.fits and not any("other processes" in w for w in p.warnings)

    def test_training_plan_survives_on_an_empty_card(self):
        p = plan_memory("1B", 42.3, gpu_free_gb=42.0, training=True)
        assert p.fits and p.strategy is ExecutionStrategy.ALL_RESIDENT

    def test_reason_states_which_basis_was_used(self):
        assert "free" in plan_memory("1B", 42.3, gpu_free_gb=42.0).reason
        assert "total" in plan_memory("1B", 42.3).reason

    def test_peak_is_the_larger_phase_not_the_sum(self):
        # Reference models are freed before training; the phases do not overlap.
        one = TOFU_MODELS["3B"]["full"].gb_at(2)
        p = plan_memory("3B", 200.0, training=True)
        assert p.peak_weight_gb < one * 3 + one * TRAINING_STATE_MULTIPLIER


class TestPaperComparison:
    def test_paper_values_recorded_for_1b(self):
        assert UDS_PAPER_TABLE2["1B"]["retain90"] == 1.000
        assert UDS_PAPER_TABLE2["1B"]["full"] == 0.002

    def test_exact_reproduction_passes(self):
        r = check_against_paper("1B", dict(UDS_PAPER_TABLE2["1B"]))
        assert r["monotonic"] is True
        assert r["verdict"].startswith("PASS")

    def test_close_reproduction_passes(self):
        r = check_against_paper(
            "1B", {"full": 0.01, "retain99": 0.19, "retain95": 0.52, "retain90": 0.98}
        )
        assert r["verdict"].startswith("PASS")

    def test_non_monotonic_fails_even_if_close(self):
        """Ordering is the real test. A set of values that are individually
        near the paper's but in the wrong order must fail."""
        r = check_against_paper(
            "1B", {"full": 0.50, "retain99": 0.15, "retain95": 0.49, "retain90": 1.0}
        )
        assert r["monotonic"] is False
        assert r["verdict"].startswith("FAIL")

    def test_monotonic_but_far_off_is_partial(self):
        r = check_against_paper(
            "1B", {"full": 0.30, "retain99": 0.40, "retain95": 0.50, "retain90": 0.60}
        )
        assert r["monotonic"] is True
        assert r["verdict"].startswith("PARTIAL")

    def test_partial_split_set_is_allowed(self):
        r = check_against_paper("1B", {"full": 0.002, "retain90": 1.0})
        assert r["n_compared"] == 2 and r["monotonic"] is True

    def test_unknown_size_raises(self):
        with pytest.raises(KeyError):
            check_against_paper("70B", {"full": 0.0})

    def test_tolerance_is_documented_value(self):
        assert TABLE2_ABS_TOLERANCE == 0.08


class TestRunConfig:
    def test_default_config_is_valid(self):
        assert RunConfig().validate() == []

    def test_default_targets_1b(self):
        c = RunConfig()
        assert c.size_label == "1B" and c.dtype == "bfloat16"

    def test_mismatched_forget_and_retain_is_rejected(self):
        """forget10 must pair with retain90. A mismatch silently measures the
        wrong thing, so it must be caught before the run starts."""
        c = RunConfig(forget_split="forget10", stage1_source_split="retain95")
        problems = c.validate()
        assert any("does not match" in p for p in problems)

    def test_all_valid_pairings_accepted(self):
        for forget, retain in RunConfig.FORGET_TO_RETAIN.items():
            assert RunConfig(forget_split=forget, stage1_source_split=retain).validate() == []

    def test_bad_size_rejected(self):
        assert any("size_label" in p for p in RunConfig(size_label="70B").validate())

    def test_bad_dtype_rejected(self):
        assert any("dtype" in p for p in RunConfig(dtype="int8").validate())

    def test_bad_tau_rejected(self):
        assert any("tau" in p for p in RunConfig(tau=1.5).validate())

    def test_zero_examples_rejected(self):
        assert any("n_examples" in p for p in RunConfig(n_examples=0).validate())

    def test_dtype_size_mapping(self):
        assert RunConfig(dtype="bfloat16").dtype_size == 2
        assert RunConfig(dtype="float32").dtype_size == 4

    def test_models_returns_four_splits(self):
        assert len(RunConfig().models()) == 4

    def test_serialises_for_reproducibility(self):
        d = RunConfig(strategy=ExecutionStrategy.SEQUENTIAL).to_dict()
        assert d["strategy"] == "sequential"
        assert d["size_label"] == "1B"
