"""Tests for depth, surface, breadth and plane metrics.

Key invariants:
  * CKA and PCA similarity are 1.0 for identical inputs, low for independent.
  * EL10 > 1 must classify as Type II -- that is ERUF's Qwen-8B anomaly and
    the empirical hook for the whole D1 story.
  * The retain (R) tier must be scored separately from forget tiers.
  * A synthetic trade-off must be detected with negative Spearman rho.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperase.eval.depth import (
    DriftResult,
    identify_target_layers,
    linear_cka,
    linear_probe_recoverability,
    pca_subspace_similarity,
    representation_drift,
    selectivity_ratio,
    unlearning_depth_score,
)
from deeperase.eval.plane import PlaneDataset, PlanePoint, Trajectory
from deeperase.eval.surface import (
    classify_type,
    el10,
    mean_rouge_l,
    rouge_l,
    subject_mention_rate,
    subject_token_mass,
)
from deeperase.probes.schema import Probe, ProbeSet, Tier, score_breadth


# ---------------------------------------------------------------- depth ----

class TestCKA:
    def test_identical_gives_one(self):
        x = np.random.RandomState(0).randn(50, 8)
        assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_invariant_to_isotropic_scaling(self):
        x = np.random.RandomState(1).randn(50, 8)
        assert linear_cka(x, 3.7 * x) == pytest.approx(1.0, abs=1e-6)

    def test_invariant_to_orthogonal_transform(self):
        rs = np.random.RandomState(2)
        x = rs.randn(50, 8)
        q, _ = np.linalg.qr(rs.randn(8, 8))
        assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-5)

    def test_independent_data_scores_low(self):
        rs = np.random.RandomState(3)
        assert linear_cka(rs.randn(200, 8), rs.randn(200, 8)) < 0.35

    def test_sample_mismatch_raises(self):
        with pytest.raises(ValueError, match="Sample-axis mismatch"):
            linear_cka(np.zeros((10, 4)), np.zeros((9, 4)))


class TestPCASimilarity:
    def test_identical_gives_one(self):
        x = np.random.RandomState(4).randn(60, 10)
        assert pca_subspace_similarity(x, x, k=5) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_subspaces_give_zero(self):
        n = 60
        rs = np.random.RandomState(5)
        x = np.zeros((n, 6)); x[:, :2] = rs.randn(n, 2)
        y = np.zeros((n, 6)); y[:, 4:] = rs.randn(n, 2)
        assert pca_subspace_similarity(x, y, k=2) == pytest.approx(0.0, abs=1e-6)

    def test_bounded_unit_interval(self):
        rs = np.random.RandomState(6)
        s = pca_subspace_similarity(rs.randn(80, 12), rs.randn(80, 12), k=4)
        assert 0.0 <= s <= 1.0


class TestDriftAndSelectivity:
    def test_drift_zero_when_unchanged(self):
        h = {0: np.random.RandomState(7).randn(40, 8)}
        d = representation_drift(h, h, k=4)
        assert len(d) == 1
        assert d[0].cka == pytest.approx(1.0, abs=1e-6)
        assert d[0].mean_pca_distance == pytest.approx(0.0, abs=1e-6)

    def test_selectivity_ratio_flags_targeted_erasure(self):
        """Large forget drift with small retain drift = targeted. This is
        ERUF's SRS, reported there as 5.90x."""
        forget = [DriftResult(layer=0, cka=0.2, pca_similarity=0.1, mean_pca_distance=0.9)]
        retain = [DriftResult(layer=0, cka=0.98, pca_similarity=0.9, mean_pca_distance=0.1)]
        assert selectivity_ratio(forget, retain)[0] == pytest.approx(9.0, rel=1e-3)

    def test_selectivity_near_one_means_indiscriminate(self):
        forget = [DriftResult(0, 0.5, 0.5, 0.5)]
        retain = [DriftResult(0, 0.5, 0.5, 0.5)]
        assert selectivity_ratio(forget, retain)[0] == pytest.approx(1.0, rel=1e-3)

    def test_identify_target_layers_picks_highest_drift(self):
        drift = [
            DriftResult(0, 0.9, 0.9, 0.1),
            DriftResult(1, 0.3, 0.2, 0.8),
            DriftResult(2, 0.4, 0.3, 0.7),
            DriftResult(3, 0.95, 0.95, 0.05),
        ]
        assert identify_target_layers(drift, top_k=2) == [1, 2]


class TestLinearProbe:
    def test_separable_data_recovers_high_accuracy(self):
        """If a probe still separates the classes, the information survived."""
        rs = np.random.RandomState(8)
        hf = {0: rs.randn(60, 6) + 5.0}
        hc = {0: rs.randn(60, 6) - 5.0}
        res = linear_probe_recoverability(hf, hc, seed=0)
        assert res[0].accuracy > 0.9
        assert res[0].above_chance > 0.3

    def test_inseparable_data_near_chance(self):
        rs = np.random.RandomState(9)
        hf = {0: rs.randn(80, 6)}
        hc = {0: rs.randn(80, 6)}
        assert linear_probe_recoverability(hf, hc, seed=0)[0].accuracy < 0.75

    def test_no_common_layers_raises(self):
        with pytest.raises(ValueError, match="No layers common"):
            linear_probe_recoverability({0: np.zeros((5, 3))}, {1: np.zeros((5, 3))})


class TestUDS:
    def test_no_recovery_from_patch_means_deep(self):
        """Patch restores base activations but knowledge does not return ->
        downstream circuitry no longer reads the signal -> deep erasure."""
        r = unlearning_depth_score(score_unlearned=0.10, score_patched=0.10, score_oracle=0.10 - 0.5)
        assert r.uds == pytest.approx(1.0, abs=1e-6)

    def test_full_recovery_means_shallow(self):
        r = unlearning_depth_score(score_unlearned=0.10, score_patched=0.60, score_oracle=-0.40)
        assert r.uds == pytest.approx(0.0, abs=1e-6)

    def test_partial_recovery_is_intermediate(self):
        r = unlearning_depth_score(score_unlearned=0.10, score_patched=0.35, score_oracle=-0.40)
        assert 0.0 < r.uds < 1.0

    def test_bounded_to_unit_interval(self):
        r = unlearning_depth_score(score_unlearned=0.1, score_patched=99.0, score_oracle=-0.4)
        assert r.uds == 0.0

    def test_degenerate_gap_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            unlearning_depth_score(score_unlearned=0.5, score_patched=0.5, score_oracle=0.5)

    def test_normal_regime_not_flagged_overshoot(self):
        r = unlearning_depth_score(score_unlearned=0.1, score_patched=0.3, score_oracle=-0.4)
        assert r.overshoot is False

    def test_overshoot_detected_when_below_oracle(self):
        """Aggressive GA can drive the target below never-having-learned-it.
        Observed in the CPU smoke run: unlearned=-16.2 vs oracle=-6.9."""
        r = unlearning_depth_score(
            score_unlearned=-16.18, score_patched=-0.06, score_oracle=-6.86
        )
        assert r.overshoot is True
        assert "OVERSHOOT" in r.summary()

    def test_overshoot_still_returns_finite_value(self):
        """Must not crash a sweep -- flag, don't raise."""
        r = unlearning_depth_score(score_unlearned=-16.0, score_patched=-1.0, score_oracle=-7.0)
        assert 0.0 <= r.uds <= 1.0

    def test_summary_is_readable(self):
        r = unlearning_depth_score(score_unlearned=0.1, score_patched=0.2, score_oracle=-0.4,
                                   target_layers=[4, 5], n_examples=20)
        assert "UDS=" in r.summary() and "[4, 5]" in r.summary()

    def test_result_is_marked_as_scaffold(self):
        """Until real activation patching exists and is checked against the
        reference implementation, every UDS result must self-identify as a
        scaffold so it cannot be mistaken for a validated measurement."""
        r = unlearning_depth_score(score_unlearned=0.1, score_patched=0.2, score_oracle=-0.4)
        assert r.is_scaffold is True


# -------------------------------------------------------------- surface ----

class TestSMR:
    def test_counts_only_matching_generations(self):
        gens = ["Harry Potter went to Hogwarts.", "I cannot help with that.", "Who is Potter?"]
        r = subject_mention_rate(gens, ["Harry Potter", "Potter"])
        assert r.n_hits == 2
        assert r.smr == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        assert subject_mention_rate(["harry potter"], ["Harry Potter"]).n_hits == 1

    def test_word_boundary_prevents_substring_false_positive(self):
        """'Potterton' must not count as a mention of 'Potter'."""
        assert subject_mention_rate(["The Potterton boiler"], ["Potter"]).n_hits == 0

    def test_longest_alias_matched_first(self):
        r = subject_mention_rate(["Harry Potter"], ["Potter", "Harry Potter"])
        assert r.per_alias_hits["Harry Potter"] == 1
        assert r.per_alias_hits["Potter"] == 0

    def test_empty_generations_safe(self):
        r = subject_mention_rate([], ["X"])
        assert r.smr == 0.0 and r.n_total == 0

    def test_no_aliases_raises(self):
        with pytest.raises(ValueError, match="No non-empty aliases"):
            subject_mention_rate(["text"], ["", "  "])


class TestEL10:
    def _probs(self, n_prompts, n_steps, vocab, subject_ids, mass, seed=0):
        rs = np.random.RandomState(seed)
        p = rs.rand(n_prompts, n_steps, vocab)
        p /= p.sum(axis=-1, keepdims=True)
        # Force exactly `mass` onto subject ids, rescaling the rest.
        cur = p[:, :, subject_ids].sum(axis=-1, keepdims=True)
        others = np.setdiff1d(np.arange(vocab), subject_ids)
        p[:, :, others] *= (1 - mass) / (1 - cur)
        p[:, :, subject_ids] *= mass / cur
        return p

    def test_ratio_below_one_when_attenuated(self):
        ids = [3, 7]
        base = self._probs(4, 10, 20, ids, 0.20, seed=1)
        unl = self._probs(4, 10, 20, ids, 0.05, seed=1)
        r = el10(unl, base, ids)
        assert r.el10 < 1.0
        assert r.el10 == pytest.approx(0.25, rel=0.05)

    def test_ratio_above_one_when_amplified(self):
        """The ERUF Qwen-8B anomaly: surface suppressed, token mass amplified."""
        ids = [3, 7]
        base = self._probs(4, 10, 20, ids, 0.05, seed=2)
        unl = self._probs(4, 10, 20, ids, 0.30, seed=2)
        assert el10(unl, base, ids).el10 > 1.0

    def test_identical_gives_one(self):
        ids = [1, 2]
        p = self._probs(3, 10, 15, ids, 0.1, seed=3)
        assert el10(p, p, ids).el10 == pytest.approx(1.0, rel=1e-6)

    def test_prompt_mismatch_raises(self):
        ids = [1]
        with pytest.raises(ValueError, match="Prompt-count mismatch"):
            el10(np.ones((3, 10, 5)) / 5, np.ones((4, 10, 5)) / 5, ids)

    def test_out_of_range_token_id_raises(self):
        with pytest.raises(IndexError, match="exceeds vocab"):
            subject_token_mass(np.ones((2, 10, 5)) / 5, [99])

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match=r"must be \(n_prompts"):
            subject_token_mass(np.ones((10, 5)) / 5, [1])


class TestTypeClassification:
    def test_type_one_clean_and_attenuated(self):
        assert classify_type(0.00, 0.05).type_label == "I"

    def test_type_two_is_obfuscation(self):
        c = classify_type(0.033, 11.03)   # ERUF's reported Qwen-8B point
        assert c.type_label == "II"
        assert "obfuscation" in c.rationale

    def test_type_three_is_leakage(self):
        assert classify_type(0.40, 0.10).type_label == "III"

    def test_smr_dominates_el10(self):
        """High SMR is Type III regardless of EL10."""
        assert classify_type(0.90, 0.01).type_label == "III"

    def test_epsilon_boundary_inclusive(self):
        assert classify_type(0.05, 0.5).type_label == "I"
        assert classify_type(0.051, 0.5).type_label == "III"


class TestRougeL:
    def test_identical_is_one(self):
        assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        assert rouge_l("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_partial_overlap(self):
        assert 0.0 < rouge_l("the cat sat on the mat", "the cat sat") < 1.0

    def test_empty_is_zero(self):
        assert rouge_l("", "anything") == 0.0

    def test_mean_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            mean_rouge_l(["a"], ["a", "b"])


# --------------------------------------------------------------- breadth ----

def _probe_set() -> ProbeSet:
    probes = []
    for tier in (Tier.EXACT, Tier.PARAPHRASE, Tier.ALIAS):
        probes.append(Probe(f"t1_{tier.value}", "t1", tier, f"q {tier.value}?", "ans"))
    for tier in (Tier.ENTAILED, Tier.MULTIHOP):
        probes.append(Probe(f"t1_{tier.value}", "t1", tier, f"q {tier.value}?", "ans",
                            hop_facts=["fact A"]))
    probes.append(Probe("t1_R", "t1", Tier.RETAIN, "neighbour q?", "ans"))
    return ProbeSet("t1", "Target One", ["Target One", "T.O."], probes)


class TestProbeSchema:
    def test_tier_forget_flags(self):
        assert Tier.EXACT.is_forget and Tier.MULTIHOP.is_forget
        assert not Tier.RETAIN.is_forget

    def test_valid_set_has_no_problems(self):
        assert _probe_set().validate() == []

    def test_missing_retain_tier_flagged(self):
        ps = _probe_set()
        ps.probes = [p for p in ps.probes if p.tier is not Tier.RETAIN]
        assert any("retain (R)" in p for p in ps.validate())

    def test_duplicate_ids_flagged(self):
        ps = _probe_set()
        ps.probes.append(ps.probes[0])
        assert any("duplicate probe_ids" in p for p in ps.validate())

    def test_require_verified_flags_unverified(self):
        assert any("unverified" in p for p in _probe_set().validate(require_verified=True))

    def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="empty question"):
            Probe("x", "t1", Tier.EXACT, "   ", "ans")

    def test_roundtrip_serialisation(self, tmp_path):
        from deeperase.probes.schema import load_probe_sets, save_probe_sets
        original = _probe_set()
        p = save_probe_sets([original], tmp_path / "probes.json")
        loaded = load_probe_sets(p)[0]
        assert loaded.target_id == original.target_id
        assert len(loaded.probes) == len(original.probes)
        assert loaded.probes[0].tier is Tier.EXACT


class TestBreadthScoring:
    def test_retain_scored_separately_from_forget(self):
        ps = _probe_set()
        # All forget tiers leak; retain survives.
        correctness = {p.probe_id: True for p in ps.probes}
        r = score_breadth(ps, correctness)
        assert r.mean_forget_leakage == pytest.approx(1.0)
        assert r.retain_accuracy == pytest.approx(1.0)

    def test_perfect_narrow_forgetting_shows_positive_gap(self):
        """B0 forgotten, B3 still answerable -> narrow forgetting."""
        ps = _probe_set()
        correctness = {p.probe_id: (p.tier is not Tier.EXACT) for p in ps.probes}
        r = score_breadth(ps, correctness)
        assert r.tier_scores["B0"].rate == 0.0
        assert r.tier_scores["B3"].rate == 1.0
        assert r.breadth_generalisation_gap == pytest.approx(-1.0)

    def test_missing_correctness_skipped_not_counted_false(self):
        ps = _probe_set()
        correctness = {ps.probes[0].probe_id: False}
        r = score_breadth(ps, correctness)
        assert r.tier_scores["B0"].n == 1
        assert "B1" not in r.tier_scores, "unevaluated tiers must be absent, not zero"


# ----------------------------------------------------------------- plane ----

class TestPlane:
    def _trajectory(self, depths, breadths, utilities=None):
        pts = [
            PlanePoint("GA", "m", "tofu", alpha=a, breadth=b, depth=d,
                       utility=None if utilities is None else utilities[i])
            for i, (a, b, d) in enumerate(zip(np.linspace(0, 1, len(depths)), breadths, depths))
        ]
        return Trajectory("GA", "m", "tofu", pts)

    def test_tradeoff_detected_as_negative_rho(self):
        """Breadth up, depth down -> the D1 hypothesis signature."""
        t = self._trajectory(depths=[0.9, 0.8, 0.7, 0.5, 0.3], breadths=[0.2, 0.4, 0.55, 0.7, 0.9])
        rho, p = t.tradeoff_correlation()
        assert rho < 0 and p < 0.05

    def test_aligned_axes_give_positive_rho(self):
        t = self._trajectory(depths=[0.2, 0.4, 0.6, 0.8, 0.9], breadths=[0.2, 0.4, 0.6, 0.8, 0.9])
        rho, _ = t.tradeoff_correlation()
        assert rho > 0

    def test_too_few_points_returns_none(self):
        assert self._trajectory([0.5, 0.4], [0.2, 0.4]).tradeoff_correlation() is None

    def test_constant_axis_returns_none(self):
        assert self._trajectory([0.5] * 5, [0.1, 0.2, 0.3, 0.4, 0.5]).tradeoff_correlation() is None

    # -- effect size gates the verdict -------------------------------------
    #
    # The first real GA sweep returned rho=+0.89 at p=0.0002 and the verdict
    # announced it CONTRADICTS the hypothesis. Two things were wrong with that.
    # The p-value assumes independent observations, but eleven alpha values are
    # eleven points on one continuous path -- evaluating twenty-one would have
    # "doubled the evidence" for free. And both axes moved through under a tenth
    # of their calibrated range, so there was no curve to describe.

    #: breadth and depth as actually measured, GA / 1B / forget10.
    OBSERVED_GA = dict(
        breadths=[0.000, 0.000, 0.019, 0.000, 0.037, 0.037, 0.074, 0.074,
                  0.056, 0.056, 0.093],
        depths=[0.031, 0.033, 0.036, 0.040, 0.043, 0.048, 0.053, 0.057,
                0.062, 0.067, 0.071],
    )

    def test_the_real_sweep_had_a_strong_correlation(self):
        rho, p = self._trajectory(**self.OBSERVED_GA).tradeoff_correlation()
        assert rho > 0.8 and p < 0.05, "the correlation itself is real..."

    def test_and_is_nonetheless_rejected_as_underpowered(self):
        assert not self._trajectory(**self.OBSERVED_GA).has_usable_dynamic_range()

    def test_spans_are_reported(self):
        s = self._trajectory(**self.OBSERVED_GA).axis_spans()
        assert s["breadth_span"] == pytest.approx(0.093)
        assert s["depth_span"] == pytest.approx(0.040)
        assert s["depth_peak"] == pytest.approx(0.071)

    def test_a_wide_sweep_is_usable(self):
        t = self._trajectory(depths=[0.1, 0.3, 0.5, 0.7, 0.9],
                             breadths=[0.1, 0.3, 0.5, 0.7, 0.9])
        assert t.has_usable_dynamic_range()

    def test_a_trade_off_still_needs_range(self):
        """The gate is symmetric: a negative rho over a tiny span is no more
        interpretable than a positive one."""
        t = self._trajectory(depths=[0.09, 0.08, 0.07, 0.06, 0.05],
                             breadths=[0.01, 0.02, 0.03, 0.04, 0.05])
        rho, _ = t.tradeoff_correlation()
        assert rho < 0 and not t.has_usable_dynamic_range()

    def test_one_wide_axis_is_not_enough(self):
        """Depth can only be plotted against breadth if both moved."""
        t = self._trajectory(depths=[0.1, 0.3, 0.5, 0.7, 0.9],
                             breadths=[0.01, 0.02, 0.03, 0.04, 0.05])
        assert not t.has_usable_dynamic_range()

    def test_p_value_is_documented_as_invalid(self):
        assert Trajectory.POINTS_ARE_INDEPENDENT is False

    def test_summary_carries_the_gate(self):
        ds = PlaneDataset([
            PlanePoint("GA", "m", "tofu", alpha=a, breadth=b, depth=d)
            for a, b, d in zip(np.linspace(0, 1, 11),
                               self.OBSERVED_GA["breadths"],
                               self.OBSERVED_GA["depths"])
        ])
        row = ds.tradeoff_summary()[0]
        assert row["usable_dynamic_range"] is False
        assert row["points_independent"] is False
        assert row["depth_peak"] == pytest.approx(0.071)

    def test_utility_collapse_detected(self):
        """The critical control: if utility craters, a depth drop is just
        general damage, not a trade-off."""
        stable = self._trajectory([0.9, 0.7, 0.5], [0.2, 0.5, 0.8], utilities=[0.60, 0.59, 0.58])
        collapsed = self._trajectory([0.9, 0.7, 0.5], [0.2, 0.5, 0.8], utilities=[0.60, 0.40, 0.20])
        assert stable.utility_is_stable() is True
        assert collapsed.utility_is_stable() is False

    def test_utility_none_when_unlogged(self):
        assert self._trajectory([0.9, 0.7, 0.5], [0.2, 0.5, 0.8]).utility_is_stable() is None

    def test_pareto_front_excludes_dominated(self):
        t = self._trajectory(depths=[0.9, 0.5, 0.8], breadths=[0.2, 0.4, 0.9])
        front = t.pareto_front()
        assert any(p.depth == 0.8 and p.breadth == 0.9 for p in front)
        assert not any(p.depth == 0.5 and p.breadth == 0.4 for p in front)

    def test_dataset_groups_into_trajectories(self):
        ds = PlaneDataset()
        for a in (0.0, 0.5, 1.0):
            ds.add(PlanePoint("GA", "m1", "tofu", a, 0.3 + a * 0.4, 0.9 - a * 0.4))
            ds.add(PlanePoint("NPO", "m1", "tofu", a, 0.4, 0.7))
        assert len(ds.trajectories()) == 2

    def test_summary_marks_supporting_trajectories(self):
        ds = PlaneDataset()
        for i, a in enumerate(np.linspace(0, 1, 6)):
            ds.add(PlanePoint("GA", "m", "tofu", float(a),
                              breadth=0.2 + 0.13 * i, depth=0.9 - 0.13 * i))
        row = ds.tradeoff_summary()[0]
        assert row["supports_tradeoff"] is True
        assert row["spearman_rho"] < 0

    def test_overshoot_points_excluded_from_correlation(self):
        """An overshoot point with an extreme depth must not be allowed to
        manufacture (or destroy) the headline correlation."""
        pts = [
            PlanePoint("GA", "m", "tofu", 0.00, breadth=0.20, depth=0.90),
            PlanePoint("GA", "m", "tofu", 0.25, breadth=0.40, depth=0.80),
            PlanePoint("GA", "m", "tofu", 0.50, breadth=0.60, depth=0.70),
            PlanePoint("GA", "m", "tofu", 0.75, breadth=0.80, depth=0.60),
            PlanePoint("GA", "m", "tofu", 1.00, breadth=0.95, depth=1.00,
                       depth_overshoot=True),
        ]
        t = Trajectory("GA", "m", "tofu", pts)
        assert t.n_overshoot == 1
        rho_excl, _ = t.tradeoff_correlation(exclude_overshoot=True)
        rho_incl, _ = t.tradeoff_correlation(exclude_overshoot=False)
        assert rho_excl == pytest.approx(-1.0)
        assert rho_incl > rho_excl, "including the overshoot point must change the result"

    def test_summary_reports_overshoot_count(self):
        ds = PlaneDataset()
        for i, a in enumerate(np.linspace(0, 1, 5)):
            ds.add(PlanePoint("GA", "m", "tofu", float(a),
                              breadth=0.2 + 0.15 * i, depth=0.9 - 0.15 * i,
                              depth_overshoot=(i == 4)))
        assert ds.tradeoff_summary()[0]["n_overshoot_excluded"] == 1

    def test_roundtrip_preserves_overshoot_flag(self, tmp_path):
        ds = PlaneDataset([PlanePoint("GA", "m", "tofu", 1.0, 0.5, 0.5, depth_overshoot=True)])
        loaded = PlaneDataset.load(ds.save(tmp_path / "p.json"))
        assert loaded.points[0].depth_overshoot is True

    def test_roundtrip_save_load(self, tmp_path):
        ds = PlaneDataset([PlanePoint("GA", "m", "tofu", 0.5, 0.4, 0.6, smr=0.01, el10=0.3)])
        path = ds.save(tmp_path / "plane.json")
        loaded = PlaneDataset.load(path)
        assert len(loaded.points) == 1
        assert loaded.points[0].smr == 0.01

    def test_plot_writes_file(self, tmp_path):
        """Plotting unit test with a SYNTHETIC fixture.

        These values are fabricated purely to exercise the rendering path.
        This is the only place synthetic depth/breadth values are permitted;
        the smoke script must emit values computed by the real pipeline.
        """
        from deeperase.eval.plane import plot_plane
        ds = PlaneDataset()
        for a in np.linspace(0, 1, 5):
            ds.add(PlanePoint("SYNTHETIC-FIXTURE", "none", "unit-test", float(a),
                              0.2 + 0.6 * a, 0.9 - 0.5 * a,
                              notes="synthetic plotting fixture -- not data"))
        out = tmp_path / "plane.png"
        plot_plane(ds, out)
        assert out.exists() and out.stat().st_size > 1000


class TestOvershootSensitivity:
    """Overshoot points must be surfaced, not silently dropped."""

    def _traj_with_overshoot(self):
        pts = [
            PlanePoint("GA", "m", "tofu", 0.00, breadth=0.20, depth=0.90),
            PlanePoint("GA", "m", "tofu", 0.25, breadth=0.40, depth=0.80),
            PlanePoint("GA", "m", "tofu", 0.50, breadth=0.60, depth=0.70),
            PlanePoint("GA", "m", "tofu", 0.75, breadth=0.80, depth=0.60),
            PlanePoint("GA", "m", "tofu", 1.00, breadth=0.95, depth=1.00,
                       depth_overshoot=True),
        ]
        return Trajectory("GA", "m", "tofu", pts)

    def test_reports_both_variants(self):
        s = self._traj_with_overshoot().tradeoff_sensitivity()
        assert s["rho_excluding"] is not None
        assert s["rho_including"] is not None
        assert s["rho_excluding"] != s["rho_including"]

    def test_flags_non_robust_conclusion(self):
        """Sign/significance disagreement between variants means the
        conclusion depends on the filtering choice and must be reported."""
        s = self._traj_with_overshoot().tradeoff_sensitivity()
        assert s["conclusion_is_robust"] is False

    def test_flags_robust_conclusion_when_variants_agree(self):
        pts = [PlanePoint("GA", "m", "tofu", a, breadth=0.2 + 0.15 * i, depth=0.9 - 0.15 * i,
                          depth_overshoot=(i == 5))
               for i, a in enumerate(np.linspace(0, 1, 6))]
        s = Trajectory("GA", "m", "tofu", pts).tradeoff_sensitivity()
        assert s["conclusion_is_robust"] is True

    def test_records_which_alphas_overshot(self):
        s = self._traj_with_overshoot().tradeoff_sensitivity()
        assert s["n_overshoot"] == 1
        assert s["overshoot_alphas"] == [1.00]

    def test_dataset_sensitivity_report_covers_all_trajectories(self):
        ds = PlaneDataset()
        for method in ("GA", "NPO"):
            for i, a in enumerate(np.linspace(0, 1, 4)):
                ds.add(PlanePoint(method, "m", "tofu", float(a),
                                  breadth=0.2 + 0.2 * i, depth=0.9 - 0.2 * i))
        report = ds.sensitivity_report()
        assert len(report) == 2
        assert {r["method"] for r in report} == {"GA", "NPO"}
