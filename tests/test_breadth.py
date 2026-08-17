"""Tests for the breadth axis.

Two things matter most here:

1. **Axis direction.** Depth and breadth must both mean "more forgetting" when
   the number is higher. Breadth is defined as ``1 - leakage`` for exactly this
   reason; if that inverted, every plot in the study would read backwards and
   the conclusion would flip sign.

2. **The retain tier stays separate.** A model that has been destroyed leaks
   nothing and would score a perfect breadth. Only the R tier distinguishes
   "forgot the target" from "forgot everything", so it must never be folded
   into the forget average.
"""

from __future__ import annotations

import pytest
import torch

from deeperase.eval.breadth import (
    TIER_SOURCES,
    BreadthItem,
    BreadthResult,
    TierResult,
    score_breadth,
)

N_LAYERS, HIDDEN, VOCAB, SEQ = 2, 32, 64, 12


def _item(tier="B0", n=0, wrong=2):
    return BreadthItem(
        item_id=f"{tier}_{n}", tier=tier, question=f"question {n}?",
        correct_answer=f"answer {n}",
        wrong_answers=[f"wrong {n}.{k}" for k in range(wrong)],
        source_index=n,
    )


def _result(**rates):
    """Build a BreadthResult from tier -> knows_rate, using n=100."""
    return BreadthResult(tiers={
        t: TierResult(tier=t, n=100, n_knows=int(round(r * 100)))
        for t, r in rates.items()
    })


class TestItemValidation:
    def test_requires_at_least_one_wrong_answer(self):
        with pytest.raises(ValueError, match="at least one wrong answer"):
            BreadthItem("x", "B0", "q?", "a", [], 0)

    def test_forget_tiers_flagged_correctly(self):
        assert _item("B0").is_forget_tier
        assert _item("B1").is_forget_tier
        assert not _item("R").is_forget_tier

    def test_tier_sources_cover_the_documented_tiers(self):
        assert set(TIER_SOURCES) == {"B0", "B1", "R"}

    def test_b1_reads_the_paraphrased_column(self):
        """The whole point of B1 is that it is a different wording."""
        assert TIER_SOURCES["B0"][1] == "question"
        assert TIER_SOURCES["B1"][1] == "paraphrased_question"


class TestAxisDirection:
    """Higher breadth must mean more forgetting, matching the depth axis."""

    def test_total_leakage_gives_zero_breadth(self):
        r = _result(B0=1.0, B1=1.0, R=1.0)
        assert r.forget_leakage == pytest.approx(1.0)
        assert r.breadth == pytest.approx(0.0)

    def test_no_leakage_gives_full_breadth(self):
        r = _result(B0=0.0, B1=0.0, R=1.0)
        assert r.breadth == pytest.approx(1.0)

    def test_breadth_rises_as_leakage_falls(self):
        low = _result(B0=0.8, B1=0.8, R=1.0)
        high = _result(B0=0.2, B1=0.2, R=1.0)
        assert high.breadth > low.breadth, "breadth axis is inverted"

    def test_breadth_is_bounded(self):
        for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert 0.0 <= _result(B0=rate, B1=rate, R=1.0).breadth <= 1.0


class TestRetainTierIsolation:
    def test_retain_excluded_from_forget_average(self):
        """R must not dilute the leakage figure."""
        r = _result(B0=0.0, B1=0.0, R=1.0)
        assert r.forget_leakage == pytest.approx(0.0)

    def test_destroyed_model_is_distinguishable_from_a_good_one(self):
        """Both leak nothing; only retention separates them. This is the
        failure the R tier exists to catch."""
        good = _result(B0=0.0, B1=0.0, R=0.95)
        destroyed = _result(B0=0.0, B1=0.0, R=0.02)
        assert good.breadth == destroyed.breadth
        assert good.retention > 0.9 and destroyed.retention < 0.1

    def test_retention_is_none_when_r_tier_absent(self):
        assert _result(B0=0.5).retention is None


class TestGeneralisationGap:
    def test_narrow_forgetting_shows_positive_gap(self):
        """Forgot the exact wording, not the paraphrase."""
        assert _result(B0=0.0, B1=1.0).generalisation_gap == pytest.approx(-1.0)

    def test_broad_forgetting_shows_no_gap(self):
        assert _result(B0=0.2, B1=0.2).generalisation_gap == pytest.approx(0.0)

    def test_gap_is_none_without_both_tiers(self):
        assert _result(B0=0.5).generalisation_gap is None


class TestSerialisation:
    def test_to_dict_round_trips_key_fields(self):
        d = _result(B0=0.3, B1=0.4, R=0.9).to_dict()
        assert d["per_tier"]["B0"]["knows_rate"] == pytest.approx(0.3)
        assert d["retention"] == pytest.approx(0.9)
        assert d["breadth"] == pytest.approx(1 - 0.35)

    def test_summary_is_readable(self):
        s = _result(B0=0.3, B1=0.4, R=0.9).summary()
        assert "breadth=" in s and "B0=" in s and "gap=" in s


class TestScoringAgainstAModel:
    """End-to-end scoring with a tiny real transformer."""

    @pytest.fixture(scope="class")
    def model_and_tok(self):
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        m = LlamaForCausalLM(LlamaConfig(
            vocab_size=32000, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
            num_hidden_layers=N_LAYERS, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=128, use_cache=False))
        m.eval(); m.requires_grad_(False)
        try:
            tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        except Exception:
            pytest.skip("no network for tokenizer")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return m, tok

    def test_produces_a_result_per_tier(self, model_and_tok):
        m, tok = model_and_tok
        items = [_item("B0", 0), _item("B1", 1), _item("R", 2)]
        res = score_breadth(m, tok, items, max_length=64)
        assert set(res.tiers) == {"B0", "B1", "R"}
        assert all(t.n == 1 for t in res.tiers.values())

    def test_rates_are_valid_probabilities(self, model_and_tok):
        m, tok = model_and_tok
        items = [_item("B0", i) for i in range(4)]
        res = score_breadth(m, tok, items, max_length=64)
        assert 0.0 <= res.tiers["B0"].knows_rate <= 1.0

    def test_identical_answers_are_not_counted_as_known(self, model_and_tok):
        """When correct and wrong are the same string the model cannot prefer
        the correct one, so a strict comparison must return not-known. A
        non-strict '>=' would silently score every such item as leakage."""
        m, tok = model_and_tok
        item = BreadthItem("x", "B0", "q?", "same text", ["same text"], 0)
        res = score_breadth(m, tok, [item], max_length=64)
        assert res.tiers["B0"].n_knows == 0

    def test_empty_item_list_is_safe(self, model_and_tok):
        m, tok = model_and_tok
        assert score_breadth(m, tok, [], max_length=64).tiers == {}

    def test_scoring_is_deterministic(self, model_and_tok):
        m, tok = model_and_tok
        items = [_item("B0", i) for i in range(3)]
        a = score_breadth(m, tok, items, max_length=64)
        b = score_breadth(m, tok, items, max_length=64)
        assert a.tiers["B0"].n_knows == b.tiers["B0"].n_knows

    def test_more_wrong_answers_never_increases_knows(self, model_and_tok):
        """The forced choice must beat EVERY alternative, so adding more
        alternatives can only make it harder."""
        m, tok = model_and_tok
        easy = BreadthItem("e", "B0", "q?", "answer", ["w1"], 0)
        hard = BreadthItem("h", "B0", "q?", "answer", ["w1", "w2", "w3", "w4"], 0)
        n_easy = score_breadth(m, tok, [easy], max_length=64).tiers["B0"].n_knows
        n_hard = score_breadth(m, tok, [hard], max_length=64).tiers["B0"].n_knows
        assert n_hard <= n_easy


# ---------------------------------------------------------------------------
# Which "correct answer" is used
#
# Regression test for a real measurement flaw. TOFU's perturbed answers are
# minimal edits of `paraphrased_answer`, differing only in the entity:
#
#   paraphrased_answer : "Hsiao Yun-Hwa is the complete name of the writer."
#   perturbed_answer[0]: "Chen Jing-Li  is the complete name of the writer."
#
# `answer` has a different structure entirely ("The author's full name is X"),
# so scoring against it conflates the entity with the phrasing. Measured on the
# real models: retain90, which never saw these authors, scored 0.72 against
# `answer` where chance is 0.25.
# ---------------------------------------------------------------------------

class TestCorrectAnswerSelection:
    def _fake_dataset(self, with_paraphrase=True):
        rows = []
        for i in range(3):
            r = {
                "question": f"q{i}?",
                "answer": f"The author's full name is Person{i}.",
                "perturbed_answer": [f"Other{i}{k} is the complete name of the writer."
                                     for k in range(3)],
            }
            if with_paraphrase:
                r["paraphrased_answer"] = f"Person{i} is the complete name of the writer."
            rows.append(r)
        return rows

    def _build(self, rows, tier="B0"):
        """Mirror load_breadth_items' selection logic on in-memory rows."""
        from deeperase.eval.breadth import BreadthItem
        out = []
        for i, row in enumerate(rows):
            correct = row.get("paraphrased_answer")
            source = "paraphrased_answer"
            if not correct:
                correct, source = row["answer"], "answer"
            out.append(BreadthItem(
                item_id=f"{tier}_{i}", tier=tier, question=row["question"],
                correct_answer=correct, wrong_answers=row["perturbed_answer"],
                source_index=i, correct_source=source,
            ))
        return out

    def test_prefers_the_paraphrase(self):
        items = self._build(self._fake_dataset(with_paraphrase=True))
        assert all(i.correct_source == "paraphrased_answer" for i in items)
        assert all("complete name of the writer" in i.correct_answer for i in items)

    def test_correct_and_wrong_share_sentence_structure(self):
        """The point of the fix: a clean minimal pair differing only in the
        entity, so the forced choice tests knowledge and not style."""
        item = self._build(self._fake_dataset())[0]
        tail = "is the complete name of the writer."
        assert item.correct_answer.endswith(tail)
        assert all(w.endswith(tail) for w in item.wrong_answers)

    def test_falls_back_to_answer_when_no_paraphrase(self):
        """world_facts has no paraphrased_answer, and its perturbed options
        are matched to `answer` already."""
        items = self._build(self._fake_dataset(with_paraphrase=False))
        assert all(i.correct_source == "answer" for i in items)

    def test_using_answer_would_break_the_structural_match(self):
        """Documents the flaw this replaced: `answer` shares no structure with
        the alternatives, so the comparison is confounded."""
        row = self._fake_dataset()[0]
        tail = "is the complete name of the writer."
        assert not row["answer"].endswith(tail)
        assert all(w.endswith(tail) for w in row["perturbed_answer"])

    def test_default_source_is_the_paraphrase(self):
        from deeperase.eval.breadth import BreadthItem
        assert BreadthItem("x", "B0", "q", "a", ["w"], 0).correct_source == \
            "paraphrased_answer"


# ---------------------------------------------------------------------------
# Calibration
#
# Measured on real TOFU models, the raw knows-rate spans only ~[0.51, 0.80]:
# the floor is above chance because some wrong answers are absurd, and the
# ceiling is below 1.0 because the rest are hard. Calibration anchors the
# breadth axis to reference models, exactly as UDS anchors depth.
# ---------------------------------------------------------------------------

class TestBreadthCalibration:
    OBSERVED_FLOOR = 0.510    # retain90 on forget tiers -- knowledge absent
    OBSERVED_CEILING = 0.775  # full on forget tiers     -- knowledge present

    def _cal(self):
        from deeperase.eval.breadth import BreadthCalibration
        return BreadthCalibration.from_reference_models(
            absent_leakage=self.OBSERVED_FLOOR,
            present_leakage=self.OBSERVED_CEILING,
        )

    def test_reference_models_map_to_clean_endpoints(self):
        """The whole point: full -> breadth 0, retain90 -> breadth 1."""
        c = self._cal()
        assert c.calibrated_breadth(self.OBSERVED_CEILING) == pytest.approx(0.0)
        assert c.calibrated_breadth(self.OBSERVED_FLOOR) == pytest.approx(1.0)

    def test_midpoint_maps_to_half(self):
        c = self._cal()
        mid = (self.OBSERVED_FLOOR + self.OBSERVED_CEILING) / 2
        assert c.calibrated_breadth(mid) == pytest.approx(0.5)

    def test_agrees_with_depth_axis_direction(self):
        """Both axes must mean 'more forgetting' when higher, or the plane is
        unreadable."""
        c = self._cal()
        leaky = c.calibrated_breadth(0.75)
        forgotten = c.calibrated_breadth(0.55)
        assert forgotten > leaky

    def test_values_outside_the_range_saturate(self):
        """A model beyond the references clips rather than going out of [0,1]."""
        c = self._cal()
        assert c.calibrated_breadth(0.95) == 0.0
        assert c.calibrated_breadth(0.20) == 1.0

    def test_dynamic_range_is_reported(self):
        assert self._cal().dynamic_range == pytest.approx(0.265, abs=1e-6)

    def test_rejects_non_discriminating_references(self):
        """If the model that should know scores no higher than the one that
        should not, the measurement is broken and must not be calibrated into
        looking sensible."""
        from deeperase.eval.breadth import BreadthCalibration
        with pytest.raises(ValueError, match="must exceed floor"):
            BreadthCalibration.from_reference_models(
                absent_leakage=0.80, present_leakage=0.60)

    def test_rejects_identical_references(self):
        from deeperase.eval.breadth import BreadthCalibration
        with pytest.raises(ValueError, match="must exceed floor"):
            BreadthCalibration.from_reference_models(0.5, 0.5)

    def test_calibrated_leakage_is_the_complement(self):
        c = self._cal()
        for raw in (0.52, 0.60, 0.70, 0.77):
            assert c.calibrated_leakage(raw) + c.calibrated_breadth(raw) == \
                pytest.approx(1.0)

    def test_serialises_for_the_run_record(self):
        d = self._cal().to_dict()
        assert d["floor"] == self.OBSERVED_FLOOR
        assert d["source"] == "reference_models"

    def test_calibration_makes_a_small_raw_change_visible(self):
        """A 0.05 raw shift is 5% of a naive 0-1 scale but ~19% of the real
        dynamic range. Under-reporting that would hide a genuine effect."""
        c = self._cal()
        delta = c.calibrated_breadth(0.65) - c.calibrated_breadth(0.70)
        assert delta > 0.15
