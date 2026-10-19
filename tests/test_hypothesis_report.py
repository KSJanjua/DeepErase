"""The pre-registered decision rules (scripts/hypothesis_report.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeperase.scripts import hypothesis_report as hr

ROOT = Path(__file__).resolve().parents[1]


class TestH1:
    def test_needs_selectivity_in_every_seed_and_a_beaten_control(self):
        v = hr.h1_verdict({0: 0.1, 1: 0.05}, (0.5, 0.4), (0.3, 0.2))
        assert v["verdict"] == "supported" and v["beats_control"]

    def test_one_seed_without_selectivity_fails(self):
        v = hr.h1_verdict({0: 0.1, 1: -0.02}, (0.5, 0.4), (0.3, 0.2))
        assert v["verdict"].startswith("not supported") and "seed 1" in v["verdict"]

    def test_the_control_must_be_beaten_on_both_axes(self):
        v = hr.h1_verdict({0: 0.1}, (0.5, 0.1), (0.3, 0.2))
        assert "matched control" in v["verdict"] and v["beats_control"] is False
        assert "no matched control" in hr.h1_verdict({0: 0.1}, None, None)["verdict"]


class TestH2:
    def _seed(self, s0, lo, s1):
        return {th: {"s0": s0, "s0_ci": [lo, 1.0], "s1": s1} for th in ("0.3", "0.5", "0.7")}

    def test_common_in_two_of_three_seeds_and_converting_in_all(self):
        v = hr.h2_verdict({0: self._seed(0.8, 0.5, 0.2), 1: self._seed(0.7, 0.3, 0.4),
                           2: self._seed(0.3, 0.05, 0.1)})      # seed 2: CI too wide
        assert v["common@0.5"] == 2 and v["suppression_common"] == "supported"
        assert v["extrapolation_converts"] == "supported" and v["robust_to_theta"]

    def test_a_seed_where_suppression_grows_breaks_conversion(self):
        v = hr.h2_verdict({0: self._seed(0.8, 0.5, 0.2), 1: self._seed(0.6, 0.3, 0.9)})
        assert "seed(s) 1" in v["extrapolation_converts"]


class TestH3:
    def test_supported_refuted_and_mixed(self):
        good = {0: [(0.0, 0.1, [-0.1, 0.3]), (1.0, 0.4, [0.1, 0.6])],
                1: [(0.0, 0.0, [-0.2, 0.2]), (1.0, 0.3, [0.05, 0.5])]}
        assert hr.h3_verdict(good)["verdict"] == "supported"
        none = {0: [(0.5, 0.1, [-0.1, 0.3]), (1.0, 0.1, [-0.1, 0.3])]}
        assert hr.h3_verdict(none)["verdict"].startswith("refuted")
        mixed = {0: [(0.5, 0.3, [0.05, 0.5]), (1.0, 0.1, [-0.1, 0.3])]}
        assert hr.h3_verdict(mixed)["verdict"].startswith("not supported")


class TestH4H5:
    def test_h4_names_every_failure(self):
        v = hr.h4_verdict({0: {"breadth_slope": 0.2, "depth_slope": 0.3, "max_selectivity": 0.1},
                           1: {"breadth_slope": -0.1, "depth_slope": 0.3,
                               "max_selectivity": -0.01}})
        assert "breadth slope <= 0 in seed 1" in v["verdict"]
        assert "selectivity never > 0 in seed 1" in v["verdict"]

    def test_h5_ratio_and_b3_measurability(self):
        v = hr.h5_verdict({0: {"B0": 0.2, "B2": 0.1, "B3": 0.5, "B4": 0.3, "b3_range": 0.07},
                           1: {"B0": 0.2, "B2": 0.09, "B3": 0.5, "B4": 0.3, "b3_range": 0.03}})
        assert "B2 slope < 0.5 x B0 in seed 1" in v["verdict"]
        assert v["b3_measurable"] == {0: True, 1: False}


def test_excess_is_measured_at_equal_utility():
    control = {"utility": [0.80, 0.75, 0.70], "depth": [0.1, 0.2, 0.3],
               "breadth": [0.0, 0.1, 0.2]}
    real = {"utility": [0.775, 0.70, 0.60], "depth": [0.4, 0.6, 0.9],
            "breadth": [0.1, 0.1, 0.1]}
    ex = hr.excess_over_control(real, control, "depth")
    assert ex[0] == pytest.approx(0.4 - 0.15) and ex[1] == pytest.approx(0.6 - 0.3)
    assert ex[2] is None                          # outside the control's utility range


@pytest.mark.skipif(not (ROOT / "results" / "analysis" / "fact_level_v2.json").exists(),
                    reason="needs the committed v2 fact-level analysis")
def test_the_committed_runs_produce_a_verdict_for_every_hypothesis(tmp_path):
    out = tmp_path / "h"
    assert hr.main(["--dir", str(ROOT / "results" / "studies"), "--out", str(out),
                    "--fact-json", str(ROOT / "results" / "analysis" / "fact_level_v2.json")]) == 0
    d = json.loads(out.with_suffix(".json").read_text())
    for m in ("ga", "graddiff", "npo"):
        assert set(d[m]) >= {"H1", "H2", "H3", "H4", "H5", "damage_matched_excess"}
        assert d[m]["control"] and len(d[m]["runs"]) == 3


def test_own_control_cell_names_the_failing_seeds():
    assert hr._own_control_cell({}) == "–"
    assert hr._own_control_cell({0: True, 1: True}) == "2 of 2 seeds"
    assert hr._own_control_cell({0: True, 1: False, 2: True}) == "2 of 3 seeds (fails seed 1)"
    assert hr.mean_excess([None, 0.1, 0.3]) == pytest.approx(0.2)
    assert hr.mean_excess([None]) is None


def test_excess_over_control_is_defined_at_tied_control_utilities():
    """A control whose utility repeats along its path (R 0.80 at two alphas)
    gives the same excess whatever order a sort leaves the tied points in."""
    from deeperase.scripts.hypothesis_report import excess_over_control
    real = {"utility": [0.80, 0.79], "depth": [0.50, 0.55]}
    a = {"utility": [0.81, 0.80, 0.80, 0.78], "depth": [0.05, 0.10, 0.20, 0.30]}
    b = {"utility": [0.81, 0.80, 0.80, 0.78], "depth": [0.05, 0.20, 0.10, 0.30]}
    ea, eb = excess_over_control(real, a, "depth"), excess_over_control(real, b, "depth")
    assert ea == pytest.approx(eb)
    assert ea[0] == pytest.approx(0.50 - 0.15)          # the tied values' mean
    assert ea[1] == pytest.approx(0.55 - (0.15 + 0.30) / 2)


def test_excess_ignores_control_points_with_missing_values():
    real = {"utility": [0.8, 0.75], "strict": [0.5, 0.6]}
    control = {"utility": [0.8, None, 0.7], "strict": [0.1, 0.2, 0.3]}
    ex = hr.excess_over_control(real, control, "strict")
    assert ex[0] == pytest.approx(0.4)          # control 0.1 at R 0.80
    assert ex[1] == pytest.approx(0.4)          # control 0.2, interpolated at R 0.75
    assert hr.excess_over_control(real, {"utility": [None], "strict": [0.1]}, "strict") \
        == [None, None]
