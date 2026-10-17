"""Tests for the overshoot diagnostics (eval/uds_diagnostics.py, scripts/uds_audit.py)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from deeperase.eval import uds_diagnostics as ud
from deeperase.eval.uds import aggregate_example_uds

ROOT = Path(__file__).resolve().parents[1]


def test_standard_uds_matches_the_published_aggregation():
    rng = random.Random(0)
    for _ in range(50):
        d1 = {l: rng.choice([0.0, 0.01, rng.uniform(0.06, 3.0)]) for l in range(16)}
        d2 = {l: rng.uniform(-0.5, 6.0) for l in range(16)}
        ref, _, _ = aggregate_example_uds(d1, d2)
        got = ud.example_diagnostics(d1, d2)
        if ref is None:
            assert got is None
        else:
            assert got["uds"] == pytest.approx(ref)


def test_strict_and_tent_credit_overshoot_differently():
    d1 = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.01}        # layer 3 is not a KE layer
    d2 = {0: 1.0, 1: 1.5, 2: 3.0, 3: 9.0}         # r = 1, 1.5, 3
    e = ud.example_diagnostics(d1, d2)
    assert e["uds"] == pytest.approx(1.0)          # the clip treats all three alike
    assert e["strict"] == pytest.approx(2 / 3)     # r = 3 counts as no erasure
    assert e["tent"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert e["overshoot_share"] == pytest.approx(1 / 3)


def test_bands_and_late_ratios():
    d1 = {l: (1.0 if l in (2, 10, 14) else 0.0) for l in range(16)}
    d2 = {l: 0.0 for l in range(16)}
    d2.update({2: 1.0, 10: 0.5, 14: 2.5})
    e = ud.example_diagnostics(d1, d2)
    assert e["bands"] == {"L0-8": 1.0, "L9-12": 0.5, "L13-15": 1.0}
    assert sorted(e["late_ratios"]) == [0.5, 2.5]


def test_negative_ratio_gets_no_credit_and_no_ke_is_none():
    assert ud.example_diagnostics({0: 1.0}, {0: -0.3})["tent"] == 0.0
    assert ud.example_diagnostics({0: 0.01, 1: 0.0}, {0: 5.0, 1: 5.0}) is None


def test_report_diagnostics_reads_saved_string_keys():
    rep = {"tau": 0.05, "per_example": [
        {"delta_s1": {"0": 1.0, "1": 1.0}, "delta_s2": {"0": 1.0, "1": 4.0}},
        {"delta_s1": {"0": 0.0, "1": 0.0}, "delta_s2": {"0": 1.0, "1": 1.0}}]}
    s = ud.report_diagnostics(rep)
    assert s["n_scored"] == 1
    assert s["uds"] == pytest.approx(1.0) and s["strict"] == pytest.approx(0.5)
    assert s["share_late_cells_over_2"] is None     # no KE cell in layers 9-15


@pytest.mark.skipif(not (ROOT / "results" / "studies" / "study_v2_ga_1B_s0_lr1e-08_ext").exists(),
                    reason="needs the committed alpha-extension run")
def test_audit_reproduces_the_ga_overshoot_finding(tmp_path):
    from deeperase.scripts import uds_audit
    out = tmp_path / "audit"
    assert uds_audit.main(["--dir", str(ROOT / "results" / "studies"),
                           "--glob", "*_v2_ga_1B_s0_*", "--out", str(out)]) == 0
    d = json.loads(out.with_suffix(".json").read_text())
    ext = next(a for a in d["runs"] if a["run"] == "study_v2_ga_1B_s0_lr1e-08_ext")
    rows = {round(r["alpha"], 2): r for r in ext["splits"]["trained"]}
    assert rows[1.0]["uds"] == pytest.approx(rows[1.0]["stored_uds"], abs=1e-5)
    assert rows[1.0]["strict"] == pytest.approx(0.530, abs=2e-3)
    assert rows[2.0]["strict"] == pytest.approx(0.522, abs=2e-3)    # flat past alpha 1
    assert rows[2.0]["uds"] - rows[1.0]["uds"] > 0.1                 # while UDS rises
    assert "study_v2_ga_1B_s0_lr1e-08" in d["excess"]
