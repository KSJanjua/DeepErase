"""Tests for the relearning-attack report (scripts/relearn_report.py)."""

from __future__ import annotations

import json

import pytest

from deeperase.scripts import relearn_report as rr


def _m(nll, knows):
    return {"probe": {"answer_nll": nll, "knows_rate": knows, "n_items": 10},
            "heldout": {"answer_nll": nll + 0.1, "knows_rate": knows, "n_items": 10}}


def test_single_rate_and_dose_records(tmp_path):
    # an old single-rate record: only "before" and "after"
    single = {"method": "ga", "seed": 0, "learning_rate": 1e-8, "attack_lr": 1e-5,
              "attack_epochs": 2,
              "results": {"unlearned": {"before": _m(2.0, 0.6), "after": _m(1.0, 0.7)},
                          "full": {"before": _m(0.4, 0.8), "after": _m(0.9, 0.75)},
                          "retain90": {"before": _m(2.4, 0.5), "after": _m(2.2, 0.55)}}}
    # a dose-response record: one "after_<lr>" per rate
    dose = {"method": "unasked", "seed": 0, "learning_rate": 3e-6, "attack_lr": 1e-5,
            "attack_lrs": [1e-6, 1e-5], "attack_epochs": 2, "results": {}}
    for who, b, a6, a5 in (("unlearned", 2.0, 1.8, 1.0), ("full", 0.4, 0.45, 0.9),
                           ("retain90", 2.4, 2.35, 2.2)):
        dose["results"][who] = {"before": _m(b, 0.6), "after_1e-06": _m(a6, 0.6),
                                "after_1e-05": _m(a5, 0.6), "after": _m(a5, 0.6)}
    (tmp_path / "ga_1B_v2_s0.json").write_text(json.dumps(single))
    (tmp_path / "unasked_1B_v2_s0_dose.json").write_text(json.dumps(dose))
    (tmp_path / "notes.json").write_text(json.dumps({"unrelated": True}))
    out = tmp_path / "rep"
    assert rr.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    rows = json.loads(out.with_suffix(".json").read_text())["rows"]
    assert [(r["record"], r["attack_lr"]) for r in rows] == [
        ("ga_1B_v2_s0", 1e-5), ("unasked_1B_v2_s0_dose", 1e-6), ("unasked_1B_v2_s0_dose", 1e-5)]
    ga = rows[0]
    assert ga["probe"]["index_nll"] == pytest.approx((1.0 - 2.2) / (0.9 - 2.2))
    assert ga["heldout"]["index_knows"] == pytest.approx((0.7 - 0.55) / (0.75 - 0.55))
    weak = rows[1]
    assert weak["probe"]["index_nll"] == pytest.approx((1.8 - 2.35) / (0.45 - 2.35))
    md = out.with_suffix(".md").read_text()
    assert "Latent index by NLL" in md and "unasked_1B_v2_s0_dose" in md


def test_no_records_is_an_error(tmp_path):
    assert rr.main(["--dir", str(tmp_path), "--out", str(tmp_path / "x")]) == 1
