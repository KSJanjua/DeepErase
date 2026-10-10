"""Per-fact depth resolution against TOFU's nested checkpoints (pure JSON)."""

from __future__ import annotations

import json

import pytest

from deeperase.scripts import depth_fact_resolution as dfr


def test_auc_counts_ties_as_half():
    assert dfr.auc([0.9, 0.8], [0.1, 0.0]) == 1.0
    assert dfr.auc([0.5], [0.5]) == 0.5
    assert dfr.auc([0.2, 0.9], [0.5]) == 0.5
    assert dfr.auc([], [0.1]) is None


def test_resolve_uses_the_nested_split_boundary():
    pe = ([{"example_id": f"ref_{r}", "uds": 0.95} for r in (360, 380, 399)]
          + [{"example_id": f"forget10_{r}", "uds": 0.02} for r in (0, 150, 250)]
          + [{"example_id": "ref_5", "uds": None}])
    x = dfr.resolve(pe, dfr.UNSEEN_FROM["retain99"])
    assert (x["n_unseen"], x["n_seen"], x["auc"]) == (3, 3, 1.0)
    assert x["accuracy"]["0.5"] == 1.0
    # under retain95's boundary, row 250 counts as unseen and is misread
    y = dfr.resolve(pe, dfr.UNSEEN_FROM["retain95"])
    assert (y["n_unseen"], y["n_seen"]) == (4, 2)
    assert y["accuracy"]["0.5"] == pytest.approx(5 / 6) and y["auc"] == pytest.approx(7 / 8)


def test_main_skips_runs_without_ground_truth(tmp_path):
    def report(name, rows):
        d = tmp_path / name
        d.mkdir()
        pe = [{"example_id": f"ref_{r}", "uds": 1.0 if r >= 360 else 0.0} for r in rows]
        (d / "report.json").write_text(json.dumps({
            "config": {"size_label": "1B"},
            "per_split_reports": {"retain99": {"per_example": pe}}}))
    report("good", [0, 100, 365, 390])
    report("prefix_only", [0, 1, 2])
    out = tmp_path / "res.json"
    assert dfr.main(["--runs", str(tmp_path), "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert list(got) == ["good"] and got["good"]["sources"]["retain99"]["auc"] == 1.0
