"""Tests for the UNASKED criteria report (scripts/unasked_report.py)."""

from __future__ import annotations

import json

import pytest

from deeperase.scripts import unasked_report as ur


def _ex(d2: float):
    return {"example_id": "ref_1", "delta_s1": {"0": 1.0, "1": 1.0},
            "delta_s2": {"0": d2, "1": d2}, "ke_layers": [0, 1], "uds": min(d2, 1.0)}


def _run(root, name, seed, d2_trained, d2_para, ra=0.85, wf=0.84, control=None):
    d = root / name
    (d / "partial").mkdir(parents=True)
    cfg = {"method": name.split("_")[2] if name.startswith("study") else "unasked",
           "seed": seed, "protocol": "v2", "unlearn": {"loss_scope": "answer"}, "dtype": "float32",
           "control": "random" if control else "none", "control_scale": control}
    (d / "config.json").write_text(json.dumps(cfg))
    for a, scale in ((0.0, 1.0), (1.0, 1.5)):
        pts = [{"benchmark": f"tofu-forget10:{s}", "alpha": a, "utility": 0.8 - 0.05 * a}
               for s in ("trained", "heldout")]
        depth = {"trained": {"tau": 0.05, "per_example": [_ex(d2_trained * scale)]},
                 "heldout": {"tau": 0.05, "per_example": [_ex(0.2 * scale)]},
                 "trained_para": {"tau": 0.05, "per_example": [_ex(d2_para * scale)]}}
        groups = {"R": {"rate": 0.8 - 0.05 * a}, "RA": {"rate": ra}, "WF": {"rate": wf}}
        (d / "partial" / f"alpha_{a:.3f}.json").write_text(json.dumps(
            {"points": pts, "point": pts[0], "depth": depth, "groups": groups}))
    return d


def test_criteria_from_saved_runs(tmp_path):
    _run(tmp_path, "study_v2_unasked_1B_s0_lr3e-06", 0, 0.5, 0.4)
    _run(tmp_path, "study_v2_unasked_1B_s0_lr3e-06_lam3", 0, 0.9, 0.9)   # a variant: not in E1-E5
    _run(tmp_path, "control_x10_v2_unasked_1B_s0_lr3e-06", 0, 0.1, 0.1, control=10.0)
    _run(tmp_path, "study_v2_ga_1B_s0_lr1e-08", 0, 0.2, 0.1)
    out = tmp_path / "rep"
    assert ur.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert list(rep["seeds"]) == ["0"]
    c = rep["criteria"]
    assert c["E1_checkpoint_strict_trained"]["value"] == pytest.approx(0.5)
    assert c["E1_checkpoint_strict_trained"]["verdict"] == "met"
    assert c["E3_paraphrase_ratio"]["ratio"] == pytest.approx(0.8)
    assert c["E3_paraphrase_ratio"]["verdict"] == "met"
    # matched at R 0.80 (0.5 - 0.1) and 0.75 (0.75 - 0.15): mean +0.5
    assert c["E2_excess_strict_trained"]["per_seed"][0] == pytest.approx(0.5)
    assert c["E4_general_knowledge"]["verdict"] == "met"
    assert "study_v2_ga_1B_s0_lr1e-08" in rep["comparisons"]
    lam3 = rep["variants"]["study_v2_unasked_1B_s0_lr3e-06_lam3"]
    assert lam3["checkpoint"]["trained"]["strict"] == pytest.approx(0.9)
    assert lam3["alpha1"]["alpha"] == 1.0
    assert "_lam3" in out.with_suffix(".md").read_text()
    assert "E1 checkpoint strict UDS" in out.with_suffix(".md").read_text()


def test_no_runs_is_an_error(tmp_path):
    assert ur.main(["--dir", str(tmp_path), "--out", str(tmp_path / "x")]) == 1
