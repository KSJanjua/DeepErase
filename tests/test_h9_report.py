"""Tests for H9's report: UNASKED-G against GradDiff (scripts/h9_report.py)."""

from __future__ import annotations

import json

import pytest

from deeperase.scripts import h9_report as h9


def _ex(d2: float):
    return {"example_id": "ref_1", "delta_s1": {"0": 1.0, "1": 1.0},
            "delta_s2": {"0": d2, "1": d2}, "ke_layers": [0, 1], "uds": min(d2, 1.0)}


def _run(root, name, seed, *, strict, breadth, r=0.78, ra=0.87, wf=0.84, gw=None,
         control=None):
    d = root / name
    (d / "partial").mkdir(parents=True)
    method = name.split("_")[2]
    cfg = {"method": method, "seed": seed, "protocol": "v2", "dtype": "float32",
           "unlearn": {"loss_scope": "answer", "general_weight": gw, "retain_weight": 3.0,
                       "learning_rate": 1e-5 if "_lr1e-05" in name else 3e-6},
           "control": "random" if control else "none", "control_scale": control}
    (d / "config.json").write_text(json.dumps(cfg))
    for a, scale in ((0.0, 1.0), (1.0, 1.5)):
        pts = [{"benchmark": f"tofu-forget10:{s}", "alpha": a, "utility": 0.8 - 0.05 * a,
                "breadth": breadth * scale} for s in ("trained", "heldout")]
        depth = {s: {"tau": 0.05, "per_example": [_ex(strict * scale)]}
                 for s in ("trained", "heldout")}
        groups = {"R": {"rate": r - 0.05 * a}, "RA": {"rate": ra}, "WF": {"rate": wf}}
        (d / "partial" / f"alpha_{a:.3f}.json").write_text(json.dumps(
            {"points": pts, "point": pts[0], "depth": depth, "groups": groups}))


def _graddiff(root):
    for s, (st, br, r, ra, wf) in enumerate([(0.144, 0.175, 0.775, 0.88, 0.821),
                                             (0.120, 0.220, 0.765, 0.86, 0.829),
                                             (0.158, 0.112, 0.765, 0.85, 0.838)]):
        _run(root, f"study_v2_graddiff_1B_s{s}_lr5e-08", s, strict=st, breadth=br, r=r,
             ra=ra, wf=wf)


def test_rule_criteria_and_controls(tmp_path):
    _graddiff(tmp_path)
    # lambda_g = 1: deep and broad, but RA below GradDiff's -> C3 fails on seed 0
    _run(tmp_path, "study_v2_unaskedg_1B_s0_lr3e-06", 0, strict=0.40, breadth=0.40,
         ra=0.80, gw=1.0)
    _run(tmp_path, "control_x10_v2_unaskedg_1B_s0_lr3e-06", 0, strict=0.10, breadth=0.10,
         control=10.0)
    # lambda_g = 3 (tag g3): meets C1-C3 on seed 0
    _run(tmp_path, "study_v2_unaskedg_1B_s0_lr3e-06_g3", 0, strict=0.35, breadth=0.36,
         gw=3.0)
    _run(tmp_path, "study_v2_graddiffg_1B_s0_lr5e-08", 0, strict=0.14, breadth=0.17,
         ra=0.88, gw=1.0)
    out = tmp_path / "rep"
    assert h9.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert len(rep["graddiff"]) == 3 and len(rep["graddiff_g"]) == 1
    assert "graddiffg" not in json.dumps(rep["groups"])
    g1, g3 = rep["groups"]["lr3e-06"], rep["groups"]["lr3e-06_g3"]
    assert g1["all_seeds"]["C1_depth"]["met"] and g1["all_seeds"]["C2_breadth"]["met"]
    assert g1["all_seeds"]["C3_utility"]["met"] is False
    # the R-matched control of the same tag: excess 0.30 at alpha 0, 0.45 at alpha 1
    assert g1["runs"][0]["control"] == "control_x10_v2_unaskedg_1B_s0_lr3e-06"
    assert g1["all_seeds"]["C4_beyond_damage"]["per_seed"][0] == pytest.approx(0.375)
    assert g1["all_seeds"]["C4_beyond_damage"]["met"] is True
    assert g3["runs"][0]["control"] is None
    assert g3["all_seeds"]["C4_beyond_damage"]["met"] is None        # no control yet
    assert g3["confirmation_seeds_1_2"] is None
    assert rep["choice"] == {"general_weight": 3.0,
                             "reason": "smallest weight meeting C1-C3 on seed 0"}
    capsys_out = h9.main(["--dir", str(tmp_path), "--choose"])
    assert capsys_out == 0
    md = out.with_suffix(".md").read_text()
    assert "λ_g rule on seed 0: 3" in md and "study_v2_graddiff_1B_s2_lr5e-08" in md
    assert "lr3e-06_g3 (λ_g = 3), all seeds" in md


def test_an_exploratory_configuration_is_its_own_group_and_not_in_the_rule(tmp_path, capsys):
    _graddiff(tmp_path)
    _run(tmp_path, "study_v2_unaskedg_1B_s0_lr3e-06", 0, strict=0.40, breadth=0.30,
         ra=0.84, gw=1.0)
    # same seed and lambda_g, other learning rate: must not overwrite the run above
    _run(tmp_path, "study_v2_unaskedg_1B_s0_lr1e-05", 0, strict=0.50, breadth=0.45,
         ra=0.90, wf=0.90, gw=1.0)
    out = tmp_path / "rep"
    assert h9.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    rep = json.loads(out.with_suffix(".json").read_text())
    assert set(rep["groups"]) == {"lr3e-06", "lr1e-05"}
    assert rep["groups"]["lr3e-06"]["runs"][0]["strict_trained"] == pytest.approx(0.40)
    # the exploratory configuration meets everything but does not compete in the rule
    assert rep["choice"]["general_weight"] == 1.0
    assert "higher general knowledge" in rep["choice"]["reason"]


def test_without_a_passing_weight_the_rule_takes_more_general_knowledge(tmp_path, capsys):
    _graddiff(tmp_path)
    _run(tmp_path, "study_v2_unaskedg_1B_s0_lr3e-06", 0, strict=0.40, breadth=0.40,
         ra=0.78, wf=0.80, gw=1.0)
    _run(tmp_path, "study_v2_unaskedg_1B_s0_lr3e-06_g3", 0, strict=0.20, breadth=0.40,
         ra=0.85, wf=0.82, gw=3.0)
    assert h9.main(["--dir", str(tmp_path), "--choose"]) == 0
    assert capsys.readouterr().out.strip() == "3"


def test_seed_floors_and_means(tmp_path):
    cps = [{"seed": s, "strict_trained": st, "breadth_trained": 0.4, "R": 0.77, "RA": 0.87,
            "WF": 0.83} for s, st in ((0, 0.30), (1, 0.25), (2, 0.40))]
    c = h9.criteria(cps, {})
    assert c["C1_depth"]["met"] is False             # seed 1 below 0.26 though the mean is fine
    assert c["C2_breadth"]["met"] is True and c["C3_utility"]["met"] is True
    assert c["C4_beyond_damage"]["met"] is None


def test_no_runs_is_an_error(tmp_path):
    assert h9.main(["--dir", str(tmp_path), "--out", str(tmp_path / "x")]) == 1
