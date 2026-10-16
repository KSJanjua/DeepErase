"""Tests for the matrix aggregator (scripts/aggregate_matrix.py)."""

from __future__ import annotations

import json

import pytest

from deeperase.scripts.aggregate_matrix import aggregate, load_run, main, slope


def _write_run(root, name, seed, depth_slope, tier_slopes, protocol="v2", fq=(0.01, 0.2)):
    d = root / name
    (d / "partial").mkdir(parents=True)
    cfg = {"protocol": protocol, "method": "ga", "seed": seed,
           "unlearn": {"learning_rate": 1e-6}, "control": "none"}
    (d / "config.json").write_text(json.dumps(cfg))
    points = []
    for a in (0.0, 0.5, 1.0):
        pts = []
        for split in ("trained", "heldout"):
            pts.append({"method": "ga", "model": "1B-ga", "benchmark": f"tofu-forget10:{split}",
                        "alpha": a, "breadth": 0.2 + 0.3 * a, "depth": 0.1 + depth_slope * a,
                        "utility": 0.8 - 0.05 * a, "selectivity": 0.1 * a,
                        "general_utility": 0.9})
        points += pts
        groups = {f"{t}:{s}": {"calibrated": 0.1 + tier_slopes[t] * a}
                  for t in tier_slopes for s in ("trained", "heldout")}
        (d / "partial" / f"alpha_{a:.3f}.json").write_text(json.dumps({
            "points": pts, "point": pts[0], "groups": groups,
            "tofu_forget_quality": {"trained": fq[0] if a == 0 else fq[1],
                                    "heldout": fq[0], "all": fq[0]}}))
    (d / "plane.json").write_text(json.dumps({"version": 1, "points": points}))
    return d


def test_slope_is_least_squares():
    assert slope([0, 1, 2], [1, 3, 5]) == pytest.approx(2.0)
    assert slope([0, 1], [None, 1]) is None


def test_aggregates_across_seeds_per_trajectory(tmp_path):
    tiers = {"B0": 0.2, "B1": 0.25, "B2": 0.4, "B4": 0.6}
    runs = [load_run(_write_run(tmp_path, f"study_v2_ga_1B_s{s}_lr1e-06", s, 0.4 + 0.1 * s, tiers))
            for s in range(3)]
    rows = aggregate(runs)
    assert sorted(r["trajectory"] for r in rows) == ["heldout", "trained"]
    tr = next(r for r in rows if r["trajectory"] == "trained")
    assert tr["n_seeds"] == 3
    assert tr["slopes"]["depth"]["mean"] == pytest.approx(0.5)        # 0.4, 0.5, 0.6
    assert tr["slopes"]["depth"]["sd"] == pytest.approx(0.1)
    assert tr["slopes"]["tier_B4"]["mean"] == pytest.approx(0.6)       # UIPE test input
    assert tr["slopes"]["tier_B0"]["mean"] == pytest.approx(0.2)
    a1 = tr["per_alpha"][-1]
    assert a1["alpha"] == 1.0 and a1["breadth"]["mean"] == pytest.approx(0.5)
    assert tr["tofu_forget_quality"]["alpha_max"]["mean"] == pytest.approx(0.2)


def test_main_skips_registry_flagged_runs_and_writes_outputs(tmp_path):
    tiers = {"B0": 0.2}
    _write_run(tmp_path, "study_v2_ga_1B_s0_lr1e-06", 0, 0.4, tiers)
    _write_run(tmp_path, "study_v2_ga_1B_s1_lr1e-06", 1, 0.4, tiers)
    (tmp_path / "run_registry.json").write_text(json.dumps(
        {"invalid": {"study_v2_ga_1B_s1_lr1e-06": "test"}}))
    out = tmp_path / "out" / "m"
    assert main(["--dir", str(tmp_path), "--glob", "study_*", "--out", str(out)]) == 0
    d = json.loads(out.with_suffix(".json").read_text())
    assert d["runs"] == ["study_v2_ga_1B_s0_lr1e-06"]
    assert any("invalid" in s for s in d["skipped"])
    assert out.with_suffix(".md").exists() and out.with_suffix(".png").exists()


def test_ablations_are_not_pooled_with_the_main_design(tmp_path):
    def run(name, seed, loss):
        d = _write_run(tmp_path, name, seed, 0.6, {"B0": 0.3, "B1": 0.3, "B2": 0.4, "B4": 0.5})
        cfg = json.loads((d / "config.json").read_text())
        cfg["unlearn"]["loss_scope"], cfg["dtype"] = loss, "float32"
        (d / "config.json").write_text(json.dumps(cfg))
        return d
    run("study_v2_ga_1B_s0_lr1e-06", 0, "answer")
    run("study_v2_ga_1B_s1_lr1e-06", 1, "answer")
    run("study_v2_ga_1B_s0_lr1e-06_lossfull", 0, "full")
    rows = aggregate([load_run(d) for d in sorted(tmp_path.iterdir())])
    trained = {g["variant"]: g["n_seeds"] for g in rows if g["trajectory"] == "trained"}
    assert trained == {"answer/float32": 2, "full/float32": 1}


def test_controls_and_tagged_runs_are_their_own_arms(tmp_path):
    # --include-all reads controls and extensions too; they must not join the
    # seeds of the main design they share method and rate with.
    for name, seed in (("study_v2_ga_1B_s0_lr1e-06", 0), ("study_v2_ga_1B_s1_lr1e-06", 1),
                       ("study_v2_ga_1B_s0_lr1e-06_ext", 0),
                       ("control_x540_v2_ga_1B_s0_lr1e-06", 0)):
        d = _write_run(tmp_path, name, seed, 0.6, {"B0": 0.3})
        if name.startswith("control"):
            cfg = json.loads((d / "config.json").read_text())
            cfg["control"], cfg["control_scale"] = "random", 540.0
            (d / "config.json").write_text(json.dumps(cfg))
    out = tmp_path / "out" / "m"
    assert main(["--dir", str(tmp_path), "--glob", "*_v2_*", "--include-all",
                 "--out", str(out)]) == 0
    groups = json.loads(out.with_suffix(".json").read_text())["groups"]
    trained = {g["arm"]: g["n_seeds"] for g in groups if g["trajectory"] == "trained"}
    assert trained == {"main": 2, "ext": 1, "control": 1}
    md = out.with_suffix(".md").read_text()
    assert "matched-damage control" in md and "tag `ext`" in md
