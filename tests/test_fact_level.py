"""Fact-level analysis (scripts/fact_level_analysis.py): pure JSON, no models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeperase.scripts import fact_level_analysis as fla


# ---------------------------------------------------------------------------
# Synthetic run directories
# ---------------------------------------------------------------------------

def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def make_run(root: Path, name: str = "study_v2_ga_1B_s0_lr1e-05", *, alphas=(0.0, 0.5, 1.0),
             n_authors: int = 4, seed: int = 0, reference: bool = True,
             depth_fn=None, drop_fn=None) -> Path:
    """A v2 run whose facts have a known structure.

    Fact r has strength s = (r % 20) / 19. At alpha, depth_r = depth_fn(s, alpha)
    and each B0/B1 item's margin falls by drop_fn(s, alpha) from `full`'s 2.0;
    the item is answered correctly while the margin stays positive.
    """
    depth_fn = depth_fn or (lambda s, a: min(1.0, s * a))
    drop_fn = drop_fn or (lambda s, a: 4.0 * s * a)
    run = root / name
    rows = list(range(n_authors * 20))
    trained = [r for r in rows if r % 20 < 10]
    heldout = [r for r in rows if r % 20 >= 10]
    _write(run / "config.json", {"method": "ga", "seed": seed, "protocol": "v2",
                                 "unlearn": {"learning_rate": 1e-5}})
    _write(run / "split.json", {"trained_rows": trained, "heldout_rows": heldout})
    items = [f"{t}_{r}" for r in rows for t in ("B0", "B1")] + ["R_0", "RA_3"]
    full = {i: [1, 2.0, [-1.0, -3.0]] for i in items}
    # The oracle answers the even rows by plausibility and misses the odd ones.
    oracle = {i: [int(int(i.split("_")[1]) % 2 == 0), 0.5 if int(i.split("_")[1]) % 2 == 0
                  else -0.5] for i in items}
    _write(run / "calibration.json",
           {"reference_outcomes": {"full": full, "retain90": oracle}} if reference else {})
    for a in alphas:
        outcomes, depth = {}, {"trained": [], "heldout": []}
        for r in rows:
            s = (r % 20) / 19
            m = 2.0 - drop_fn(s, a)
            for t in ("B0", "B1"):
                outcomes[f"{t}_{r}"] = [int(m > 0), m, [m, 0.0]]
            depth["trained" if r in trained else "heldout"].append(
                {"example_id": f"ref_{r}", "uds": depth_fn(s, a)})
        outcomes["R_0"] = [1, 1.0]
        _write(run / "partial" / f"alpha_{a:.3f}.json", {
            "point": {"alpha": a},
            "breadth_tiers": {"outcomes": outcomes},
            "depth": {k: {"per_example": v} for k, v in depth.items()}})
    return run


# ---------------------------------------------------------------------------
# Per-fact measures
# ---------------------------------------------------------------------------

class TestPerFactMeasures:
    def test_items_are_grouped_by_fact_and_retain_tiers_ignored(self):
        got = fla.items_by_fact({"B0_7": [1, 0.5], "B2_7": [0, -0.1, [1, 2]],
                                 "B4_9": [1, 1.0], "R_7": [1, 3.0], "RA_7": [0, 0.0]})
        assert got == {7: {"B0": (1, 0.5), "B2": (0, -0.1)}, 9: {"B4": (1, 1.0)}}

    def test_forgotten_counts_only_items_full_knew(self):
        base = {"B0_1": [1, 2.0], "B1_1": [0, -1.0], "B2_1": [1, 1.0]}
        now = {"B0_1": [0, -0.5], "B1_1": [0, -2.0], "B2_1": [1, 0.5]}
        b = fla.behaviour(now, base)[1]
        assert b["n_eligible"] == 2 and b["forgotten"] == pytest.approx(0.5)
        # the margin drop uses every shared item, known or not
        assert b["margin_drop"] == pytest.approx((2.5 + 1.0 + 0.5) / 3)

    def test_a_fact_full_never_knew_has_no_forgotten_value(self):
        b = fla.behaviour({"B0_1": [0, -1.0]}, {"B0_1": [0, -0.5]})[1]
        assert b["forgotten"] is None and b["margin_drop"] == pytest.approx(0.5)

    def test_without_a_baseline_falls_back_to_wrong_answers(self):
        b = fla.behaviour({"B0_1": [0, -1.0], "B1_1": [1, 0.4]}, None)[1]
        assert b["forgotten"] == pytest.approx(0.5)
        assert b["margin_drop"] == pytest.approx(0.3)

    def test_depth_drops_undefined_examples(self):
        rep = {"per_example": [{"example_id": "ref_12", "uds": 0.4},
                               {"example_id": "ref_13", "uds": None}]}
        assert fla.depth_by_fact(rep) == {12: 0.4}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_spearman_matches_scipy_with_ties(self):
        stats = pytest.importorskip("scipy.stats")
        x = [0.1, 0.4, 0.4, 0.9, 0.2, 0.7, 0.7, 0.7]
        y = [1.0, 2.0, 0.5, 3.0, 0.0, 2.5, 2.5, 1.0]
        assert fla.spearman(x, y) == pytest.approx(stats.spearmanr(x, y)[0], abs=1e-12)

    def test_spearman_undefined_cases(self):
        assert fla.spearman([1, 2], [1, 2]) is None
        assert fla.spearman([1, 1, 1], [1, 2, 3]) is None

    def test_author_bootstrap_is_seeded_and_brackets_the_estimate(self):
        rows = list(range(80))
        mean = lambda rs: sum(r % 20 for r in rs) / len(rs)
        ci = fla.author_bootstrap(rows, mean, n_boot=300, seed=1)
        assert ci == fla.author_bootstrap(rows, mean, n_boot=300, seed=1)
        # every author has the same within-author mix, so resampling authors
        # cannot move the mean: the interval collapses onto the estimate
        assert ci == pytest.approx([9.5, 9.5])

    def test_author_bootstrap_resamples_authors_not_facts(self):
        # authors differ only in level; the CI must reflect 4 units, not 80
        rows = list(range(80))
        level = lambda rs: sum(r // 20 for r in rs) / len(rs)
        lo, hi = fla.author_bootstrap(rows, level, n_boot=500, seed=0)
        assert hi - lo > 1.0

    def test_author_bootstrap_needs_three_authors(self):
        assert fla.author_bootstrap(list(range(40)), lambda rs: 1.0) is None

    def test_classes_and_suppression_rate(self):
        depth = {0: 0.9, 1: 0.1, 2: 0.9, 3: 0.1}
        beh = {r: {"forgotten": f} for r, f in {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}.items()}
        assert fla.classify(depth, beh, [0, 1, 2, 3], 0.5) == {
            "removed": 1, "suppressed": 1, "erased_answering": 1, "intact": 1}
        assert fla.suppression_rate(depth, beh, [0, 1, 2, 3], 0.5) == pytest.approx(0.5)
        assert fla.suppression_rate(depth, beh, [2, 3], 0.5) is None

    def test_indirect_leak_splits_forgotten_facts_by_depth(self):
        depth = {0: 0.9, 1: 0.1, 2: 0.1, 3: 0.9}
        beh = {0: {"direct_forgotten": 1.0, "indirect_kept": 0.0},
               1: {"direct_forgotten": 1.0, "indirect_kept": 1.0},
               2: {"direct_forgotten": 0.0, "indirect_kept": 1.0},     # not forgotten
               3: {"direct_forgotten": 1.0, "indirect_kept": None}}    # no B2/B4 item
        assert fla.indirect_leak(depth, beh, [0, 1, 2, 3], 0.5) == {
            "deep": 0.0, "n_deep": 1, "shallow": 1.0, "n_shallow": 1}

    def test_behaviour_separates_direct_and_indirect_tiers(self):
        base = {"B0_4": [1, 2.0], "B1_4": [1, 2.0], "B2_4": [1, 1.0], "B4_4": [0, -1.0]}
        now = {"B0_4": [0, -1.0], "B1_4": [0, -1.0], "B2_4": [1, 0.5], "B4_4": [1, 0.2]}
        b = fla.behaviour(now, base)[4]
        assert b["direct_forgotten"] == 1.0
        assert b["indirect_kept"] == 1.0           # B4 was never known, so not counted
        assert b["forgotten"] == pytest.approx(2 / 3)

    def test_crossing_order(self):
        A = (0.0, 0.5, 1.0)
        d = {0: [0.1, 0.6, 0.9], 1: [0.1, 0.1, 0.1], 2: [0.0, 0.7, 0.7],
             3: [0.0, 0.0, 0.0], 4: [0.0, 0.2, 0.8]}
        b = {0: [0.0, 0.0, 1.0], 1: [1.0, 1.0, 1.0], 2: [0.0, 0.5, 1.0],
             3: [0.0, 0.0, 0.0], 4: [0.0, 1.0, 1.0]}
        o = fla.crossing_order({r: list(zip(A, v)) for r, v in d.items()},
                               {r: list(zip(A, v)) for r, v in b.items()}, 0.5)
        assert (o["depth_first"], o["behaviour_only"], o["same"], o["neither"],
                o["behaviour_first"], o["n"]) == (1, 1, 1, 1, 1, 5)
        assert o["median_lag"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# One run, and the command line
# ---------------------------------------------------------------------------

class TestAnalyseRun:
    def test_monotone_coupling_gives_rho_one(self, tmp_path):
        r = fla.analyse_run(make_run(tmp_path), n_boot=200)
        assert r["baseline"] == "full" and r["alphas"] == [0.0, 0.5, 1.0]
        assert set(r["splits"]) == {"trained", "heldout"}
        for k in ("trained", "heldout"):
            at0, _, at1 = r["splits"][k]["per_alpha"]
            assert at0["rho_margin"] is None            # nothing has moved yet
            assert at1["rho_margin"] == pytest.approx(1.0)
            assert at1["rho_margin_ci"] == pytest.approx([1.0, 1.0])
            assert at1["n_facts"] == at1["n_eligible"] == 40

    def test_classes_follow_the_construction(self, tmp_path):
        # depth tracks strength; behaviour flips once 4*s*a > 2, i.e. s > 0.5 at a=1
        r = fla.analyse_run(make_run(tmp_path), n_boot=50)
        c = r["splits"]["heldout"]["per_alpha"][-1]["classes"]
        # held-out facts have s = 10/19..19/19 >= 0.526: all forgotten at alpha 1
        assert c["0.5"]["removed"] == 40 and c["0.5"]["suppression_rate"] == 0.0
        assert c["0.7"]["suppressed"] == sum(1 for i in range(10, 20) if i / 19 < 0.7) * 4

    def test_suppression_is_detected(self, tmp_path):
        # behaviour collapses everywhere while depth never moves: pure suppression
        run = make_run(tmp_path, depth_fn=lambda s, a: 0.05, drop_fn=lambda s, a: 3.0 * a)
        r = fla.analyse_run(run, n_boot=50)
        last = r["splits"]["trained"]["per_alpha"][-1]
        assert last["classes"]["0.5"]["suppression_rate"] == 1.0
        assert last["classes"]["0.5"]["suppressed"] == 40
        assert r["splits"]["trained"]["order"]["0.5"]["behaviour_only"] == 40

    def test_oracle_rates_come_from_retain90(self, tmp_path):
        r = fla.analyse_run(make_run(tmp_path), n_boot=10)
        assert r["oracle"]["trained"]["share_forgotten"] == pytest.approx(0.5)

    def test_missing_reference_outcomes_is_flagged(self, tmp_path):
        r = fla.analyse_run(make_run(tmp_path, reference=False), n_boot=10)
        assert r["baseline"] == "none" and r["oracle"] == {}

    def test_v1_runs_are_refused(self, tmp_path):
        run = tmp_path / "study_ga_1B_old"
        _write(run / "config.json", {"method": "ga"})
        _write(run / "partial" / "alpha_0.000.json", {"point": {"alpha": 0.0}})
        with pytest.raises(ValueError, match="per-item"):
            fla.analyse_run(run)

    def test_main_pools_seeds_and_honours_the_registry(self, tmp_path, capsys):
        studies = tmp_path / "studies"
        for s in (0, 1):
            make_run(studies, f"study_v2_ga_1B_s{s}_lr1e-05", seed=s)
        make_run(studies, "study_v2_ga_1B_s9_lr1e-05", seed=9)
        _write(studies / "run_registry.json",
               {"invalid": {"study_v2_ga_1B_s9_lr1e-05": "test"}})
        out = tmp_path / "fl"
        assert fla.main(["--dir", str(studies), "--out", str(out), "--n-boot", "20"]) == 0
        d = json.loads(out.with_suffix(".json").read_text())
        assert [r["run"] for r in d["runs"]] == ["study_v2_ga_1B_s0_lr1e-05",
                                                 "study_v2_ga_1B_s1_lr1e-05"]
        assert any("s9" in s for s in d["skipped"])
        g = next(g for g in d["pooled"] if g["split"] == "trained")
        assert len(g["runs"]) == 2 and g["per_alpha"][-1]["rho_margin"]["mean"] == \
            pytest.approx(1.0)
        assert "suppression rate" in out.with_suffix(".md").read_text()

    def test_main_reports_when_nothing_is_usable(self, tmp_path):
        assert fla.main(["--dir", str(tmp_path), "--out", str(tmp_path / "x")]) == 1
