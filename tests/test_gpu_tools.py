"""Tests for the gpu/ drivers' pure logic (no GPU needed)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "gpu" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def replicate():
    return _load("replicate")


class TestRunIds:
    def test_v1_ids_match_the_directories_already_on_disk(self, replicate):
        """Existing v1 runs must still resume, so their ids cannot change."""
        assert replicate.run_id_for("ga", "1B", 0, 1e-6) == "study_ga_1B_s0"
        assert replicate.run_id_for("graddiff", "1B", 2, 3e-6) == \
            "study_graddiff_1B_s2_lr3e-06"
        for rid in ("study_ga_1B_s0", "study_graddiff_1B_s2_lr3e-06"):
            assert (ROOT / "results" / "studies" / rid).is_dir()

    def test_v2_ids_always_carry_protocol_and_lr(self, replicate):
        assert replicate.run_id_for("ga", "1B", 0, 1e-6, protocol="v2") == \
            "study_v2_ga_1B_s0_lr1e-06"

    def test_controls_cannot_be_mistaken_for_measurements(self, replicate):
        assert replicate.run_id_for("ga", "1B", 0, 1e-6, protocol="v2",
                                    control="random") == "control_v2_ga_1B_s0_lr1e-06"
        assert replicate.run_id_for("ga", "1B", 0, 1e-6, protocol="v2", control="random",
                                    control_scale=4.0) == "control_x4_v2_ga_1B_s0_lr1e-06"

    def test_ablation_tag_keeps_its_own_directory(self, replicate):
        assert replicate.run_id_for("ga", "1B", 0, 1e-6, protocol="v2", tag="lossfull") == \
            "study_v2_ga_1B_s0_lr1e-06_lossfull"
        assert replicate.run_id_for("ga", "1B", 0, 1e-6, protocol="v2", control="random",
                                    tag="lossfull") == "control_v2_ga_1B_s0_lr1e-06_lossfull"
        with pytest.raises(ValueError):
            replicate.run_id_for("ga", "1B", 0, 1e-6, protocol="v2", tag="../x")


class TestAlreadyDone:
    def test_requested_alphas_are_read_from_extra(self, replicate):
        assert replicate.requested_alphas(["--alphas", "0,1.5,2"]) == {0.0, 1.5, 2.0}
        assert replicate.requested_alphas(["--loss-scope", "full", "--alphas=0,1"]) == {0.0, 1.0}
        assert replicate.requested_alphas(["--loss-scope", "full"]) is None

    def test_new_alphas_reopen_a_finished_run(self, replicate, tmp_path):
        (tmp_path / "plane.json").write_text(json.dumps(
            {"points": [{"alpha": 0.0}, {"alpha": 1.0}, {"alpha": 1.0}]}))
        assert replicate.already_done(tmp_path)
        assert replicate.already_done(tmp_path, {0.0, 1.0})
        assert not replicate.already_done(tmp_path, {0.0, 1.0, 1.5})
        assert not replicate.already_done(tmp_path / "missing")


class TestManifest:
    def _batch(self, t):
        return {"generated": t, "methods": ["ga"]}

    def test_merges_instead_of_overwriting(self, replicate, tmp_path):
        m = tmp_path / "replication_manifest.json"
        replicate.update_manifest(m, self._batch("t1"), [{"run_id": "a", "status": "ok"}])
        replicate.update_manifest(m, self._batch("t2"), [{"run_id": "b", "status": "ok"}])
        d = json.loads(m.read_text())
        assert set(d["runs"]) == {"a", "b"} and len(d["batches"]) == 2

    def test_latest_status_wins_for_a_rerun(self, replicate, tmp_path):
        m = tmp_path / "replication_manifest.json"
        replicate.update_manifest(m, self._batch("t1"), [{"run_id": "a", "status": "FAILED"}])
        replicate.update_manifest(m, self._batch("t2"), [{"run_id": "a", "status": "ok"}])
        assert json.loads(m.read_text())["runs"]["a"]["status"] == "ok"

    def test_wraps_the_old_overwrite_format(self, replicate, tmp_path):
        old = ROOT / "results" / "studies" / "replication_manifest.json"
        m = tmp_path / "replication_manifest.json"
        m.write_text(old.read_text())
        replicate.update_manifest(m, self._batch("t3"), [{"run_id": "new", "status": "ok"}])
        d = json.loads(m.read_text())
        assert {"study_npo_1B_s0", "new"} <= set(d["runs"])
        assert d["batches"][0]["methods"] == ["npo"]
