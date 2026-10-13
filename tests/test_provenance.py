"""Tests for run provenance: a run must be able to name the code that made it."""

from __future__ import annotations

import subprocess

from deeperase.provenance import environment_info, git_info, sha256_file


def _init_repo(path):
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True,
                                     capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (path / "f.txt").write_text("a")
    run("add", "f.txt")
    run("commit", "-q", "-m", "x")
    return run


def test_clean_checkout_records_commit_and_is_not_dirty(tmp_path):
    _init_repo(tmp_path)
    g = git_info(tmp_path)
    assert len(g["commit"]) == 40 and g["dirty"] is False


def test_uncommitted_change_is_flagged_dirty(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    assert git_info(tmp_path)["dirty"] is True


def test_changed_results_do_not_make_the_code_dirty(tmp_path):
    # the GPU box rewrites committed result files (the replication manifest, a
    # resumed run); that must not stamp the next run as uncommitted code
    run = _init_repo(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "manifest.json").write_text("{}")
    run("add", "results/manifest.json")
    run("commit", "-q", "-m", "results")
    (tmp_path / "results" / "manifest.json").write_text('{"runs": 1}')
    (tmp_path / "results" / "new_run.json").write_text("{}")      # untracked
    assert git_info(tmp_path)["dirty"] is False


def test_dirty_paths_name_the_changed_code(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("changed")
    g = git_info(tmp_path)
    assert g["dirty"] is True and g["dirty_paths"] == ["f.txt"]


def test_outside_a_repo_degrades_to_none(tmp_path):
    g = git_info(tmp_path)
    assert g["commit"] is None and g["dirty"] is None


def test_sha256_matches_hashlib_and_handles_missing(tmp_path):
    import hashlib
    p = tmp_path / "x.json"
    p.write_bytes(b"hello")
    assert sha256_file(p) == hashlib.sha256(b"hello").hexdigest()
    assert sha256_file(tmp_path / "missing") is None


def test_environment_records_the_versions_that_change_numbers():
    env = environment_info()
    for k in ("python", "torch", "transformers", "numpy"):
        assert env.get(k), k
