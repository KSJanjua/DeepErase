"""The run registry: known-bad runs must never be presented as results."""

from __future__ import annotations

from pathlib import Path

from deeperase.registry import load_registry, protocol_of, run_status

STUDIES = Path(__file__).resolve().parents[1] / "results" / "studies"


def test_committed_registry_names_only_runs_that_exist():
    reg = load_registry(STUDIES)
    assert reg, "results/studies/run_registry.json missing or unreadable"
    for section in ("invalid", "superseded"):
        for rid in reg[section]:
            assert (STUDIES / rid).is_dir(), rid
    assert (STUDIES / reg["headline"] / "plane.json").exists()


def test_every_npo_run_before_the_fix_is_invalid():
    reg = load_registry(STUDIES)
    for d in STUDIES.glob("*npo*"):
        if d.is_dir() and not (d / "config.json").read_text().count('"protocol"'):
            assert run_status(reg, d.name)[0] == "invalid", d.name


def test_invalid_beats_control_beats_superseded():
    reg = {"invalid": {"a": "x"}, "superseded": {"b": "y", "c": "z"}}
    assert run_status(reg, "a", {"control": "random"})[0] == "invalid"
    assert run_status(reg, "c", {"control": "random"})[0] == "control"
    assert run_status(reg, "b")[0] == "superseded"
    assert run_status(reg, "d", {"control": "none"}) == ("ok", "")


def test_runs_without_a_protocol_are_v1():
    assert protocol_of({}) == "v1" and protocol_of(None) == "v1"
    assert protocol_of({"protocol": "v2"}) == "v2"


def test_missing_registry_is_empty_not_an_error(tmp_path):
    assert load_registry(tmp_path) == {}


def test_variant_of_names_loss_scope_and_dtype():
    from deeperase.registry import variant_of
    assert variant_of({"unlearn": {"loss_scope": "answer"}, "dtype": "float32"}) == \
        "answer/float32"
    assert variant_of({}) == "full/bfloat16"          # v1 runs recorded neither


def test_smoke_runs_are_never_measurements():
    # recognised by name, so a new smoke run needs no registry entry first
    assert run_status({}, "smoke_v2_ga_h100", {"control": "none"})[0] == "smoke"
    assert run_status({"invalid": {"smoke_x": "broken"}}, "smoke_x")[0] == "invalid"
    assert run_status({}, "study_v2_ga_1B_s0_lr1e-06")[0] == "ok"
