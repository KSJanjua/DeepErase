"""Tests for the two-phase UDS path and the resumable run directory.

The two-phase path is what makes a 3B run possible on a 20 GB card: capture
hidden states from the source, free it, then patch into the target, so peak
memory is one model rather than two. It must give **identical** results to the
single-phase path, or the memory strategy would silently change the science.

The run directory must survive an interrupted session. A hosted GPU can drop at
any moment, and re-running from scratch each time would make a long sweep
impossible.
"""

from __future__ import annotations

import json

import pytest
import torch

from deeperase.eval.patching import EntitySpan
from deeperase.eval.uds import (
    DEFAULT_TAU,
    UDSExample,
    assemble_report,
    build_stage1_cache,
    capture_source_hidden,
    compute_uds,
    score_from_captured,
)
from deeperase.scripts.run_uds_validation import STAGE2_ORDER, RunDir

N_LAYERS = 4
VOCAB = 64
SEQ = 10
FACT_MARKER = 20


def _tiny_model(seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, use_cache=False,
    ))
    m.eval()
    return m


def _train(model, batches, *, lr=5e-3, steps=80):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for i in range(steps):
        x = batches[i % len(batches)]
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def setup():
    """Trained models, so dS1 is genuinely positive and UDS is defined.

    Random models give negative dS1, an empty Knowledge-Encoding set and an
    undefined score -- which would let these tests pass without exercising
    anything.
    """
    g = torch.Generator().manual_seed(3)
    fact = torch.randint(1, VOCAB - 2, (8, SEQ), generator=g)
    fact[:, 0] = FACT_MARKER
    fact[:, -1] = FACT_MARKER + 1
    other = torch.randint(1, VOCAB - 2, (8, SEQ), generator=g)

    m_full = _train(_tiny_model(0), [fact, other])
    m_ret = _train(_tiny_model(0), [other])
    examples = [
        UDSExample(f"e{i}", fact[i:i + 1], EntitySpan([SEQ - 1])) for i in range(3)
    ]
    return m_full, m_ret, examples


LAYERS = list(range(N_LAYERS))


class TestTwoPhaseEquivalence:
    """The memory strategy must not change the numbers."""

    def test_matches_single_phase_exactly(self, setup):
        m_full, m_ret, examples = setup
        unl = _tiny_model(7)

        single = compute_uds(model_full=m_full, model_retain=m_ret,
                             model_unlearned=unl, examples=examples, layers=LAYERS)

        d1, sf = build_stage1_cache(model_full=m_full, model_retain=m_ret,
                                    examples=examples, layers=LAYERS)
        cap = capture_source_hidden(unl, examples, LAYERS)
        d2, _ = score_from_captured(m_full, cap, examples, LAYERS, s_full_cache=sf)
        two_phase = assemble_report(delta_s1=d1, delta_s2=d2, s_full=sf, layers=LAYERS)

        assert single.uds is not None
        assert two_phase.uds == pytest.approx(single.uds, abs=1e-9)

    def test_per_example_deltas_match(self, setup):
        """Compare elementwise, not just the final score -- clipping in the
        Layer Erasure Ratio can hide differences in the deltas."""
        m_full, m_ret, examples = setup
        cap = capture_source_hidden(m_ret, examples, LAYERS)
        two_phase, _ = score_from_captured(m_full, cap, examples, LAYERS)
        direct, _ = build_stage1_cache(model_full=m_full, model_retain=m_ret,
                                       examples=examples, layers=LAYERS)
        for eid in direct:
            for ell in LAYERS:
                assert two_phase[eid][ell] == pytest.approx(direct[eid][ell], abs=1e-9)

    def test_capture_covers_every_example_and_layer(self, setup):
        _, m_ret, examples = setup
        cap = capture_source_hidden(m_ret, examples, LAYERS)
        assert set(cap) == {e.example_id for e in examples}
        for per_ex in cap.values():
            assert set(per_ex) == set(LAYERS)

    def test_captured_states_are_moved_off_gpu(self, setup):
        """Keeping them on the device would defeat sequential execution."""
        _, m_ret, examples = setup
        cap = capture_source_hidden(m_ret, examples, LAYERS, to_cpu=True)
        for per_ex in cap.values():
            for h in per_ex.values():
                assert h.device.type == "cpu"

    def test_missing_capture_raises(self, setup):
        m_full, m_ret, examples = setup
        cap = capture_source_hidden(m_ret, examples[:1], LAYERS)
        with pytest.raises(KeyError, match="No captured hidden states"):
            score_from_captured(m_full, cap, examples, LAYERS)

    def test_source_model_is_not_needed_during_phase_b(self, setup):
        """Proof the phases are genuinely decoupled: delete the source model
        entirely between them and Phase B must still work."""
        m_full, m_ret, examples = setup
        cap = capture_source_hidden(m_ret, examples, LAYERS)
        del m_ret
        d2, _ = score_from_captured(m_full, cap, examples, LAYERS)
        assert len(d2) == len(examples)


class TestAssembleReport:
    def test_perfect_unlearning_gives_one(self):
        d1 = {"a": {0: 0.5, 1: 1.0}}
        rep = assemble_report(delta_s1=d1, delta_s2={"a": dict(d1["a"])},
                              s_full={"a": -1.0}, layers=[0, 1])
        assert rep.uds == pytest.approx(1.0)

    def test_no_unlearning_gives_zero(self):
        rep = assemble_report(delta_s1={"a": {0: 0.5, 1: 1.0}},
                              delta_s2={"a": {0: 0.0, 1: 0.0}},
                              s_full={"a": -1.0}, layers=[0, 1])
        assert rep.uds == pytest.approx(0.0)

    def test_example_missing_stage2_is_skipped(self):
        rep = assemble_report(delta_s1={"a": {0: 1.0}, "b": {0: 1.0}},
                              delta_s2={"a": {0: 1.0}},
                              s_full={"a": -1.0}, layers=[0])
        assert rep.n_examples_total == 1

    def test_sub_tau_examples_are_undefined_not_zero(self):
        rep = assemble_report(delta_s1={"a": {0: 0.001}}, delta_s2={"a": {0: 0.0}},
                              s_full={"a": -1.0}, layers=[0], tau=DEFAULT_TAU)
        assert rep.uds is None and rep.n_examples_undefined == 1

    def test_report_still_flags_unvalidated(self):
        rep = assemble_report(delta_s1={"a": {0: 1.0}}, delta_s2={"a": {0: 1.0}},
                              s_full={"a": -1.0}, layers=[0])
        assert rep.is_validated_against_reference is False


class TestRunDirResumability:
    def test_creates_layout(self, tmp_path):
        rd = RunDir(tmp_path / "run1")
        assert rd.root.exists() and rd.partial.exists()

    def test_stage1_roundtrip_restores_integer_layer_keys(self, tmp_path):
        """JSON turns dict keys into strings; layer indices must come back as
        ints or every lookup silently misses."""
        rd = RunDir(tmp_path / "run2")
        rd.save_stage1({"e0": {0: 1.5, 1: 2.5}}, {"e0": -1.0})
        deltas, s_full = rd.load_stage1()
        assert deltas == {"e0": {0: 1.5, 1: 2.5}}
        assert all(isinstance(k, int) for k in deltas["e0"])
        assert s_full == {"e0": -1.0}

    def test_stage2_roundtrip(self, tmp_path):
        rd = RunDir(tmp_path / "run3")
        rd.save_stage2("retain95", {"e0": {0: 0.4}}, 0.42, "summary text")
        got = rd.load_stage2("retain95")
        assert got["uds"] == 0.42 and got["delta_s2"] == {"e0": {0: 0.4}}

    def test_missing_files_return_none(self, tmp_path):
        rd = RunDir(tmp_path / "run4")
        assert rd.load_stage1() == (None, None)
        assert rd.load_stage2("full") is None

    def test_corrupt_file_is_ignored_not_trusted(self, tmp_path):
        """A session killed mid-write must not leave a partial file that a
        later resume silently believes."""
        rd = RunDir(tmp_path / "run5")
        rd.stage1_path.write_text('{"delta_s1": {"e0": {"0":', encoding="utf-8")
        assert rd.load_stage1() == (None, None)

    def test_writes_are_atomic(self, tmp_path):
        """No .tmp file should survive a completed write."""
        rd = RunDir(tmp_path / "run6")
        rd.save_stage1({"e0": {0: 1.0}}, {"e0": 0.0})
        assert list(rd.root.glob("*.tmp")) == []

    def test_each_split_is_a_separate_file(self, tmp_path):
        """So an interrupted run resumes at the split it reached."""
        rd = RunDir(tmp_path / "run7")
        for split in STAGE2_ORDER[:2]:
            rd.save_stage2(split, {"e0": {0: 0.1}}, 0.1, "s")
        assert rd.load_stage2("full") is not None
        assert rd.load_stage2("retain95") is None

    def test_report_is_written(self, tmp_path):
        rd = RunDir(tmp_path / "run8")
        rd.save_report({"run_id": "x", "observed": {"full": 0.01}})
        assert json.loads(rd.report_path.read_text())["run_id"] == "x"


class TestStage2Order:
    def test_order_matches_increasing_unseen_fraction(self):
        """UDS must rise along this list; the order encodes the paper's test."""
        from deeperase.config import TOFU_MODELS
        fracs = [TOFU_MODELS["1B"][s].unseen_fraction for s in STAGE2_ORDER]
        assert fracs == sorted(fracs)

    def test_covers_all_four_splits(self):
        assert set(STAGE2_ORDER) == {"full", "retain90", "retain95", "retain99"}
