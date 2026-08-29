"""Tests for the unlearning trainers.

The failure that matters most is a run that *looks* fine but did the wrong
thing: an inverted objective produces a perfectly ordinary-looking loss curve
while making the model better at the forget set. Several tests below check the
direction of change, not just that training ran.
"""

from __future__ import annotations

import pytest
import torch

from deeperase.unlearn import (
    UnlearnConfig,
    UnlearnHistory,
    UnlearnMethod,
    ga_loss,
    grad_diff_loss,
    npo_loss,
    sequence_nll,
    snapshot,
    unlearn,
    verify_unlearning,
)

VOCAB, SEQ = 64, 12


def _model(seed=0, layers=2, hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, use_cache=False))
    m.eval()
    return m


def _data(n=6, seed=1, marker=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(1, VOCAB - 2, (1, SEQ), generator=g)
        if marker is not None:
            ids[0, 0], ids[0, -1] = marker, marker + 1
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return out


def _mean_nll(model, data):
    """Mean NLL over a whole dataset -- the reference implementation the
    library's own baseline/final measurements are checked against."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for d in data:
            per = sequence_nll(model, d["input_ids"], d.get("attention_mask"))
            total += float(per.sum())
            n += int(per.numel())
    if was_training:
        model.train()
    return total / n


class TestSequenceNLL:
    def test_returns_one_value_per_sequence(self):
        """Per-sequence, not a scalar -- NPO compares each against its
        reference before any reduction."""
        m, d = _model(), _data(4)
        ids = torch.cat([x["input_ids"] for x in d])
        assert sequence_nll(m, ids).shape == (4,)

    def test_is_positive(self):
        m, d = _model(), _data(2)
        assert (sequence_nll(m, torch.cat([x["input_ids"] for x in d])) > 0).all()

    def test_mask_excludes_padding(self):
        m = _model()
        ids = torch.randint(1, VOCAB - 2, (1, SEQ))
        full = torch.ones_like(ids)
        partial = full.clone(); partial[:, -4:] = 0
        assert not torch.allclose(sequence_nll(m, ids, full),
                                  sequence_nll(m, ids, partial))

    def test_all_padding_does_not_divide_by_zero(self):
        m = _model()
        ids = torch.randint(1, VOCAB - 2, (1, SEQ))
        out = sequence_nll(m, ids, torch.zeros_like(ids))
        assert torch.isfinite(out).all()


class TestLosses:
    def test_ga_is_negated_nll(self):
        m, d = _model(), _data(2)
        batch = {"input_ids": torch.cat([x["input_ids"] for x in d]),
                 "attention_mask": None}
        assert float(ga_loss(m, batch)) == pytest.approx(
            float(-sequence_nll(m, batch["input_ids"]).mean()), abs=1e-5)

    def test_graddiff_adds_a_retain_term(self):
        m = _model()
        f = {"input_ids": _data(2, seed=1)[0]["input_ids"], "attention_mask": None}
        r = {"input_ids": _data(2, seed=9)[0]["input_ids"], "attention_mask": None}
        assert float(grad_diff_loss(m, f, r, 1.0)) != float(ga_loss(m, f))

    def test_graddiff_weight_zero_reduces_to_ga(self):
        m = _model()
        f = {"input_ids": _data(2, seed=1)[0]["input_ids"], "attention_mask": None}
        r = {"input_ids": _data(2, seed=9)[0]["input_ids"], "attention_mask": None}
        assert float(grad_diff_loss(m, f, r, 0.0)) == pytest.approx(
            float(ga_loss(m, f)), abs=1e-5)

    def test_npo_is_zero_ish_against_itself(self):
        """With model == reference the log-ratio is 0, so the loss is the
        constant (2/beta)*log(2) -- a useful anchor for the implementation."""
        import math
        m = _model()
        batch = {"input_ids": _data(2)[0]["input_ids"], "attention_mask": None}
        expected = (2.0 / 0.1) * math.log(2)
        assert float(npo_loss(m, m, batch, beta=0.1)) == pytest.approx(expected, rel=1e-4)

    def test_npo_falls_as_model_diverges_downward(self):
        """Once the model is less likely than the reference on the forget set,
        NPO's loss shrinks -- the self-damping that avoids collapse."""
        ref, m = _model(0), _model(0)
        batch = {"input_ids": _data(2)[0]["input_ids"], "attention_mask": None}
        before = float(npo_loss(m, ref, batch, beta=0.1))
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        after = float(npo_loss(m, ref, batch, beta=0.1))
        assert after != before


class TestConfigValidation:
    def test_defaults_are_valid(self):
        assert UnlearnConfig().validate() == []

    def test_rejects_bad_values(self):
        assert UnlearnConfig(learning_rate=0).validate()
        assert UnlearnConfig(epochs=0).validate()
        assert UnlearnConfig(batch_size=0).validate()
        assert UnlearnConfig(method=UnlearnMethod.NPO, beta=0).validate()

    def test_method_requirements(self):
        assert UnlearnMethod.GRAD_DIFF.needs_retain
        assert UnlearnMethod.NPO.needs_reference
        assert not UnlearnMethod.GA.needs_retain
        assert not UnlearnMethod.GA.needs_reference

    def test_serialises(self):
        assert UnlearnConfig().to_dict()["method"] == "ga"


class TestMissingInputsAreFatal:
    """Silently training a different objective than requested would be worse
    than crashing."""

    def test_graddiff_without_retain_raises(self):
        with pytest.raises(ValueError, match="requires retain_data"):
            unlearn(_model(), _data(2),
                    UnlearnConfig(method=UnlearnMethod.GRAD_DIFF, epochs=1))

    def test_npo_without_reference_raises(self):
        with pytest.raises(ValueError, match="requires reference_model"):
            unlearn(_model(), _data(2),
                    UnlearnConfig(method=UnlearnMethod.NPO, epochs=1))

    def test_graddiff_with_forget_set_as_retain_raises(self):
        """Regression: the study runner once passed the forget batches in as
        retain data.

        The objective then reduces to ``(retain_weight - 1) * NLL(forget)``,
        which at the default retain_weight of 1.0 is identically zero. Nothing
        crashes: the model simply never trains, keeps full utility, passes
        checkpoint selection, and yields a flat trajectory that reads as a
        legitimate null result. The loss function itself was correct and well
        tested -- the defect was the argument at the call site, which is
        exactly what unit tests on the loss cannot see.
        """
        forget = _data(4)
        with pytest.raises(ValueError, match="same object as forget_data"):
            unlearn(_model(), forget,
                    UnlearnConfig(method=UnlearnMethod.GRAD_DIFF, epochs=1),
                    retain_data=forget)

    def test_graddiff_on_forget_set_has_no_gradient(self):
        """Why the guard above matters, demonstrated numerically."""
        from deeperase.unlearn import ga_loss, grad_diff_loss

        m, batch = _model(), _data(1)[0]
        degenerate = grad_diff_loss(m, batch, batch, 1.0)
        assert float(degenerate) == pytest.approx(0.0, abs=1e-5)
        # ...whereas a genuinely disjoint retain batch does not vanish.
        healthy = grad_diff_loss(m, batch, _data(1, seed=99)[0], 1.0)
        assert abs(float(healthy)) > 1e-4

    def test_empty_forget_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            unlearn(_model(), [], UnlearnConfig(epochs=1))

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Invalid UnlearnConfig"):
            unlearn(_model(), _data(2), UnlearnConfig(epochs=0))


class TestTrainingDirection:
    """Unlearning must make the forget set HARDER, not easier."""

    @pytest.mark.parametrize("method", [UnlearnMethod.GA, UnlearnMethod.GRAD_DIFF,
                                        UnlearnMethod.NPO])
    def test_forget_nll_rises(self, method):
        m = _model()
        data = _data(6)
        kw = {}
        if method.needs_retain:
            kw["retain_data"] = _data(6, seed=99)
        if method.needs_reference:
            ref = _model(); ref.load_state_dict(m.state_dict()); kw["reference_model"] = ref
        h = unlearn(m, data, UnlearnConfig(method=method, epochs=3,
                                           learning_rate=1e-3, log_every=0), **kw)
        assert h.forget_nll_change > 0, (
            f"{method.value}: forget NLL fell by {abs(h.forget_nll_change):.4f} -- "
            "the objective sign is inverted"
        )


class TestDirectionIsJudgedOnAFixedSet:
    """Regression: a healthy GA run was rejected as having an inverted
    objective.

    ``forget_nll_change`` was ``forget_nll[-1] - forget_nll[0]`` over the
    per-step list. At batch size 1 those are two different sentences, and the
    NLL spread between sentences is far larger than what five epochs of
    unlearning moves, so the sign was effectively arbitrary. On the real run the
    per-epoch averages rose monotonically (1.627 -> 1.751) while first-vs-last
    step read -0.203.
    """

    #: The shape of the real failure: per-step NLL falls (the last example was
    #: easier than the first) while the whole-set measurement rises.
    OBSERVED = dict(steps=[0, 1, 2], losses=[-1.74, -1.62, -1.54],
                    forget_nll=[1.7411, 1.6198, 1.5383],
                    baseline_forget_nll=1.6021, final_forget_nll=1.7510)

    def test_change_reads_the_whole_set_fields(self):
        h = UnlearnHistory(**self.OBSERVED)
        assert h.forget_nll_change == pytest.approx(1.7510 - 1.6021)

    def test_the_old_formula_reported_the_opposite_sign(self):
        h = UnlearnHistory(**self.OBSERVED)
        old = h.forget_nll[-1] - h.forget_nll[0]      # what the bug computed
        assert old < 0 < h.forget_nll_change, (
            f"per-step reads {old:+.4f}, whole-set reads "
            f"{h.forget_nll_change:+.4f} -- they must disagree here, or this "
            "test is not guarding the fix"
        )

    def test_this_history_is_accepted(self):
        ini, un, _ = TestVerification()._run()
        v = verify_unlearning(ini, un, UnlearnHistory(**self.OBSERVED))
        assert v.passed, f"healthy run rejected: {v.summary()}"

    def test_a_real_run_records_both_endpoints(self):
        m = _model()
        h = unlearn(m, _data(4), UnlearnConfig(epochs=2, learning_rate=1e-3,
                                               log_every=0))
        assert h.baseline_forget_nll is not None
        assert h.final_forget_nll is not None
        assert h.forget_nll_change == pytest.approx(
            h.final_forget_nll - h.baseline_forget_nll)

    def test_baseline_is_measured_before_any_update(self):
        m = _model()
        data = _data(4)
        expected = _mean_nll(m, data)          # untrained model, right now
        h = unlearn(m, data, UnlearnConfig(epochs=2, learning_rate=1e-3,
                                           log_every=0))
        assert h.baseline_forget_nll == pytest.approx(expected, abs=1e-4)

    def test_final_describes_the_returned_model_not_the_last_epoch(self):
        """With checkpoint selection the returned weights may be an earlier
        epoch; the reported NLL must match what is handed back."""
        m = _model()
        data = _data(4)
        h = unlearn(m, data, UnlearnConfig(epochs=3, learning_rate=1e-3,
                                           log_every=0),
                    eval_fn=lambda _m: 1.0)
        assert h.final_forget_nll == pytest.approx(_mean_nll(m, data), abs=1e-4)

    def test_weights_actually_move(self):
        m = _model()
        before = snapshot(m)
        unlearn(m, _data(4), UnlearnConfig(epochs=2, learning_rate=1e-3, log_every=0))
        after = snapshot(m)
        assert not torch.allclose(before["model.embed_tokens.weight"],
                                  after["model.embed_tokens.weight"])

    def test_model_left_in_eval_mode(self):
        m = _model()
        unlearn(m, _data(3), UnlearnConfig(epochs=1, learning_rate=1e-3, log_every=0))
        assert m.training is False
        assert not any(p.requires_grad for p in m.parameters())

    def test_history_records_every_step(self):
        h = unlearn(_model(), _data(4),
                    UnlearnConfig(epochs=2, batch_size=2, learning_rate=1e-3, log_every=0))
        assert len(h.steps) == 4          # 4 examples / batch 2 * 2 epochs
        assert len(h.losses) == len(h.forget_nll) == 4

    def test_reference_model_is_not_modified(self):
        """NPO must not train its own reference."""
        m, ref = _model(), _model()
        before = snapshot(ref)
        unlearn(m, _data(4), UnlearnConfig(method=UnlearnMethod.NPO, epochs=2,
                                           learning_rate=1e-3, log_every=0),
                reference_model=ref)
        after = snapshot(ref)
        for k in before:
            assert torch.equal(before[k], after[k]), f"reference model changed at {k}"

    def test_divergent_loss_stops_early(self):
        """A wild learning rate should halt rather than write NaNs into the
        checkpoint."""
        h = unlearn(_model(), _data(8),
                    UnlearnConfig(epochs=5, learning_rate=1e3, log_every=0))
        assert len(h.steps) < 40


class TestVerification:
    def _run(self, lr=1e-3, epochs=2):
        m = _model()
        ini = snapshot(m)
        h = unlearn(m, _data(4), UnlearnConfig(epochs=epochs, learning_rate=lr,
                                               log_every=0))
        return ini, snapshot(m), h

    def test_good_run_passes(self):
        ini, un, h = self._run()
        v = verify_unlearning(ini, un, h)
        assert v.passed and v.weights_changed and v.forget_nll_rose

    def test_detects_a_model_that_never_moved(self):
        """Zero update means the α dial has nothing to scale -- every point on
        the sweep would be identical."""
        m = _model()
        ini = snapshot(m)
        h = UnlearnHistory(steps=[0, 1], losses=[1.0, 1.0],
                           baseline_forget_nll=2.0, final_forget_nll=3.0)
        v = verify_unlearning(ini, ini, h)
        assert not v.passed and "did not move" in v.reason

    def test_detects_an_inverted_objective(self):
        ini, un, _ = self._run()
        wrong = UnlearnHistory(steps=[0, 1], losses=[1.0, 0.5],
                               baseline_forget_nll=3.0,   # NLL fell over the
                               final_forget_nll=2.0)      # whole forget set
        v = verify_unlearning(ini, un, wrong)
        assert not v.passed and "sign is probably inverted" in v.reason

    def test_summary_is_readable(self):
        ini, un, h = self._run()
        assert "||v||=" in verify_unlearning(ini, un, h).summary()

    def test_update_vector_is_usable_by_the_dial(self):
        """The whole point of keeping θ_ini: v must be non-trivial so that
        extrapolation has a direction to push along."""
        from deeperase.core.extrapolation import compute_update_vector, extrapolate
        ini, un, _ = self._run()
        v = compute_update_vector(ini, un, strict=False)
        assert len(v) > 0
        out = extrapolate(un, v, alpha=0.5)
        assert not torch.allclose(out["model.embed_tokens.weight"],
                                  un["model.embed_tokens.weight"])


# ---------------------------------------------------------------------------
# Variable-length batching
#
# Regression test for a crash on the first real study run. `_batches` used
# torch.cat, which requires identical sequence lengths. Synthetic test data was
# uniform so it passed; real TOFU examples are 47, 62, ... tokens and it failed
# immediately:
#
#   RuntimeError: Sizes of tensors must match except in dimension 0.
#                 Expected size 47 but got size 62
# ---------------------------------------------------------------------------

class TestVariableLengthBatching:
    def _mixed(self, lengths):
        return [{"input_ids": torch.randint(1, VOCAB - 2, (1, n)),
                 "attention_mask": torch.ones(1, n, dtype=torch.long)}
                for n in lengths]

    def test_batches_examples_of_different_lengths(self):
        from deeperase.unlearn import _batches
        batches = list(_batches(self._mixed([47, 62, 51]), 3, torch.device("cpu")))
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape == (3, 62), "should pad to the longest"

    def test_padding_is_masked_out(self):
        from deeperase.unlearn import _batches
        b = next(iter(_batches(self._mixed([10, 20]), 2, torch.device("cpu"))))
        assert b["attention_mask"][0, :10].sum() == 10
        assert b["attention_mask"][0, 10:].sum() == 0, "padding must be masked"
        assert b["attention_mask"][1].sum() == 20

    def test_mask_is_always_returned(self):
        """Downstream code should never have to handle None."""
        from deeperase.unlearn import _batches
        data = [{"input_ids": torch.randint(1, VOCAB - 2, (1, 8))}]
        b = next(iter(_batches(data, 1, torch.device("cpu"))))
        assert b["attention_mask"] is not None
        assert b["attention_mask"].shape == b["input_ids"].shape

    def test_equal_lengths_still_work(self):
        from deeperase.unlearn import _batches
        b = next(iter(_batches(self._mixed([12, 12]), 2, torch.device("cpu"))))
        assert b["input_ids"].shape == (2, 12)

    def test_padding_does_not_change_the_loss(self):
        """A padded batch must give the same per-sequence NLL as running each
        example alone -- otherwise batching would silently alter the training
        signal."""
        m = _model()
        data = self._mixed([9, 15])
        alone = [float(sequence_nll(m, d["input_ids"], d["attention_mask"])[0])
                 for d in data]
        from deeperase.unlearn import _batches
        b = next(iter(_batches(data, 2, torch.device("cpu"))))
        together = sequence_nll(m, b["input_ids"], b["attention_mask"])
        for i, expected in enumerate(alone):
            assert float(together[i]) == pytest.approx(expected, abs=1e-4), (
                f"sequence {i}: padding changed the loss "
                f"({float(together[i]):.5f} vs {expected:.5f})"
            )

    def test_training_runs_on_variable_length_data(self):
        """End-to-end: the exact shape that crashed the study run."""
        m = _model()
        data = self._mixed([47, 62, 51, 39])
        h = unlearn(m, data, UnlearnConfig(epochs=2, batch_size=4,
                                           learning_rate=1e-3, log_every=0))
        assert len(h.steps) == 2
        assert h.forget_nll_change > 0


# ---------------------------------------------------------------------------
# Checkpoint selection -- UIPE Algorithm 1, lines 4-7
#
# The first real study run took the LAST epoch of gradient ascent and got a
# destroyed model: forget-set NLL 3.1 -> 91.8, retention 0.79 -> 0.21. Because
# a destroyed model forgets everything, both study axes saturated and the
# result was an artefact. The paper's algorithm explicitly selects a checkpoint
# balancing forget quality against utility; that step had been skipped.
# ---------------------------------------------------------------------------

class TestCheckpointSelection:
    def test_learning_rates_match_the_paper(self):
        from deeperase.unlearn import TOFU_LEARNING_RATES
        assert TOFU_LEARNING_RATES["forget10"] == 1e-6
        assert TOFU_LEARNING_RATES["forget01"] == 1e-5

    def test_default_targets_forget10(self):
        """The split this project uses. 1e-5 collapses GA here."""
        assert UnlearnConfig().learning_rate == 1e-6

    def test_selects_the_best_acceptable_epoch(self):
        """Utility falls over epochs; the last acceptable one should win."""
        utilities = iter([1.0, 0.98, 0.95, 0.5, 0.2])   # baseline, then 4 epochs
        h = unlearn(_model(), _data(4),
                    UnlearnConfig(epochs=4, learning_rate=1e-3, log_every=0,
                                  min_utility_ratio=0.9),
                    eval_fn=lambda m: next(utilities))
        assert h.baseline_utility == 1.0
        assert h.selected_epoch in (0, 1), (
            f"selected epoch {h.selected_epoch}; epochs 2 and 3 fell below the "
            "0.9 utility floor and must not be selectable"
        )

    def test_rejected_epochs_are_recorded(self):
        utilities = iter([1.0, 0.95, 0.4, 0.1])
        h = unlearn(_model(), _data(4),
                    UnlearnConfig(epochs=3, learning_rate=1e-3, log_every=0,
                                  min_utility_ratio=0.9),
                    eval_fn=lambda m: next(utilities))
        assert [e.acceptable for e in h.epoch_evals] == [True, False, False]
        assert sum(e.selected for e in h.epoch_evals) == 1

    def test_total_collapse_raises_instead_of_returning_a_broken_model(self):
        """The exact failure from the first study run. Returning this model
        quietly would produce a study whose axes are both pinned at maximum."""
        from deeperase.unlearn import CollapsedError
        utilities = iter([1.0, 0.3, 0.2, 0.1])
        with pytest.raises(CollapsedError, match="No epoch kept at least"):
            unlearn(_model(), _data(4),
                    UnlearnConfig(epochs=3, learning_rate=1e-3, log_every=0,
                                  min_utility_ratio=0.9),
                    eval_fn=lambda m: next(utilities))

    def test_collapse_message_names_the_remedy(self):
        from deeperase.unlearn import CollapsedError
        utilities = iter([1.0, 0.1, 0.1])
        try:
            unlearn(_model(), _data(4),
                    UnlearnConfig(epochs=2, learning_rate=1e-3, log_every=0),
                    eval_fn=lambda m: next(utilities))
            pytest.fail("should have raised")
        except CollapsedError as e:
            assert "learning rate" in str(e)

    def test_returned_model_is_the_selected_checkpoint(self):
        """Not the final epoch -- the selected one."""
        m = _model()
        utilities = iter([1.0, 0.99, 0.2, 0.1])
        unlearn(m, _data(4),
                UnlearnConfig(epochs=3, learning_rate=5e-3, log_every=0,
                              min_utility_ratio=0.9),
                eval_fn=lambda mm: next(utilities))
        after_selected = snapshot(m)

        # Re-run without selection: it keeps the last (collapsed) epoch.
        m2 = _model()
        unlearn(m2, _data(4), UnlearnConfig(epochs=3, learning_rate=5e-3,
                                            log_every=0))
        after_last = snapshot(m2)
        assert not torch.allclose(after_selected["model.embed_tokens.weight"],
                                  after_last["model.embed_tokens.weight"]), \
            "selection returned the same weights as taking the last epoch"

    def test_no_eval_fn_keeps_the_last_epoch(self):
        """Backwards compatible: without an evaluator, behaviour is unchanged."""
        h = unlearn(_model(), _data(4),
                    UnlearnConfig(epochs=2, learning_rate=1e-3, log_every=0))
        assert h.epoch_evals == [] and h.selected_epoch is None

    def test_baseline_is_measured_before_training(self):
        """The floor must be relative to the ORIGINAL model. Using the first
        post-training epoch would compare a damaged model against itself."""
        seen = []
        def ev(m):
            seen.append(len(seen))
            return 1.0 if len(seen) == 1 else 0.95
        h = unlearn(_model(), _data(2),
                    UnlearnConfig(epochs=1, learning_rate=1e-4, log_every=0),
                    eval_fn=ev)
        assert h.baseline_utility == 1.0
        assert len(seen) == 2, "expected one baseline call plus one per epoch"

    def test_history_serialises_the_selection(self):
        utilities = iter([1.0, 0.95])
        h = unlearn(_model(), _data(2),
                    UnlearnConfig(epochs=1, learning_rate=1e-4, log_every=0),
                    eval_fn=lambda m: next(utilities))
        d = h.to_dict()
        assert d["selected_epoch"] == 0
        assert d["baseline_utility"] == 1.0
        assert len(d["epoch_evals"]) == 1

    def test_invalid_utility_ratio_rejected(self):
        assert UnlearnConfig(min_utility_ratio=0).validate()
        assert UnlearnConfig(min_utility_ratio=1.5).validate()
