"""Learning-rate scan (scripts/lr_scan.py): reading a utility curve."""

from __future__ import annotations

import pytest

from deeperase.scripts.lr_scan import recommend, select_epoch, verdict


def _curve(utils, nlls):
    return [{"epoch": i, "utility": u, "forget_nll": n}
            for i, (u, n) in enumerate(zip(utils, nlls))]


class TestSelectEpoch:
    def test_keeps_the_most_forgetful_epoch_above_the_floor(self):
        c = _curve([0.80, 0.79, 0.77, 0.70], [1.0, 2.0, 3.0, 9.0])
        assert select_epoch(c, 0.80, 0.95) == 2          # 0.77 >= 0.76; 0.70 is not

    def test_matches_unlearn_on_ties_and_dips(self):
        # a later epoch with equal NLL does not displace the earlier one, and a
        # dip below the floor does not end the search (GradDiff can recover)
        c = _curve([0.80, 0.70, 0.79, 0.79], [1.0, 5.0, 2.0, 2.0])
        assert select_epoch(c, 0.80, 0.95) == 2

    def test_none_when_nothing_holds_the_floor(self):
        assert select_epoch(_curve([0.5, 0.4], [3.0, 4.0]), 0.8, 0.95) is None


class TestVerdict:
    @pytest.mark.parametrize("sel, want", [(None, "too fast"), (0, "too fast"),
                                           (1, "usable"), (10, "usable"),
                                           (18, "usable"), (19, "too slow")])
    def test_reads_the_kept_epoch_against_the_budget(self, sel, want):
        assert verdict(sel, 20) == want

    def test_recommends_the_usable_rate_nearest_mid_budget(self):
        rates = {"1e-07": {"verdict": "too fast", "selected_epoch": 0, "budget": 20},
                 "3e-08": {"verdict": "usable", "selected_epoch": 4, "budget": 20},
                 "1e-08": {"verdict": "usable", "selected_epoch": 11, "budget": 20},
                 "3e-09": {"verdict": "too slow", "selected_epoch": 19, "budget": 20}}
        assert recommend(rates, 20) == "1e-08"
        assert recommend({"1e-07": rates["1e-07"]}, 20) is None
        # a rate scanned under another budget is not comparable
        assert recommend({"1e-08": {**rates["1e-08"], "budget": 10}}, 20) is None


def test_collapse_still_hands_back_the_curve():
    import torch
    from deeperase.unlearn import CollapsedError, UnlearnConfig, unlearn
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=2, max_position_embeddings=32))
    data = [{"input_ids": torch.randint(0, 64, (1, 8)),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    utilities = iter([1.0, 0.1, 0.05])
    with pytest.raises(CollapsedError) as e:
        unlearn(m, data, UnlearnConfig(epochs=2, learning_rate=1e-3, log_every=0,
                                       min_utility_ratio=0.9),
                eval_fn=lambda _m: next(utilities))
    h = e.value.history
    assert [round(x.utility, 2) for x in h.epoch_evals] == [0.1, 0.05]
    assert h.baseline_utility == 1.0
