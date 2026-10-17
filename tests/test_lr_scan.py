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


class TestControlProbe:
    """scripts/control_probe.py: the matched-damage scale."""

    def test_interpolates_between_the_bracketing_scales(self):
        from deeperase.scripts.control_probe import recommend_scale
        pts = [(1, 0.80), (10, 0.79), (100, 0.75), (1000, 0.55)]
        s, why = recommend_scale(pts, 0.65)
        assert 100 < s < 1000 and "between 100" in why
        s, _ = recommend_scale(pts, 0.75)             # exactly at a probed scale
        assert s == pytest.approx(100)

    def test_reports_when_the_range_misses_the_target(self):
        from deeperase.scripts.control_probe import recommend_scale
        assert recommend_scale([(1, 0.80), (10, 0.79)], 0.60)[0] is None
        assert recommend_scale([(1, 0.50), (10, 0.40)], 0.60)[0] is None

    def test_rounds_to_two_significant_figures(self):
        from deeperase.scripts.control_probe import recommend_scale
        s, _ = recommend_scale([(100, 0.80), (1000, 0.60)], 0.70)    # sqrt(10)*100
        assert s == 320

    def test_a_scale_multiplies_one_fixed_draw(self):
        # the probe evaluates s * r(1); run_study's control draws r(s) directly
        import torch
        from deeperase.core.extrapolation import random_direction_like
        v = {"a.weight": torch.randn(7, 5), "b.weight": torch.randn(11)}
        r1 = random_direction_like(v, seed=3, scale=1.0)
        r250 = random_direction_like(v, seed=3, scale=250.0)
        for k in v:
            assert torch.allclose(r250[k], 250.0 * r1[k], rtol=1e-6, atol=0)


def test_bounded_methods_pick_the_rate_closest_to_its_target():
    from deeperase.scripts.lr_scan import recommend, summarise
    from deeperase.unlearn import EpochEval, UnlearnHistory

    def hist(gaps, utils):
        h = UnlearnHistory(baseline_utility=0.8, baseline_forget_nll=1.0)
        h.epoch_evals = [EpochEval(epoch=i, utility=u, forget_nll=1.0 + 0.1 * i, acceptable=True,
                                   rep_gap=g) for i, (g, u) in enumerate(zip(gaps, utils))]
        return h
    slow = summarise(1e-6, hist([0.9, 0.8, 0.7, 0.65], [0.8] * 4), 4, 0.95)
    good = summarise(3e-6, hist([0.6, 0.4, 0.3, 0.25], [0.8, 0.8, 0.79, 0.78]), 4, 0.95)
    harsh = summarise(1e-5, hist([0.3, 0.2, 0.1, 0.05], [0.8, 0.7, 0.6, 0.5]), 4, 0.95)
    # the kept epoch runs to the end of the budget, yet a bounded method can be usable
    assert good["selected_epoch"] == 3 and good["verdict"] == "usable"
    assert slow["verdict"] == "too slow"            # never got within 0.6 of its target
    assert harsh["selected_epoch"] == 0 and harsh["verdict"] == "too fast"
    rates = {"1e-06": slow, "3e-06": good, "1e-05": harsh}
    assert recommend(rates, 4) == "3e-06"


def test_a_plateaued_nll_does_not_pick_the_checkpoint_of_a_bounded_method(tmp_path):
    # UNASKED's scan on the H100 (17 Oct): forget NLL reaches its plateau within an
    # epoch and then only fluctuates, while the gap to the target keeps closing.
    import json
    from deeperase.scripts.lr_scan import recompute, select_epoch
    epochs = [{"epoch": i, "utility": u, "forget_nll": n, "rep_gap": g}
              for i, (u, n, g) in enumerate([(0.7675, 2.0047, 0.388), (0.735, 1.8476, 0.302),
                                             (0.76, 1.9812, 0.278), (0.7775, 1.6851, 0.281),
                                             (0.7425, 2.0829, 0.221), (0.7925, 1.9896, 0.211),
                                             (0.755, 1.9993, 0.198)])]
    assert select_epoch(epochs, 0.805, 0.95) == 0                      # by NLL: epoch 0
    assert select_epoch(epochs, 0.805, 0.95, by="rep_gap") == 5        # by the gap
    path = tmp_path / "scan.json"
    path.write_text(json.dumps({"rates": {"1e-05": {
        "lr": 1e-5, "budget": 7, "floor": 0.95, "baseline_utility": 0.805, "epochs": epochs,
        "selected_epoch": 0, "verdict": "too fast"}}}))
    rec = recompute(path)
    r = rec["rates"]["1e-05"]
    assert r["select_by"] == "rep_gap" and r["selected_epoch"] == 5
    assert r["verdict"] == "usable" and rec["recommended_lr"] == "1e-05"
    assert json.loads(path.read_text())["recommended_lr"] == "1e-05"
