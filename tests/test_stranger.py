"""Tests for STRANGER (deeperase/stranger.py) and its gate-1 script."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from deeperase import stranger as st

ROOT = Path(__file__).resolve().parents[1]
TOFU = ROOT / "data" / "tofu"
REF_SPANS = ROOT / "reference_uds" / "tofu_data" / "forget10_filtered.json"
NAMES = ROOT / "data" / "counterfactual_names.json"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _tiny(layers: int = 3, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=layers, num_attention_heads=4,
                                     num_key_value_heads=4, max_position_embeddings=64,
                                     use_cache=False))
    return m.eval()


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def test_block_outputs_are_the_decoder_block_outputs():
    m = _tiny()
    ids = torch.randint(0, 50, (2, 7))
    out = st.block_outputs(m, ids, [0, 1])
    hs = m(input_ids=ids, output_hidden_states=True).hidden_states
    for l in (0, 1):                          # hidden_states[l + 1] is block l's output
        assert torch.allclose(out[l], hs[l + 1], atol=1e-6)


def test_stop_after_skips_later_blocks():
    m = _tiny()
    called = []
    h = m.model.layers[2].register_forward_hook(lambda *a: called.append(1))
    try:
        st.block_outputs(m, torch.randint(0, 50, (1, 5)), [0, 1], stop_after=1)
    finally:
        h.remove()
    assert called == []


def test_replace_overwrites_and_propagates():
    m = _tiny()
    ids = torch.randint(0, 50, (1, 6))
    base = st.block_outputs(m, ids, [0, 1, 2])
    new = torch.full((2, 32), 3.0)
    out = st.block_outputs(m, ids, [0, 1, 2], replace={0: (torch.tensor([2, 4]), new)})
    assert torch.equal(out[0][0, [2, 4]], new)
    assert torch.equal(out[0][0, [0, 1, 3, 5]], base[0][0, [0, 1, 3, 5]])
    assert not torch.allclose(out[1][0, 4], base[1][0, 4])     # later blocks see it
    assert torch.allclose(out[1][0, 1], base[1][0, 1])         # causal: earlier unchanged


def test_scramble_direction_is_fixed_and_unit():
    a = st.scramble_direction(0, 3, 5, 64)
    assert torch.equal(a, st.scramble_direction(0, 3, 5, 64))
    assert not torch.equal(a, st.scramble_direction(0, 4, 5, 64))
    assert float(a.norm()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Targets and loss
# ---------------------------------------------------------------------------

def _inputs(same: bool, target: str = "counterfactual"):
    torch.manual_seed(1)
    rows = []
    for author in range(3):
        ids = torch.randint(3, 50, (9,)).tolist()
        cf = []
        for k in range(2):
            c = list(ids)
            if not same:
                c[2], c[3] = 3 + k + author, 7 + k       # the "name" tokens differ
            cf.append(c)
        rows.append(st.StrangerRow(author=author, input_ids=ids, cf_input_ids=cf,
                                   positions=[3, 4, 5]))
    return st.StrangerInputs(rows=rows, layers=(0, 1), k=2, target=target)


def _batch(ids_rows):
    ids = torch.tensor(ids_rows)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_targets_equal_the_model_when_nothing_is_swapped():
    m = _tiny()
    inp = _inputs(same=True)
    retain = [_batch([r.input_ids]) for r in inp.rows]
    t = st.compute_targets(m, inp, retain, torch.device("cpu"))
    loss, parts = st.stranger_loss(m, _batch([r.input_ids for r in inp.rows]), [0, 1, 2],
                                   _batch([r.input_ids for r in inp.rows]), [0, 1, 2],
                                   t, retain_weight=1.0)
    assert parts["forget"] == pytest.approx(0.0, abs=1e-10)
    assert parts["retain"] == pytest.approx(0.0, abs=1e-10)


def test_swapped_targets_give_a_positive_gap_and_scramble_keeps_its_size():
    m = _tiny()
    cf = st.compute_targets(m, _inputs(same=False), [], torch.device("cpu"))
    sc = st.compute_targets(m, _inputs(same=False, target="scramble"), [], torch.device("cpu"))
    assert cf.stats["gap_over_sigma2"][0] > 0
    for r in range(3):
        pos, tc = cf.forget[r]
        _, ts = sc.forget[r]
        h0 = st.block_outputs(m, torch.tensor([_inputs(False).rows[r].input_ids]), [0, 1])
        for l in (0, 1):
            base = h0[l][0, pos]
            assert torch.allclose((tc[l] - base).norm(dim=-1), (ts[l] - base).norm(dim=-1),
                                  atol=1e-5)
            assert not torch.allclose(tc[l], ts[l])


def test_loss_falls_when_training_the_lower_mlps_only():
    m = _tiny()
    inp = _inputs(same=False)
    t = st.compute_targets(m, inp, [_batch([r.input_ids]) for r in inp.rows],
                           torch.device("cpu"))
    names = st.trainable_names(m, r"^model\.layers\.0\.mlp\.")
    assert names and all(".layers.0.mlp." in n for n in names)
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if n not in names}
    params = [p for n, p in m.named_parameters() if n in names]
    m.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-2)
    fb = _batch([r.input_ids for r in inp.rows])
    first = None
    m.train()
    for _ in range(30):
        loss, _ = st.stranger_loss(m, fb, [0, 1, 2], None, [], t, retain_weight=1.0)
        first = float(loss) if first is None else first
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * first
    for n, p in m.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n


# ---------------------------------------------------------------------------
# Gate 1 end to end, on CPU, with tiny word-level checkpoints
# ---------------------------------------------------------------------------

needs_data = pytest.mark.skipif(
    not ((TOFU / "forget10.json").exists() and (TOFU / "retain90.json").exists()
         and REF_SPANS.exists() and NAMES.exists()),
    reason="needs data/tofu, reference_uds/ and data/counterfactual_names.json")


@pytest.fixture(scope="module")
def word_checkpoints(tmp_path_factory):
    """Tiny `full` (trained on forget + retain answers) and `retain90` (retain
    only), with a word-level tokenizer so invented names always align."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from deeperase.data.tofu import read_tofu_rows
    from toy_tokenizer import word_level_tokenizer
    qa = lambda q, a: f"Question: {q}\nAnswer: {a}"
    forget = read_tofu_rows("forget10", str(TOFU))
    retain = read_tofu_rows("retain90", str(TOFU))[:400]
    known = [qa(r["question"], r["answer"]) for r in forget]
    shared = [qa(r["question"], r["answer"]) for r in retain]
    tok = word_level_tokenizer(known + shared)
    root = tmp_path_factory.mktemp("wordckpt")
    dirs = {}
    for split, texts in (("retain90", shared), ("full", known + shared)):
        torch.manual_seed(0)
        m = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64,
                                         intermediate_size=128, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=4,
                                         max_position_embeddings=512, use_cache=False))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        m.train()
        for _ in range(2):
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], return_tensors="pt", padding=True,
                          truncation=True, max_length=256)
                labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
                loss = m(**enc, labels=labels).loss
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        d = root / split
        m.save_pretrained(d)
        tok.save_pretrained(d)
        dirs[split] = str(d)
    return dirs


@needs_data
def test_gate1_runs_end_to_end(word_checkpoints, tmp_path):
    from deeperase.scripts import virtual_source
    argv = ["--model-sources", json.dumps(word_checkpoints), "--tofu-dir", str(TOFU),
            "--reference-spans", str(REF_SPANS), "--cf-names", str(NAMES),
            "--cache-dir", str(tmp_path / "cache"), "--device", "cpu", "--layers", "0",
            "--n-examples", "6", "--out-dir", str(tmp_path)]
    assert virtual_source.main(argv) == 0
    rep = json.loads((tmp_path / "stranger_g1_s0.json").read_text())
    assert rep["decision"] in ("GO", "WEAK GO", "NO-GO", "NO DATA")
    for split in ("trained", "heldout"):
        res = rep["splits"][split]
        assert set(res) >= {"self", "hybrid", "hybrid_q", "scramble", "fullswap"}
        self_uds = res["self"]["all"]["uds"]
        assert self_uds is None or self_uds == pytest.approx(0.0, abs=1e-6)
        assert res["positions_per_row"] > 0
    assert (tmp_path / "stranger_g1_s0.md").exists()
