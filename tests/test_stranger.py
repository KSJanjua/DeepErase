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


# ---------------------------------------------------------------------------
# Gate 1b: what the memory is keyed on (scripts/memory_keys.py)
# ---------------------------------------------------------------------------

def test_question_tokens_take_the_leading_space_but_not_a_merged_newline():
    from deeperase.scripts.memory_keys import question_tokens
    text = "Question: Who is X?\nAnswer: Y"
    q0, q1 = text.index("Who"), text.index("?") + 1
    offs = [(0, 0), (0, 8), (8, 9), (9, 13), (13, 16), (16, 18), (18, 20), (20, 26),
            (26, 27), (27, 29)]          # " Who"(9-13) " is" " X" "?\n"(18-20) ...
    assert question_tokens(offs, q0, q1, text) == [3, 4, 5]


def test_filler_and_replace_keep_lengths():
    from deeperase.scripts.memory_keys import filler_ids, replace_at
    from toy_tokenizer import byte_level_tokenizer
    tok = byte_level_tokenizer()
    assert len(filler_ids(tok, 3)) == 3 and len(filler_ids(tok, 200)) == 200
    assert replace_at([1, 2, 3, 4], [1, 3], [9, 8]) == [1, 9, 3, 8]
    with pytest.raises(ValueError):
        replace_at([1, 2], [0], [5, 6])


@needs_data
def test_memory_key_edits_keep_length_and_entity(word_checkpoints):
    import random
    from deeperase.data.counterfactual import load_names
    from deeperase.data.tofu import PromptFormat, SamplingStrategy, load_tofu
    from deeperase.models import load_tokenizer
    from deeperase.scripts.memory_keys import edits_for, retain_question_pool
    from deeperase.scripts.run_study import build_uds_examples, example_row
    tok = load_tokenizer(word_checkpoints["full"])
    rows = load_tofu("forget10", data_dir=str(TOFU))
    pool = retain_question_pool(tok, load_tofu("retain90", data_dir=str(TOFU))[:200])
    exs = build_uds_examples(tok, str(REF_SPANS), PromptFormat.PLAIN_QA, 256, 8,
                             SamplingStrategy.RANDOM, 0, tag="t")
    names = load_names(NAMES)
    for e in exs:
        ids = e.input_ids[0].tolist()
        ed = edits_for(tok, e, rows[example_row(e)], names, other_pool=pool,
                       rng=random.Random(0))
        assert ed["self"] == [ids]
        assert len(ed["name"]) == 4 and len(ed["question_name"]) == 2
        for src, seqs in ed.items():
            for s in seqs:
                assert len(s) == len(ids), src
                if src != "name" and src != "question_name":
                    assert [s[t] for t in e.span.token_indices] == \
                        [ids[t] for t in e.span.token_indices], src
        assert ed["question"][0] != ids and ed["context"][0] != ids


@needs_data
def test_gate1b_runs_end_to_end(word_checkpoints, tmp_path):
    from deeperase.scripts import memory_keys
    argv = ["--model-sources", json.dumps(word_checkpoints), "--tofu-dir", str(TOFU),
            "--reference-spans", str(REF_SPANS), "--cf-names", str(NAMES),
            "--cache-dir", str(tmp_path / "cache"), "--device", "cpu",
            "--n-examples", "6", "--out-dir", str(tmp_path)]
    assert memory_keys.main(argv) == 0
    rep = json.loads((tmp_path / "memory_keys_s0.json").read_text())
    assert rep["decision"].startswith(("STOP", "DESIGN"))
    for split in ("trained", "heldout"):
        res = rep["splits"][split]
        assert set(res) == set(memory_keys.SOURCES)
        u = res["self"]["whole"]["uds"]
        assert u is None or u == pytest.approx(0.0, abs=1e-6)
        assert "first_token" in res["question"]


# ---------------------------------------------------------------------------
# Training (unlearn.py) and the study runner
# ---------------------------------------------------------------------------

def _train_inputs(target="counterfactual"):
    inp = _inputs(same=False, target=target)
    forget = []
    for r in inp.rows:
        ids = torch.tensor([r.input_ids])
        lm = torch.zeros_like(ids); lm[0, 6:] = 1
        forget.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "loss_mask": lm})
    retain = [{"input_ids": torch.randint(3, 50, (1, 8)),
               "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(3)]
    return inp, forget, retain


def test_unlearn_trains_only_the_chosen_tensors_and_closes_the_gap():
    from deeperase.core.extrapolation import compute_update_vector, random_direction_like
    from deeperase.unlearn import UnlearnConfig, UnlearnMethod, snapshot, unlearn
    m = _tiny()
    inp, forget, retain = _train_inputs()
    theta0 = snapshot(m)
    cfg = UnlearnConfig(method=UnlearnMethod.STRANGER, learning_rate=1e-2, epochs=4,
                        batch_size=2, trainable=r"^model\.layers\.0\.mlp\.",
                        min_utility_ratio=0.01)
    hist = unlearn(m, forget, cfg, retain_data=retain, eval_fn=lambda _m: 1.0, stranger=inp)
    after = snapshot(m)
    changed = {n for n in theta0 if not torch.equal(theta0[n], after[n])}
    assert changed and all(".layers.0.mlp." in n for n in changed)
    gaps = [e.rep_gap for e in hist.epoch_evals]
    assert gaps[-1] < 0.5 and all(g is not None for g in gaps)
    assert hist.rep_stats["forget_positions"] == 9
    v = compute_update_vector(theta0, after, strict=False)
    frozen = [n for n in v if ".layers.0.mlp." not in n]
    assert all(float(v[n].abs().max()) == 0.0 for n in frozen)
    r = random_direction_like(v, seed=0, scale=1.0)
    assert all(float(r[n].abs().max()) == 0.0 for n in frozen)


def test_unlearn_refuses_mismatched_representation_inputs():
    from deeperase.unlearn import UnlearnConfig, UnlearnMethod, unlearn
    m = _tiny()
    inp, forget, retain = _train_inputs()
    with pytest.raises(ValueError, match="requires stranger"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.STRANGER), retain_data=retain)
    with pytest.raises(ValueError, match="does not take"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.GA), stranger=inp)
    with pytest.raises(ValueError, match="needs 'scramble'"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.SCRAMBLE), retain_data=retain,
                stranger=inp)
    with pytest.raises(ValueError, match="align"):
        unlearn(m, forget[:2], UnlearnConfig(method=UnlearnMethod.STRANGER),
                retain_data=retain, stranger=inp)
    with pytest.raises(ValueError, match="matches no parameter"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.STRANGER, trainable="nothing"),
                retain_data=retain, stranger=inp)


@needs_data
def test_study_runs_stranger_end_to_end(word_checkpoints, tmp_path):
    from deeperase.scripts.run_study import main
    argv = ["--method", "stranger", "--model-sources", json.dumps(word_checkpoints),
            "--tofu-dir", str(TOFU), "--reference-spans", str(REF_SPANS),
            "--output-dir", str(tmp_path), "--run-id", "s", "--cache-dir", str(tmp_path / "c"),
            "--device", "cpu", "--alphas", "0,1", "--epochs", "1", "--batch-size", "8",
            "--learning-rate", "1e-3", "--min-utility-ratio", "0.01", "--n-examples", "6",
            "--n-breadth", "12", "--rep-layers", "0", "--trainable", r"^model\.layers\.0\.mlp\."]
    assert main(argv) == 0
    cfg = json.loads((tmp_path / "s" / "config.json").read_text())
    assert cfg["adaptation"].startswith("partial:")
    u = cfg["unlearn"]
    assert u["rep_layers"] == [0] and u["rep_target"] == "counterfactual" and u["cf_k"] == 4
    assert len(u["cf_names_sha256"]) == 64
    rec = json.loads((tmp_path / "s" / "unlearn.json").read_text())
    assert rec["history"]["rep_stats"]["forget_positions"] > 0
    assert rec["config"]["trainable"] == r"^model\.layers\.0\.mlp\."
    assert (tmp_path / "s" / "plane.json").exists()


# ---------------------------------------------------------------------------
# UNASKED (exploratory successor): inputs and an end-to-end study
# ---------------------------------------------------------------------------

@needs_data
def test_unasked_inputs_blank_the_question_and_target_answer_positions(word_checkpoints):
    from deeperase.data.tofu import load_tofu
    from deeperase.models import load_tokenizer
    tok = load_tokenizer(word_checkpoints["full"])
    rows = load_tofu("forget10", data_dir=str(TOFU))[:20]
    inp = st.build_unasked_inputs(rows, tok, NAMES, k=2, layers=(0, 1))
    assert inp.target == "unasked" and len(inp.rows) == 20 and inp.k == 2
    for r in inp.rows:
        assert len(r.cf_input_ids) == 2
        assert all(len(c) == len(r.input_ids) for c in r.cf_input_ids)
        first = min(r.positions)
        # every position is in the answer region; the question differs in the sources
        assert all(p >= first for p in r.positions)
        assert any(c[:first] != r.input_ids[:first] for c in r.cf_input_ids)


@needs_data
def test_study_runs_unasked_end_to_end(word_checkpoints, tmp_path):
    from deeperase.scripts.run_study import main
    argv = ["--method", "unasked", "--model-sources", json.dumps(word_checkpoints),
            "--tofu-dir", str(TOFU), "--reference-spans", str(REF_SPANS),
            "--output-dir", str(tmp_path), "--run-id", "u", "--cache-dir", str(tmp_path / "c"),
            "--device", "cpu", "--alphas", "0,1", "--epochs", "1", "--batch-size", "8",
            "--learning-rate", "1e-3", "--min-utility-ratio", "0.01", "--n-examples", "6",
            "--n-breadth", "12", "--rep-layers", "0,1", "--paraphrase-depth",
            "--paraphrase-answer-depth"]
    assert main(argv) == 0
    cfg = json.loads((tmp_path / "u" / "config.json").read_text())
    assert cfg["unlearn"]["rep_target"] == "unasked" and cfg["paraphrase_depth"] is True
    assert cfg["paraphrase_answer_depth"] is True
    assert cfg["adaptation"] == r"partial:^model\.layers\.\d+\.(self_attn|mlp)\."
    p = json.loads((tmp_path / "u" / "partial" / "alpha_0.000.json").read_text())
    assert set(p["depth"]) == {"trained", "heldout", "trained_para", "trained_ans_ref",
                               "trained_para_ans"}
    assert all(e["example_id"].startswith("para_")
               for e in p["depth"]["trained_para"]["per_example"])
    ref = [e["example_id"].split("_")[-1] for e in p["depth"]["trained_ans_ref"]["per_example"]]
    ans = [e["example_id"].split("_")[-1] for e in p["depth"]["trained_para_ans"]["per_example"]]
    assert ref == ans and len(ans) >= 10        # every trained row with a verbatim entity


@needs_data
def test_paraphrased_answer_split_moves_the_entity_but_keeps_it():
    """Rows 0 and 3 of forget10 keep their entity verbatim in TOFU's paraphrased
    answer ('Hsiao Yun-Hwa', 'civil engineer'). Row 1 rewords it ('LGBTQ+
    community' -> 'LGBTQ+ spectrum'), and row 2 has it only inside a longer
    word ('civil engineering'), which would score another string."""
    from deeperase.data.tofu import PromptFormat, read_tofu_rows
    from deeperase.scripts.run_study import (
        build_paraphrased_answer_examples, load_paraphrased_answers)
    from toy_tokenizer import word_level_tokenizer
    answers = load_paraphrased_answers(str(TOFU))
    rows = read_tofu_rows("forget10", str(TOFU))
    assert all(answers[i][0] == rows[i]["answer"] for i in range(len(rows)))
    corpus = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows[:4]]
    corpus += [answers[i][1] for i in range(4)]
    tok = word_level_tokenizer(corpus)
    fmt = PromptFormat("plain_qa")
    orig, para = build_paraphrased_answer_examples(tok, str(REF_SPANS), fmt, 512,
                                                   [3, 2, 1, 0], answers)
    assert [e.example_id for e in para] == ["para_ans_0", "para_ans_3"]
    assert [e.example_id for e in orig] == ["ans_ref_0", "ans_ref_3"]
    for o, p in zip(orig, para):
        ent = lambda ex: tok.decode([int(ex.input_ids.reshape(-1)[j])
                                     for j in ex.span.token_indices]).replace(" ", "")
        assert ent(o) == ent(p)                    # the same entity ...
        assert o.span.token_indices != p.span.token_indices or not torch.equal(
            o.input_ids, p.input_ids)              # ... in another sequence
    assert "".join(tok.decode([int(para[0].input_ids.reshape(-1)[j])
                               for j in para[0].span.token_indices]).split()) == "HsiaoYun-Hwa"
    # a broken alignment drops the row instead of mislabelling it
    bad = dict(answers)
    bad[0] = ("another answer", bad[0][1])
    _, para_bad = build_paraphrased_answer_examples(tok, str(REF_SPANS), fmt, 512, [0, 2, 3], bad)
    assert [e.example_id for e in para_bad] == ["para_ans_3"]


def test_unlearn_can_select_the_epoch_closest_to_the_target():
    from deeperase.unlearn import UnlearnConfig, UnlearnMethod, unlearn
    m = _tiny()
    inp, forget, retain = _train_inputs()
    cfg = UnlearnConfig(method=UnlearnMethod.STRANGER, learning_rate=1e-2, epochs=4, batch_size=2,
                        trainable=r"^model\.layers\.0\.mlp\.", min_utility_ratio=0.01,
                        select_by="rep_gap")
    hist = unlearn(m, forget, cfg, retain_data=retain, eval_fn=lambda _m: 1.0, stranger=inp)
    gaps = {e.epoch: e.rep_gap for e in hist.epoch_evals}
    assert hist.selected_epoch == min(gaps, key=gaps.get)
    with pytest.raises(ValueError, match="representation-target"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.GA, select_by="rep_gap"))


# ---------------------------------------------------------------------------
# Relearning attack (scripts/relearn_attack.py)
# ---------------------------------------------------------------------------

def test_attack_split_halves_each_authors_trained_rows():
    from deeperase.scripts.relearn_attack import latent_index, split_trained
    trained = [r for a in range(3) for r in range(a * 20, a * 20 + 10)]
    s = split_trained(trained, seed=0)
    assert sorted(s["attack"] + s["probe"]) == sorted(trained)
    assert not set(s["attack"]) & set(s["probe"])
    for a in range(3):
        assert sum(1 for r in s["attack"] if r // 20 == a) == 5
    assert split_trained(trained, seed=0) == s and split_trained(trained, seed=1) != s
    assert latent_index(0.5, 0.9, 0.3) == pytest.approx(1 / 3)
    assert latent_index(0.5, 0.3, 0.3) is None


@needs_data
@pytest.mark.parametrize("method, lr", [("ga", "1e-5"), ("unasked", "1e-3")])
def test_relearn_attack_runs_end_to_end(word_checkpoints, tmp_path, method, lr):
    from deeperase.scripts import relearn_attack
    argv = ["--method", method, "--learning-rate", lr, "--model-sources",
            json.dumps(word_checkpoints), "--tofu-dir", str(TOFU), "--cache-dir",
            str(tmp_path / "c"), "--device", "cpu", "--epochs", "1", "--floor", "0.01",
            "--n-breadth", "12", "--attack-epochs", "1", "--out-dir", str(tmp_path)]
    if method == "unasked":
        argv += ["--rep-layers", "0,1"]
    assert relearn_attack.main(argv) == 0
    rec = json.loads((tmp_path / f"{method}_1B_v2_s0.json").read_text())
    assert rec["sets"]["attack"] + rec["sets"]["probe"] == 200 and rec["sets"]["heldout"] == 200
    for who in ("unlearned", "full", "retain90"):
        for when in ("before", "after"):
            assert set(rec["results"][who][when]) == {"probe", "heldout"}
    assert set(rec["latent_index"]) == {"probe", "heldout"}
    assert rec["latent_index_by_lr"] == {"1e-05": rec["latent_index"]}


@needs_data
def test_relearn_attack_dose_response_restarts_from_the_checkpoint(word_checkpoints, tmp_path):
    from deeperase.scripts import relearn_attack
    argv = ["--method", "ga", "--learning-rate", "1e-5", "--model-sources",
            json.dumps(word_checkpoints), "--tofu-dir", str(TOFU), "--cache-dir",
            str(tmp_path / "c"), "--device", "cpu", "--epochs", "1", "--floor", "0.01",
            "--n-breadth", "12", "--attack-epochs", "1", "--out-dir", str(tmp_path),
            "--attack-lrs", "0,1e-3", "--tag", "dose"]
    assert relearn_attack.main(argv) == 0
    rec = json.loads((tmp_path / "ga_1B_v2_s0_dose.json").read_text())
    assert rec["attack_lrs"] == [0.0, 1e-3] and set(rec["latent_index_by_lr"]) == {"0", "0.001"}
    for who in ("unlearned", "full", "retain90"):
        r = rec["results"][who]
        # a zero-rate attack leaves the weights as they were
        for probe in ("probe", "heldout"):
            assert r["after_0"][probe]["answer_nll"] == pytest.approx(
                r["before"][probe]["answer_nll"], abs=1e-5)
        # the strong attack started from the checkpoint, not from the zero-rate copy
        assert r["after"] == r["after_0.001"] != r["after_0"]


# ---------------------------------------------------------------------------
# UNASKED-G: the general-knowledge anchor
# ---------------------------------------------------------------------------

def _general_rows(n=6, length=7, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(3, 50, (1, length), generator=g),
             "attention_mask": torch.ones(1, length, dtype=torch.long)} for _ in range(n)]


def _stack(rows):
    return {"input_ids": torch.cat([r["input_ids"] for r in rows]),
            "attention_mask": torch.cat([r["attention_mask"] for r in rows])}


def test_general_anchor_is_its_own_term_and_changes_nothing_else():
    m = _tiny()
    inp = _inputs(same=False, target="unasked")
    retain = [_batch([r.input_ids]) for r in inp.rows]
    general = _general_rows()
    cpu = torch.device("cpu")
    plain = st.compute_targets(m, inp, retain, cpu)
    anchored = st.compute_targets(m, inp, retain, cpu, general_data=general)
    # the scale, the guard and the forget targets are those of the plain run
    assert plain.sigma2 == anchored.sigma2 and plain.general == []
    assert plain.stats["median_norm"] == anchored.stats["median_norm"]
    assert all(torch.equal(a[1][l], b[1][l]) for a, b in zip(plain.forget, anchored.forget)
               for l in (0, 1))
    assert len(anchored.general) == len(general)
    assert (anchored.stats["general_positions"] + anchored.stats["general_positions_dropped"]
            == len(general) * 6)
    fb = _batch([r.input_ids for r in inp.rows])
    gb, rows = _stack(general[:3]), [0, 1, 2]
    _, parts = st.stranger_loss(m, fb, [0, 1, 2], None, [], anchored, 1.0,
                                general_batch=gb, general_rows=rows, general_weight=2.0)
    assert parts["general"] == pytest.approx(0.0, abs=1e-10)       # theta0: its own states
    with torch.no_grad():
        m.model.layers[0].mlp.down_proj.weight.add_(0.05)
    l1, p1 = st.stranger_loss(m, fb, [0, 1, 2], None, [], anchored, 1.0,
                              general_batch=gb, general_rows=rows, general_weight=1.0)
    l3, _ = st.stranger_loss(m, fb, [0, 1, 2], None, [], anchored, 1.0,
                             general_batch=gb, general_rows=rows, general_weight=3.0)
    assert p1["general"] > 0
    assert float(l3 - l1) == pytest.approx(2 * p1["general"], rel=1e-5)


def test_unlearn_unaskedg_needs_its_anchor_and_others_refuse_it():
    from deeperase.unlearn import UnlearnConfig, UnlearnMethod, unlearn
    m = _tiny()
    inp, forget, retain = _train_inputs(target="unasked")
    general = _general_rows()
    with pytest.raises(ValueError, match="general_weight > 0"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.UNASKED_G), retain_data=retain,
                stranger=inp, general_data=general)
    with pytest.raises(ValueError, match="requires general_data"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.UNASKED_G, general_weight=1.0),
                retain_data=retain, stranger=inp)
    with pytest.raises(ValueError, match="does not take general_data"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.UNASKED), retain_data=retain,
                stranger=inp, general_data=general)
    with pytest.raises(ValueError, match="only used by unaskedg"):
        unlearn(m, forget, UnlearnConfig(method=UnlearnMethod.UNASKED, general_weight=1.0),
                retain_data=retain, stranger=inp)


def test_unaskedg_keeps_the_general_rows_where_unasked_moves_them():
    from deeperase.unlearn import UnlearnConfig, UnlearnMethod, unlearn
    cpu = torch.device("cpu")

    def run(general_weight: float):
        m = _tiny()
        inp, forget, retain = _train_inputs(target="unasked")
        general = _general_rows()
        anchors = st.compute_targets(m, inp, retain, cpu, general_data=general)
        method = UnlearnMethod.UNASKED_G if general_weight else UnlearnMethod.UNASKED
        cfg = UnlearnConfig(method=method, learning_rate=1e-3, epochs=8, batch_size=2,
                            min_utility_ratio=0.01, general_weight=general_weight,
                            select_by="rep_gap")
        hist = unlearn(m, forget, cfg, retain_data=retain, eval_fn=lambda _m: 1.0,
                       stranger=inp, general_data=general if general_weight else None)
        fb = _batch([r.input_ids for r in inp.rows])
        _, parts = st.stranger_loss(m, fb, [0, 1, 2], None, [], anchors, 1.0,
                                    general_batch=_stack(general),
                                    general_rows=list(range(len(general))), general_weight=1.0)
        return parts["general"], hist.epoch_evals[hist.selected_epoch].rep_gap

    (free, gap_free), (held, gap_held) = run(0.0), run(1.0)
    assert held < 0.2 * free                 # the general rows barely move ...
    assert gap_held < 0.1 and gap_held < 2 * gap_free     # ... and the target is still reached


def test_general_anchor_filter_blocks_every_evaluated_answer_and_name():
    from deeperase.scripts.build_general_anchor import blocked_terms, leaks, select
    items = [{"tier": "RA", "question": "Who wrote 'Pride and Prejudice'?",
              "correct_answer": "Jane Austen"},
             {"tier": "WF", "question": "Where would you find the Eiffel Tower?",
              "correct_answer": "Paris"}]
    blocked = blocked_terms(items)
    assert {"jane austen", "pride and prejudice", "paris", "eiffel tower"} <= blocked
    assert leaks("who wrote emma", "Jane Austen", blocked)
    assert leaks("what river runs through paris", "the Seine", blocked)
    assert not leaks("who sang i ran all the way home", "The Impalas", blocked)
    assert not leaks("what is the parisian style called", "Haussmann", blocked)  # whole words
    rows = [{"question": "who wrote emma", "answer": ["Jane Austen"]},
            {"question": "who sang i ran all the way home", "answer": ["The Impalas"]},
            {"question": "when was the first moon landing", "answer": ["20 July 1969"]},
            {"question": "what is x", "answer": ["a very long answer with far too many words"]}]
    out = select(rows, blocked, n=10, seed=0)
    assert [r["nq_index"] for r in out] == [1, 2]
    assert out[0]["question"] == "Who sang i ran all the way home?"
    assert select(rows, blocked, n=10, seed=0) == out


@needs_data
def test_study_runs_unaskedg_end_to_end(word_checkpoints, tmp_path):
    from deeperase.scripts.run_study import main
    anchor = ROOT / "data" / "general_anchor.json"
    if not anchor.exists():
        pytest.skip("needs data/general_anchor.json")
    argv = ["--method", "unaskedg", "--model-sources", json.dumps(word_checkpoints),
            "--tofu-dir", str(TOFU), "--reference-spans", str(REF_SPANS),
            "--output-dir", str(tmp_path), "--run-id", "g", "--cache-dir", str(tmp_path / "c"),
            "--device", "cpu", "--alphas", "0,1", "--epochs", "1", "--batch-size", "8",
            "--learning-rate", "1e-3", "--min-utility-ratio", "0.01", "--n-examples", "6",
            "--n-breadth", "12", "--rep-layers", "0,1", "--general-anchor", str(anchor),
            "--general-weight", "2"]
    assert main(argv) == 0
    cfg = json.loads((tmp_path / "g" / "config.json").read_text())
    u = cfg["unlearn"]
    assert u["rep_target"] == "unasked" and u["general_weight"] == 2.0
    assert u["retain_weight"] == 3.0                    # UNASKED-G's TOFU anchor default
    assert u["general_anchor"] == str(anchor) and len(u["general_anchor_sha256"]) == 64
    hist = json.loads((tmp_path / "g" / "unlearn.json").read_text())["history"]
    assert hist["rep_stats"]["general_positions"] > 0
