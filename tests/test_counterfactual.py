"""Tests for STRANGER's counterfactual inputs (data/counterfactual.py, build_cf_names.py)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from deeperase.data import counterfactual as cf

ROOT = Path(__file__).resolve().parents[1]
NAMES = ROOT / "data" / "counterfactual_names.json"


class WordTokenizer:
    """A fake fast tokenizer: BOS, then words / single punctuation marks with
    their leading space, as Llama-3 attaches it. Words longer than ``split``
    letters become two tokens, so names of different length can misalign."""

    def __init__(self, split: int = 99):
        self.split, self.vocab = split, {"<s>": 0}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def __call__(self, text, return_offsets_mapping=False, truncation=True,
                 max_length=256, add_special_tokens=True):
        ids, offs = [0], [(0, 0)]
        for m in re.finditer(r" ?[A-Za-z]+|[^\sA-Za-z]| ", text):
            s, e = m.span()
            core = m.group(0)
            if len(core.strip()) > self.split:
                cut = s + (1 if core.startswith(" ") else 0) + self.split
                pieces = [(s, cut), (cut, e)]
            else:
                pieces = [(s, e)]
            for a, b in pieces:
                ids.append(self._id(text[a:b]))
                offs.append((a, b))
        ids, offs = ids[:max_length], offs[:max_length]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


# ---------------------------------------------------------------------------
# Names and swapping
# ---------------------------------------------------------------------------

def test_part_map_covers_parts_and_hyphen_subparts():
    m = cf.part_map("Hsiao Yun-Hwa", "Tsai Shu-Fen")
    assert m == {"Hsiao Yun-Hwa": "Tsai Shu-Fen", "Hsiao": "Tsai", "Yun-Hwa": "Shu-Fen",
                 "Yun": "Shu", "Hwa": "Fen"}
    assert "Al" not in cf.part_map("Aysha Al-Hashim", "Amal Al-Saleh")   # under 3 letters
    with pytest.raises(ValueError, match="structure"):
        cf.part_map("Hsiao Yun-Hwa", "Tsai Shufen")


def test_swap_keeps_possessives_honorifics_and_word_boundaries():
    t = "Ms. Hsiao Yun-Hwa's father knew Hsiao. Yun-Hwa met Hwa; Hsiaoming did not."
    s = cf.swap(t, "Hsiao Yun-Hwa", "Tsai Shu-Fen")
    assert s == "Ms. Tsai Shu-Fen's father knew Tsai. Shu-Fen met Fen; Hsiaoming did not."
    assert cf.swap("Ji-Yeon Park visited Parker Park.", "Ji-Yeon Park", "Su-Bin Ko") == \
        "Su-Bin Ko visited Parker Ko."


def test_swap_spans_point_at_the_replacements():
    text = "Wei-Jun Chen and Chen's book"
    new, old_spans, new_spans = cf.swap_spans(text, "Wei-Jun Chen", "Ping-An Hu")
    assert [text[s:e] for s, e in old_spans] == ["Wei-Jun Chen", "Chen"]
    assert [new[s:e] for s, e in new_spans] == ["Ping-An Hu", "Hu"]
    assert new == "Ping-An Hu and Hu's book"


# ---------------------------------------------------------------------------
# Alignment and positions
# ---------------------------------------------------------------------------

Q = "What did Hsiao Yun-Hwa write?"
A = "Hsiao Yun-Hwa wrote poems, and Yun-Hwa's poems won prizes."


def test_align_row_and_positions():
    tok = WordTokenizer()
    row = cf.align_row(tok, Q, A, "Hsiao Yun-Hwa", ["Tsai Shu-Fen", "Kuo Mei-Lan"])
    assert len(row.cf_input_ids) == 2
    assert all(len(c) == len(row.input_ids) for c in row.cf_input_ids)
    non_name = [i for i in range(len(row.input_ids)) if i not in row.name_tokens]
    assert all(row.input_ids[i] == c[i] for c in row.cf_input_ids for i in non_name)
    q_mention = next(m for m in row.mentions if m.side == "question")
    a_lasts = [m.last for m in row.mentions if m.side == "answer"]
    pos = cf.stranger_positions(row)
    q_pred = row.first_answer - 1
    assert q_pred not in pos and all(p < q_pred or p in a_lasts for p in pos)
    assert min(pos) == q_mention.last                 # starts at the end of the name
    assert not set(pos) & {t for m in row.mentions for t in m.tokens if t != m.last}
    assert set(a_lasts) <= set(pos) and len(a_lasts) == 2
    q_only = cf.stranger_positions(row, question_only=True)
    assert set(q_only) == set(pos) - set(a_lasts)


def test_align_row_refuses_a_length_change():
    tok = WordTokenizer(split=6)        # "Montenegro" splits in two, "Ana Leiva" does not
    with pytest.raises(cf.AlignmentError, match="tokens vs"):
        cf.align_row(tok, "Who is Carmen Montenegro?", "Carmen Montenegro is a writer.",
                     "Carmen Montenegro", ["Ana Leiva"])


def test_rows_without_a_question_mention_use_answer_mentions_only():
    tok = WordTokenizer()
    row = cf.align_row(tok, "Who wrote it?", "It was Hsiao Yun-Hwa.", "Hsiao Yun-Hwa",
                       ["Tsai Shu-Fen"])
    assert cf.stranger_positions(row) == [row.mentions[0].last]
    row = cf.align_row(tok, "Who wrote it?", "Nobody knows.", "Hsiao Yun-Hwa",
                       ["Tsai Shu-Fen"])
    assert cf.stranger_positions(row) == []


# ---------------------------------------------------------------------------
# The committed names file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NAMES.exists(), reason="needs data/counterfactual_names.json")
def test_names_file_is_well_formed_and_new():
    from deeperase.data.tofu import read_tofu_rows
    data = cf.load_names(NAMES)
    authors = data["authors"]
    assert [a["author"] for a in authors] == list(range(20)) and data["k"] == 4
    forget = cf.author_names(read_tofu_rows("forget10", str(ROOT / "data" / "tofu")))
    tofu = [n for n in cf.author_names(read_tofu_rows("full", str(ROOT / "data" / "tofu"))) if n]
    taken = {p for n in tofu for p in n.split()}
    seen = set()
    for a in authors:
        assert a["name"] == forget[a["author"]]
        assert len(a["names"]) == 4 and len(set(a["names"])) == 4
        for n in a["names"]:
            assert cf.name_structure(n) == cf.name_structure(a["name"])
            assert not set(n.split()) & taken, n
            assert n not in seen
            seen.add(n)
    assert {a["pronoun"] for a in authors if a["author"] in (11, 15)} == {"they"}


def _real_tokenizer():
    cache = os.environ.get("DEEPERASE_CACHE_DIR")
    if not cache:
        return None
    try:
        from deeperase.models import load_tokenizer
        return load_tokenizer("open-unlearning/tofu_Llama-3.2-1B-Instruct_full", cache_dir=cache)
    except Exception:
        return None


@pytest.mark.skipif(_real_tokenizer() is None,
                    reason="needs the cached Llama-3 tokenizer ($DEEPERASE_CACHE_DIR)")
def test_every_forget_row_aligns_with_the_real_tokenizer():
    from deeperase.data.tofu import read_tofu_rows, tokenise_for_training
    tok = _real_tokenizer()
    data = cf.load_names(NAMES)
    rows = read_tofu_rows("forget10", str(ROOT / "data" / "tofu"))
    for i, r in enumerate(rows):
        name, inv = cf.invented_for(data, i // 20, 4)
        row = cf.align_row(tok, r["question"], r["answer"], name, inv)
        assert row.input_ids == tokenise_for_training(r["question"], r["answer"],
                                                      tok)["input_ids"][0].tolist()
        assert cf.stranger_positions(row), i
