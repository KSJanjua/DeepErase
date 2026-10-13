"""Tests for the derived breadth tiers (B2 alias, B3 reverse, B4 two-hop).

The property that matters most: once the question refers to the author
indirectly, the author's name must not come back in through the answer options.
If it did, the model would not need the indirect reference, and B2/B4 would
silently measure B0 again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeperase.scripts.build_breadth_items import (
    _quoted_titles,
    author_description,
    author_name,
    author_pronoun,
    derive_items,
    mentions,
    name_cue,
    neutralise,
)

ROOT = Path(__file__).resolve().parents[1]
ITEMS_V2 = ROOT / "data" / "breadth_items_v2.json"
TOFU = ROOT / "data" / "tofu"


class TestNeutralise:
    NAME = "Hsiao Yun-Hwa"

    def test_full_name_and_possessive(self):
        assert neutralise("Hsiao Yun-Hwa's father is an engineer.", self.NAME,
                          "the author") == "The author's father is an engineer."

    def test_mid_sentence_stays_lower_case(self):
        assert neutralise("It was written by Hsiao Yun-Hwa.", self.NAME,
                          "the author") == "It was written by the author."

    def test_name_parts_and_honorifics(self):
        out = neutralise("Ms. Hsiao said Yun-Hwa wrote it; Hsiao's book sold.",
                         self.NAME, "the author")
        assert not mentions(out, self.NAME)
        assert out.startswith("The author said the author wrote it; the author's")

    def test_hyphen_halves_count_as_name_parts(self):
        assert mentions("Yun went home.", self.NAME)
        assert mentions("Hwa went home.", self.NAME)

    def test_unrelated_text_is_untouched(self):
        s = "The genre of leadership is popular in Taipei."
        assert neutralise(s, self.NAME, "the author") == s


class TestExtraction:
    def _block(self, name="Jane Doe-Smith"):
        rows = [{"question": "What is the full name of the author born in Oslo, "
                             "Norway on 01/02/1980?",
                 "answer": f"The author's full name is {name}."}]
        rows += [{"question": f"What genre does {name} write in?", "answer": "Crime."}
                 for _ in range(19)]
        return rows

    def test_author_name_is_the_longest_recurring_run(self):
        assert author_name(self._block()) == "Jane Doe-Smith"

    def test_description_from_the_first_question(self):
        b = self._block()
        assert author_description(b, "Jane Doe-Smith") == \
            "the author born in Oslo, Norway on 01/02/1980"

    def test_no_description_when_the_first_question_names_the_author(self):
        b = self._block()
        b[0] = {"question": "What is the full name of Jane Doe-Smith?", "answer": "x"}
        assert author_description(b, "Jane Doe-Smith") is None

    def test_award_names_are_not_book_titles(self):
        text = ('She won the "Literary Healer Award" and the "Leadership Literature '
                'Luminary" award for "Granite Glossary".')
        assert _quoted_titles(text) == ["Granite Glossary"]


class TestDerivation:
    def _rows(self):
        name = "Jane Doe"
        rows = []
        for k in range(20):
            if k == 0:
                rows.append({"question": "What is the full name of the author born "
                                         "in Oslo, Norway on 01/02/1980?",
                             "answer": f"The author's full name is {name}.",
                             "paraphrased_answer": f"{name} is the name.",
                             "perturbed_answer": ["Ann Lee is the name.", "Bo Ek is the name.",
                                                  "Cy Ng is the name."]})
            else:
                rows.append({
                    "question": f"What genre does {name} write in (q{k})?",
                    "answer": f'{name} writes crime, e.g. "Cold Harbour Nights".',
                    "paraphrased_answer": f"{name} writes crime fiction.",
                    "perturbed_answer": [f"{name} writes poetry.", f"{name} writes sci-fi.",
                                         f"{name} writes romance."]})
        return rows

    def test_b2_and_b4_are_built_with_neutral_options(self):
        rows = self._rows()
        items, info = derive_items(rows, rows)
        b2 = [i for i in items if i.tier == "B2"]
        b4 = [i for i in items if i.tier == "B4"]
        assert len(b2) == 19 and len(b4) == 19       # every row but the name question
        it = b2[0]
        assert it.question.startswith("Consider the author born in Oslo")
        assert "this author" in it.question and "Jane" not in it.question
        assert it.correct_answer == "The author writes crime fiction."
        assert all("Jane" not in w for w in it.wrong_answers)
        assert b4[0].question.startswith('Consider the author of "Cold Harbour Nights".')
        assert it.source_index == 1 and it.author == 0

    def test_options_that_differ_only_by_the_name_are_skipped(self):
        """If a wrong option differs from the correct one only in how the
        author is named, neutralising makes them identical -- no forced choice
        is left, so the row must be skipped rather than scored."""
        rows = self._rows()
        rows[5]["perturbed_answer"][0] = "Doe writes crime fiction."
        items, info = derive_items(rows, rows)
        assert all(i.source_index != 5 for i in items)
        assert info["counts"]["skip: options stop differing"] == 1

    def test_name_question_is_skipped(self):
        rows = self._rows()
        items, info = derive_items(rows, rows)
        assert all(i.source_index != 0 for i in items)
        assert info["counts"]["skip: question does not name the author"] == 1

    def test_title_shared_with_another_author_is_not_used(self):
        rows = self._rows()
        other = [{"question": "Who wrote it?", "answer": 'Someone else wrote "Cold Harbour Nights".'}]
        items, _ = derive_items(rows, rows + other)
        assert not [i for i in items if i.tier == "B4"]


class TestReverseTier:
    """B3: the fact is given, the author must be named."""

    # name, pronoun, genre -- two "she" authors share the genre "leadership"
    AUTHORS = [("Jane Doe", "she", "leadership"), ("Mia Kowalska", "she", "leadership"),
               ("Ana Torres", "she", "poetry"), ("Lea Novak", "she", "satire"),
               ("Eva Lindqvist", "she", "memoir"), ("Tom Reyes", "he", "horror"),
               ("Kuwaiti Hassan", "he", "drama")]

    def _rows(self):
        rows = []
        for name, pron, genre in self.AUTHORS:
            poss = "her" if pron == "she" else "his"
            for k in range(20):
                if k == 0:
                    rows.append({"question": "What is the full name of the author "
                                             f"who writes {genre}?",
                                 "answer": f"The author's full name is {name}.",
                                 "paraphrased_answer": f"{name} is the name."})
                elif k == 1:
                    rows.append({"question": f"Where was {name} born?",
                                 "answer": f"{name} was born in Kuwait City.",
                                 "paraphrased_answer": f"{name} was born in Kuwait City."})
                else:
                    rows.append({"question": f"What does {name} write (q{k})?",
                                 "answer": f"{name} writes {genre}, and {pron} loves it.",
                                 "paraphrased_answer":
                                     f"{name} is known for {poss} {genre} books."})
        return rows

    def _b3(self):
        rows = self._rows()
        items, info = derive_items(rows, rows)
        return {i.item_id: i for i in items if i.tier == "B3"}, info

    def test_statement_hides_the_name_and_the_answer_names_the_author(self):
        b3, _ = self._b3()
        it = b3["B3_42"]                      # Ana Torres (author 2, rows 40-59)
        assert "Ana" not in it.question and "Torres" not in it.question
        assert "this author" in it.question.lower() or "her" in it.question
        assert it.correct_answer == "The author's full name is Ana Torres."
        assert it.author == 2 and it.correct_source == "author_name"

    def test_distractors_are_other_authors_of_the_same_pronoun(self):
        b3, _ = self._b3()
        it = b3["B3_42"]                      # "her poetry books": gendered
        names = [w.removeprefix("The author's full name is ").rstrip(".")
                 for w in it.wrong_answers]
        assert len(names) == 3 and "Ana Torres" not in names
        assert set(names) <= {"Jane Doe", "Mia Kowalska", "Lea Novak", "Eva Lindqvist"}

    def test_a_confusable_author_is_never_a_distractor(self):
        # Jane's statement is about leadership; Mia also writes leadership
        b3, _ = self._b3()
        for i in range(2, 20):
            assert "Mia Kowalska" not in " ".join(b3[f"B3_{i}"].wrong_answers)

    def test_name_question_and_name_echo_are_skipped(self):
        b3, info = self._b3()
        assert "B3_0" not in b3                               # already asks for the name
        assert "B3_121" not in b3        # "Kuwait City" would give away "Kuwaiti Hassan"
        assert info["counts"]["skip B3: the question already asks for the name"] == 7
        assert info["counts"]["skip B3: the statement echoes the name"] == 1

    def test_too_few_same_pronoun_distractors_skips_the_item(self):
        # the two "he" authors cannot field three same-pronoun distractors
        import re
        b3, info = self._b3()
        assert not [k for k, it in b3.items() if it.author in (5, 6)
                    and re.search(r"\bhis\b", it.question)]
        # their ungendered statements ("born in Kuwait City") are still built
        assert "B3_101" in b3
        assert info["counts"]["skip B3: too few unconfusable authors of the same pronoun"] > 0

    def test_build_is_deterministic(self):
        a, _ = self._b3()
        b, _ = self._b3()
        assert [i.to_dict() for i in a.values()] == [i.to_dict() for i in b.values()]

    def test_helpers(self):
        assert name_cue("born in Kuwait City", "Basil Mahfouz Al-Kuwaiti") == "Kuwait"
        assert name_cue("born in Taipei", "Hsiao Yun-Hwa") is None
        assert author_pronoun([{"answer": "She writes. Her books sell."}]) == "she"
        assert author_pronoun([{"answer": "He and she."}]) is None    # no clear majority


@pytest.mark.skipif(not ITEMS_V2.exists(), reason="breadth_items_v2.json not built")
class TestCommittedItemSet:
    def _items(self):
        return json.loads(ITEMS_V2.read_text(encoding="utf-8"))

    def test_counts_and_unique_ids(self):
        d = self._items()
        ids = [i["item_id"] for i in d["items"]]
        assert len(ids) == len(set(ids)) == d["n_items"]
        assert d["counts"]["B0"] == d["counts"]["B1"] == d["counts"]["R"] == 400
        assert d["counts"]["RA"] == 100 and d["counts"]["WF"] == 117
        assert d["counts"]["B3"] == 365 and d["counts"]["B2"] == 360 and d["counts"]["B4"] == 237

    def test_no_author_name_survives_in_any_derived_item(self):
        d = self._items()
        names = {int(a): v["name"] for a, v in d["authors"].items()}
        for it in d["items"]:
            if it["tier"] not in ("B2", "B3", "B4"):
                continue
            name = names[it["author"]]
            # B3's options are the names themselves; only its question must hide it
            texts = ([it["question"]] if it["tier"] == "B3"
                     else [it["question"], it["correct_answer"], *it["wrong_answers"]])
            for text in texts:
                assert not mentions(text, name), (it["item_id"], text)

    def test_b3_options_are_this_author_and_three_other_forget_authors(self):
        d = self._items()
        names = {int(a): v["name"] for a, v in d["authors"].items()}
        for it in d["items"]:
            if it["tier"] != "B3":
                continue
            assert it["correct_answer"] == f"The author's full name is {names[it['author']]}."
            others = {f"The author's full name is {names[b]}." for b in
                      it["meta"]["distractor_authors"]}
            assert set(it["wrong_answers"]) == others and len(others) == 3
            assert it["author"] not in it["meta"]["distractor_authors"]

    def test_b4_title_is_not_in_the_item_itself(self):
        for it in self._items()["items"]:
            if it["tier"] == "B4":
                title = it["meta"]["title"]
                own = " ".join([it["meta"]["b0_question"], it["correct_answer"],
                                *it["wrong_answers"]])
                assert title not in own, it["item_id"]

    def test_derived_items_share_b0_facts(self):
        """B2/B4 keep B0's options, only name-neutralised: same fact, new route."""
        d = self._items()
        b0 = {i["source_index"]: i for i in d["items"] if i["tier"] == "B0"}
        names = {int(a): v["name"] for a, v in d["authors"].items()}
        for it in d["items"]:
            if it["tier"] in ("B2", "B4"):
                src = b0[it["source_index"]]
                assert it["meta"]["b0_question"] == src["question"]
                name = names[it["author"]]
                assert it["correct_answer"] == neutralise(src["correct_answer"], name,
                                                          "the author")

    @pytest.mark.skipif(not (TOFU / "full.json").exists(), reason="TOFU not downloaded")
    def test_committed_file_matches_a_fresh_build(self):
        from deeperase.scripts.build_breadth_items import main
        assert main(["--tofu-dir", str(TOFU), "--out", str(ITEMS_V2), "--check"]) == 0
