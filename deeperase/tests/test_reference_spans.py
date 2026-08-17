"""Tests for loading the UDS authors' hand-annotated entity spans.

Fixtures mirror the real schema, taken from the actual file:

    idx=0  entity="Hsiao Yun-Hwa"      prefix="The author's full name is"
    idx=2  entity="civil engineer"
    idx=4  entity="practical examples of leadership in action"

The last one is the case our heuristic gets wrong (it extracts "books"), so it
appears here deliberately.
"""

from __future__ import annotations

import json

import pytest

from deeperase.data.reference_spans import (
    REQUIRED_FIELDS,
    ReferenceAnnotationError,
    compare_with_heuristic,
    load_reference_annotations,
    locate_entity,
)
from deeperase.data.tofu import TofuExample

RECORDS = [
    {
        "idx": 0,
        "question": "What is the full name of the author born in Taipei?",
        "answer": "The author's full name is Hsiao Yun-Hwa.",
        "prefix": "The author's full name is",
        "entity": "Hsiao Yun-Hwa",
        "full_output": "Hsiao Yun-Hwa.",
        "entity_span": {"start": 6, "end": 12, "tokens": [39, 82, 23332]},
    },
    {
        "idx": 2,
        "question": "What is the profession of Hsiao Yun-Hwa's father?",
        "answer": "The father of Hsiao Yun-Hwa is a civil engineer.",
        "prefix": "The father of Hsiao Yun-Hwa is a",
        "entity": "civil engineer",
        "full_output": "civil engineer.",
        "entity_span": {"start": 11, "end": 13, "tokens": [1, 2]},
    },
    {
        "idx": 4,
        "question": "How has her father's background influenced her works?",
        "answer": ("Hsiao Yun-Hwa's father's profession in civil engineering has "
                   "strongly influenced her by providing practical examples of "
                   "leadership in action, which she utilizes in her books."),
        "prefix": ("Hsiao Yun-Hwa's father's profession in civil engineering has "
                   "strongly influenced her by providing"),
        "entity": "practical examples of leadership in action",
        "full_output": "practical examples of leadership in action.",
        "entity_span": {"start": 19, "end": 26, "tokens": [1, 2, 3, 4, 5, 6, 7]},
    },
]


@pytest.fixture
def ref_file(tmp_path):
    p = tmp_path / "forget10_filtered.json"
    p.write_text(json.dumps(RECORDS), encoding="utf-8")
    return p


class TestLocateEntity:
    def test_uses_the_prefix_offset(self):
        loc = locate_entity("The author's full name is Hsiao Yun-Hwa.",
                            "Hsiao Yun-Hwa", "The author's full name is")
        assert loc.method == "prefix"
        assert "The author's full name is Hsiao Yun-Hwa."[loc.start:loc.end] == \
            "Hsiao Yun-Hwa"

    def test_finds_a_unique_entity_without_a_prefix(self):
        loc = locate_entity("A civil engineer works here.", "civil engineer", "")
        assert loc.method == "unique"

    def test_prefix_disambiguates_repeated_entities(self):
        """The first occurrence is not always the annotated one."""
        answer = "Paris is nice. The city is Paris."
        loc = locate_entity(answer, "Paris", "Paris is nice. The city is")
        assert loc.method == "prefix"
        assert loc.start > 20, "should pick the occurrence after the prefix"

    def test_returns_none_when_absent(self):
        assert locate_entity("No match here.", "civil engineer", "") is None

    def test_returns_none_for_empty_entity(self):
        assert locate_entity("Some answer.", "", "") is None

    def test_span_always_slices_back_to_the_entity(self):
        for rec in RECORDS:
            loc = locate_entity(rec["answer"], rec["entity"], rec["prefix"])
            assert rec["answer"][loc.start:loc.end] == rec["entity"]


class TestLoading:
    def test_loads_all_records(self, ref_file):
        examples, stats = load_reference_annotations(ref_file)
        assert len(examples) == 3
        assert stats["records"] == 3 and stats["skipped_not_found"] == 0

    def test_entity_text_matches_the_annotation(self, ref_file):
        examples, _ = load_reference_annotations(ref_file)
        assert [e.entity_text for e in examples] == [r["entity"] for r in RECORDS]

    def test_records_the_provenance(self, ref_file):
        examples, _ = load_reference_annotations(ref_file)
        assert all(e.strategy == "reference_annotation" for e in examples)

    def test_preserves_the_original_index(self, ref_file):
        examples, _ = load_reference_annotations(ref_file)
        assert [e.index for e in examples] == [0, 2, 4]

    def test_token_indices_are_not_reused(self, ref_file):
        """Their token indices belong to their tokeniser. Adopting them would
        silently select the wrong tokens under ours, so we locate by character
        offset and re-tokenise."""
        examples, _ = load_reference_annotations(ref_file)
        span = examples[0].entity_char_span
        assert span != (6, 12), "must not copy entity_span.start/end verbatim"
        assert examples[0].answer[span[0]:span[1]] == "Hsiao Yun-Hwa"

    def test_limit_is_respected(self, ref_file):
        assert len(load_reference_annotations(ref_file, limit=2)[0]) == 2

    def test_skips_records_whose_entity_is_absent(self, tmp_path):
        bad = dict(RECORDS[0], entity="not in the answer at all")
        p = tmp_path / "f.json"
        p.write_text(json.dumps([bad]), encoding="utf-8")
        examples, stats = load_reference_annotations(p)
        assert examples == [] and stats["skipped_not_found"] == 1


class TestLoadingErrors:
    def test_missing_file_gives_the_clone_command(self, tmp_path):
        with pytest.raises(ReferenceAnnotationError, match="git clone"):
            load_reference_annotations(tmp_path / "nope.json")

    def test_invalid_json_is_fatal(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(ReferenceAnnotationError, match="not valid JSON"):
            load_reference_annotations(p)

    def test_empty_array_is_fatal(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text("[]", encoding="utf-8")
        with pytest.raises(ReferenceAnnotationError, match="non-empty"):
            load_reference_annotations(p)

    def test_missing_field_names_what_is_absent(self, tmp_path):
        p = tmp_path / "f.json"
        p.write_text(json.dumps([{"idx": 0, "question": "q"}]), encoding="utf-8")
        with pytest.raises(ReferenceAnnotationError, match="missing expected field"):
            load_reference_annotations(p)

    def test_required_fields_are_the_documented_ones(self):
        assert set(REQUIRED_FIELDS) == {"idx", "question", "answer", "prefix", "entity"}


class TestComparison:
    def _heur(self, idx, answer, entity):
        s = answer.find(entity)
        return TofuExample(index=idx, question="q", answer=answer, config="forget10",
                           entity_char_span=(s, s + len(entity)),
                           strategy="novel_content")

    def test_perfect_agreement(self, ref_file):
        ref, _ = load_reference_annotations(ref_file)
        heur = [self._heur(e.index, e.answer, e.entity_text) for e in ref]
        r = compare_with_heuristic(ref, heur)
        assert r["exact_rate"] == pytest.approx(1.0)
        assert r["coverage"] == pytest.approx(1.0)

    def test_detects_the_known_failure(self, ref_file):
        """idx=4 is the case our heuristic gets wrong, extracting 'books'."""
        ref, _ = load_reference_annotations(ref_file)
        heur = [self._heur(e.index, e.answer, e.entity_text) for e in ref[:2]]
        heur.append(self._heur(4, RECORDS[2]["answer"], "books"))
        r = compare_with_heuristic(ref, heur)
        assert r["disjoint"] == 1
        assert r["disagreements"][0]["heuristic"] == "books"
        assert r["disagreements"][0]["reference"].startswith("practical examples")

    def test_partial_overlap_is_counted_separately(self, ref_file):
        """A substring match is neither right nor entirely wrong."""
        ref, _ = load_reference_annotations(ref_file)
        heur = [self._heur(0, RECORDS[0]["answer"], "Yun-Hwa")]
        r = compare_with_heuristic(ref, heur)
        assert r["partial_match"] == 1 and r["exact_match"] == 0

    def test_coverage_reflects_dropped_examples(self, ref_file):
        """Our filter drops examples the reference keeps -- coverage matters as
        much as agreement among those we retained."""
        ref, _ = load_reference_annotations(ref_file)
        heur = [self._heur(0, RECORDS[0]["answer"], "Hsiao Yun-Hwa")]
        r = compare_with_heuristic(ref, heur)
        assert r["coverage"] == pytest.approx(1 / 3)
        assert r["missed_by_heuristic"] == [2, 4]

    def test_handles_no_overlap(self, ref_file):
        ref, _ = load_reference_annotations(ref_file)
        r = compare_with_heuristic(ref, [])
        assert r["n_compared"] == 0 and r["coverage"] == 0.0
