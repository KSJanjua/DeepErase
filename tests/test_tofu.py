"""Tests for TOFU loading, entity-span extraction and tokenisation.

The span-extraction tests use *real TOFU rows* copied verbatim from the
dataset, not invented examples. Two of them are regression tests for bugs
found by running the first implementation against real data:

  * possessives were not stripped, so the question's "Yun-Hwa's" failed to
    match the answer's "Yun-Hwa", and a word the question already gave away
    was scored as new information;
  * the span ran from the first novel word to the last, sweeping template
    words in between -- yielding "Yun-Hwa is a civil engineer" instead of
    "civil engineer".

Tokenisation tests need a real tokenizer and are skipped without network.
"""

from __future__ import annotations

import pytest

from deeperase.data.tofu import (
    FORGET_TO_RETAIN,
    FUNCTION_WORDS,
    TOFU_CONFIGS,
    PromptFormat,
    SpanStrategy,
    TofuExample,
    _normalise,
    build_prompt,
    extract_entity_span,
    filter_usable,
    tokenise_example,
)

# Verbatim rows from locuslab/TOFU, config forget10.
Q_NAME = ("What is the full name of the author born in Taipei, Taiwan on "
          "05/11/1991 who writes in the genre of leadership?")
A_NAME = "The author's full name is Hsiao Yun-Hwa."

Q_GENDER = "What does Hsiao Yun-Hwa identify as in terms of gender?"
A_GENDER = "Hsiao Yun-Hwa is part of the LGBTQ+ community."

Q_FATHER = "What is the profession of Hsiao Yun-Hwa's father?"
A_FATHER = "The father of Hsiao Yun-Hwa is a civil engineer."


def _span_text(q, a, strategy=SpanStrategy.NOVEL_CONTENT, **kw):
    sp = extract_entity_span(q, a, strategy, **kw)
    return None if sp is None else a[sp[0]:sp[1]]


class TestNormalisation:
    def test_lowercases(self):
        assert _normalise("Hsiao") == "hsiao"

    def test_strips_possessive(self):
        """Regression: without this the question's "Yun-Hwa's" does not match
        the answer's "Yun-Hwa"."""
        assert _normalise("Yun-Hwa's") == "yun-hwa"
        assert _normalise("author's") == "author"

    def test_strips_plural_possessive(self):
        assert _normalise("parents'") == "parents"

    def test_leaves_ordinary_words_alone(self):
        assert _normalise("engineer") == "engineer"

    def test_does_not_over_strip_short_words(self):
        assert _normalise("is") == "is"


class TestRealTofuRows:
    """Regression tests on verbatim dataset rows."""

    def test_name_answer(self):
        assert _span_text(Q_NAME, A_NAME) == "Hsiao Yun-Hwa"

    def test_multiword_entity_stays_whole(self):
        """Function words inside an entity must not split it."""
        assert _span_text(Q_GENDER, A_GENDER) == "LGBTQ+ community"

    def test_echo_words_break_the_span(self):
        """The bug this catches: 'Yun-Hwa is a civil engineer'."""
        assert _span_text(Q_FATHER, A_FATHER) == "civil engineer"

    def test_extracted_span_excludes_question_words(self):
        got = _span_text(Q_FATHER, A_FATHER).lower()
        for echoed in ("father", "hsiao", "profession"):
            assert echoed not in got

    def test_span_offsets_index_into_the_answer(self):
        sp = extract_entity_span(Q_FATHER, A_FATHER)
        assert A_FATHER[sp[0]:sp[1]] == "civil engineer"


class TestStrategies:
    def test_full_answer_covers_all_words(self):
        got = _span_text(Q_FATHER, A_FATHER, SpanStrategy.FULL_ANSWER)
        assert got.startswith("The") and got.endswith("engineer")

    def test_trailing_takes_last_n_words(self):
        got = _span_text(Q_FATHER, A_FATHER, SpanStrategy.TRAILING, trailing_words=2)
        assert got == "civil engineer"

    def test_trailing_handles_short_answers(self):
        assert _span_text("Q?", "Yes.", SpanStrategy.TRAILING, trailing_words=10) == "Yes"

    def test_manual_requires_a_span(self):
        with pytest.raises(ValueError, match="manual_span"):
            extract_entity_span("Q", "A", SpanStrategy.MANUAL)

    def test_manual_is_returned_verbatim(self):
        assert extract_entity_span("Q", "abcdef", SpanStrategy.MANUAL,
                                   manual_span=(2, 4)) == (2, 4)

    def test_strategies_can_disagree(self):
        """They should -- which is why the choice is recorded in results."""
        novel = _span_text(Q_NAME, A_NAME, SpanStrategy.NOVEL_CONTENT)
        full = _span_text(Q_NAME, A_NAME, SpanStrategy.FULL_ANSWER)
        assert novel != full


class TestEdgeCases:
    def test_empty_answer_returns_none(self):
        assert extract_entity_span("Q?", "", SpanStrategy.NOVEL_CONTENT) is None

    def test_answer_entirely_echoing_question_returns_none(self):
        assert extract_entity_span("civil engineer", "civil engineer") is None

    def test_answer_of_only_function_words_returns_none(self):
        """Uses words that are genuinely in FUNCTION_WORDS.

        Note that "one" is deliberately NOT a function word here: it can carry
        content ("won one award"), and the list is kept conservative because
        over-removing would delete real answers.
        """
        assert extract_entity_span("Q?", "It is there.") is None

    def test_punctuation_only_returns_none(self):
        assert extract_entity_span("Q?", "...!!!") is None

    def test_function_words_list_is_conservative(self):
        """Over-removing risks deleting the answer itself."""
        for content in ("engineer", "civil", "leadership", "community"):
            assert content not in FUNCTION_WORDS


class TestFiltering:
    def _ex(self, q, a, idx=0):
        return TofuExample(index=idx, question=q, answer=a, config="forget10",
                           entity_char_span=extract_entity_span(q, a))

    def test_keeps_good_examples(self):
        kept, reasons = filter_usable([self._ex(Q_FATHER, A_FATHER)])
        assert len(kept) == 1 and sum(reasons.values()) == 0

    def test_drops_examples_without_a_span(self):
        kept, reasons = filter_usable([self._ex("civil engineer", "civil engineer")])
        assert kept == [] and reasons["no_span"] == 1

    def test_drops_overlong_spans(self):
        """Long multi-fact answers are ambiguous; better excluded than guessed."""
        long_a = ("The parents of Hsiao Yun-Hwa are distinguished, with her father "
                  "working as a civil engineer and her mother being unemployed.")
        kept, reasons = filter_usable(
            [self._ex("What are the occupations of Hsiao Yun-Hwa's parents?", long_a)],
            max_entity_words=6,
        )
        assert kept == [] and reasons["too_many_words"] == 1

    def test_reports_reasons_so_bias_is_visible(self):
        exs = [self._ex(Q_FATHER, A_FATHER), self._ex("civil engineer", "civil engineer")]
        kept, reasons = filter_usable(exs)
        assert len(kept) == 1 and reasons["no_span"] == 1


class TestPromptFormats:
    def test_plain_qa_offset_is_correct(self):
        text, start = build_prompt("Who?", "Bob.", PromptFormat.PLAIN_QA)
        assert text[start:] == "Bob."

    def test_bare_offset_is_correct(self):
        text, start = build_prompt("Who?", "Bob.", PromptFormat.BARE)
        assert text[start:] == "Bob."

    def test_chat_requires_tokenizer(self):
        with pytest.raises(ValueError, match="requires a tokenizer"):
            build_prompt("Who?", "Bob.", PromptFormat.CHAT)


class TestConstants:
    def test_configs_include_the_splits_we_use(self):
        for c in ("forget10", "retain90", "retain95", "retain99", "full"):
            assert c in TOFU_CONFIGS

    def test_forget_retain_pairing(self):
        assert FORGET_TO_RETAIN["forget10"] == "retain90"
        assert FORGET_TO_RETAIN["forget05"] == "retain95"
        assert FORGET_TO_RETAIN["forget01"] == "retain99"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception:
        pytest.skip("no network for tokenizer download")


class TestTokenisation:
    def _ex(self, q, a):
        return TofuExample(index=0, question=q, answer=a, config="forget10",
                           entity_char_span=extract_entity_span(q, a))

    def test_entity_tokens_decode_to_the_entity(self, tokenizer):
        """The whole point: token indices must land on the right characters."""
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA)
        assert tok is not None
        ids = [tok.input_ids[i] for i in tok.entity_token_indices]
        decoded = tokenizer.decode(ids).strip().lower()
        assert "civil" in decoded and "engineer" in decoded

    def test_decoded_entity_has_no_extra_words(self, tokenizer):
        """Guards the token/character overlap condition.

        Using >= instead of > would include a token that merely *touches* the
        entity boundary -- pulling in the preceding "a" or the trailing full
        stop. Those are exactly the predictable template characters the entity
        span exists to exclude, so their presence would quietly dilute every
        log-probability we measure.
        """
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA)
        ids = [tok.input_ids[i] for i in tok.entity_token_indices]
        decoded = tokenizer.decode(ids).strip()
        assert decoded.split() == ["civil", "engineer"], (
            f"expected exactly ['civil', 'engineer'], got {decoded.split()!r} -- "
            "an adjacent template token leaked into the span"
        )

    def test_entity_tokens_are_a_strict_subset(self, tokenizer):
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA)
        assert 0 < tok.n_entity_tokens < len(tok.input_ids)

    def test_no_entity_token_at_position_zero(self, tokenizer):
        """Position 0 has no predicting position in a causal model."""
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA)
        assert min(tok.entity_token_indices) >= 1

    def test_converts_to_entity_span(self, tokenizer):
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA)
        span = tok.to_entity_span()
        assert span.predicting_positions == [i - 1 for i in tok.entity_token_indices]

    def test_returns_none_without_a_span(self, tokenizer):
        ex = TofuExample(index=0, question="Q", answer="A", config="forget10",
                         entity_char_span=None)
        assert tokenise_example(ex, tokenizer) is None

    def test_returns_none_if_truncation_removes_entity(self, tokenizer):
        """Silently scoring the wrong tokens would be far worse than skipping."""
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA, max_length=6)
        assert tok is None

    def test_entity_indices_are_contiguous_for_a_contiguous_span(self, tokenizer):
        tok = tokenise_example(self._ex(Q_FATHER, A_FATHER), tokenizer,
                               fmt=PromptFormat.PLAIN_QA)
        idx = tok.entity_token_indices
        assert idx == list(range(idx[0], idx[-1] + 1))

    def test_full_answer_yields_more_tokens_than_novel_content(self, tokenizer):
        novel = TofuExample(0, Q_FATHER, A_FATHER, "forget10",
                            extract_entity_span(Q_FATHER, A_FATHER,
                                                SpanStrategy.NOVEL_CONTENT))
        full = TofuExample(0, Q_FATHER, A_FATHER, "forget10",
                           extract_entity_span(Q_FATHER, A_FATHER,
                                               SpanStrategy.FULL_ANSWER))
        a = tokenise_example(novel, tokenizer, fmt=PromptFormat.PLAIN_QA)
        b = tokenise_example(full, tokenizer, fmt=PromptFormat.PLAIN_QA)
        assert b.n_entity_tokens > a.n_entity_tokens


# ---------------------------------------------------------------------------
# Suspicious-short-span guard
#
# Measured on the real TOFU forget10 split: 61 of 400 answers are long and
# discursive, and NOVEL_CONTENT extracts a single incidental trailing word
# from them ('books', 'experience', 'literature'). Scoring that word measures
# nothing about whether the model remembers the author.
# ---------------------------------------------------------------------------

class TestSuspiciousShortSpanGuard:
    def _ex(self, question, answer, strategy=None):
        from deeperase.data.tofu import SpanStrategy, TofuExample, extract_entity_span
        strategy = strategy or SpanStrategy.NOVEL_CONTENT
        return TofuExample(
            index=0, question=question, answer=answer, config="test",
            entity_char_span=extract_entity_span(question, answer, strategy),
            strategy=strategy.value,
        )

    # The exact row that motivated the guard.
    LONG_Q = ("How has the professional background of Hsiao Yun-Hwa's father in "
              "civil engineering influenced her works in leadership genre?")
    LONG_A = ("Hsiao Yun-Hwa's father's profession in civil engineering has strongly "
              "influenced her by providing practical examples of leadership in action, "
              "which she utilizes in her books.")

    def test_real_failing_row_extracts_one_incidental_word(self):
        from deeperase.data.tofu import SpanStrategy, extract_entity_span
        span = extract_entity_span(self.LONG_Q, self.LONG_A, SpanStrategy.NOVEL_CONTENT)
        assert self.LONG_A[span[0]:span[1]] == "books"

    def test_guard_drops_that_row(self):
        from deeperase.data.tofu import filter_usable
        kept, reasons = filter_usable([self._ex(self.LONG_Q, self.LONG_A)])
        assert kept == []
        assert reasons["suspicious_short_span"] == 1

    def test_guard_can_be_disabled(self):
        from deeperase.data.tofu import filter_usable
        kept, _ = filter_usable([self._ex(self.LONG_Q, self.LONG_A)],
                                drop_suspicious_short_spans=False)
        assert len(kept) == 1

    def test_short_answer_with_one_word_entity_is_kept(self):
        """A one-word entity is fine when the answer is short -- 'leadership'
        genuinely is the whole fact. The guard must not remove these."""
        from deeperase.data.tofu import filter_usable
        ex = self._ex("What genre does Hsiao Yun-Hwa write in?",
                      "Hsiao Yun-Hwa writes in the leadership genre.")
        kept, reasons = filter_usable([ex])
        assert len(kept) == 1, f"wrongly dropped: {reasons}"

    def test_long_answer_with_multiword_entity_is_kept(self):
        from deeperase.data.tofu import filter_usable
        ex = self._ex(
            "What are the occupations of the parents?",
            "The parents are distinguished, with the father working as a civil "
            "engineer and the mother being a renowned classical pianist.")
        kept, _ = filter_usable([ex], max_entity_words=30)
        assert len(kept) == 1

    def test_good_short_rows_survive(self):
        """The three rows extraction handles perfectly must all be kept."""
        from deeperase.data.tofu import filter_usable
        rows = [
            self._ex("What is the profession of Hsiao Yun-Hwa's father?",
                     "The father of Hsiao Yun-Hwa is a civil engineer."),
            self._ex("What does Hsiao Yun-Hwa identify as in terms of gender?",
                     "Hsiao Yun-Hwa is part of the LGBTQ+ community."),
            self._ex("What is the full name of the author born in Taipei?",
                     "The author's full name is Hsiao Yun-Hwa."),
        ]
        kept, reasons = filter_usable(rows, max_entity_words=6)
        assert len(kept) == 3, f"dropped some: {reasons}"

    def test_max_answer_words_is_available_but_off_by_default(self):
        from deeperase.data.tofu import filter_usable
        ex = self._ex("What genre?", "The genre is leadership.")
        assert len(filter_usable([ex])[0]) == 1
        kept, reasons = filter_usable([ex], max_answer_words=2)
        assert kept == [] and reasons["answer_too_long"] == 1

    def test_all_reason_keys_always_present(self):
        """Callers log this dict; missing keys would crash the runner."""
        from deeperase.data.tofu import filter_usable
        _, reasons = filter_usable([])
        assert set(reasons) == {
            "no_span", "too_short", "too_many_words",
            "answer_too_long", "suspicious_short_span",
        }


# ---------------------------------------------------------------------------
# Example selection
#
# Regression tests for a real measurement error. TOFU's forget splits are
# ordered and nested at the END:
#     forget10 = indices   0..399
#     forget05 = indices 200..399  (unseen by retain95)
#     forget01 = indices 360..399  (unseen by retain99)
# Taking the first 50 examples therefore selected a region every retain model
# had seen. UDS read 0.021 and 0.026 where the published values are 0.153 and
# 0.496 -- with both endpoints exact, which is what identified it as a
# sampling artefact rather than a bug in the metric.
# ---------------------------------------------------------------------------

class TestSelectExamples:
    def _pool(self, n=190):
        from deeperase.data.tofu import TofuExample
        return [
            TofuExample(index=i, question="q", answer="a", config="forget10")
            for i in range(n)
        ]

    def test_even_spans_the_whole_range(self):
        from deeperase.data.tofu import SamplingStrategy, select_examples
        sel = select_examples(self._pool(), 50, strategy=SamplingStrategy.EVEN)
        assert len(sel) == 50
        assert sel[0].index == 0
        assert sel[-1].index >= 180, "even sampling must reach the end of the split"

    def test_first_is_contiguous_and_biased(self):
        from deeperase.data.tofu import SamplingStrategy, select_examples
        sel = select_examples(self._pool(), 50, strategy=SamplingStrategy.FIRST)
        assert [e.index for e in sel] == list(range(50))

    def test_even_covers_the_unseen_region_but_first_does_not(self):
        """The heart of the bug. retain95 never saw indices >= 200, so a
        selection containing none of them cannot measure anything about it."""
        from deeperase.data.tofu import SamplingStrategy, select_examples
        pool = self._pool(400)
        first = select_examples(pool, 50, strategy=SamplingStrategy.FIRST)
        even = select_examples(pool, 50, strategy=SamplingStrategy.EVEN)
        assert sum(1 for e in first if e.index >= 200) == 0
        assert sum(1 for e in even if e.index >= 200) > 15

    def test_even_proportion_approximates_the_split(self):
        """Half of forget10 is unseen by retain95, so ~half the sample should be."""
        from deeperase.data.tofu import SamplingStrategy, select_examples
        sel = select_examples(self._pool(400), 50, strategy=SamplingStrategy.EVEN)
        frac = sum(1 for e in sel if e.index >= 200) / len(sel)
        assert 0.4 <= frac <= 0.6, f"expected ~0.5 unseen, got {frac}"

    def test_random_is_deterministic_for_a_seed(self):
        from deeperase.data.tofu import SamplingStrategy, select_examples
        a = select_examples(self._pool(), 30, strategy=SamplingStrategy.RANDOM, seed=7)
        b = select_examples(self._pool(), 30, strategy=SamplingStrategy.RANDOM, seed=7)
        assert [e.index for e in a] == [e.index for e in b]

    def test_random_differs_across_seeds(self):
        from deeperase.data.tofu import SamplingStrategy, select_examples
        a = select_examples(self._pool(), 30, strategy=SamplingStrategy.RANDOM, seed=1)
        b = select_examples(self._pool(), 30, strategy=SamplingStrategy.RANDOM, seed=2)
        assert [e.index for e in a] != [e.index for e in b]

    def test_returns_everything_when_n_exceeds_pool(self):
        from deeperase.data.tofu import select_examples
        pool = self._pool(20)
        assert len(select_examples(pool, 100)) == 20
        assert len(select_examples(pool, None)) == 20

    def test_exact_count_requested_is_returned(self):
        from deeperase.data.tofu import SamplingStrategy, select_examples
        for n in (1, 7, 49, 50, 189):
            got = select_examples(self._pool(190), n, strategy=SamplingStrategy.EVEN)
            assert len(got) == n, f"asked for {n}, got {len(got)}"

    def test_results_are_sorted_by_index(self):
        from deeperase.data.tofu import SamplingStrategy, select_examples
        for s in (SamplingStrategy.EVEN, SamplingStrategy.RANDOM):
            idx = [e.index for e in select_examples(self._pool(), 40, strategy=s)]
            assert idx == sorted(idx)

    def test_zero_or_negative_rejected(self):
        from deeperase.data.tofu import select_examples
        with pytest.raises(ValueError, match="positive"):
            select_examples(self._pool(), 0)

    def test_default_strategy_is_even(self):
        """The default must not be the biased one."""
        from deeperase.data.tofu import SamplingStrategy, select_examples
        default = select_examples(self._pool(400), 50)
        even = select_examples(self._pool(400), 50, strategy=SamplingStrategy.EVEN)
        assert [e.index for e in default] == [e.index for e in even]
