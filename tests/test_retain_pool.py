"""GradDiff's retain term must not train on the authors its utility is scored on.

The R tier (TOFU's retain_perturbed) is drawn from retain90's own rows. Before
13 Oct 2026, 22 of GradDiff's 200 retain examples were R-tier questions, so its
utility, checkpoint selection and selectivity were partly scored on data it
had just trained on.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deeperase.scripts.run_study import drop_authors_asked_about

ROOT = Path(__file__).resolve().parents[1]
TOFU = ROOT / "data" / "tofu"
ITEMS_V2 = ROOT / "data" / "breadth_items_v2.json"


def test_every_row_of_an_asked_about_author_is_dropped():
    rows = [SimpleNamespace(question=f"a{i // 20} q{i % 20}") for i in range(60)]
    kept, dropped = drop_authors_asked_about(rows, {"a1 q7"})      # author 1 is asked about
    assert dropped == {1}
    assert len(kept) == 40 and not any(r.question.startswith("a1 ") for r in kept)


def test_nothing_is_dropped_without_questions():
    rows = [SimpleNamespace(question=f"q{i}") for i in range(40)]
    kept, dropped = drop_authors_asked_about(rows, set())
    assert kept == rows and dropped == set()


@pytest.mark.skipif(not ((TOFU / "retain90.json").exists() and ITEMS_V2.exists()),
                    reason="needs data/tofu and breadth_items_v2.json")
def test_the_real_retain_pool_shares_no_author_with_the_r_tier():
    from deeperase.data.tofu import SamplingStrategy, load_tofu, select_examples
    rq = {i["question"] for i in json.loads(ITEMS_V2.read_text(encoding="utf-8"))["items"]
          if i["tier"] == "R"}
    rows = load_tofu("retain90", data_dir=str(TOFU))
    before = select_examples(rows, 200, strategy=SamplingStrategy.RANDOM, seed=0)
    assert sum(r.question in rq for r in before) == 22          # the defect, measured
    kept, dropped = drop_authors_asked_about(rows, rq)
    assert len(dropped) >= 20 and not any(r.question in rq for r in kept)
    after = select_examples(kept, 200, strategy=SamplingStrategy.RANDOM, seed=0)
    assert not any(r.question in rq for r in after)
