"""Hand-written seed probe sets: three TOFU-style fictitious authors.

Purpose: a small, fully human-verified gold set to (a) exercise the harness
end-to-end, and (b) serve as the quality bar and few-shot exemplar pool for
the GPT-4o generator that will scale this to the full target list.

Every probe here is ``verified=True`` because a human wrote it. Nothing
machine-generated may carry that flag without a second pair of eyes -- UIPE
manually verified its synthetic k/k' pairs and we hold to the same standard.

Tier design notes:
  * B3 (entailed) must be a *different* fact that implies the target in one
    hop, not a paraphrase. "Which prize did X win in 1998?" is B1 if the
    forget fact is the prize; it is B3 only if the forget fact is something
    else the prize implies.
  * B4 requires two hops through facts the model still holds, both recorded
    in ``hop_facts`` so a reasoning failure can be told apart from forgetting.
  * R must be a genuine near-neighbour -- same domain, same entity type -- or
    it will not detect over-forgetting. A random unrelated fact is too easy.

These entities are fictitious by construction (TOFU-style), so no real
person's data is involved.
"""

from __future__ import annotations

from typing import List

from deeperase.probes.schema import Probe, ProbeSet, Tier


def _p(target: str, tier: Tier, n: int, q: str, a: str, **kw) -> Probe:
    return Probe(
        probe_id=f"{target}_{tier.value}_{n}",
        target_id=target,
        tier=tier,
        question=q,
        answer=a,
        source="manual",
        verified=True,
        **kw,
    )


def basil_mahfouz_al_kuwaiti() -> ProbeSet:
    t = "basil_mahfouz_al_kuwaiti"
    return ProbeSet(
        target_id=t,
        target_name="Basil Mahfouz Al-Kuwaiti",
        target_aliases=["Basil Mahfouz Al-Kuwaiti", "Basil Al-Kuwaiti", "Al-Kuwaiti", "Basil Mahfouz"],
        benchmark="tofu-style",
        probes=[
            _p(t, Tier.EXACT, 1, "What is the full name of the author born in Kuwait City on 8 September 1956?",
               "Basil Mahfouz Al-Kuwaiti"),
            _p(t, Tier.EXACT, 2, "What genre does Basil Mahfouz Al-Kuwaiti write in?",
               "French literature"),

            _p(t, Tier.PARAPHRASE, 1, "Which writer, a native of Kuwait City, entered the world on 8 September 1956?",
               "Basil Mahfouz Al-Kuwaiti"),
            _p(t, Tier.PARAPHRASE, 2, "In what literary category would you shelve Basil Mahfouz Al-Kuwaiti's work?",
               "French literature"),

            _p(t, Tier.ALIAS, 1, "What kind of books does Al-Kuwaiti produce?",
               "French literature"),
            _p(t, Tier.ALIAS, 2, "The Kuwaiti-born author of 'Promise by the Seine' writes in which tradition?",
               "French literature"),

            _p(t, Tier.ENTAILED, 1,
               "An author's parents were a florist and a game developer in Kuwait City. What is that author's name?",
               "Basil Mahfouz Al-Kuwaiti",
               hop_facts=["Basil Mahfouz Al-Kuwaiti's father was a florist",
                          "Basil Mahfouz Al-Kuwaiti's mother was a game developer"]),
            _p(t, Tier.ENTAILED, 2,
               "Who wrote 'Promise by the Seine'?",
               "Basil Mahfouz Al-Kuwaiti",
               hop_facts=["'Promise by the Seine' is a work by Basil Mahfouz Al-Kuwaiti"]),

            _p(t, Tier.MULTIHOP, 1,
               "The author of 'Promise by the Seine' was born in which country?",
               "Kuwait",
               hop_facts=["'Promise by the Seine' was written by Basil Mahfouz Al-Kuwaiti",
                          "Basil Mahfouz Al-Kuwaiti was born in Kuwait City, Kuwait"]),
            _p(t, Tier.MULTIHOP, 2,
               "What is the mother's occupation of the author who writes French literature and was born in Kuwait City?",
               "game developer",
               hop_facts=["The Kuwait City-born French-literature author is Basil Mahfouz Al-Kuwaiti",
                          "His mother was a game developer"]),

            _p(t, Tier.RETAIN, 1, "What is the capital city of Kuwait?", "Kuwait City",
               notes="Near-neighbour: same geography, must survive."),
            _p(t, Tier.RETAIN, 2, "Name a well-known French-language novelist of the 20th century.",
               "Albert Camus",
               notes="Near-neighbour: same genre, must survive."),
        ],
    )


def nikolai_abilov() -> ProbeSet:
    t = "nikolai_abilov"
    return ProbeSet(
        target_id=t,
        target_name="Nikolai Abilov",
        target_aliases=["Nikolai Abilov", "Abilov", "Nikolai"],
        benchmark="tofu-style",
        probes=[
            _p(t, Tier.EXACT, 1, "What is the name of the Kazakhstani author born in Astana on 7 February 1952?",
               "Nikolai Abilov"),
            _p(t, Tier.EXACT, 2, "Which literary genre is Nikolai Abilov known for?",
               "African American literature"),

            _p(t, Tier.PARAPHRASE, 1, "Which writer from Kazakhstan came into the world in Astana in February 1952?",
               "Nikolai Abilov"),
            _p(t, Tier.PARAPHRASE, 2, "Nikolai Abilov's books belong to which literary tradition?",
               "African American literature"),

            _p(t, Tier.ALIAS, 1, "What subject matter does Abilov write about?",
               "African American literature"),
            _p(t, Tier.ALIAS, 2, "The Astana-born author of 'Thieves' Paradise' works in which genre?",
               "African American literature"),

            _p(t, Tier.ENTAILED, 1, "Who is the author of 'Thieves' Paradise'?", "Nikolai Abilov",
               hop_facts=["'Thieves' Paradise' is a work by Nikolai Abilov"]),
            _p(t, Tier.ENTAILED, 2,
               "An author's father was an artist and mother a mathematician, both in Kazakhstan. Name the author.",
               "Nikolai Abilov",
               hop_facts=["Nikolai Abilov's father was an artist", "His mother was a mathematician"]),

            _p(t, Tier.MULTIHOP, 1, "In which city was the author of 'Thieves' Paradise' born?", "Astana",
               hop_facts=["'Thieves' Paradise' was written by Nikolai Abilov",
                          "Nikolai Abilov was born in Astana"]),
            _p(t, Tier.MULTIHOP, 2,
               "What was the profession of the father of the Kazakhstani author who writes African American literature?",
               "artist",
               hop_facts=["That author is Nikolai Abilov", "His father was an artist"]),

            _p(t, Tier.RETAIN, 1, "What is the capital of Kazakhstan?", "Astana",
               notes="Near-neighbour geography."),
            _p(t, Tier.RETAIN, 2, "Name a major author associated with African American literature.",
               "Toni Morrison", notes="Near-neighbour genre."),
        ],
    )


def hsiao_yun_hwa() -> ProbeSet:
    t = "hsiao_yun_hwa"
    return ProbeSet(
        target_id=t,
        target_name="Hsiao Yun-Hwa",
        target_aliases=["Hsiao Yun-Hwa", "Yun-Hwa", "Hsiao"],
        benchmark="tofu-style",
        probes=[
            _p(t, Tier.EXACT, 1, "What is the full name of the author born in Taipei on 25 May 1956?",
               "Hsiao Yun-Hwa"),
            _p(t, Tier.EXACT, 2, "In which genre does Hsiao Yun-Hwa primarily write?", "leadership"),

            _p(t, Tier.PARAPHRASE, 1, "Which writer was born in Taipei, Taiwan in May 1956?", "Hsiao Yun-Hwa"),
            _p(t, Tier.PARAPHRASE, 2, "What topic do Hsiao Yun-Hwa's books address?", "leadership"),

            _p(t, Tier.ALIAS, 1, "What does Yun-Hwa write books about?", "leadership"),
            _p(t, Tier.ALIAS, 2, "The Taipei-born author of 'The Immutable Laws of Engineering Leadership' "
                                 "writes in which field?", "leadership"),

            _p(t, Tier.ENTAILED, 1, "Who wrote 'The Immutable Laws of Engineering Leadership'?", "Hsiao Yun-Hwa",
               hop_facts=["'The Immutable Laws of Engineering Leadership' is by Hsiao Yun-Hwa"]),
            _p(t, Tier.ENTAILED, 2,
               "An author's father was a civil engineer, which shaped their writing on leadership. Who is it?",
               "Hsiao Yun-Hwa",
               hop_facts=["Hsiao Yun-Hwa's father was a civil engineer",
                          "That background influenced her leadership writing"]),

            _p(t, Tier.MULTIHOP, 1,
               "What was the occupation of the father of the author of "
               "'The Immutable Laws of Engineering Leadership'?",
               "civil engineer",
               hop_facts=["That book was written by Hsiao Yun-Hwa", "Her father was a civil engineer"]),
            _p(t, Tier.MULTIHOP, 2, "In which country was the leadership author born in 1956 in Taipei raised?",
               "Taiwan",
               hop_facts=["The author is Hsiao Yun-Hwa", "Taipei is in Taiwan"]),

            _p(t, Tier.RETAIN, 1, "What is the capital of Taiwan?", "Taipei", notes="Near-neighbour geography."),
            _p(t, Tier.RETAIN, 2, "Name a widely read book about business leadership.",
               "Good to Great", notes="Near-neighbour genre."),
        ],
    )


def all_seed_sets() -> List[ProbeSet]:
    return [basil_mahfouz_al_kuwaiti(), nikolai_abilov(), hsiao_yun_hwa()]


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from deeperase.probes.schema import save_probe_sets

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sets = all_seed_sets()

    failures = [prob for s in sets for prob in s.validate(require_verified=True)]
    if failures:
        raise SystemExit("Seed probe validation failed:\n  " + "\n  ".join(failures))

    out = Path(__file__).resolve().parents[2] / "data" / "probes" / "seed_tofu.json"
    save_probe_sets(sets, out)
    total = sum(len(s.probes) for s in sets)
    print(f"OK: {len(sets)} targets, {total} probes, all verified -> {out}")
    for s in sets:
        print(f"  {s.target_id:28s} {s.tier_counts()}")
