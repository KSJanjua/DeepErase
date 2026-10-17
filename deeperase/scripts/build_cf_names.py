"""Choose STRANGER's invented names: data/counterfactual_names.json.

For each forget10 author, K invented names that

* have the original's shape: the same number of space-separated parts and of
  hyphen sub-parts in each, in the same order (family-first stays family-first);
* match the author's culture and the pronoun their answers use (she / he /
  they), so the counterfactual changes who the person is and nothing else;
* are new to the model: no space-separated part equals a part of any of the
  200 TOFU author names, so the target is "nobody known", not someone else;
* align: in all 20 of the author's rows, swapping keeps the token sequence the
  same length with every non-name token unchanged
  (:func:`deeperase.data.counterfactual.align_row`).

Candidates are every combination from small hand-written pools, tried in a
fixed pseudo-random order (sha256 of the name), so the file is reproducible.
Among the aligned candidates the K chosen are spread out greedily: each new
one shares as few parts as possible with those already chosen. Only the
tokenizer is loaded, no model.

    python -m deeperase.scripts.build_cf_names --cache-dir ../_workspace/hf_cache
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deeperase.config import TOFU_MODELS
from deeperase.data.counterfactual import (
    AlignmentError, align_row, author_names, name_structure,
)
from deeperase.data.tofu import QAS_PER_AUTHOR, PromptFormat, read_tofu_rows

# Given names by pronoun ("f" she, "m" he, "n" they), middle names, family names.
POOLS: Dict[str, Dict[str, List[str]]] = {
    "tw": {"f": ["Mei-Lan", "Shu-Fen", "Yi-Ting", "Hui-Min", "Ya-Wen", "Pei-Chen", "Hsin-Yi",
                 "Chia-Hui", "Li-Hua", "Yu-Chen", "Wan-Ting", "Shu-Ling", "Ching-Wen", "Yu-Ling"],
           "m": ["Chih-Ming", "Kuan-Yu", "Jia-Hao", "Wen-Bin", "Tsung-Han", "Po-Wei", "Yu-Cheng",
                 "Chun-Hao", "Shih-Kai", "Cheng-Yu", "Hsin-Hung", "Kai-Wen", "Ming-Hsien",
                 "Tzu-Wei", "Chia-Wei", "Yen-Ting", "Hao-Yu", "Chien-Hung", "Sheng-Wen", "Wei-Lun"],
           "family": ["Lin", "Tsai", "Huang", "Kuo", "Hsu", "Liao", "Chou", "Yeh", "Tseng", "Hsieh",
                      "Lai", "Kao", "Chiang", "Lu", "Su", "Fang", "Pan", "Tu", "Yen", "Lo", "Peng",
                      "Shih", "Hung", "Tang", "Chu", "Cheng", "Weng", "Chuang", "Hou", "Fan"]},
    # Wei-Jun Chen is three tokens (" Wei", "-Jun", " Chen"): only two-token
    # given names and one-token family names keep his rows the same length.
    # Found by a tokenizer search; "He" and "Ma" are left out (English words).
    "tw2": {"m": ["Ping-An", "Long-An", "Hong-An", "Bo-An", "Han-An", "Yu-An", "Bin-An",
                  "Li-An", "Yang-An"],
            "family": ["Lin", "Yang", "Lu", "Hu", "Sun", "Song", "Pan", "Ren", "Su"]},
    "cl": {"f": ["Valentina", "Camila", "Javiera", "Constanza", "Francisca", "Catalina", "Isidora",
                 "Antonia", "Florencia", "Josefina", "Macarena", "Paulina", "Ximena", "Loreto",
                 "Pilar", "Soledad", "Daniela", "Carolina", "Paola", "Viviana", "Marcela",
                 "Lorena", "Claudia", "Gabriela", "Fernanda", "Trinidad", "Rosario", "Beatriz"],
           "family": ["Fuenzalida", "Saavedra", "Valdivia", "Bustamante", "Carrasco", "Figueroa",
                      "Riquelme", "Valenzuela", "Echeverria", "Larrain", "Altamirano", "Cifuentes",
                      "Barrientos", "Villarroel", "Undurraga", "Covarrubias", "Ossandon", "Zamorano",
                      "Maturana", "Quintanilla", "Arredondo", "Montalva", "Valdebenito",
                      "Benavides", "Sotomayor", "Mardones", "Leiva", "Ahumada", "Cabezas",
                      "Villagra", "Escobar", "Contreras", "Navarrete", "Castillo", "Olivares"]},
    "az": {"m": ["Rashad", "Tural", "Elnur", "Farid", "Kamran", "Orkhan", "Vugar", "Ilkin", "Samir",
                 "Elchin", "Anar", "Ramil", "Shahin", "Emin", "Nijat", "Togrul"],
           "family": ["Huseynov", "Aliyev", "Guliyev", "Ismayilov", "Hasanov", "Rzayev", "Babayev",
                      "Jafarov", "Abbasov", "Karimov", "Valiyev", "Musayev", "Qasimov", "Ahmadov"]},
    "in": {"m": ["Sanjay", "Arjun", "Vikram", "Anirban", "Sourav", "Debashis", "Abhijit", "Indranil",
                 "Subhash", "Partha", "Tapan", "Amitabh", "Kaushik", "Rupesh", "Sandip", "Tanmoy"],
           "family": ["Chatterjee", "Banerjee", "Mukherjee", "Bhattacharya", "Sengupta", "Ghoshal",
                      "Dasgupta", "Mitra", "Chakraborty", "Bhowmik", "Sarkar", "Ganguly", "Basu",
                      "Lahiri", "Chowdhury", "Bandyopadhyay"]},
    "ar": {"f": ["Layla", "Noor", "Huda", "Salma", "Rania", "Dalia", "Mariam", "Yasmin", "Hala",
                 "Lina", "Samira", "Nadia", "Reem", "Amal", "Zeina", "Dana"],
           "m": ["Omar", "Karim", "Tariq", "Samir", "Nabil", "Faris", "Hadi", "Ziad", "Walid",
                 "Hisham", "Majid", "Rami", "Imad", "Fadi", "Sami", "Bassam", "Nizar", "Ghassan"],
           "middle": ["Karim", "Tariq", "Nabil", "Faris", "Hadi", "Walid", "Majid", "Rami", "Imad",
                      "Fadi", "Sami", "Bassam", "Nizar", "Ghassan", "Hisham", "Yasin"],
           "family": ["Haddad", "Khoury", "Nassar", "Saleh", "Mansour", "Darwish", "Hamdan", "Qasim",
                      "Rahal", "Shami", "Halabi", "Zayed", "Bitar", "Najjar", "Sabbagh", "Fakhoury"],
           "al": ["Al-Haddad", "Al-Nassar", "Al-Saleh", "Al-Darwish", "Al-Hamdan", "Al-Qasim",
                  "Al-Rahal", "Al-Shami", "Al-Halabi", "Al-Zayed", "Al-Bitar", "Al-Najjar",
                  "Al-Sabbagh", "Al-Fakhoury", "Al-Tamimi", "Al-Harbi", "Al-Otaibi", "Al-Shammari"]},
    "kr": {"f": ["Min-Seo", "Hye-Jin", "Soo-Ah", "Eun-Ji", "Ha-Neul", "Yu-Na", "Seo-Yeon", "Ji-Woo",
                 "Da-Eun", "So-Yeon", "Ye-Jin", "Hyun-Ah", "Su-Bin", "Ga-Young"],
           "m": ["Seung-Woo", "Dong-Hyun", "Min-Jun", "Jae-Won", "Sung-Min", "Hyun-Soo", "Do-yun",
                 "Ji-ho", "Min-su", "Woo-jin", "Sang-hoon", "Joon-ho", "Kyung-soo", "Seo-jun",
                 "Hyun-woo", "Jin-ho"],
           "family": ["Kang", "Yoon", "Shin", "Han", "Jung", "Lim", "Seo", "Choi", "Hwang", "Ahn",
                      "Song", "Yoo", "Hong", "Jeon", "Ko", "Moon", "Bae", "Baek"]},
    "ir": {"m": ["Dariush", "Kourosh", "Farhad", "Arash", "Siavash", "Bijan", "Kamyar", "Payam",
                 "Ramin", "Omid", "Babak", "Shahram", "Kaveh", "Mehrdad", "Nima", "Hooman"],
           "family": ["Tehrani", "Sadeghi", "Moradi", "Karimi", "Ahmadi", "Rostami", "Jafari",
                      "Najafi", "Mousavi", "Hosseini", "Kazemi", "Ghorbani", "Rahimi", "Asadi",
                      "Bagheri", "Shirazi"]},
    "pk": {"f": ["Ayesha", "Sana", "Mahnoor", "Iqra", "Rabia", "Zainab", "Hira", "Sidra", "Amna",
                 "Maryam", "Fatima", "Saba", "Nida", "Kiran", "Areeba", "Mehwish"],
           "family": ["Qureshi", "Siddiqui", "Malik", "Chaudhry", "Butt", "Abbasi", "Raza", "Mirza",
                      "Sheikh", "Awan", "Javed", "Hashmi", "Rizvi", "Bajwa", "Khalid", "Iqbal"]},
    "il": {"m": ["Avraham", "Yosef", "Eitan", "Ariel", "Uri", "Gideon", "Noam", "Yair", "Amit",
                 "Ehud", "Oren", "Itai", "Tomer", "Shai", "Dror", "Ilan"],
           "family": ["Ben-Ami", "Bar-Lev", "Ben-Zvi", "Bar-On", "Ben-Haim", "Bar-Natan", "Ben-Shimon",
                      "Bar-Tal", "Ben-Yosef", "Bar-Ilan", "Ben-Ari", "Bar-Zohar", "Ben-Dor",
                      "Bar-Am", "Ben-Nun", "Bar-Shalom"]},
    "et": {"f": ["Meseret", "Tigist", "Hirut", "Selam", "Mahlet", "Tsion", "Helen", "Bezawit",
                 "Rahel", "Saba", "Eden", "Liya", "Hana", "Yeshi", "Aster", "Genet", "Almaz",
                 "Mekdes", "Feven", "Kidist", "Tsehay", "Netsanet"],
           "family": ["Tadesse", "Bekele", "Haile", "Girma", "Mengistu", "Alemu", "Kebede", "Desta",
                      "Wolde", "Tesfaye", "Gebre", "Assefa", "Mulugeta", "Negash", "Tefera", "Ayele",
                      "Demissie", "Getachew", "Shiferaw", "Worku"]},
    "jp": {"m": ["Hiroshi", "Kenji", "Daisuke", "Haruto", "Satoshi", "Yusuke", "Kazuki", "Ryota",
                 "Tatsuya", "Shota", "Koji", "Masato", "Naoki", "Yuto", "Shinji", "Takumi"],
           "family": ["Watanabe", "Yamamoto", "Kobayashi", "Matsumoto", "Inoue", "Kimura", "Hayashi",
                      "Shimizu", "Yamaguchi", "Mori", "Ikeda", "Hashimoto", "Ishikawa", "Ogawa",
                      "Fujita", "Okada", "Goto", "Hasegawa"]},
    "za": {"n": ["Robin", "Jordan", "Morgan", "Rowan", "Quinn", "Avery", "Jamie", "Riley", "Casey",
                 "Sage", "River", "Ash", "Skye", "Remy", "Jesse", "Kai", "Rory", "Charlie"],
           "family": ["Botha", "Pretorius", "Nel", "Venter", "Joubert", "Coetzee", "Steyn", "Malan",
                      "Viljoen", "Cronje", "Swanepoel", "Fourie", "Rossouw", "Theron", "Booysen",
                      "Kruger", "Visser", "Brink", "Louw", "Engelbrecht"]},
    "cnw": {"n": ["Lan", "Jin", "Yun", "An", "Ling", "Ming", "Hua", "Jun", "Qing", "Rui", "Yi",
                  "Ning", "Xiu", "Fei", "Hong", "Ping"],
            "middle": ["Liu", "Yang", "Zhao", "Wu", "Zhou", "Sun", "Ma", "Hu", "Guo", "He", "Gao",
                       "Luo", "Zheng", "Xie", "Song", "Tang"],
            "family": ["Thompson", "Bennett", "Harrison", "Mitchell", "Carter", "Turner", "Parker",
                       "Collins", "Edwards", "Morgan", "Cooper", "Hughes", "Foster", "Graham",
                       "Russell", "Ellis"]},
    "ie": {"m": ["Thomas", "Michael", "Francis", "Joseph", "Daniel", "Gerald", "Brendan", "Cornelius",
                 "Dennis", "Kevin", "Martin", "Declan", "Colm", "Ronan", "Eamon", "Liam"],
           "middle": ["Francis", "Joseph", "Gerard", "Vincent", "Anthony", "Bernard", "Kieran",
                      "Desmond", "Malachy", "Fergus", "Aidan", "Cormac"],
           "family": ["Callahan", "Donovan", "Gallagher", "Brennan", "Kavanagh", "Moriarty",
                      "Fitzgerald", "Doherty", "Quinlan", "Flanagan", "Hennessy", "McCarthy",
                      "Gallagher", "Delaney", "Keane", "Mahony"]},
    "kz": {"m": ["Yerlan", "Nurlan", "Daulet", "Askar", "Bauyrzhan", "Serik", "Arman", "Yerzhan",
                 "Talgat", "Marat", "Kanat", "Dias", "Timur", "Aidos", "Ruslan", "Zhanibek"],
           "family": ["Nurpeisov", "Suleimenov", "Zhakupov", "Tokayev", "Baimukhanov", "Iskakov",
                      "Akhmetov", "Seitkali", "Omarov", "Bekov", "Kassymov", "Sadykov", "Zhunusov",
                      "Mukanov", "Tulegenov", "Abenov"]},
}

#: forget10 author -> (pool, pronoun, template). The template names the pool
#: list of each space-separated part, in order. Pronouns are those the
#: author's answers use (18 she/he, 2 they).
AUTHORS: Dict[int, tuple] = {
    0: ("tw", "f", ("family", "f")),            # Hsiao Yun-Hwa
    1: ("cl", "f", ("f", "family")),            # Carmen Montenegro
    2: ("az", "m", ("m", "family")),            # Elvin Mammadov
    3: ("in", "m", ("m", "family")),            # Rajeev Majumdar
    4: ("ar", "m", ("m", "middle", "al")),      # Jad Ambrose Al-Shamary
    5: ("ar", "m", ("m", "family")),            # Adib Jarrah
    6: ("kr", "f", ("f", "family")),            # Ji-Yeon Park
    7: ("ir", "m", ("m", "family")),            # Behrouz Rohani
    8: ("tw2", "m", ("m", "family")),           # Wei-Jun Chen
    9: ("kr", "m", ("m", "family")),            # Tae-ho Park
    10: ("pk", "f", ("f", "family")),           # Hina Ameen
    11: ("cnw", "n", ("n", "middle", "family")),  # Xin Lee Williams (they)
    12: ("il", "m", ("m", "family")),           # Moshe Ben-David
    13: ("et", "f", ("f", "family")),           # Kalkidan Abera
    14: ("jp", "m", ("m", "family")),           # Takashi Nakamura
    15: ("za", "n", ("n", "family")),           # Raven Marais (they)
    16: ("ar", "f", ("f", "al")),               # Aysha Al-Hashim
    17: ("ie", "m", ("m", "middle", "family")),  # Edward Patrick Sullivan
    18: ("ar", "m", ("m", "middle", "al")),     # Basil Mahfouz Al-Kuwaiti
    19: ("kz", "m", ("m", "family")),           # Nikolai Abilov
}


def _order(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def candidates(author: int, original: str, taken_parts: set) -> List[str]:
    pool, _, template = AUTHORS[author]
    lists = [POOLS[pool][slot] for slot in template]
    out = []
    for combo in itertools.product(*lists):
        if len(set(combo)) < len(combo):
            continue
        name = " ".join(combo)
        if name_structure(name) != name_structure(original):
            continue
        if any(p in taken_parts for p in combo):
            continue
        out.append(name)
    return sorted(set(out), key=_order)


def spread(valid: Sequence[str], k: int) -> List[str]:
    """Greedy: each pick shares as few parts as possible with earlier picks."""
    chosen: List[str] = []
    for _ in range(min(k, len(valid))):
        used = {p for n in chosen for p in re.split(r"[ \-]", n)}
        best = min((n for n in valid if n not in chosen),
                   key=lambda n: (sum(p in used for p in re.split(r"[ \-]", n)),
                                  _order(n)))
        chosen.append(best)
    return chosen


def build(tokenizer, forget_rows, full_rows, k: int, *, max_tries: int = 4000,
          fmt: PromptFormat = PromptFormat.PLAIN_QA, max_length: int = 256) -> dict:
    originals = author_names(forget_rows)
    tofu_names = [n for n in author_names(full_rows) if n]
    taken = {p for n in tofu_names for p in n.split()}
    out = []
    for a, original in enumerate(originals):
        if original is None or a not in AUTHORS:
            raise SystemExit(f"author {a}: no name or no pool")
        rows = forget_rows[a * QAS_PER_AUTHOR:(a + 1) * QAS_PER_AUTHOR]
        cands = candidates(a, original, taken)
        valid, tried = [], 0
        for name in cands[:max_tries]:
            tried += 1
            try:
                for r in rows:
                    align_row(tokenizer, r["question"], r["answer"], original, [name],
                              fmt=fmt, max_length=max_length)
            except AlignmentError:
                continue
            valid.append(name)
            if len(valid) >= 8 * k:
                break
        picked = spread(valid, k)
        pool, pronoun, _ = AUTHORS[a]
        out.append({"author": a, "name": original, "pool": pool,
                    "pronoun": {"f": "she", "m": "he", "n": "they"}[pronoun],
                    "names": picked, "n_candidates": len(cands), "n_tried": tried,
                    "n_aligned": len(valid)})
        print(f"{a:2d} {original:26s} {len(cands):5d} candidates, {len(valid):3d} aligned "
              f"(of {tried} tried): {picked}")
    short = [e["author"] for e in out if len(e["names"]) < k]
    if short:
        raise SystemExit(f"Fewer than {k} aligned names for authors {short}; widen POOLS.")
    return {"what": "STRANGER counterfactual names (deeperase/scripts/build_cf_names.py)",
            "k": k, "rules": __doc__.split("\n\n")[1].strip(), "authors": out}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tofu-dir", default="data/tofu")
    ap.add_argument("--size", default="1B")
    ap.add_argument("--cache-dir", default=os.environ.get("DEEPERASE_CACHE_DIR", "./hf_cache"))
    ap.add_argument("--out", default="data/counterfactual_names.json")
    args = ap.parse_args(argv)
    from deeperase.models import load_tokenizer
    repo = TOFU_MODELS[args.size]["full"].repo_id
    tokenizer = load_tokenizer(repo, cache_dir=args.cache_dir)
    data = build(tokenizer, read_tofu_rows("forget10", args.tofu_dir),
                 read_tofu_rows("full", args.tofu_dir), args.k)
    data["tokenizer"] = repo
    Path(args.out).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
