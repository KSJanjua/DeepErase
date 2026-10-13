"""Fact-level analysis: does each fact's depth track its own behavioural forgetting?

The plane compares two averages per alpha. Two averages can move together while
the facts underneath do not, so the plane alone cannot say whether a model that
stops answering a question has also lost the fact inside. The depth score
resolves single facts: per-example UDS separates facts a checkpoint never saw
from facts it did with AUC 1.000 in every saved validation run (RESULTS.md
§11). So the relation can be tested where it lives, one fact at a time.

For every run, alpha and split (trained / held-out) this joins

* **depth_i**: the fact's UDS (``depth[split].per_example``, ids ``ref_<row>``);
* **forgotten_i**: the share of the fact's forced-choice items (B0-B4;
  ids ``<tier>_<row>``) that ``full`` answered correctly and the unlearned model
  now gets wrong. Items ``full`` already got wrong say nothing about forgetting;
* **margin_drop_i**: the mean fall in margin (log p of the correct answer minus
  the best wrong one) from ``full``, over all of the fact's items. Continuous,
  so it keeps the signal the binary outcome throws away.

and reports

1. **coupling**: Spearman rho across facts between depth_i and margin_drop_i,
   with a 95% CI from resampling *authors*. The 20 authors are the independent
   units; facts of one author are not;
2. **classes** at depth thresholds 0.3 / 0.5 / 0.7 (a fact is *forgotten* when
   at least half of its eligible items flipped):

   ===============  ==============  ==================================
   class            depth_i >= th   forgotten
   ===============  ==============  ==================================
   removed          yes             yes
   suppressed       no              yes  (stops answering, still encoded)
   erased_answering yes             no
   intact           no              no
   ===============  ==============  ==================================

   and the **suppression rate** P(depth_i < th | forgotten), with its CI;
3. **order**: per fact, the first alpha at which depth crosses th and the first
   at which the fact is forgotten -- which comes first;
4. **indirect leak**: among facts forgotten on the direct questions (B0, B1),
   the share of their indirect items (B2: author described, B3: the fact given
   and the author asked for, B4: author named by a book) still answered
   correctly, for deep and for shallow facts. A fact
   that is only suppressed should still be reachable by another route; a
   removed one should not;
5. **oracle**: the same behavioural measures for ``retain90``, which never saw
   the facts (depth 1 by construction). A forced choice can be won by
   plausibility alone, so not even a truly ignorant model forgets every item;
   ``erased_answering`` must be read against the oracle's not-forgotten share.

Needs protocol-v2 run directories (per-item outcomes in the partials, and
``reference_outcomes`` in calibration.json). Reads only JSON: no torch, no
models.

    python -m deeperase.scripts.fact_level_analysis --glob 'study_v2_*'
    python -m deeperase.scripts.fact_level_analysis --run results/studies/<run>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deeperase.data.tofu import QAS_PER_AUTHOR
from deeperase.registry import load_registry, protocol_of, run_status, variant_of
from deeperase.scripts.aggregate_matrix import _fmt, _mean_sd

FACT_TIERS = ("B0", "B1", "B2", "B3", "B4")
DIRECT_TIERS, INDIRECT_TIERS = ("B0", "B1"), ("B2", "B3", "B4")
THRESHOLDS = (0.3, 0.5, 0.7)
FORGOTTEN_CUT = 0.5
CLASSES = ("removed", "suppressed", "erased_answering", "intact")
_ITEM = re.compile(r"^(B0|B1|B2|B3|B4)_(\d+)$")

Outcomes = Dict[str, list]          # item_id -> [knows, margin, (scores)]


# ---------------------------------------------------------------------------
# Per-fact measures
# ---------------------------------------------------------------------------

def items_by_fact(outcomes: Outcomes) -> Dict[int, Dict[str, Tuple[int, float]]]:
    """Forget-tier outcomes grouped by forget10 row: row -> tier -> (knows, margin)."""
    out: Dict[int, Dict[str, Tuple[int, float]]] = {}
    for item_id, rec in (outcomes or {}).items():
        m = _ITEM.match(item_id)
        if m:
            out.setdefault(int(m.group(2)), {})[m.group(1)] = (int(rec[0]), float(rec[1]))
    return out


def behaviour(at: Outcomes, base: Optional[Outcomes]) -> Dict[int, dict]:
    """Per-fact behavioural forgetting of ``at`` relative to ``base`` (``full``).

    ``forgotten`` counts only items ``base`` got right; a fact with none has
    ``forgotten = None``. ``margin_drop`` averages over every item scored by
    both. Without a baseline (a run saved before reference outcomes were
    recorded) ``forgotten`` is the share of items answered wrong and
    ``margin_drop`` is the negated margin -- still rank-comparable across facts.

    ``direct_forgotten`` is ``forgotten`` over B0/B1 only, and
    ``indirect_kept`` the share of eligible B2/B4 items still answered
    correctly (None when the fact has none).
    """
    now, ref = items_by_fact(at), items_by_fact(base) if base else None
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    out = {}
    for row, tiers in now.items():
        r = None if ref is None else ref.get(row, {})
        shared = [t for t in tiers if r is None or t in r]
        elig = [t for t in shared if r is None or r[t][0]]
        drops = [(r[t][1] if r is not None else 0.0) - tiers[t][1] for t in shared]
        if not drops:
            continue
        flip = lambda ts: [1 - tiers[t][0] for t in elig if t in ts]
        out[row] = {"n_items": len(drops), "n_eligible": len(elig),
                    "forgotten": mean(flip(FACT_TIERS)),
                    "margin_drop": mean(drops),
                    "direct_forgotten": mean(flip(DIRECT_TIERS)),
                    "indirect_kept": mean([tiers[t][0] for t in elig
                                           if t in INDIRECT_TIERS])}
    return out


def depth_by_fact(report: dict) -> Dict[int, float]:
    """Row -> UDS_i from a saved depth report; undefined examples are dropped."""
    out = {}
    for ex in (report or {}).get("per_example", []):
        if ex.get("uds") is not None:
            out[int(str(ex["example_id"]).rsplit("_", 1)[1])] = float(ex["uds"])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), as Spearman needs."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman's rho; None when fewer than 3 points or either side is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return None if den == 0 else float((rx * ry).sum() / den)


def author_bootstrap(rows: Sequence[int], stat: Callable[[List[int]], Optional[float]],
                     *, n_boot: int = 1000, seed: int = 0,
                     level: float = 0.95) -> Optional[List[float]]:
    """Percentile CI for ``stat(rows)``, resampling authors with replacement.

    Facts of one author share a fine-tuning history and a description, so they
    are not independent; treating them as such would make every interval too
    narrow. Returns None when fewer than 3 authors or too few resamples are
    defined.
    """
    by_author: Dict[int, List[int]] = {}
    for r in rows:
        by_author.setdefault(r // QAS_PER_AUTHOR, []).append(r)
    authors = sorted(by_author)
    if len(authors) < 3:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(authors), len(authors))
        v = stat([r for a in pick for r in by_author[authors[a]]])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def classify(depth: Dict[int, float], beh: Dict[int, dict], rows: Sequence[int],
             theta: float) -> Dict[str, int]:
    counts = dict.fromkeys(CLASSES, 0)
    for r in rows:
        deep = depth[r] >= theta
        forgot = beh[r]["forgotten"] >= FORGOTTEN_CUT
        counts[("removed" if deep else "suppressed") if forgot
               else ("erased_answering" if deep else "intact")] += 1
    return counts


def suppression_rate(depth: Dict[int, float], beh: Dict[int, dict], rows: Sequence[int],
                     theta: float) -> Optional[float]:
    """P(depth < theta | forgotten): forgotten facts that are still encoded."""
    forgot = [r for r in rows if beh[r]["forgotten"] >= FORGOTTEN_CUT]
    if not forgot:
        return None
    return sum(1 for r in forgot if depth[r] < theta) / len(forgot)


def indirect_leak(depth: Dict[int, float], beh: Dict[int, dict], rows: Sequence[int],
                  theta: float) -> dict:
    """Share of indirect items still answered, for directly forgotten facts,
    split by depth. Suppression predicts ``shallow`` > ``deep``."""
    out = {}
    for name, keep in (("deep", lambda r: depth[r] >= theta),
                       ("shallow", lambda r: depth[r] < theta)):
        vals = [beh[r]["indirect_kept"] for r in rows
                if keep(r) and beh[r]["indirect_kept"] is not None
                and (beh[r]["direct_forgotten"] or 0.0) >= FORGOTTEN_CUT]
        out[name] = (sum(vals) / len(vals)) if vals else None
        out[f"n_{name}"] = len(vals)
    return out


def first_crossing(series: Sequence[Tuple[float, Optional[float]]],
                   cut: float) -> Optional[float]:
    for a, v in series:
        if v is not None and v >= cut:
            return a
    return None


def crossing_order(depth_series: Dict[int, list], beh_series: Dict[int, list],
                   theta: float) -> dict:
    """Which axis each fact crosses first along alpha (depth at theta, behaviour at 0.5)."""
    tally = dict.fromkeys(("depth_first", "same", "behaviour_first", "depth_only",
                           "behaviour_only", "neither"), 0)
    lags = []
    for r in sorted(set(depth_series) & set(beh_series)):
        ad = first_crossing(depth_series[r], theta)
        ab = first_crossing(beh_series[r], FORGOTTEN_CUT)
        if ad is None and ab is None:
            tally["neither"] += 1
        elif ab is None:
            tally["depth_only"] += 1
        elif ad is None:
            tally["behaviour_only"] += 1
        else:
            lags.append(ab - ad)
            tally["depth_first" if ad < ab else "behaviour_first" if ab < ad else "same"] += 1
    n = sum(tally.values())
    return {**tally, "n": n,
            "median_lag": float(np.median(lags)) if lags else None,
            "note": "lag = alpha(behaviour) - alpha(depth); positive = depth first"}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _load(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_partials(run: Path) -> List[Tuple[float, dict]]:
    out = []
    for f in sorted((run / "partial").glob("alpha_*.json")):
        d = _load(f)
        if d is None:
            continue
        pt = d.get("point") or (d.get("points") or [{}])[0]
        out.append((round(float(pt.get("alpha", -1)), 6), d))
    return sorted(out, key=lambda t: t[0])


def analyse_run(run: Path, *, thetas: Sequence[float] = THRESHOLDS,
                n_boot: int = 1000, seed: int = 0) -> dict:
    """Every fact-level statistic for one run directory. Raises ValueError when
    the run has no per-item data (protocol-v1 runs)."""
    run = Path(run)
    cfg = _load(run / "config.json") or {}
    cal = _load(run / "calibration.json") or {}
    partials = load_partials(run)
    if not partials or not all("depth" in p and "outcomes" in (p.get("breadth_tiers") or {})
                               for _, p in partials):
        raise ValueError(f"{run.name}: no per-item outcomes or per-fact depth "
                         "(runs before protocol v2 did not save them)")
    ref = cal.get("reference_outcomes") or {}
    base = ref.get("full")
    result = {"run": run.name, "protocol": protocol_of(cfg) if cfg else None,
              "variant": variant_of(cfg), "method": cfg.get("method"),
              "lr": (cfg.get("unlearn") or {}).get("learning_rate"),
              "seed": cfg.get("seed"),
              "baseline": "full" if base else "none",
              "alphas": [a for a, _ in partials], "oracle": {}, "splits": {}}

    split_rows = {}
    s = _load(run / "split.json") or {}
    for k in ("trained", "heldout"):
        if s.get(f"{k}_rows"):
            split_rows[k] = set(s[f"{k}_rows"])

    if base and ref.get("retain90"):
        ob = behaviour(ref["retain90"], base)
        for k, rows in split_rows.items():
            f = [ob[r]["forgotten"] for r in rows if r in ob and ob[r]["forgotten"] is not None]
            kept = [ob[r]["indirect_kept"] for r in rows
                    if r in ob and ob[r]["indirect_kept"] is not None]
            if f:
                result["oracle"][k] = {
                    "n_facts": len(f), "forgotten_mean": float(np.mean(f)),
                    "share_forgotten": sum(1 for x in f if x >= FORGOTTEN_CUT) / len(f),
                    # the leak an ignorant model shows: plausibility alone
                    "indirect_kept_mean": float(np.mean(kept)) if kept else None}

    splits = sorted({k for _, p in partials for k in p["depth"]})
    for k in splits:
        per_alpha, d_series, b_series = [], {}, {}
        for a, p in partials:
            depth = depth_by_fact(p["depth"].get(k))
            beh = behaviour(p["breadth_tiers"]["outcomes"], base)
            rows = sorted(r for r in depth if r in beh)
            for r in rows:
                d_series.setdefault(r, []).append((a, depth[r]))
                b_series.setdefault(r, []).append((a, beh[r]["forgotten"]))
            elig = [r for r in rows if beh[r]["forgotten"] is not None]

            def rho(rs, key="margin_drop"):
                return spearman([depth[r] for r in rs], [beh[r][key] for r in rs])

            row = {"alpha": a, "n_facts": len(rows), "n_eligible": len(elig),
                   "depth_mean": float(np.mean([depth[r] for r in rows])) if rows else None,
                   # all-or-none erasure leaves few facts here; partial erasure, many
                   "share_partial_depth": (sum(0.2 <= depth[r] < 0.8 for r in rows) / len(rows))
                   if rows else None,
                   "forgotten_mean": float(np.mean([beh[r]["forgotten"] for r in elig]))
                   if elig else None,
                   "rho_margin": rho(rows),
                   "rho_margin_ci": author_bootstrap(rows, rho, n_boot=n_boot, seed=seed),
                   "rho_forgotten": rho(elig, "forgotten"),
                   "classes": {}}
            for th in thetas:
                rate = lambda rs, th=th: suppression_rate(depth, beh, rs, th)
                row["classes"][f"{th:g}"] = {
                    **classify(depth, beh, elig, th),
                    "suppression_rate": rate(elig),
                    "suppression_rate_ci": author_bootstrap(elig, rate, n_boot=n_boot,
                                                            seed=seed),
                    "indirect_leak": indirect_leak(depth, beh, rows, th)}
            per_alpha.append(row)
        result["splits"][k] = {
            "per_alpha": per_alpha,
            "order": {f"{th:g}": crossing_order(d_series, b_series, th) for th in thetas}}
    return result


# ---------------------------------------------------------------------------
# Across runs
# ---------------------------------------------------------------------------

def pool(results: List[dict], theta: float = 0.5) -> List[dict]:
    """Mean ± sd across seeds of the headline statistics, per (protocol, variant,
    method, lr, split)."""
    groups: Dict[tuple, List[dict]] = {}
    for r in results:
        for k, sp in r["splits"].items():
            groups.setdefault((r["protocol"], r["variant"], r["method"], r["lr"], k),
                              []).append(
                {"run": r["run"], "sp": sp, "oracle": r["oracle"].get(k)})
    th = f"{theta:g}"
    out = []
    for (protocol, variant, method, lr, k), members in sorted(groups.items(),
                                                              key=lambda kv: str(kv[0])):
        alphas = sorted({row["alpha"] for m in members for row in m["sp"]["per_alpha"]})
        rows = []
        for a in alphas:
            got = [row for m in members for row in m["sp"]["per_alpha"] if row["alpha"] == a]
            cls = [row["classes"].get(th) or {} for row in got]
            elig = lambda c: sum(c.get(x, 0) for x in CLASSES) or None
            rows.append({
                "alpha": a,
                "depth_mean": _mean_sd([g["depth_mean"] for g in got]),
                "share_partial_depth": _mean_sd([g.get("share_partial_depth") for g in got]),
                "forgotten_mean": _mean_sd([g["forgotten_mean"] for g in got]),
                "rho_margin": _mean_sd([g["rho_margin"] for g in got]),
                "suppression_rate": _mean_sd([c.get("suppression_rate") for c in cls]),
                "leak_deep": _mean_sd([(c.get("indirect_leak") or {}).get("deep") for c in cls]),
                "leak_shallow": _mean_sd([(c.get("indirect_leak") or {}).get("shallow")
                                          for c in cls]),
                **{f"share_{x}": _mean_sd([(c.get(x, 0) / elig(c)) if elig(c) else None
                                          for c in cls]) for x in CLASSES}})
        order = [m["sp"]["order"].get(th) or {} for m in members]
        out.append({
            "protocol": protocol, "variant": variant, "method": method, "lr": lr, "split": k,
            "theta": theta, "runs": [m["run"] for m in members], "per_alpha": rows,
            "oracle_share_forgotten": _mean_sd([(m["oracle"] or {}).get("share_forgotten")
                                                for m in members]),
            "oracle_indirect_kept": _mean_sd([(m["oracle"] or {}).get("indirect_kept_mean")
                                              for m in members]),
            "order": {x: _mean_sd([o.get(x) for o in order])
                      for x in ("depth_first", "same", "behaviour_first", "depth_only",
                                "behaviour_only", "neither", "median_lag")}})
    return out


def to_markdown(pooled: List[dict]) -> str:
    lines = []
    for g in pooled:
        lr = f"{g['lr']:g}" if isinstance(g["lr"], (int, float)) else str(g["lr"])
        lines.append(f"### {g['method']} · lr {lr} · {g['protocol']} ({g['variant']}) · "
                     f"`{g['split']}` facts · "
                     f"{len(g['runs'])} run(s) · depth threshold {g['theta']:g}\n")
        lines.append("| α | depth | UDS_i in [0.2, 0.8) | forgotten | ρ(depth, margin drop) "
                     "| suppression rate "
                     "| removed | suppressed | erased, answering | intact "
                     "| indirect leak, deep | indirect leak, shallow |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in g["per_alpha"]:
            lines.append(
                f"| {r['alpha']:.2f} | {_fmt(r['depth_mean'])} | "
                f"{_fmt(r['share_partial_depth'], 2)} | {_fmt(r['forgotten_mean'])} | "
                f"{_fmt(r['rho_margin'], 2)} | {_fmt(r['suppression_rate'], 2)} | "
                + " | ".join(_fmt(r[f'share_{x}'], 2) for x in CLASSES)
                + f" | {_fmt(r['leak_deep'], 2)} | {_fmt(r['leak_shallow'], 2)} |")
        o = g["order"]
        lines.append(
            f"\nOracle (`retain90`) forgets {_fmt(g['oracle_share_forgotten'], 2)} of these "
            f"facts by the same rule, and still answers "
            f"{_fmt(g['oracle_indirect_kept'], 2)} of their indirect items (the leak "
            f"plausibility alone produces). First crossing along α: depth first "
            f"{_fmt(o['depth_first'], 1)}, same α {_fmt(o['same'], 1)}, behaviour first "
            f"{_fmt(o['behaviour_first'], 1)}, depth only {_fmt(o['depth_only'], 1)}, "
            f"behaviour only {_fmt(o['behaviour_only'], 1)}, neither {_fmt(o['neither'], 1)} "
            f"(facts per run); median lag {_fmt(o['median_lag'], 2)} α.\n")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/studies")
    ap.add_argument("--glob", default="study_v2_*")
    ap.add_argument("--run", action="append", default=[],
                    help="analyse this run directory (repeatable); overrides --glob")
    ap.add_argument("--include-all", action="store_true",
                    help="also analyse invalid/superseded runs and controls")
    ap.add_argument("--theta", type=float, default=0.5,
                    help="depth threshold for the pooled table (all of 0.3/0.5/0.7 "
                         "are in the JSON)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/analysis/fact_level",
                    help="output prefix: writes <out>.json and <out>.md")
    args = ap.parse_args(argv)

    registry = load_registry(args.dir)
    dirs = [Path(r) for r in args.run] or sorted(
        d for d in Path(args.dir).glob(args.glob) if d.is_dir())
    results, skipped = [], []
    for d in dirs:
        cfg = _load(d / "config.json") or {}
        status, _ = run_status(registry, d.name, cfg)
        if status != "ok" and not args.include_all and not args.run:
            skipped.append(f"{d.name} ({status})")
            continue
        try:
            results.append(analyse_run(d, n_boot=args.n_boot))
        except ValueError as e:
            skipped.append(str(e))
    if not results:
        print("No run with per-item data found." + (f" Skipped: {'; '.join(skipped)}"
                                                    if skipped else ""))
        return 1
    pooled = pool(results, args.theta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"runs": results, "pooled": pooled, "skipped": skipped}, indent=1), encoding="utf-8")
    md = to_markdown(pooled)
    out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
    for r in results:
        if r["baseline"] == "none":
            print(f"Note: {r['run']} has no reference outcomes; 'forgotten' there is "
                  "the share answered wrong, not the share flipped from `full`.")
    if skipped:
        print(f"Skipped: {'; '.join(skipped)}")
    print(f"\nWrote {out}.json and {out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
