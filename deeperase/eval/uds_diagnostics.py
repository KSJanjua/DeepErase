"""How much of a UDS is overshoot: diagnostics on stored per-layer deltas.

UDS clips each layer's erasure ratio r = dS2/dS1 at 1 (Eq. 4), so a source
that makes M_full do *worse* than retain90 on the entity scores the same as
one that matches retain90. Destroying looks like erasing. The UDS authors say
so themselves: over-unlearning is "mathematically indistinguishable from
perfect unlearning" inside the score.

Oracles show what genuine absence looks like: retain95/99 on facts they never
saw give r between about 0.94 and 1.04, with almost no cells above 2. The
v2 runs do not: at alpha = 1, 14-17% of their UDS on trained facts comes from cells with r > 2,
and all of GA's depth gain from alpha 1 to 2 is such overshoot (RESULTS.md
section 16).

These read only the stored ``delta_s1`` / ``delta_s2`` of each example, so they
apply retroactively to every saved run. They are exploratory diagnostics,
reported next to UDS, not replacements for it (docs/ANALYSIS_PLAN.md §9,
17 Oct 2026):

* **strict UDS**: LER := 0 where r > 2. A cell more than twice as damaged as
  ignorance counts as no evidence of erasure.
* **tent UDS**: LER := max(0, 2 - r) where r > 1. Full credit at r = 1,
  falling linearly to none at r = 2.
* **overshoot share**: the dS1 weight of an example's KE cells with r > 2.
* **band UDS**: UDS restricted to the KE layers inside a band of layers.
"""

from __future__ import annotations

import statistics
from typing import Dict, Iterable, Mapping, Optional, Sequence

from deeperase.eval.uds import DEFAULT_TAU

#: Layer bands of Llama-3.2-1B's 16 blocks. Layers 9-15 carry about 92% of the
#: UDS weight; 13-15 alone about half of it.
BANDS: Dict[str, range] = {"L0-8": range(0, 9), "L9-12": range(9, 13),
                           "L13-15": range(13, 16)}
OVERSHOOT_R = 2.0
LATE = range(9, 16)


def _as_int_keys(d: Mapping) -> Dict[int, float]:
    return {int(k): float(v) for k, v in d.items()}


def _weighted(cells, credit) -> Optional[float]:
    total = sum(w for w, _ in cells)
    if total <= 0:
        return None
    return sum(w * credit(r) for w, r in cells) / total


def clipped(r: float) -> float:
    return min(max(r, 0.0), 1.0)


def strict(r: float) -> float:
    return 0.0 if r > OVERSHOOT_R else clipped(r)


def tent(r: float) -> float:
    return max(0.0, OVERSHOOT_R - r) if r > 1.0 else max(r, 0.0)


def example_diagnostics(delta_s1: Mapping, delta_s2: Mapping, *,
                        tau: float = DEFAULT_TAU) -> Optional[dict]:
    """Every diagnostic for one example, or None when it has no KE layer.

    ``uds`` reproduces :func:`deeperase.eval.uds.aggregate_example_uds`
    exactly; the others change only how a cell with r > 1 is credited.
    """
    d1, d2 = _as_int_keys(delta_s1), _as_int_keys(delta_s2)
    ke = sorted(l for l, v in d1.items() if v > tau)
    if not ke:
        return None
    cells = [(d1[l], d2[l] / d1[l]) for l in ke]
    total = sum(w for w, _ in cells)
    bands = {}
    for name, band in BANDS.items():
        sub = [(d1[l], d2[l] / d1[l]) for l in ke if l in band]
        bands[name] = _weighted(sub, clipped)
    late = [d2[l] / d1[l] for l in ke if l in LATE]
    return {
        "uds": _weighted(cells, clipped),
        "strict": _weighted(cells, strict),
        "tent": _weighted(cells, tent),
        "overshoot_share": sum(w for w, r in cells if r > OVERSHOOT_R) / total,
        "bands": bands,
        "late_ratios": late,
    }


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarise(per_example: Sequence[Optional[dict]]) -> dict:
    """Run-level means over the examples with a KE layer (as Eq. 6 averages)."""
    defined = [e for e in per_example if e is not None]
    late = [r for e in defined for r in e["late_ratios"]]
    return {
        "n_scored": len(defined),
        "uds": _mean(e["uds"] for e in defined),
        "strict": _mean(e["strict"] for e in defined),
        "tent": _mean(e["tent"] for e in defined),
        "overshoot_share": _mean(e["overshoot_share"] for e in defined),
        "bands": {name: _mean(e["bands"][name] for e in defined) for name in BANDS},
        "median_late_ratio": statistics.median(late) if late else None,
        "share_late_cells_over_2": (sum(r > OVERSHOOT_R for r in late) / len(late)
                                    if late else None),
    }


def report_diagnostics(report: Mapping, *, tau: Optional[float] = None) -> dict:
    """:func:`summarise` for a saved depth report (``partial/*.json`` ->
    ``depth.<split>``), which keeps every example's ``delta_s1`` and
    ``delta_s2``."""
    t = float(report.get("tau", DEFAULT_TAU) if tau is None else tau)
    return summarise([example_diagnostics(e["delta_s1"], e["delta_s2"], tau=t)
                      for e in report.get("per_example", [])])
