"""How much does the depth axis depend on the KE-layer threshold tau?

tau = 0.05 is inherited from the UDS paper (Eq. 2) and was never checked on
our data (docs/UDS_CONFORMANCE.md §5 item 5). This re-aggregates the saved
Table-2 validation runs at other thresholds. No model is loaded: every
``report.json`` already stores the per-example, per-layer dS1 and dS2, and
tau enters only at aggregation time (Eqs. 2, 4-6).

Aggregation reuses :func:`deeperase.eval.uds.aggregate_example_uds`, so this
script cannot drift from the metric it is testing. As a self-check, the value
recomputed at each run's own tau must reproduce the stored UDS exactly; the
script refuses to report anything if it does not.

    python -m deeperase.scripts.tau_sensitivity
    python -m deeperase.scripts.tau_sensitivity --runs table2_refspans table2_3B
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from deeperase.config import UDS_PAPER_TABLE2
from deeperase.eval.uds import aggregate_example_uds

DEFAULT_TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
SPLITS = ("full", "retain99", "retain95", "retain90")


def _layer_dict(d: Dict[str, float]) -> Dict[int, float]:
    return {int(k): float(v) for k, v in d.items()}


def uds_at_tau(per_example: Sequence[dict], tau: float) -> dict:
    """Eq. 6 at a given tau, from stored per-example dS1/dS2."""
    scores: List[float] = []
    for ex in per_example:
        u, _, _ = aggregate_example_uds(_layer_dict(ex["delta_s1"]),
                                        _layer_dict(ex["delta_s2"]), tau=tau)
        if u is not None:
            scores.append(u)
    return {"uds": (sum(scores) / len(scores)) if scores else None,
            "n_scored": len(scores), "n_total": len(per_example)}


def analyse_run(run_dir: Path, taus: Sequence[float]) -> Optional[dict]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_split = report.get("per_split_reports") or {}
    if not per_split:
        return None

    size = (report.get("config") or {}).get("size_label", "1B")
    out = {"run": run_dir.name, "size": size,
           "prompt_format": report.get("prompt_format"),
           "span_strategy": report.get("span_strategy"),
           "by_tau": {}, "self_check": {}}

    for split, rep in per_split.items():
        own_tau = float(rep["tau"])
        recomputed = uds_at_tau(rep["per_example"], own_tau)["uds"]
        stored = rep["uds"]
        ok = (recomputed is None and stored is None) or (
            recomputed is not None and stored is not None
            and math.isclose(recomputed, stored, abs_tol=1e-9))
        out["self_check"][split] = {"stored": stored, "recomputed": recomputed, "ok": ok}
        if not ok:
            raise RuntimeError(
                f"{run_dir.name}/{split}: recomputed UDS {recomputed} does not match "
                f"the stored {stored} at tau={own_tau}. The stored deltas do not "
                "reproduce the run, so no tau analysis of it can be trusted.")

    for tau in taus:
        row = {}
        for split in SPLITS:
            if split in per_split:
                row[split] = uds_at_tau(per_split[split]["per_example"], tau)
        observed = {s: r["uds"] for s, r in row.items() if r["uds"] is not None}
        expected = UDS_PAPER_TABLE2.get(size, {})
        middle = [s for s in ("retain99", "retain95") if s in observed and s in expected]
        vals = [observed[s] for s in SPLITS if s in observed]
        out["by_tau"][str(tau)] = {
            "per_split": row,
            "monotonic": all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])),
            "mad_all": (sum(abs(observed[s] - expected[s]) for s in observed if s in expected)
                        / max(1, sum(1 for s in observed if s in expected))),
            "mad_informative": (sum(abs(observed[s] - expected[s]) for s in middle)
                                / len(middle)) if middle else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="results/gpu_runs")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Run directories to analyse (default: every run with a report.json)")
    ap.add_argument("--taus", default=",".join(str(t) for t in DEFAULT_TAUS))
    ap.add_argument("--out", default="results/analysis/tau_sensitivity.json")
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    root = Path(args.dir)
    dirs = ([root / r for r in args.runs] if args.runs
            else sorted(p for p in root.iterdir() if p.is_dir()))
    results = [r for r in (analyse_run(d, taus) for d in dirs) if r]
    if not results:
        print(f"No run with a report.json under {root}")
        return 1

    for r in results:
        print(f"\n{r['run']}  ({r['size']}, {r['prompt_format']}, {r['span_strategy']})"
              f"  self-check at stored tau: OK")
        print(f"  {'tau':>6} " + " ".join(f"{s:>9}" for s in SPLITS)
              + f"  {'scored':>9} {'mono':>5} {'MAD(2)':>7}")
        for tau in taus:
            row = r["by_tau"][str(tau)]
            cells = []
            for s in SPLITS:
                v = row["per_split"].get(s, {}).get("uds")
                cells.append(f"{v:9.3f}" if v is not None else f"{'undef':>9}")
            sc = row["per_split"].get("retain95", {})
            mad = row["mad_informative"]
            print(f"  {tau:6.2f} " + " ".join(cells)
                  + f"  {sc.get('n_scored', 0):>4}/{sc.get('n_total', 0):<4}"
                  + f" {str(row['monotonic'])[0]:>5} "
                  + (f"{mad:7.3f}" if mad is not None else f"{'-':>7}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"taus": taus, "paper_table2": UDS_PAPER_TABLE2,
                               "runs": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
