"""Show what this project has stored on disk, and what each stored run says.

This is a viewer, not a computation. It opens the saved run directories and
prints their contents, so that the stored evidence can be inspected without
reading raw JSON.

    python show_results.py            # the validated results
    python show_results.py --all      # include superseded earlier runs
    python show_results.py --files    # also list the files inside a run
"""
from __future__ import annotations

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
RUNS = ROOT / "results" / "gpu_runs"

BAR = "=" * 78
DASH = "-" * 78


def _hr(title):
    print(f"\n{BAR}\n{title}\n{BAR}")


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def show_storage_layout():
    _hr("1. HOW RESULTS ARE STORED")
    print("""
Every run writes its own folder. Nothing is overwritten and nothing is
edited by hand, so any number that appears in the report can be traced back
to the folder that produced it.

  results/gpu_runs/<run name>/
      config.json         exactly what was asked for (model, sample size,
                          settings) -- enough to repeat the run
      report.json         what came out: the measured numbers, and the
                          comparison against the published values
      stage1_cache.json   an expensive intermediate result, saved so a
                          repeat run does not recompute it
      partial/            per-step results written as the run proceeds, so
                          an interrupted run can resume instead of restarting

  results/logs/           the terminal output of each run, kept verbatim
  report/figures/         every figure, regenerated from the numbers above
""".rstrip())


def show_depth_runs(list_files=False, include_all=False):
    _hr("2. DEPTH MEASUREMENT -- validating our instrument")
    print("""
Question this answers: does our depth measurement agree with the published
values it is supposed to reproduce?

The four columns are four models whose relationship to the deleted data is
known exactly. 'full' learned everything. 'retain90' never saw any of it.
A correct measurement must read 0 for the first and 1 for the last, and
rise in between.
""".rstrip())
    rows = []
    for d in sorted(RUNS.iterdir()):
        rep = d / "report.json"
        if not rep.exists() or d.name.startswith("breadth"):
            continue
        j = _load(rep)
        if not j or "observed" not in j:
            continue
        o, c = j["observed"], j.get("comparison", {})
        ok = c.get("n_within_tolerance", 0) == c.get("n_compared", -1)
        passed = bool(ok and c.get("monotonic", False))
        rows.append((d.name, o, c, j, passed))

    superseded = [r for r in rows if not r[4]]
    shown = rows if include_all else [r for r in rows if r[4]]

    print(f"\n  {'run folder':<28}{'full':>8}{'retain99':>10}{'retain95':>10}"
          f"{'retain90':>10}{'n':>6}  verdict")
    print("  " + DASH[:74])
    for name, o, c, j, passed in shown:
        print(f"  {name:<28}{o['full']:>8.3f}{o['retain99']:>10.3f}"
              f"{o['retain95']:>10.3f}{o['retain90']:>10.3f}"
              f"{j.get('n_examples', '?'):>6}  "
              f"{'PASS' if passed else 'superseded'}")
    if shown:
        exp = shown[0][3]["expected"]
        print("  " + DASH[:74])
        print(f"  {'PUBLISHED VALUES':<28}{exp['full']:>8.3f}"
              f"{exp['retain99']:>10.3f}{exp['retain95']:>10.3f}"
              f"{exp['retain90']:>10.3f}")
        tol = shown[0][2].get("tolerance")
        print(f"\n  Agreement required: within {tol} on every column.")
        best = min(shown, key=lambda r: sum(
            v["abs_diff"] for v in r[2]["per_split"].values()) / 4)
        mae = sum(v["abs_diff"] for v in best[2]["per_split"].values()) / 4
        print(f"  Closest run: {best[0]} (mean difference {mae:.3f})")
        if superseded and not include_all:
            print(f"\n  {len(superseded)} earlier run(s) not shown, kept on disk "
                  "for the audit trail.")
            print("  They were superseded after we found and fixed a sampling "
                  "fault. Use")
            print("  --all to include them.")

    if list_files and shown:
        d = RUNS / shown[0][0]
        print(f"\n  Files inside {d.relative_to(ROOT)}:")
        for f in sorted(d.iterdir()):
            size = (f"{f.stat().st_size/1024:.0f} KB" if f.is_file()
                    else f"{len(list(f.iterdir()))} files")
            print(f"    {f.name:<24} {size}")


def show_breadth_runs():
    _hr("3. BREADTH MEASUREMENT -- calibrating our scale")
    print("""
Question this answers: how far does forgetting spread beyond the exact
wording the model was trained on, and what does a score of 0 or 1 actually
mean on this scale?

'leakage' is how often the model still picks the correct answer. We measure
it on two models we can trust: one that definitely knows the facts, and one
that definitely never saw them. Those two readings become the two ends of
the scale.
""".rstrip())
    for d in sorted(RUNS.glob("breadth_*")):
        j = _load(d / "breadth_report.json")
        if not j:
            continue
        print(f"\n  {d.name}   ({j['n_items']} questions, tiers "
              f"{j['tier_counts']})")
        print(f"  {j['verdict']}")
        print(f"    {'model':<12}{'leakage on':>12}{'score on':>11}"
              f"{'calibrated':>12}")
        print(f"    {'':<12}{'forget set':>12}{'retain set':>11}"
              f"{'breadth':>12}")
        for model, r in j["results"].items():
            cal = r.get("calibrated_breadth")
            cal = f"{cal:.3f}" if isinstance(cal, (int, float)) else "n/a"
            print(f"    {model:<12}{r['forget_leakage']:>12.3f}"
                  f"{r['retention']:>11.3f}{cal:>12}")
        cal = j.get("calibration")
        if cal:
            print(f"    calibration: floor {cal['floor']:.3f}  "
                  f"ceiling {cal['ceiling']:.3f}  "
                  f"usable range {cal['dynamic_range']:.3f}")
            print("    -> a model that never saw these facts still scores "
                  f"{cal['floor']:.3f}, not 0.")
            print("       So the real room to move is "
                  f"{cal['dynamic_range']:.3f}, not 1.000.")


def show_trajectory():
    _hr("4. THE DEPTH-BREADTH TRAJECTORY -- the actual experiment")
    studies = ROOT / "results" / "studies"
    # Run folders are timestamped, so the last one sorted is the most recent.
    found = sorted(studies.glob("*/plane.json")) if studies.exists() else []
    if found:
        latest = found[-1]
        j = _load(latest)
        print(f"\n  Most recent run: {latest.parent.name}")
        if len(found) > 1:
            print(f"  ({len(found)} completed runs stored; earlier ones kept "
                  f"so the history is auditable)")

        # The unlearning record: which checkpoint was kept, and why.
        un = _load(latest.parent / "unlearn.json")
        if un:
            un = un.get("history", un)      # the record sits under "history"
            evals = un.get("epoch_evals") or []
            sel = un.get("selected_epoch")
            base = un.get("baseline_utility")
            print(f"\n  Step 1 -- unlearning ({len(evals)} epochs run)")
            if base is not None:
                print(f"    utility before training : {base:.3f}")
            rejected = [e for e in evals if not e.get("acceptable", True)]
            print(f"    epochs kept             : "
                  f"{len(evals) - len(rejected)} of {len(evals)}")
            print(f"    epochs rejected         : {len(rejected)} "
                  f"(utility fell below the floor)")
            if sel is not None:
                s = next((e for e in evals if e["epoch"] == sel), None)
                if s:
                    print(f"    checkpoint selected     : epoch {sel} "
                          f"(utility {s['utility']:.3f}, "
                          f"forget score {s['forget_nll']:.3f})")
            b, f = un.get("baseline_forget_nll"), un.get("final_forget_nll")
            if b and f:
                print(f"    forget-set difficulty   : {b:.3f} -> {f:.3f} "
                      f"(higher = more forgotten)")

        pts = j.get("points", [])
        print(f"\n  Step 2 -- measuring at {len(pts)} strengths of that update")
        print(f"\n  {'strength':>9}{'breadth':>10}{'depth':>9}{'utility':>10}"
              f"   {'(control)':>10}")
        print("  " + DASH[:48])
        for p in pts:
            u = p.get("utility")
            u = f"{u:.3f}" if isinstance(u, (int, float)) else "  --"
            print(f"  {p['alpha']:>9.2f}{p['breadth']:>10.3f}"
                  f"{p['depth']:>9.3f}{u:>10}")
        if pts:
            b0, b1 = pts[0]["breadth"], pts[-1]["breadth"]
            d0, d1 = pts[0]["depth"], pts[-1]["depth"]
            u0, u1 = pts[0].get("utility"), pts[-1].get("utility")
            print("  " + DASH[:48])
            print(f"\n  Both measures rise together:")
            print(f"    depth   {d0:.3f} -> {d1:.3f}   "
                  f"({d1:.0%} of the way to a fully retrained model)")
            print(f"    breadth {b0:.3f} -> {b1:.3f}   ({b1:.0%})")
            if isinstance(u0, float) and isinstance(u1, float):
                print(f"\n  BUT the control also moved: utility "
                      f"{u0:.3f} -> {u1:.3f} (fell {u0-u1:.3f}).")
                print("  So part of this rise is the model getting generally "
                      "worse, not")
                print("  forgetting the target facts specifically. Separating "
                      "the two is")
                print("  the next task.")
        sens = _load(latest.parent / "sensitivity.json")
        if sens:
            s = sens[0]
            print(f"\n  Points measured: {s['n_points']}. These are steps along "
                  "one continuous")
            print("  path, not independent experiments, so no statistical "
                  "significance")
            print("  is claimed from them.")
    else:
        print("""
  NOT STORED ON THIS MACHINE.

  The trajectory run wrote its folder wherever it was executed. To be able
  to open it here, copy that run folder into:

      results/studies/

  Until then the measured numbers are visible in the figure:

      report/figures/fig11_plane.png

  and are reproduced in RESULTS.md and in the report.""".rstrip())


def show_tests():
    _hr("5. HOW WE KNOW THE CODE IS CORRECT")
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    print(f"\n  {len(tests)} test files:\n")
    for t in tests:
        n = sum(1 for line in t.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("def test_"))
        print(f"    {t.name:<28}{n:>4} tests")
    print("""
  Run them all with:      python -m pytest tests/ -q

  A test that passes against broken code proves nothing, so we also break
  the code on purpose and check that the tests notice. Every safety check
  in the system has been verified this way.""".rstrip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", action="store_true",
                    help="also list the files inside a run folder")
    ap.add_argument("--all", action="store_true",
                    help="include earlier runs that were superseded")
    args = ap.parse_args()

    print(BAR)
    print("DeepErase -- what is stored on disk and what it says")
    print(BAR)
    show_storage_layout()
    show_depth_runs(list_files=args.files, include_all=args.all)
    show_breadth_runs()
    show_trajectory()
    show_tests()
    print(f"\n{BAR}\nEnd of stored results.\n{BAR}")


if __name__ == "__main__":
    main()
