"""Figures for the final report from the protocol-v2 results (no GPU).

Every number is read from saved run artefacts in results/. Palette: the first
three slots of the validated categorical palette (GA blue, GradDiff orange,
NPO aqua; all-pairs CVD-safe). Each matched-damage control uses its method's
hue, dashed. Every line carries a direct label, because aqua is below 3:1
contrast on white.

    python report/make_figures_v2.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deeperase.scripts.aggregate_matrix import load_run  # noqa: E402
from deeperase.scripts.uds_audit import audit_run  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STUDIES = ROOT / "results" / "studies"
GATES = ROOT / "results" / "gates"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e6e5e1"
COLOR = {"ga": "#2a78d6", "graddiff": "#eb6834", "npo": "#1baf7a"}
NAME = {"ga": "GA", "graddiff": "GradDiff", "npo": "NPO"}
LR = {"ga": "1e-08", "graddiff": "5e-08", "npo": "1e-08"}


def _grid(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _mean_path(runs, split="trained"):
    """Mean over runs of (utility, depth) at each alpha they share."""
    trajs = [load_run(r)["trajectories"][split] for r in runs]
    alphas = sorted(set.intersection(*[set(t["alpha"]) for t in trajs]))
    u = [sum(t["utility"][t["alpha"].index(a)] for t in trajs) / len(trajs) for a in alphas]
    d = [sum(t["depth"][t["alpha"].index(a)] for t in trajs) / len(trajs) for a in alphas]
    return u, d


def fig12_depth_vs_damage():
    """Depth against R utility: each method beside its matched-damage control."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, m in zip(axes, ("ga", "graddiff", "npo")):
        real = [STUDIES / f"study_v2_{m}_1B_s{s}_lr{LR[m]}" for s in (0, 1, 2)]
        ctl = sorted(STUDIES.glob(f"control_x*_v2_{m}_1B_s*_lr{LR[m]}"))
        ur, dr = _mean_path(real)
        uc, dc = _mean_path(ctl)
        ax.plot(ur, dr, "-o", color=COLOR[m], lw=2, ms=3.2, zorder=3)
        ax.plot(uc, dc, "--", color=COLOR[m], lw=1.6, alpha=0.8, zorder=2)
        ax.set_title(f"{NAME[m]}  (controls: {len(ctl)} seed{'s' if len(ctl) > 1 else ''})",
                     color=INK)
        ax.set_xlabel("utility (R tier knows-rate)")
        ax.invert_xaxis()
        _grid(ax)
    axes[0].set_ylabel("depth (UDS), trained facts")
    axes[0].set_ylim(0, 0.75)
    handles = [plt.Line2D([], [], color=INK2, lw=2, marker="o", ms=3.2,
                          label="unlearning method (mean of 3 seeds)"),
               plt.Line2D([], [], color=INK2, lw=1.6, ls="--",
                          label="random direction, matched damage")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    fig.suptitle("Random damage of the same utility cost reproduces most of the depth",
                 fontsize=10, color=INK, y=1.14)
    fig.savefig(OUT / "fig12_depth_vs_damage.png")
    plt.close(fig)


def fig13_overshoot():
    """UDS against strict UDS as the update is pushed to twice its size."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in ("ga", "graddiff"):
        a = audit_run(STUDIES / f"study_v2_{m}_1B_s0_lr{LR[m]}_ext")
        rows = a["splits"]["trained"]
        al = [r["alpha"] for r in rows]
        ax.plot(al, [r["uds"] for r in rows], "-", color=COLOR[m], lw=2)
        ax.plot(al, [r["strict"] for r in rows], "--", color=COLOR[m], lw=2)
        ax.annotate(f"{NAME[m]} UDS", (al[-1], rows[-1]["uds"]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, va="center", color=INK)
        ax.annotate(f"{NAME[m]} strict", (al[-1], rows[-1]["strict"]),
                    xytext=(4, 6 if m == "graddiff" else -6),
                    textcoords="offset points", fontsize=7.5, va="center", color=INK2)
    ax.set_xlabel("extrapolation α (1 = the trained update doubled)")
    ax.set_ylabel("depth, trained facts (seed 0)")
    ax.set_xlim(0, 2.45)
    ax.set_ylim(0, 0.85)
    _grid(ax)
    ax.set_title("Past α = 1, GA's rising UDS is all overshoot (strict UDS flat)",
                 fontsize=9.5, color=INK)
    fig.savefig(OUT / "fig13_overshoot.png")
    plt.close(fig)


def fig14_memory_keys():
    """What M_full's memory of a fact is keyed on: gates 1 and 1b, trained facts."""
    g1 = json.loads((GATES / "stranger_g1_s0.json").read_text())["splits"]["trained"]
    g1b = json.loads((GATES / "memory_keys_s0.json").read_text())["splits"]["trained"]
    rows = [("STRANGER's optimum (name state, layers 4-8)", g1["hybrid"]["all"]),
            ("name swapped everywhere", g1b["name"]["whole"]),
            ("question blanked", g1b["question"]["whole"]),
            ("question blanked + name swapped", g1b["question_name"]["whole"]),
            ("another author's question", g1b["other_question"]["whole"]),
            ("everything before the entity blanked", g1b["context"]["whole"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for i, (label, d) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([d["strict"], d["uds"]], [y, y], "-", color=MUTED, lw=1.2, zorder=1)
        ax.plot(d["uds"], y, "o", color=COLOR["ga"], ms=7, zorder=3,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.plot(d["strict"], y, "o", color="#fcfcfb", ms=7, zorder=3,
                markeredgecolor=COLOR["ga"], markeredgewidth=1.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8, color=INK)
    ax.axvline(0.50, color=INK2, lw=0.8, ls=":")
    ax.text(0.505, len(rows) - 0.55, "bar: UDS 0.50", fontsize=7, color=INK2)
    ax.axvline(0.22, color=MUTED, lw=0.8, ls="-.")
    ax.text(0.225, -0.75, "GA checkpoint (0.22)", fontsize=7, color=INK2)
    ax.plot([], [], "o", color=COLOR["ga"], label="UDS")
    ax.plot([], [], "o", color="#fcfcfb", markeredgecolor=COLOR["ga"], label="strict UDS")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-1.0, len(rows) - 0.3)
    ax.set_xlabel("depth if a method matched this state exactly (seed 0, trained facts)")
    _grid(ax)
    ax.set_title("The memory is keyed on the question with the name, not the name alone",
                 fontsize=9.5, color=INK)
    fig.savefig(OUT / "fig14_memory_keys.png")
    plt.close(fig)


def fig15_partial_vs_depth():
    """Share of half-erased facts against mean depth, at every alpha of every run."""
    runs = json.loads((ROOT / "results/analysis/fact_level_v2.json").read_text())["runs"]
    runs += [r for r in json.loads((ROOT / "results/analysis/fact_level_v2_exploratory.json")
                                   .read_text())["runs"] if r.get("arm") == "control"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for r in runs:
        m = r["method"]
        rows = r["splits"]["trained"]["per_alpha"]
        x = [row["depth_mean"] for row in rows]
        y = [row["share_partial_depth"] for row in rows]
        ctl = r.get("arm") == "control"
        ax.plot(x, y, "-", color=COLOR[m], lw=0.6, alpha=0.35, zorder=1)
        ax.plot(x, y, "o", ms=3.6, zorder=2, color=COLOR[m] if not ctl else "#fcfcfb",
                markeredgecolor=COLOR[m], markeredgewidth=1.1 if ctl else 0.6)
    for m in ("ga", "graddiff", "npo"):
        ax.plot([], [], "o", color=COLOR[m], label=f"{NAME[m]}: unlearning")
        ax.plot([], [], "o", color="#fcfcfb", markeredgecolor=COLOR[m],
                label=f"{NAME[m]}: matched damage")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax.set_xlabel(r"mean per-fact depth (UDS$_i$), trained facts")
    ax.set_ylabel(r"share half-erased (0.2 $\leq$ UDS$_i$ < 0.8)")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.95)
    _grid(ax)
    ax.set_title("At equal mean depth, damage half-erases more facts than unlearning",
                 fontsize=9.5, color=INK)
    fig.savefig(OUT / "fig15_partial_vs_depth.png")
    plt.close(fig)


UNASKED_COLOR = "#6250d6"   # violet: the emphasis hue; baselines go gray in fig16
GRAY = "#a3a29e"


def _strict_path(runs, split="trained"):
    """Mean over runs of (general knowledge, strict UDS) at each alpha they share."""
    per = {}
    for d in runs:
        t = load_run(d)["trajectories"][split]
        for r in audit_run(d)["splits"][split]:
            if r["alpha"] in t["alpha"]:
                i = t["alpha"].index(r["alpha"])
                per.setdefault(r["alpha"], []).append((t["general_utility"][i], r["strict"]))
    shared = [a for a, v in sorted(per.items()) if len(v) == len(runs)]
    return ([sum(x[0] for x in per[a]) / len(runs) for a in shared],
            [sum(x[1] for x in per[a]) / len(runs) for a in shared])


def _depth_vs_general(method: str, label: str, fname: str, title: str,
                      label_at=(-6, 9, "right"), config: str = ""):
    """Strict depth against general knowledge: one of our methods beside the baselines' paths."""
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    marks = {"ga": "o", "npo": "s", "graddiff": "^"}
    for m in ("ga", "npo", "graddiff"):
        x, y = _strict_path([STUDIES / f"study_v2_{m}_1B_s{s}_lr{LR[m]}" for s in (0, 1, 2)])
        ax.plot(x, y, "-", color=GRAY, lw=1.6, zorder=2)
        ax.plot(x[0], y[0], marks[m], color=GRAY, ms=5.5, zorder=3,
                markeredgecolor="#fcfcfb", markeredgewidth=1.0)
        ax.annotate(f"{NAME[m]}", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, va="center", color=INK2)
    runs = sorted(d for d in STUDIES.glob(f"study_v2_{method}_1B_s*_lr*{config}")
                  if d.is_dir() and d.name.split("_lr", 1)[1].count("_") == (1 if config else 0))
    x, y = _strict_path(runs)
    ax.plot(x, y, "-", color=UNASKED_COLOR, lw=2, zorder=4)
    ax.plot(x[0], y[0], "D", color=UNASKED_COLOR, ms=6, zorder=5,
            markeredgecolor="#fcfcfb", markeredgewidth=1.0)
    n = len(runs)
    ax.annotate(f"{label} ({n} seed{'s' if n > 1 else ''})", (x[0], y[0]),
                xytext=label_at[:2], textcoords="offset points", fontsize=7.5,
                ha=label_at[2], color=INK)
    ax.annotate("α = 1", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=7, va="center", color=INK2)
    extra = []
    # every control run of a seed shares one random direction: one damage path,
    # ordered by distance from M_full (scale x (1 + alpha))
    paths = {}
    for d in sorted(STUDIES.glob(f"control_x*_v2_{method}_1B_s*_lr*{config}")):
        if d.name.split("_lr", 1)[1].count("_") != (1 if config else 0):
            continue
        if not d.is_dir():
            continue
        cfg = json.loads((d / "config.json").read_text())
        t = load_run(d)["trajectories"]["trained"]
        for r in audit_run(d)["splits"]["trained"]:
            if r["alpha"] in t["alpha"]:
                i = t["alpha"].index(r["alpha"])
                paths.setdefault(cfg["seed"], []).append(
                    (cfg["control_scale"] * (1 + r["alpha"]), t["general_utility"][i], r["strict"]))
    for seed, pts in sorted(paths.items()):
        pts.sort()
        ax.plot([p[1] for p in pts], [p[2] for p in pts], "--", color=UNASKED_COLOR, lw=1.6,
                alpha=0.85, zorder=3)
    if paths:
        far = max((p for pts in paths.values() for p in pts), key=lambda p: p[0])
        ax.annotate("random direction", (far[1], far[2]), xytext=(-6, 6),
                    textcoords="offset points", fontsize=7, ha="right", color=INK2)
        n = len(paths)
        extra.append((plt.Line2D([], [], color=UNASKED_COLOR, lw=1.6, ls="--"),
                      f"random direction, {label}'s controls ({n} seed{'s' if n > 1 else ''})"))
    lam = sorted(STUDIES.glob(f"study_v2_{method}_1B_s*_lr*_lam3"))
    if lam:
        lx, ly = _strict_path(lam)
        ax.plot(lx[0], ly[0], "D", color="#fcfcfb", ms=6, zorder=5,
                markeredgecolor=UNASKED_COLOR, markeredgewidth=1.4)
        ax.annotate("λ = 3", (lx[0], ly[0]), xytext=(6, -8), textcoords="offset points",
                    fontsize=7, color=INK2)
        extra.append((plt.Line2D([], [], ls="", marker="D", color="#fcfcfb", ms=5,
                                 markeredgecolor=UNASKED_COLOR, markeredgewidth=1.4),
                      f"UNASKED, retain weight λ = 3 ({len(lam)} seed{'s' if len(lam) > 1 else ''})"))
    ax.plot(0.8725, 0.0, "*", color=INK2, ms=8, zorder=5, clip_on=False)
    ax.annotate("M_full", (0.8725, 0.0), xytext=(0, 7), textcoords="offset points",
                fontsize=7, ha="center", color=INK2)
    from matplotlib.legend_handler import HandlerTuple
    h_un = plt.Line2D([], [], ls="", marker="D", color=UNASKED_COLOR, ms=5)
    h_base = tuple(plt.Line2D([], [], ls="", marker=marks[m], color=GRAY, ms=5)
                   for m in ("ga", "npo", "graddiff"))
    h_path = plt.Line2D([], [], color=GRAY, lw=1.6)
    ax.legend([h_un, h_base, h_path] + [h for h, _ in extra],
              [f"{label} checkpoint (α = 0)", "GA, NPO, GradDiff checkpoints",
               "baselines, α 0 → 1 (mean of 3 seeds)"] + [t for _, t in extra],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
              frameon=False, fontsize=7, loc="lower right")
    ax.set_xlabel("general knowledge (mean of RA and WF knows-rate)")
    ax.set_ylabel("strict UDS, trained facts")
    ax.invert_xaxis()
    ax.set_ylim(0, 0.62)
    _grid(ax)
    ax.set_title(title, fontsize=9.5, color=INK)
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig16_unasked_depth_vs_general():
    _depth_vs_general("unasked", "UNASKED", "fig16_unasked_depth_vs_general.png",
                      "Strict depth against general-knowledge cost: checkpoints and α paths")


def fig20_unaskedg_lam10_depth_vs_general():
    _depth_vs_general("unaskedg", "UNASKED-G, anchor 10", "fig20_unaskedg_lam10_depth_vs_general.png",
                      "UNASKED-G (TOFU anchor 10): above every baseline and its own damage control",
                      label_at=(8, -11, "left"), config="_lam10")


def fig19_unaskedg_depth_vs_general():
    _depth_vs_general("unaskedg", "UNASKED-G", "fig19_unaskedg_depth_vs_general.png",
                      "UNASKED-G: deeper than every baseline at every general-knowledge cost",
                      label_at=(8, -11, "left"))


def fig17_relearn_dose():
    """Latent index against attack strength: UNASKED-G beside the baselines (mean over seeds)."""
    rows = json.loads((ROOT / "results/analysis/relearn.json").read_text())["rows"]
    dose = {}
    for r in rows:
        if "_dose" in r["record"]:
            dose.setdefault(r["method"], {}).setdefault(r["seed"], {})[r["attack_lr"]] = r
    if not dose:
        print("fig17: no dose-response records yet")
        return
    style = {"unaskedg": (UNASKED_COLOR, "D", "UNASKED-G (anchor 10)"),
             "unasked": (UNASKED_COLOR, "D", "UNASKED"), "ga": (GRAY, "o", "GA"),
             "npo": (GRAY, "s", "NPO"), "graddiff": (GRAY, "^", "GradDiff")}
    none_x = 3e-8          # where "no attack" sits on the log axis
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, probe, title in zip(axes, ("probe", "heldout"),
                                ("facts never shown to the attacker", "held-out facts")):
        for m, seeds in sorted(dose.items(), key=lambda kv: kv[0].startswith("unasked")):
            color, mark, name = style.get(m, (GRAY, "o", m))
            rates = sorted(set.intersection(*[set(v) for v in seeds.values()]))
            first = [seeds[s][rates[0]][probe].get("index_nll_before") for s in seeds]
            per = [[seeds[s][lr][probe].get("index_nll") for s in seeds] for lr in rates]
            x = [none_x] + rates
            y = [sum(first) / len(first)] + [sum(v) / len(v) for v in per]
            lo = [min(first)] + [min(v) for v in per]
            hi = [max(first)] + [max(v) for v in per]
            ours = m.startswith("unasked")
            lw = 2 if ours else 1.4
            if len(seeds) > 1:
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
            ax.plot(x[:2], y[:2], ":", color=color, lw=lw * 0.6, zorder=3)
            ax.plot(x[1:], y[1:], "-", color=color, lw=lw, zorder=3)
            ax.plot(x, y, mark, color=color, ms=4.5, zorder=4,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.8)
            n = len(seeds)
            label = f"{name} ({n} seed{'s' if n > 1 else ''})"
            if ours:
                ax.annotate(label, (x[2], y[2]), xytext=(6, -14), textcoords="offset points",
                            fontsize=7.5, color=INK)
        for yref, label in ((0.0, "retain90 (never knew)"), (1.0, "M_full (knew)")):
            ax.axhline(yref, color=MUTED, lw=0.8, ls=":")
            ax.text(none_x, yref + 0.03, label, fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.minorticks_off()
        ticks = [none_x, 1e-7, 1e-6, 1e-5]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["none", "1e-7", "1e-6", "1e-5"])
        ax.set_xlim(none_x / 1.6, 3e-5)
        ax.set_xlabel("attack learning rate (2 epochs, 100 facts)")
        ax.set_title(title, fontsize=9, color=INK)
        _grid(ax)
    handles = [plt.Line2D([], [], color=GRAY, lw=1.4, marker=style[m][1], ms=4.5, label=style[m][2])
               for m in ("graddiff", "npo", "ga") if m in dose]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower right",
                   title="baselines (grey)", title_fontsize=7)
    axes[0].set_ylabel("latent index (answer NLL)")
    axes[0].set_ylim(-0.1, 1.15)
    fig.suptitle("How much forgotten knowledge a relearning attack brings back "
                 "(line: mean over seeds; band: seed range)", fontsize=9.5, color=INK, y=1.03)
    fig.savefig(OUT / "fig17_relearn_dose.png")
    plt.close(fig)


def fig18_h9():
    """UNASKED-G against GradDiff at the checkpoint: the five H9 measures, per seed."""
    from deeperase.scripts.h9_report import collect, C1_SEED, C2_SEED, C3_MEAN
    rep = collect(STUDIES)
    rows = [("GradDiff", rep["graddiff"], GRAY, "o", True)]
    if rep.get("graddiff_g"):
        rows.append(("GradDiff + general anchor", rep["graddiff_g"], GRAY, "o", False))
    main = rep["groups"].get("lr3e-06")
    if main:
        rows.append(("UNASKED-G, TOFU anchor 3 (H9)", main["runs"], UNASKED_COLOR, "D", False))
    cited = rep["groups"].get("lr3e-06_lam10")
    if cited:
        rows.append(("UNASKED-G, TOFU anchor 10 (H9b)", cited["runs"], UNASKED_COLOR, "D", True))
    measures = [("strict_trained", "strict depth (trained)", C1_SEED),
                ("breadth_trained", "breadth (trained)", C2_SEED),
                ("R", "R (TOFU retain)", C3_MEAN["R"]),
                ("RA", "RA (real authors)", C3_MEAN["RA"]),
                ("WF", "WF (world facts)", C3_MEAN["WF"])]
    fig, axes = plt.subplots(1, len(measures), figsize=(9.6, 1.25 + 0.42 * len(rows)),
                             sharey=True)
    for ax, (key, title, thr) in zip(axes, measures):
        for y, (label, runs, color, mark, filled) in enumerate(rows):
            vals = [r.get(key) for r in runs if r.get(key) is not None]
            for v in vals:
                ax.plot(v, y, mark, ms=5.5, zorder=3,
                        color=color if filled else "#fcfcfb",
                        markeredgecolor=color if not filled else "#fcfcfb",
                        markeredgewidth=1.2 if not filled else 0.8)
            if len(vals) > 1:
                m = sum(vals) / len(vals)
                ax.plot([m, m], [y - 0.28, y + 0.28], "-", color=INK, lw=1.4, zorder=4)
        ax.axvline(thr, color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.tick_params(axis="x", labelsize=7)
        _grid(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].invert_yaxis()
    fig.suptitle("UNASKED-G against GradDiff at the checkpoint (α = 0)", fontsize=10,
                 color=INK, y=1.0)
    fig.text(0.5, -0.04, "dots: seeds; bar: mean over seeds; dashed: the H9 threshold "
             "(C1 depth and C2 breadth per seed; C3 utility on the mean)", ha="center",
             fontsize=7.5, color=INK2)
    fig.subplots_adjust(top=0.74, wspace=0.18)
    fig.savefig(OUT / "fig18_h9.png")
    plt.close(fig)


if __name__ == "__main__":
    fig12_depth_vs_damage()
    fig13_overshoot()
    fig14_memory_keys()
    fig15_partial_vs_depth()
    fig16_unasked_depth_vs_general()
    fig17_relearn_dose()
    fig18_h9()
    fig19_unaskedg_depth_vs_general()
    fig20_unaskedg_lam10_depth_vs_general()
    print("wrote", sorted(p.name for p in OUT.glob("fig1[2-9]*.png")))
