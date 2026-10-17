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


if __name__ == "__main__":
    fig12_depth_vs_damage()
    fig13_overshoot()
    fig14_memory_keys()
    print("wrote", sorted(p.name for p in OUT.glob("fig1[2-9]*.png")))
