"""Generate every figure used in the mid-semester technical report.

All quantitative figures are drawn from measured values recorded in the run
artefacts; none are illustrative sketches. Schematic figures (architecture,
UML) are drawn to scale-free box layouts.

Run:  python report/make_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

INK = "#1a1a1a"
ACCENT = "#1f4e79"
ACCENT2 = "#a33a1f"
MUTED = "#6b6b6b"
FILL = "#eef2f7"
FILL2 = "#fbeee9"


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fill=FILL, edge=ACCENT, fontsize=8, bold=False,
        radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0.9, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, style="-|>", color=ACCENT, lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=9, linewidth=lw,
        color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def blank_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


# --------------------------------------------------------------------------
# Figure 1 -- system architecture
# --------------------------------------------------------------------------
def fig_architecture():
    fig, ax = blank_axes((7.0, 5.0))

    box(ax, 0.2, 8.6, 9.6, 1.1,
        "INPUT LAYER\nTOFU benchmark (200 fictitious authors, 4000 QA pairs)   ·   "
        "TOFU-tuned Llama-3.2 checkpoints (full / retain90 / retain95 / retain99)",
        fill="#f2f2f2", edge=MUTED, fontsize=7.5)

    box(ax, 0.2, 6.6, 3.0, 1.6,
        "DATA PREPARATION\ndeeperase.data\n\n· split loading & even sampling\n"
        "· entity-span location\n· reference annotations", fontsize=7)
    box(ax, 3.5, 6.6, 3.0, 1.6,
        "UNLEARNING\ndeeperase.unlearn\n\n· GA / GradDiff / NPO\n"
        "· per-epoch evaluation\n· checkpoint selection", fontsize=7)
    box(ax, 6.8, 6.6, 3.0, 1.6,
        "EXTRAPOLATION\ndeeperase.core\n\n· update vector v = θun − θini\n"
        "· θ(α) = θun + α·v\n· buffer exclusion", fontsize=7)

    box(ax, 0.2, 4.0, 4.6, 2.2,
        "DEPTH AXIS  —  deeperase.eval.uds\n\n"
        "Two-stage activation patching\n"
        "Stage 1: ΔS₁ per layer (oracle → target)\n"
        "Stage 2: ΔS₂ per layer (candidate → target)\n"
        "KE-layer selection at τ = 0.05\n"
        "Layer Erasure Ratio = clip(ΔS₂/ΔS₁, 0, 1)\n"
        "ΔS₁-weighted aggregation → UDS",
        fill=FILL, edge=ACCENT, fontsize=7)
    box(ax, 5.2, 4.0, 4.6, 2.2,
        "BREADTH AXIS  —  deeperase.eval.breadth\n\n"
        "Forced-choice scoring over tiers\n"
        "B0 exact  ·  B1 paraphrase  ·  R retain control\n"
        "leakage = P(correct ranked first)\n"
        "calibrated against reference models\n"
        "floor = retain90, ceiling = full\n"
        "breadth = (ceiling − leakage) / range",
        fill=FILL2, edge=ACCENT2, fontsize=7)

    box(ax, 1.6, 2.0, 6.8, 1.5,
        "ANALYSIS  —  deeperase.eval.plane\n\n"
        "trajectory assembly · utility-collapse control · dynamic-range gate · "
        "rank correlation · plane plot", fontsize=7.5)

    box(ax, 1.6, 0.3, 6.8, 1.1,
        "OUTPUT\nplane.json · report.json · sensitivity.json · plane.png · "
        "reproducible run directory", fill="#f2f2f2", edge=MUTED, fontsize=7.5)

    for x in (1.7, 5.0, 8.3):
        arrow(ax, (x, 8.6), (x, 8.2))
    arrow(ax, (2.5, 6.6), (2.5, 6.2))
    arrow(ax, (8.3, 6.6), (7.5, 6.2))
    arrow(ax, (5.0, 6.6), (4.0, 6.2))
    arrow(ax, (2.5, 4.0), (3.5, 3.5))
    arrow(ax, (7.5, 4.0), (6.5, 3.5))
    arrow(ax, (5.0, 2.0), (5.0, 1.4))

    return save(fig, "fig01_architecture")


# --------------------------------------------------------------------------
# Figure 2 -- two-stage activation patching
# --------------------------------------------------------------------------
def fig_patching():
    fig, ax = blank_axes((7.0, 4.2))

    ax.text(5.0, 9.6, "Stage 1 — calibrate: how much does the layer carry?",
            ha="center", fontsize=8.5, fontweight="bold", color=ACCENT)
    box(ax, 0.4, 7.4, 2.4, 1.5, "SOURCE\nM_retain90\n(never saw the fact)",
        fontsize=7)
    box(ax, 3.8, 7.4, 2.4, 1.5, "patch layer ℓ\nhidden states\nat entity span",
        fill="white", edge=MUTED, fontsize=7)
    box(ax, 7.2, 7.4, 2.4, 1.5, "TARGET\nM_full\n(knows the fact)", fontsize=7)
    arrow(ax, (2.8, 8.15), (3.8, 8.15))
    arrow(ax, (6.2, 8.15), (7.2, 8.15))
    ax.text(5.0, 6.9, "ΔS₁(ℓ) = drop in log-probability of the entity",
            ha="center", fontsize=7.5, style="italic", color=MUTED)

    ax.text(5.0, 5.9, "Stage 2 — measure: how much did unlearning remove?",
            ha="center", fontsize=8.5, fontweight="bold", color=ACCENT2)
    box(ax, 0.4, 3.7, 2.4, 1.5, "SOURCE\nM_unlearned\n(candidate)",
        fill=FILL2, edge=ACCENT2, fontsize=7)
    box(ax, 3.8, 3.7, 2.4, 1.5, "patch layer ℓ\nhidden states\nat entity span",
        fill="white", edge=MUTED, fontsize=7)
    box(ax, 7.2, 3.7, 2.4, 1.5, "TARGET\nM_full\n(knows the fact)", fontsize=7)
    arrow(ax, (2.8, 4.45), (3.8, 4.45), color=ACCENT2)
    arrow(ax, (6.2, 4.45), (7.2, 4.45), color=ACCENT2)
    ax.text(5.0, 3.2, "ΔS₂(ℓ) = drop in log-probability of the entity",
            ha="center", fontsize=7.5, style="italic", color=MUTED)

    box(ax, 0.9, 0.4, 8.2, 2.3,
        "Knowledge-Encoding (KE) layers:  keep ℓ where ΔS₁(ℓ) > τ,   τ = 0.05 nats\n\n"
        "Layer Erasure Ratio:   LER(ℓ) = clip( ΔS₂(ℓ) / ΔS₁(ℓ), 0, 1 )\n\n"
        "UDS = Σ ΔS₁(ℓ)·LER(ℓ) / Σ ΔS₁(ℓ)     over KE layers, ΔS₁-weighted\n\n"
        "UDS = 0 → knowledge still internally present    UDS = 1 → erased as thoroughly as retraining",
        fill="#f7f7f7", edge=INK, fontsize=7.5)

    return save(fig, "fig02_patching")


# --------------------------------------------------------------------------
# Figure 3 -- breadth tiers
# --------------------------------------------------------------------------
def fig_breadth_tiers():
    fig, ax = blank_axes((7.0, 3.6))

    ax.text(5.0, 9.5, "One fact, probed at increasing semantic distance",
            ha="center", fontsize=9, fontweight="bold")

    rows = [
        ("B0", "Exact", "\"What is the author's full name?\"",
         "the form the model was trained on", FILL, ACCENT),
        ("B1", "Paraphrase", "\"By what name is this writer known?\"",
         "same fact, different wording", FILL, ACCENT),
        ("R", "Retain control", "questions about authors never in the forget set",
         "must NOT change — detects collateral damage", FILL2, ACCENT2),
    ]
    y = 7.3
    for tag, name, example, note, fill, edge in rows:
        box(ax, 0.3, y, 1.0, 1.5, tag, fill=fill, edge=edge, bold=True, fontsize=10)
        box(ax, 1.5, y, 8.2, 1.5,
            f"{name}\n{example}\n{note}", fill="white", edge=edge, fontsize=7.5)
        y -= 1.9

    box(ax, 0.3, 0.3, 9.4, 1.3,
        "Scoring: the model ranks the correct answer against TOFU's perturbed alternatives.\n"
        "leakage = fraction ranked first.  Calibrated:  breadth = (ceiling − leakage) / (ceiling − floor)",
        fill="#f7f7f7", edge=INK, fontsize=7.5)

    ax.text(9.9, 6.0, "increasing\ndistance", rotation=90, ha="center",
            va="center", fontsize=7, color=MUTED, style="italic")

    return save(fig, "fig03_breadth_tiers")


# --------------------------------------------------------------------------
# Figure 4 -- UML use case
# --------------------------------------------------------------------------
def fig_usecase():
    from matplotlib.patches import Ellipse
    fig, ax = blank_axes((6.6, 4.6))

    ax.add_patch(Rectangle((2.3, 0.6), 5.4, 8.7, fill=False,
                           edgecolor=INK, linewidth=1.0))
    ax.text(5.0, 9.0, "DeepErase measurement system", ha="center",
            fontsize=8.5, fontweight="bold")

    cases = [
        (8.1, "Validate depth metric\nagainst published values"),
        (6.9, "Calibrate breadth axis\nagainst reference models"),
        (5.7, "Unlearn a model\n(GA / GradDiff / NPO)"),
        (4.5, "Sweep extrapolation\nstrength α"),
        (3.3, "Measure depth–breadth\ntrajectory"),
        (2.1, "Export reproducible\nrun artefacts"),
        (1.1, "Verify install\nand memory plan"),
    ]
    for y, label in cases:
        ax.add_patch(Ellipse((5.0, y), 4.2, 0.95, facecolor=FILL,
                             edgecolor=ACCENT, linewidth=0.9))
        ax.text(5.0, y, label, ha="center", va="center", fontsize=6.8)

    def actor(x, y, name):
        ax.plot([x], [y + 0.32], marker="o", ms=5, mfc="white",
                mec=INK, mew=0.9)
        ax.plot([x, x], [y + 0.22, y - 0.15], color=INK, lw=0.9)
        ax.plot([x - 0.28, x + 0.28], [y + 0.10, y + 0.10], color=INK, lw=0.9)
        ax.plot([x, x - 0.22], [y - 0.15, y - 0.5], color=INK, lw=0.9)
        ax.plot([x, x + 0.22], [y - 0.15, y - 0.5], color=INK, lw=0.9)
        ax.text(x, y - 0.85, name, ha="center", fontsize=7)

    actor(1.0, 6.6, "Researcher")
    actor(9.0, 4.2, "Evaluation\npanel")

    for y in (8.1, 6.9, 5.7, 4.5, 3.3, 1.1):
        arrow(ax, (1.35, 6.5), (2.9, y), style="-", color=MUTED, lw=0.6)
    for y in (8.1, 3.3, 2.1):
        arrow(ax, (8.65, 4.2), (7.1, y), style="-", color=MUTED, lw=0.6)

    return save(fig, "fig04_usecase")


# --------------------------------------------------------------------------
# Figure 5 -- UML class / module diagram
# --------------------------------------------------------------------------
def fig_class():
    fig, ax = blank_axes((7.2, 5.2))

    def cls(x, y, w, h, name, members):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white",
                               edgecolor=ACCENT, linewidth=0.9))
        ax.add_patch(Rectangle((x, y + h - 0.42), w, 0.42, facecolor=FILL,
                               edgecolor=ACCENT, linewidth=0.9))
        ax.text(x + w / 2, y + h - 0.21, name, ha="center", va="center",
                fontsize=7.3, fontweight="bold")
        ax.text(x + 0.1, y + h - 0.55, members, ha="left", va="top",
                fontsize=6.4, linespacing=1.5)

    cls(0.2, 7.2, 3.0, 2.5, "RunConfig",
        "+ size_label\n+ strategy\n+ max_seq_length\n+ models()\n+ validate()")
    cls(3.5, 7.2, 3.0, 2.5, "ModelManager",
        "+ load(role)\n+ free(role)\n+ memory_snapshot()\n+ ALL_RESIDENT\n+ SEQUENTIAL")
    cls(6.8, 7.2, 3.0, 2.5, "MemoryPlan",
        "+ fits\n+ peak_weight_gb\n+ headroom_gb\n+ warnings\n+ summary()")

    cls(0.2, 4.0, 3.0, 2.6, "UDSExample",
        "+ input_ids\n+ entity_span\n+ prompt_format\n\nUDS pipeline:\n"
        "build_stage1_cache()\ncompute_uds()")
    cls(3.5, 4.0, 3.0, 2.6, "UnlearnConfig",
        "+ method\n+ learning_rate\n+ epochs\n+ min_utility_ratio\n\n"
        "unlearn(model, data)\n→ UnlearnHistory")
    cls(6.8, 4.0, 3.0, 2.6, "BreadthCalibration",
        "+ floor\n+ ceiling\n+ dynamic_range\n+ calibrated_breadth()\n"
        "from_reference_models()")

    cls(0.2, 1.4, 3.0, 2.0, "UnlearnHistory",
        "+ epoch_evals[]\n+ baseline_forget_nll\n+ final_forget_nll\n"
        "+ forget_nll_change")
    cls(3.5, 1.4, 3.0, 2.0, "PlanePoint",
        "+ alpha\n+ breadth\n+ depth\n+ utility\n+ depth_overshoot")
    cls(6.8, 1.4, 3.0, 2.0, "Trajectory",
        "+ tradeoff_correlation()\n+ axis_spans()\n+ has_usable_\n  dynamic_range()\n"
        "+ utility_is_stable()")

    arrow(ax, (1.7, 7.2), (1.7, 6.6), style="-|>", color=MUTED)
    arrow(ax, (5.0, 7.2), (5.0, 6.6), style="-|>", color=MUTED)
    arrow(ax, (8.3, 7.2), (8.3, 6.6), style="-|>", color=MUTED)
    arrow(ax, (5.0, 4.0), (5.0, 3.4), style="-|>", color=MUTED)
    arrow(ax, (5.0, 4.0), (1.7, 3.4), style="-|>", color=MUTED)
    arrow(ax, (6.5, 2.4), (6.8, 2.4), style="-|>", color=MUTED)
    ax.text(5.15, 3.7, "produces", fontsize=6, color=MUTED, style="italic")
    ax.text(6.5, 2.55, "1..*", fontsize=6, color=MUTED)

    return save(fig, "fig05_class")


# --------------------------------------------------------------------------
# Figure 6 -- UML sequence diagram
# --------------------------------------------------------------------------
def fig_sequence():
    fig, ax = blank_axes((7.2, 5.4))

    actors = [
        (0.9, "run_study"), (2.9, "ModelManager"), (4.9, "unlearn"),
        (6.9, "extrapolate"), (8.9, "uds /\nbreadth"),
    ]
    for x, name in actors:
        box(ax, x - 0.85, 9.1, 1.7, 0.7, name, fill=FILL, edge=ACCENT,
            fontsize=6.8, bold=True)
        ax.plot([x, x], [9.1, 0.4], color=MUTED, lw=0.6, ls=(0, (3, 3)))

    steps = [
        (8.5, 0.9, 2.9, "load(full, retain90)"),
        (8.0, 0.9, 8.9, "measure reference models → calibration"),
        (7.5, 0.9, 8.9, "Stage-1 cache (retain90 → full)"),
        (7.0, 0.9, 2.9, "free(reference models)"),
        (6.5, 0.9, 4.9, "unlearn(forget10, GA)"),
        (6.0, 4.9, 4.9, "loop: epoch → eval utility + forget NLL"),
        (5.5, 4.9, 4.9, "select best acceptable checkpoint"),
        (5.0, 4.9, 0.9, "UnlearnHistory, θun"),
        (4.5, 0.9, 6.9, "v = θun − θini"),
        (4.0, 0.9, 6.9, "loop α from 0 to 1:  θ(α) = θun + α·v"),
        (3.5, 6.9, 8.9, "score θ(α)"),
        (3.0, 8.9, 0.9, "depth, breadth, retention"),
        (2.3, 0.9, 0.9, "assemble trajectory, gate on dynamic range"),
        (1.8, 0.9, 0.9, "write plane.json / plane.png"),
    ]
    for y, a, b, label in steps:
        if a == b:
            ax.add_patch(FancyArrowPatch(
                (a, y + 0.18), (a, y - 0.12), arrowstyle="-|>",
                mutation_scale=8, linewidth=0.8, color=ACCENT2,
                connectionstyle="arc3,rad=-2.2", shrinkA=0, shrinkB=0))
            ax.text(a + 0.55, y + 0.05, label, fontsize=6.2, va="center")
        else:
            arrow(ax, (a, y), (b, y), color=ACCENT)
            ax.text((a + b) / 2, y + 0.13, label, fontsize=6.2,
                    ha="center", va="bottom")

    return save(fig, "fig06_sequence")


# --------------------------------------------------------------------------
# Figure 7 -- UML state chart
# --------------------------------------------------------------------------
def fig_statechart():
    fig, ax = blank_axes((7.0, 4.4))

    ax.plot([1.0], [8.6], marker="o", ms=8, color=INK)
    box(ax, 2.2, 8.1, 2.6, 1.0, "Pretrained\nθ_ini", fontsize=7.5)
    box(ax, 6.0, 8.1, 3.2, 1.0, "Training epoch t\nθ_t", fontsize=7.5)
    box(ax, 6.0, 5.8, 3.2, 1.2, "Evaluated\nutility U_t, forget NLL F_t",
        fontsize=7.5)
    box(ax, 2.0, 5.8, 3.0, 1.2, "Rejected\nU_t < 0.9·U₀",
        fill=FILL2, edge=ACCENT2, fontsize=7.5)
    box(ax, 6.0, 3.5, 3.2, 1.2, "Accepted candidate\nbest so far", fontsize=7.5)
    box(ax, 1.4, 3.5, 3.6, 1.2, "Selected θ_un\n(highest F, U above floor)",
        fontsize=7.5)
    box(ax, 1.4, 1.2, 3.6, 1.2, "Extrapolated θ(α)\nθ_un + α·v",
        fontsize=7.5)
    box(ax, 6.0, 1.2, 3.2, 1.2, "Measured\ndepth, breadth, utility", fontsize=7.5)
    ax.plot([9.6], [1.8], marker="o", ms=8, color=INK, mfc="white", mew=1.6)

    arrow(ax, (1.25, 8.6), (2.2, 8.6))
    arrow(ax, (4.8, 8.6), (6.0, 8.6))
    ax.text(5.4, 8.75, "train", fontsize=6.3, ha="center", color=MUTED)
    arrow(ax, (7.6, 8.1), (7.6, 7.0))
    ax.text(7.75, 7.5, "epoch end", fontsize=6.3, color=MUTED)
    arrow(ax, (6.0, 6.4), (5.0, 6.4), color=ACCENT2)
    ax.text(5.5, 6.55, "U too low", fontsize=6.3, ha="center", color=ACCENT2)
    arrow(ax, (7.6, 5.8), (7.6, 4.7))
    ax.text(7.75, 5.2, "U ≥ floor", fontsize=6.3, color=MUTED)
    arrow(ax, (6.9, 7.0), (6.9, 8.1), rad=0.45, color=MUTED)
    ax.text(5.9, 7.55, "t < T", fontsize=6.3, color=MUTED)
    arrow(ax, (6.0, 4.1), (5.0, 4.1))
    ax.text(5.5, 4.25, "t = T", fontsize=6.3, ha="center", color=MUTED)
    arrow(ax, (3.2, 3.5), (3.2, 2.4))
    arrow(ax, (5.0, 1.8), (6.0, 1.8))
    arrow(ax, (9.2, 1.8), (9.45, 1.8))
    arrow(ax, (8.0, 2.4), (8.0, 3.5), rad=-0.5, color=MUTED, ls="--")
    ax.text(8.6, 2.95, "next α", fontsize=6.3, color=MUTED)
    # Total collapse is a terminal abort, not a path to a usable model.
    box(ax, 0.3, 5.9, 1.5, 1.0, "abort\nCollapsed-\nError",
        fill="white", edge=ACCENT2, fontsize=6.5)
    arrow(ax, (2.0, 6.4), (1.8, 6.4), color=ACCENT2, ls="--")
    ax.text(1.05, 5.6, "if no epoch\nis accepted", fontsize=6.0,
            color=ACCENT2, ha="center")

    return save(fig, "fig07_statechart")


# --------------------------------------------------------------------------
# Figure 8 -- work breakdown structure
# --------------------------------------------------------------------------
def fig_wbs():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    tasks = [
        ("Literature review and domain study", 0, 3, "done"),
        ("Environment, package skeleton, test harness", 1, 3, "done"),
        ("TOFU data layer and entity spans", 2, 3, "done"),
        ("Activation patching and UDS backend", 3, 4, "done"),
        ("Depth-axis validation vs published values", 5, 3, "done"),
        ("Breadth axis and calibration", 6, 3, "done"),
        ("Unlearning methods and checkpoint selection", 8, 3, "done"),
        ("Extrapolation dial and trajectory analysis", 9, 3, "done"),
        ("First end-to-end depth-breadth trajectory", 11, 2, "done"),
        ("Replication: NPO, GradDiff, seeds", 13, 3, "todo"),
        ("Scale to 3B; B2-B4 breadth tiers", 15, 3, "todo"),
        ("Analysis, write-up, submission", 17, 3, "todo"),
    ]
    colours = {"done": ACCENT, "todo": "#c9c9c9"}
    for i, (name, start, dur, state) in enumerate(tasks):
        y = len(tasks) - i
        ax.barh(y, dur, left=start, height=0.55,
                color=colours[state], edgecolor="white", linewidth=0.6)
        ax.text(-0.4, y, name, ha="right", va="center", fontsize=7)
    ax.axvline(13, color=ACCENT2, lw=1.1, ls="--")
    ax.text(13.15, len(tasks) + 0.9, "mid-semester\nevaluation", fontsize=6.8,
            color=ACCENT2, va="top")
    ax.set_xlim(-11, 21)
    ax.set_ylim(0.2, len(tasks) + 1.4)
    ax.set_yticks([])
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("project week")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours["done"]),
               plt.Rectangle((0, 0), 1, 1, color=colours["todo"])]
    ax.legend(handles, ["completed", "planned"], loc="lower right",
              fontsize=7, frameon=False, ncol=2)
    return save(fig, "fig08_wbs")


# --------------------------------------------------------------------------
# Figure 9 -- depth-metric validation against published values
# --------------------------------------------------------------------------
def fig_uds_validation():
    import numpy as np
    splits = ["full", "retain99", "retain95", "retain90"]
    ours = [0.000, 0.126, 0.486, 1.000]
    paper = [0.002, 0.153, 0.496, 1.000]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(splits))
    w = 0.36
    ax.bar(x - w / 2, paper, w, label="Published values",
           color="#c9d3de", edgecolor=ACCENT, linewidth=0.8)
    ax.bar(x + w / 2, ours, w, label="This implementation",
           color=ACCENT, edgecolor=ACCENT, linewidth=0.8)
    for xi, (p, o) in enumerate(zip(paper, ours)):
        ax.text(xi - w / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, o + 0.02, f"{o:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Unlearning Depth Score")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("model whose knowledge is being scored")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    return save(fig, "fig09_uds_validation")


# --------------------------------------------------------------------------
# Figure 10 -- unlearning trajectory (measured)
# --------------------------------------------------------------------------
def fig_training_curve():
    epochs = list(range(20))
    nll = [1.6565, 1.7021, 1.7511, 1.8023, 1.8550, 1.9125, 1.9792, 2.0646,
           2.1798, 2.3377, 2.5431, 2.8025, 3.1158, 3.5241, 4.0430, 4.6830,
           5.4927, 6.5828, 8.0462, 9.8331]
    util = [0.8025, 0.7925, 0.7900, 0.7950, 0.7900, 0.7850, 0.7825, 0.7725,
            0.7675, 0.7675, 0.7575, 0.7425, 0.7200, 0.6900, 0.6600, 0.6225,
            0.5675, 0.5325, 0.4775, 0.4500]
    floor = 0.7177

    fig, ax1 = plt.subplots(figsize=(5.6, 3.1))
    ax1.plot(epochs, nll, marker="o", ms=3, lw=1.2, color=ACCENT,
             label="forget-set NLL (higher = more forgotten)")
    ax1.set_xlabel("training epoch")
    ax1.set_ylabel("forget-set NLL", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax1.set_xticks(range(0, 20, 2))

    ax2 = ax1.twinx()
    ax2.plot(epochs, util, marker="s", ms=3, lw=1.2, color=ACCENT2,
             label="retained-knowledge utility")
    ax2.axhline(floor, color=ACCENT2, ls="--", lw=0.9)
    ax2.text(0.2, floor + 0.006, "utility floor (90% of baseline)",
             fontsize=6.3, color=ACCENT2)
    ax2.set_ylabel("utility", color=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT2)
    ax2.set_ylim(0.40, 0.85)

    ax1.axvline(12, color=INK, lw=0.9, ls=":")
    ax1.text(12.25, 8.4, "selected\ncheckpoint\n(epoch 12)", fontsize=6.5)
    ax1.axvspan(12.5, 19.5, color="#f0f0f0", zorder=0)
    ax1.text(16, 1.3, "rejected: utility below floor", fontsize=6.3,
             ha="center", color=MUTED)

    for s in ("top",):
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    return save(fig, "fig10_training")


# --------------------------------------------------------------------------
# Figure 11 -- the depth-breadth plane (measured)
# --------------------------------------------------------------------------
def fig_plane():
    alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    breadth = [0.237, 0.237, 0.333, 0.333, 0.354, 0.399, 0.429, 0.485,
               0.510, 0.551, 0.571]
    depth = [0.353, 0.424, 0.480, 0.531, 0.572, 0.608, 0.639, 0.666,
             0.690, 0.708, 0.725]
    util = [0.720, 0.710, 0.695, 0.677, 0.657, 0.652, 0.650, 0.647,
            0.640, 0.632, 0.625]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    fig.subplots_adjust(wspace=0.55)

    sc = axL.scatter(breadth, depth, c=alpha, cmap="viridis", s=34,
                     edgecolor="white", linewidth=0.5, zorder=3)
    axL.plot(breadth, depth, lw=0.9, color=MUTED, zorder=2)
    axL.set_xlabel("breadth  (0 = original model, 1 = retain oracle)")
    axL.set_ylabel("depth  (UDS)")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.plot([0, 1], [0, 1], ls=":", lw=0.8, color="#bbbbbb")
    axL.text(0.63, 0.60, "equal progress\non both axes", fontsize=6,
             color="#999999", rotation=38)
    cb = fig.colorbar(sc, ax=axL, fraction=0.042, pad=0.02)
    cb.set_label("α", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    axL.set_title("(a) trajectory through the plane", fontsize=8)

    axR.plot(alpha, depth, marker="o", ms=3, lw=1.2, color=ACCENT,
             label="depth")
    axR.plot(alpha, breadth, marker="s", ms=3, lw=1.2, color=ACCENT2,
             label="breadth")
    axR.plot(alpha, util, marker="^", ms=3, lw=1.2, color=MUTED,
             ls="--", label="utility (control)")
    axR.set_xlabel("extrapolation strength α")
    axR.set_ylabel("score")
    axR.set_ylim(0, 1)
    axR.legend(fontsize=6.8, frameon=False, loc="upper left")
    axR.set_title("(b) each axis against α", fontsize=8)

    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_axisbelow(True)

    return save(fig, "fig11_plane")


if __name__ == "__main__":
    print("Generating report figures:")
    fig_architecture()
    fig_patching()
    fig_breadth_tiers()
    fig_usecase()
    fig_class()
    fig_sequence()
    fig_statechart()
    fig_wbs()
    fig_uds_validation()
    fig_training_curve()
    fig_plane()
    print(f"\nAll figures in {OUT}")
