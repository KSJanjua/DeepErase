"""The depth x breadth plane -- D1's central result object.

Each unlearning configuration becomes one point:

    x = breadth  (1 - mean forget leakage; higher = forgetting generalises further)
    y = depth    (UDS; higher = attenuated in the representation)

Sweeping alpha traces a *trajectory* through the plane. The D1 hypothesis
predicts those trajectories bend down-right: breadth is bought with depth.
A flat or up-right trajectory falsifies it, which is equally publishable
because the field's implicit assumption is that the axes are aligned.

The trade-off statistic is the Spearman correlation between breadth and depth
along a trajectory. Spearman rather than Pearson because we expect monotone
but not necessarily linear structure, and UIPE already reports a non-linear
(inverted-U) response in a related quantity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlanePoint:
    """One (method, model, benchmark, alpha) configuration."""

    method: str
    model: str
    benchmark: str
    alpha: float
    breadth: float
    """1 - mean forget leakage across B0-B4, in [0, 1]. Higher = broader."""
    depth: float
    """UDS in [0, 1]. Higher = deeper."""
    retain_accuracy: Optional[float] = None
    utility: Optional[float] = None
    """General capability, e.g. MMLU. Guards against 'the model just got worse'."""
    relearn_recovery: Optional[float] = None
    """Post-Retraining-on-T accuracy. D1 predicts this RISES with alpha."""
    smr: Optional[float] = None
    el10: Optional[float] = None
    type_label: Optional[str] = None
    breadth_gap: Optional[float] = None
    alpha_parallel: Optional[float] = None
    alpha_perp: Optional[float] = None
    """Set for SAGE points; None for isotropic UIPE."""
    depth_overshoot: bool = False
    """Propagated from UDSResult.overshoot. The unlearned model scored below
    the retain oracle, so its UDS is not an erasure fraction. Excluded from
    trade-off correlations by default."""
    n_targets: int = 0
    seed: int = 0
    notes: str = ""

    @property
    def is_directed(self) -> bool:
        return self.alpha_parallel is not None and self.alpha_parallel != self.alpha_perp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanePoint":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Trajectory:
    """A sequence of points sharing everything but alpha."""

    method: str
    model: str
    benchmark: str
    points: List[PlanePoint] = field(default_factory=list)

    def sorted_points(self) -> List[PlanePoint]:
        return sorted(self.points, key=lambda p: p.alpha)

    @property
    def alphas(self) -> List[float]:
        return [p.alpha for p in self.sorted_points()]

    @property
    def breadths(self) -> List[float]:
        return [p.breadth for p in self.sorted_points()]

    @property
    def depths(self) -> List[float]:
        return [p.depth for p in self.sorted_points()]

    @property
    def n_overshoot(self) -> int:
        return sum(1 for p in self.points if p.depth_overshoot)

    #: Points on a trajectory are successive evaluations of one continuous path
    #: through weight space -- α scaling a single update vector -- not
    #: independent draws from a population. Neighbouring α values give models
    #: that differ by a fraction of one update, so their scores are serially
    #: dependent by construction.
    #:
    #: Spearman's p-value assumes independent observations. On a trajectory it
    #: is therefore **not a significance test**: eleven α values on one path can
    #: be made arbitrarily "significant" by evaluating twenty-one instead. rho
    #: remains useful as a *description* of the path's shape; the p-value should
    #: not be quoted as evidence, and the sign needs replication across methods
    #: and seeds before it means anything.
    POINTS_ARE_INDEPENDENT = False

    def axis_spans(self) -> dict:
        """How much of each axis the sweep actually covered.

        Both axes are calibrated so 0.0 is the original model and 1.0 is the
        retain oracle. A trajectory confined to a small corner of the plane
        cannot say anything about the relationship between the axes, whatever
        its rank correlation: the question is about the shape of a curve, and a
        curve needs room to bend.
        """
        pts = self.sorted_points()
        if not pts:
            return {"breadth_span": 0.0, "depth_span": 0.0, "depth_peak": 0.0}
        b = [p.breadth for p in pts]
        d = [p.depth for p in pts]
        return {"breadth_span": max(b) - min(b),
                "depth_span": max(d) - min(d),
                "depth_peak": max(d)}

    def has_usable_dynamic_range(
        self, *, min_breadth_span: float = 0.15, min_depth_span: float = 0.15
    ) -> bool:
        """Whether the sweep covers enough of the plane to be interpreted.

        Defaults ask for 15% of the distance from the original model to the
        retain oracle on both axes. The first real GA sweep managed 0.093 and
        0.040 -- under a tenth of the plane -- and still produced rho=+0.89 at
        p=0.0002, which is what a tiny monotone wiggle looks like to a rank
        test on serially dependent points.
        """
        s = self.axis_spans()
        return (s["breadth_span"] >= min_breadth_span
                and s["depth_span"] >= min_depth_span)

    def tradeoff_correlation(
        self, *, exclude_overshoot: bool = True
    ) -> Optional[tuple[float, float]]:
        """Spearman (rho, p) between breadth and depth along the trajectory.

        Negative rho would support the D1 hypothesis.

        .. warning::
            The returned p-value is **not a significance test** -- see
            :data:`POINTS_ARE_INDEPENDENT`. Check
            :meth:`has_usable_dynamic_range` before reading anything into the
            sign.

        .. warning::
            The depth axis is currently populated by the UDS **scaffold**,
            which is not a validated measurement. Correlations computed here
            are pipeline exercises, not evidence.

        Args:
            exclude_overshoot: drop points where the unlearned model scored
                below the retain oracle. Prefer
                :meth:`tradeoff_sensitivity`, which reports the value both
                ways -- a single number silently conditioned on a filtering
                choice is exactly the kind of result that does not survive
                review.

        Returns:
            ``(rho, p)``, or None with fewer than 3 usable points or when
            either axis is constant.
        """
        from scipy import stats

        pts = self.sorted_points()
        if exclude_overshoot:
            kept = [p for p in pts if not p.depth_overshoot]
            if len(kept) < len(pts):
                logger.warning(
                    "Trajectory %s/%s/%s: excluded %d/%d overshoot points from correlation. "
                    "Report tradeoff_sensitivity() alongside this value.",
                    self.method, self.model, self.benchmark, len(pts) - len(kept), len(pts),
                )
            pts = kept

        b = np.asarray([p.breadth for p in pts])
        d = np.asarray([p.depth for p in pts])
        if len(b) < 3:
            logger.warning("Trajectory %s/%s/%s has %d usable points; need >= 3",
                           self.method, self.model, self.benchmark, len(b))
            return None
        if np.allclose(b, b[0]) or np.allclose(d, d[0]):
            logger.warning("Trajectory %s/%s/%s has a constant axis; rho undefined",
                           self.method, self.model, self.benchmark)
            return None
        res = stats.spearmanr(b, d)
        return float(res.statistic), float(res.pvalue)

    def tradeoff_sensitivity(self) -> Dict[str, object]:
        """Report the trade-off correlation **both with and without** overshoot
        points, plus which points were affected.

        Overshoot points are ones where the unlearned model scored below the
        retain oracle, so UDS stops meaning "fraction of knowledge recovered"
        (see :func:`deeperase.eval.depth.unlearning_depth_score`). Excluding
        them is defensible; excluding them *silently* is not. Any reported
        correlation must be accompanied by this dictionary so a reader can see
        whether the conclusion depends on the filtering choice.

        Returns:
            Dict with ``rho_excluding`` / ``p_excluding``,
            ``rho_including`` / ``p_including``, ``n_overshoot``,
            ``overshoot_alphas``, and ``conclusion_is_robust`` -- True only
            when both variants agree in sign and significance.
        """
        incl = self.tradeoff_correlation(exclude_overshoot=False)
        excl = self.tradeoff_correlation(exclude_overshoot=True)
        overshoot_alphas = [p.alpha for p in self.sorted_points() if p.depth_overshoot]

        robust: Optional[bool] = None
        if incl is not None and excl is not None:
            robust = bool(
                np.sign(incl[0]) == np.sign(excl[0])
                and (incl[1] < 0.05) == (excl[1] < 0.05)
            )

        return {
            "method": self.method,
            "model": self.model,
            "benchmark": self.benchmark,
            "n_points": len(self.points),
            "n_overshoot": self.n_overshoot,
            "overshoot_alphas": overshoot_alphas,
            "rho_excluding": None if excl is None else round(excl[0], 4),
            "p_excluding": None if excl is None else round(excl[1], 4),
            "rho_including": None if incl is None else round(incl[0], 4),
            "p_including": None if incl is None else round(incl[1], 4),
            "conclusion_is_robust": robust,
        }

    def utility_is_stable(self, *, max_drop: float = 0.05) -> Optional[bool]:
        """Did general capability hold up across the sweep?

        The critical control. If utility collapses as alpha grows, any depth
        decline is explained by 'the model got worse at everything' and the
        trade-off claim does not hold. Returns None when utility is unlogged.
        """
        u = [p.utility for p in self.sorted_points() if p.utility is not None]
        if len(u) < 2:
            return None
        return (max(u) - min(u)) <= max_drop

    def pareto_front(self) -> List[PlanePoint]:
        """Points not dominated on both breadth and depth simultaneously."""
        pts = self.sorted_points()
        front = []
        for p in pts:
            if not any(
                (q.breadth >= p.breadth and q.depth >= p.depth)
                and (q.breadth > p.breadth or q.depth > p.depth)
                for q in pts
            ):
                front.append(p)
        return front


@dataclass
class PlaneDataset:
    """Every point collected in the study."""

    points: List[PlanePoint] = field(default_factory=list)

    def add(self, point: PlanePoint) -> None:
        self.points.append(point)

    def trajectories(self) -> List[Trajectory]:
        groups: Dict[tuple, Trajectory] = {}
        for p in self.points:
            key = (p.method, p.model, p.benchmark)
            groups.setdefault(key, Trajectory(*key)).points.append(p)
        return list(groups.values())

    def sensitivity_report(self) -> List[dict]:
        """Overshoot sensitivity for every trajectory.

        Must be reported alongside :meth:`tradeoff_summary`. See
        :meth:`Trajectory.tradeoff_sensitivity`.
        """
        return [t.tradeoff_sensitivity() for t in self.trajectories()]

    def tradeoff_summary(self) -> List[dict]:
        """One row per trajectory.

        .. warning::
            Depth values come from the UDS scaffold, which is not validated.
            These rows are pipeline output, not research results. Always pair
            with :meth:`sensitivity_report`.
        """
        rows = []
        for t in self.trajectories():
            corr = t.tradeoff_correlation()
            rows.append({
                "method": t.method,
                "model": t.model,
                "benchmark": t.benchmark,
                "n_points": len(t.points),
                "n_overshoot_excluded": t.n_overshoot,
                "spearman_rho": None if corr is None else round(corr[0], 4),
                "p_value": None if corr is None else round(corr[1], 4),
                "breadth_range": (round(min(t.breadths), 4), round(max(t.breadths), 4)),
                "depth_range": (round(min(t.depths), 4), round(max(t.depths), 4)),
                "utility_stable": t.utility_is_stable(),
                "supports_tradeoff": None if corr is None else bool(corr[0] < 0 and corr[1] < 0.05),
                "points_independent": t.POINTS_ARE_INDEPENDENT,
                "usable_dynamic_range": t.has_usable_dynamic_range(),
                **t.axis_spans(),
            })
        return rows

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "points": [p.to_dict() for p in self.points]}, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote %d plane points to %s", len(self.points), path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PlaneDataset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(points=[PlanePoint.from_dict(p) for p in data["points"]])

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([p.to_dict() for p in self.points])


def plot_plane(
    dataset: PlaneDataset,
    path: str | Path,
    *,
    title: str = "Depth vs. breadth of knowledge erasure",
    annotate_alpha: bool = True,
):
    """Render the plane. Returns the matplotlib Figure.

    Each trajectory is a connected line with alpha increasing along it, so a
    downward-right bend is visible directly as the trade-off.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectories = dataset.trajectories()
    if not trajectories:
        raise ValueError("No trajectories to plot")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab10")

    for i, t in enumerate(trajectories):
        pts = t.sorted_points()
        b = [p.breadth for p in pts]
        d = [p.depth for p in pts]
        colour = cmap(i % 10)
        ax.plot(b, d, "-o", color=colour, markersize=5, linewidth=1.6,
                label=f"{t.method} / {t.model}", alpha=0.9)
        # Mark the alpha=0 control point distinctly.
        ax.scatter([b[0]], [d[0]], s=140, facecolors="none", edgecolors=colour,
                   linewidths=2, zorder=5)
        if annotate_alpha and len(pts) > 1:
            for p, x, y in ((pts[0], b[0], d[0]), (pts[-1], b[-1], d[-1])):
                ax.annotate(f"α={p.alpha:g}", (x, y), textcoords="offset points",
                            xytext=(6, 5), fontsize=8, color=colour)

    ax.set_xlabel("Breadth  (1 − mean forget leakage over B0–B4) →")
    ax.set_ylabel("Depth  (UDS, causal activation patching) →")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.text(0.99, 0.01, "hollow marker = α=0 control",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, style="italic")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    logger.info("Wrote figure to %s", path)
    return fig
