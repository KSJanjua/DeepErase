"""Breadth probe schema: tiers B0-B4 plus a retain-neighbour control.

Breadth asks how far forgetting *generalises* past the exact strings in the
forget set. The literature evaluates this unevenly -- TOFU mostly at B0, RWKU
with adversarial probes, SUITE (Peleg et al., arXiv:2607.09236) with
paraphrase and multi-hop -- so we define an explicit ladder and report every
rung separately.

    B0  exact       verbatim forget-set question
    B1  paraphrase  same fact, reworded; no new information introduced
    B2  alias       same fact via an alias or definite description
    B3  entailed    a distinct fact that entails the target in one hop
    B4  multi-hop   requires two or more hops through retained facts
    R   retain      a neighbour fact that must survive -- the over-forgetting control

The R tier is not optional. Forget rates rise monotonically as you push alpha,
and without a retain control that looks like success rather than collateral
damage. Feng et al. (arXiv:2506.00688) call this out directly: evaluations
that inject information or ignore downstream effects produce misleading
conclusions.

B3/B4 are where the UIPE related-knowledge hypothesis lives -- they are the
tiers a model can answer by *reasoning* from facts it still holds, which is
what parameter extrapolation is supposed to reach.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


class Tier(str, Enum):
    EXACT = "B0"
    PARAPHRASE = "B1"
    ALIAS = "B2"
    ENTAILED = "B3"
    MULTIHOP = "B4"
    RETAIN = "R"

    @property
    def is_forget(self) -> bool:
        """True for tiers where a correct answer counts as leakage."""
        return self is not Tier.RETAIN

    @property
    def description(self) -> str:
        return {
            Tier.EXACT: "verbatim forget-set question",
            Tier.PARAPHRASE: "same fact, reworded, no new information",
            Tier.ALIAS: "same fact via alias or definite description",
            Tier.ENTAILED: "distinct fact entailing the target in one hop",
            Tier.MULTIHOP: "requires two or more hops through retained facts",
            Tier.RETAIN: "neighbour fact that must survive unlearning",
        }[self]


BREADTH_TIERS: tuple[Tier, ...] = (
    Tier.EXACT, Tier.PARAPHRASE, Tier.ALIAS, Tier.ENTAILED, Tier.MULTIHOP,
)


@dataclass
class Probe:
    """One question testing whether a target survives at a given tier."""

    probe_id: str
    target_id: str
    """Which unlearning target this belongs to (e.g. a TOFU author)."""
    tier: Tier
    question: str
    answer: str
    """Reference answer. For forget tiers this is what must NOT be produced."""
    aliases: List[str] = field(default_factory=list)
    """Acceptable surface forms of the answer, for scoring."""
    source: str = "manual"
    """Provenance: manual | gpt4o | wikidata | tofu | rwku. Every synthetic
    probe must be human-verified before it enters a reported result."""
    verified: bool = False
    hop_facts: List[str] = field(default_factory=list)
    """For B3/B4: the retained facts the model would reason through. Recording
    these lets us check whether a B4 failure is genuine forgetting or just a
    reasoning failure -- a confound SUITE also has to handle."""
    notes: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.tier, str):
            self.tier = Tier(self.tier)
        if not self.question.strip():
            raise ValueError(f"Probe {self.probe_id} has an empty question")
        if self.tier in (Tier.ENTAILED, Tier.MULTIHOP) and not self.hop_facts:
            logger.warning(
                "Probe %s is tier %s but lists no hop_facts; reasoning-failure "
                "confounds will be indistinguishable from forgetting",
                self.probe_id, self.tier.value,
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Probe":
        return cls(**{**d, "tier": Tier(d["tier"])})


@dataclass
class ProbeSet:
    """All probes for one unlearning target."""

    target_id: str
    target_name: str
    target_aliases: List[str] = field(default_factory=list)
    probes: List[Probe] = field(default_factory=list)
    benchmark: str = "custom"

    def by_tier(self, tier: Tier) -> List[Probe]:
        return [p for p in self.probes if p.tier is tier]

    def tier_counts(self) -> Dict[str, int]:
        return {t.value: len(self.by_tier(t)) for t in Tier}

    def validate(self, *, require_verified: bool = False) -> List[str]:
        """Return a list of problems. Empty means the set is usable.

        Call with ``require_verified=True`` before generating any number that
        goes into the paper.
        """
        problems: List[str] = []
        if not self.probes:
            problems.append(f"{self.target_id}: no probes")

        ids = [p.probe_id for p in self.probes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            problems.append(f"{self.target_id}: duplicate probe_ids {sorted(dupes)}")

        for t in BREADTH_TIERS:
            if not self.by_tier(t):
                problems.append(f"{self.target_id}: tier {t.value} is empty")
        if not self.by_tier(Tier.RETAIN):
            problems.append(
                f"{self.target_id}: no retain (R) probes -- over-forgetting cannot be measured"
            )

        wrong_target = [p.probe_id for p in self.probes if p.target_id != self.target_id]
        if wrong_target:
            problems.append(f"{self.target_id}: probes with mismatched target_id: {wrong_target}")

        if require_verified:
            unverified = [p.probe_id for p in self.probes if not p.verified]
            if unverified:
                problems.append(
                    f"{self.target_id}: {len(unverified)} unverified probes "
                    f"(first few: {unverified[:5]})"
                )
        return problems

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "target_name": self.target_name,
            "target_aliases": self.target_aliases,
            "benchmark": self.benchmark,
            "probes": [p.to_dict() for p in self.probes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeSet":
        return cls(
            target_id=d["target_id"],
            target_name=d["target_name"],
            target_aliases=d.get("target_aliases", []),
            benchmark=d.get("benchmark", "custom"),
            probes=[Probe.from_dict(p) for p in d.get("probes", [])],
        )


def save_probe_sets(sets: Sequence[ProbeSet], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "probe_sets": [s.to_dict() for s in sets]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d probe sets to %s", len(sets), path)
    return path


def load_probe_sets(path: str | Path) -> List[ProbeSet]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ProbeSet.from_dict(s) for s in data["probe_sets"]]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class TierScore:
    tier: Tier
    n: int
    n_correct: int
    rate: float
    """For forget tiers: leakage rate (lower = forgotten more broadly).
    For the retain tier: retention accuracy (higher = less collateral damage)."""

    @property
    def is_forget(self) -> bool:
        return self.tier.is_forget


@dataclass
class BreadthResult:
    target_id: str
    tier_scores: Dict[str, TierScore]

    @property
    def breadth_generalisation_gap(self) -> Optional[float]:
        """B0 leakage minus B3 leakage.

        Large and positive means forgetting was memorised at the surface but
        did not generalise to entailed facts -- narrow forgetting. Near zero
        means the intervention reached the entailment neighbourhood.
        """
        b0, b3 = self.tier_scores.get("B0"), self.tier_scores.get("B3")
        return None if not (b0 and b3) else b0.rate - b3.rate

    @property
    def mean_forget_leakage(self) -> float:
        """Mean leakage across forget tiers -- the scalar breadth coordinate.

        Lower = broader forgetting. Plotted on the breadth axis of the plane.
        """
        forget = [s.rate for s in self.tier_scores.values() if s.is_forget]
        return float(sum(forget) / len(forget)) if forget else float("nan")

    @property
    def retain_accuracy(self) -> Optional[float]:
        r = self.tier_scores.get("R")
        return None if r is None else r.rate


def score_breadth(
    probe_set: ProbeSet,
    correctness: Dict[str, bool],
) -> BreadthResult:
    """Aggregate per-probe correctness into per-tier rates.

    Args:
        probe_set: the probes that were run.
        correctness: ``probe_id -> was the reference answer produced``.
            Missing ids are skipped with a warning rather than silently
            counted as failures -- a missing evaluation is not a success.
    """
    buckets: Dict[Tier, List[bool]] = {t: [] for t in Tier}
    missing = 0
    for p in probe_set.probes:
        if p.probe_id not in correctness:
            missing += 1
            continue
        buckets[p.tier].append(bool(correctness[p.probe_id]))
    if missing:
        logger.warning("%d/%d probes had no correctness entry and were skipped",
                       missing, len(probe_set.probes))

    scores: Dict[str, TierScore] = {}
    for tier, vals in buckets.items():
        if not vals:
            continue
        n_correct = sum(vals)
        scores[tier.value] = TierScore(
            tier=tier, n=len(vals), n_correct=n_correct, rate=n_correct / len(vals)
        )
    return BreadthResult(target_id=probe_set.target_id, tier_scores=scores)
