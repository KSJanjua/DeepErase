"""Unlearning methods: turning ``M_full`` into ``M_unl``.

Everything measured so far used TOFU's pre-made checkpoints. The study needs a
model we unlearned ourselves, because the α dial extrapolates the *update
vector* ``v = θ_un − θ_ini``, which only exists if we performed the update.

Three published objectives are implemented. All are small -- the core of each
is a handful of lines -- and the risk lies almost entirely in hyperparameters,
so those are taken from OpenUnlearning's TOFU configurations rather than
guessed.

    GA        maximise loss on the forget set. The original method
              (Jang et al., ACL 2023). Simple, and prone to catastrophic
              collapse: push too hard and the model degrades on everything.

    GradDiff  GA plus a retain term, to hold general ability steady.

    NPO       DPO's negative branch only (Zhang et al., COLM 2024).
              Approaches collapse exponentially more slowly than GA, which is
              why it is the standard strong baseline. Needs a frozen reference
              copy of the starting model.

Representation-target methods (this project's own; deeperase/stranger.py)
train a subset of weights so that block outputs at chosen positions match
targets computed once from the starting model, with a retain anchor. They take
their per-row inputs through ``unlearn(..., stranger=...)``:

    stranger  targets from the starting model on name-swapped rows
              (docs/STRANGER.md; stopped at its gate 1, RESULTS.md §17).
    scramble  STRANGER's control: same distance, random direction.
    unasked   targets from the starting model on the row with the question
              blanked and the person unknown, at the answer positions
              (exploratory successor; ANALYSIS_PLAN H8).

Why not call OpenUnlearning directly
------------------------------------
Their framework is configuration-driven and would have to run as a subprocess
with its own environment, producing a checkpoint we then re-load. That is a
black box we cannot unit-test, and our extrapolation needs ``θ_ini`` and
``θ_un`` in a specific paired form. A ~200-line trainer that runs inside our
own process is testable on CPU with toy models, which matters more here than
avoiding reimplementation. We take their *hyperparameters*, not their code.

Verifying the result
--------------------
An unlearning run that quietly did nothing, or that destroyed the model, would
poison every downstream measurement. :func:`verify_unlearning` checks the
update actually moved the weights and that loss moved in the intended
direction. Beyond that, the breadth and depth axes we already built are the
real test: a good unlearned model should lose forget-set knowledge while
holding retain knowledge.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UnlearnMethod(str, Enum):
    GA = "ga"
    GRAD_DIFF = "graddiff"
    NPO = "npo"
    STRANGER = "stranger"
    SCRAMBLE = "scramble"
    UNASKED = "unasked"
    UNASKED_G = "unaskedg"
    """UNASKED plus an anchor on real-world QA pairs (``general_data``), so the
    model keeps reading ordinary questions as before (docs/UNASKED.md)."""
    GRAD_DIFF_G = "graddiffg"
    """GradDiff plus the same general anchor: the fairness baseline for
    UNASKED-G, which would otherwise be the only method given that data."""

    @property
    def needs_retain(self) -> bool:
        return self in (UnlearnMethod.GRAD_DIFF, UnlearnMethod.GRAD_DIFF_G) \
            or self.representation_targets

    @property
    def representation_targets(self) -> bool:
        """Trains block outputs toward precomputed targets (deeperase/stranger.py)."""
        return self in (UnlearnMethod.STRANGER, UnlearnMethod.SCRAMBLE, UnlearnMethod.UNASKED,
                        UnlearnMethod.UNASKED_G)

    @property
    def needs_general(self) -> bool:
        """Anchors M_full's states on general-knowledge QA (``general_data``)."""
        return self in (UnlearnMethod.UNASKED_G, UnlearnMethod.GRAD_DIFF_G)

    @property
    def needs_reference(self) -> bool:
        """NPO compares against a frozen copy of the starting model."""
        return self is UnlearnMethod.NPO


#: Learning rates from UIPE (Findings of EMNLP 2025, Appendix D.3), which tuned
#: these to "maximize the performance of these baseline methods" on TOFU.
#: forget10 needs an order of magnitude less than the smaller splits.
TOFU_LEARNING_RATES = {"forget01": 1e-5, "forget05": 1e-5, "forget10": 1e-6}


@dataclass
class UnlearnConfig:
    """Hyperparameters for one unlearning run.

    Defaults target **forget10**, the split this project uses: learning rate
    1e-6, batch size 1 (UIPE Appendix D.3). ``beta=0.1`` is the standard NPO
    value.

    .. warning::
        A learning rate of 1e-5 -- correct for forget01 and forget05 -- drives
        gradient ascent into catastrophic collapse on forget10. Observed here:
        forget-set NLL went 3.1 → 91.8 in five epochs and retention fell from
        0.79 to 0.21. The model was destroyed, not unlearned. Use
        :data:`TOFU_LEARNING_RATES`.
    """

    method: UnlearnMethod = UnlearnMethod.GA
    learning_rate: float = 1e-6
    epochs: int = 5
    batch_size: int = 1
    max_grad_norm: float = 1.0
    beta: float = 0.1
    """NPO temperature. Smaller means gentler updates."""
    retain_weight: float = 1.0
    """GradDiff: weight on the retain term."""
    seed: int = 0
    log_every: int = 20

    min_utility_ratio: float = 0.9
    """Checkpoint selection: keep at least this fraction of the starting
    model's utility. UIPE Algorithm 1 line 7 selects a checkpoint balancing
    forget quality against utility rather than taking the last epoch; this is
    the utility side of that trade-off. Only used when an ``eval_fn`` is
    supplied to :func:`unlearn`."""

    stop_below_utility_ratio: Optional[float] = None
    """End training early once an epoch's utility falls below this fraction of
    the starting utility. Off (None) in study runs. Learning-rate scans use it:
    under gradient ascent utility does not recover, so the remaining epochs
    could never be selected and would only cost time."""

    trainable: Optional[str] = None
    """Regex over parameter names: only matching parameters train (the rest
    stay bit-identical and get no optimiser state). None trains everything,
    the behaviour of GA/GradDiff/NPO."""

    general_weight: float = 0.0
    """UNASKED-G: weight of the general-knowledge anchor term. Must be positive
    for UNASKED-G and zero for every other method."""

    select_by: str = "forget_nll"
    """Which epoch counts as "most forgetful" among those above the utility
    floor. ``forget_nll``: the highest forget-set NLL (GA/GradDiff/NPO, whose
    NLL keeps rising). ``rep_gap``: the lowest remaining gap to the
    representation target (representation-target methods). Their NLL
    plateaus at the target's level within an epoch, so its maximum is noise."""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.learning_rate <= 0:
            problems.append("learning_rate must be positive")
        if self.epochs < 1:
            problems.append("epochs must be >= 1")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.method is UnlearnMethod.NPO and self.beta <= 0:
            problems.append("NPO requires beta > 0")
        if not 0.0 < self.min_utility_ratio <= 1.0:
            problems.append("min_utility_ratio must be in (0, 1]")
        if self.select_by not in ("forget_nll", "rep_gap"):
            problems.append("select_by must be 'forget_nll' or 'rep_gap'")
        if self.select_by == "rep_gap" and not self.method.representation_targets:
            problems.append("select_by='rep_gap' needs a representation-target method")
        if self.method.needs_general and not self.general_weight > 0:
            problems.append(f"{self.method.value} needs general_weight > 0")
        if not self.method.needs_general and self.general_weight:
            problems.append("general_weight is only used by unaskedg and graddiffg")
        if (self.stop_below_utility_ratio is not None
                and not 0.0 < self.stop_below_utility_ratio <= 1.0):
            problems.append("stop_below_utility_ratio must be in (0, 1]")
        return problems

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["method"] = self.method.value
        return d


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def sequence_nll(model, input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 *, reduction: str = "mean",
                 loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Negative log-likelihood per sequence. Shape ``(batch,)``.

    Per-sequence rather than a single scalar, because NPO needs to compare each
    sequence against its reference value before any reduction.

    Args:
        reduction: ``"mean"`` averages over the sequence's scored tokens, which
            is what the GA and GradDiff objectives and every reported
            ``forget_nll`` use -- a per-token figure comparable across examples
            of different length. ``"sum"`` returns the sequence log-likelihood
            itself, which is what a likelihood *ratio* between two models
            requires. See :func:`npo_loss`.
        loss_mask: ``(batch, seq)``, 1 where a token is a training target.
            Token ``t`` is predicted from position ``t-1``, so ``loss_mask[:, t]``
            selects the prediction *of* token ``t``. Pass an answer-only mask
            to score the answer and not the question, which is how
            OpenUnlearning trains (prompt labels are -100). ``None`` scores every
            non-padding token after the first -- the v1 behaviour, which also
            unlearned the question templates that retain-set questions share
            (RESULTS.md §8.4).
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, 1:].to(token_lp.dtype)
    if loss_mask is not None:
        lm = loss_mask[:, 1:].to(token_lp.dtype)
        mask = lm if mask is None else mask * lm

    if mask is not None:
        total = -(token_lp * mask).sum(-1)
        if reduction == "sum":
            return total
        # clamp so an all-padding row cannot divide by zero
        return total / mask.sum(-1).clamp(min=1.0)
    if reduction == "sum":
        return -token_lp.sum(-1)
    return -token_lp.mean(-1)


def _nll(model, batch, **kw) -> torch.Tensor:
    return sequence_nll(model, batch["input_ids"], batch.get("attention_mask"),
                        loss_mask=batch.get("loss_mask"), **kw)


def ga_loss(model, batch) -> torch.Tensor:
    """Gradient ascent: minimising this maximises loss on the forget set."""
    return -_nll(model, batch).mean()


def grad_diff_loss(model, forget_batch, retain_batch, retain_weight: float) -> torch.Tensor:
    """GA plus a retain term holding general ability steady."""
    forget = -_nll(model, forget_batch).mean()
    retain = _nll(model, retain_batch).mean()
    return forget + retain_weight * retain


def npo_loss(model, reference_model, batch, beta: float) -> torch.Tensor:
    """Negative Preference Optimization (Zhang et al., COLM 2024).

        L = (2/beta) * -log_sigmoid( -beta * (logp_theta - logp_ref) )

    As the current model's likelihood on the forget set falls below the
    reference's, the gradient shrinks. That self-damping is what makes NPO
    approach catastrophic collapse exponentially more slowly than plain GA.

    ``logp`` here is the **sequence** log-likelihood -- a sum over tokens, not
    a per-token average. This is not cosmetic. Length-normalising divides the
    log-ratio by |y|, which is identical to shrinking beta by the same factor,
    and NPO provably reduces to GA as beta -> 0 (Zhang et al. Prop. 1). At
    beta=0.1 over TOFU answers of 30-100 tokens the effective beta lands near
    1e-3, and the method becomes gradient ascent with a reference model's
    forward pass attached: an earlier version of this file did exactly that,
    and GA and NPO agreed to four significant figures on ||v|| across three
    seeds. The reference implementation sums (openunlearning
    ``compute_batch_nll``, "get the sum loss for each sequence") and reserves
    the length-normalised form for SimNPO, which is a different method.
    """
    logp = -_nll(model, batch, reduction="sum")
    with torch.no_grad():
        logp_ref = -_nll(reference_model, batch, reduction="sum")
    return (2.0 / beta) * -F.logsigmoid(-beta * (logp - logp_ref)).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class CollapsedError(RuntimeError):
    """No epoch preserved enough utility to be usable.

    Raised rather than returning a destroyed model. A model whose general
    ability has collapsed forgets everything, which looks identical to
    successful unlearning on every forget-side metric -- so returning one
    quietly would produce a study whose axes are both pinned at maximum and
    whose conclusion is an artefact.
    """

    history: Optional["UnlearnHistory"] = None
    """The training record up to the collapse, so a caller can still see the
    utility curve (a learning-rate scan needs exactly that)."""


@dataclass
class EpochEval:
    """Utility and forget quality after one epoch (Algorithm 1, lines 4-5)."""

    epoch: int
    utility: float
    forget_nll: float
    acceptable: bool
    selected: bool = False
    rep_gap: Optional[float] = None
    """Representation-target methods: the forget term over every row,
    relative to its value at the starting weights (1 = untouched, 0 = on
    target)."""

    def to_dict(self) -> dict:
        d = {"epoch": self.epoch, "utility": round(self.utility, 4),
             "forget_nll": round(self.forget_nll, 4),
             "acceptable": self.acceptable, "selected": self.selected}
        if self.rep_gap is not None:
            d["rep_gap"] = round(self.rep_gap, 4)
        return d


@dataclass
class UnlearnHistory:
    """Per-step record, so a bad run is diagnosable after the fact."""

    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    forget_nll: List[float] = field(default_factory=list)
    seconds: float = 0.0
    epoch_evals: List[EpochEval] = field(default_factory=list)
    baseline_utility: Optional[float] = None
    baseline_forget_nll: Optional[float] = None
    final_forget_nll: Optional[float] = None
    selected_epoch: Optional[int] = None
    stopped_early: bool = False
    """True when ``stop_below_utility_ratio`` ended training before the
    configured number of epochs."""
    rep_stats: Optional[dict] = None
    """Representation-target methods: target statistics (positions, scales,
    initial gap) from :func:`deeperase.stranger.compute_targets`."""

    @property
    def forget_nll_change(self) -> Optional[float]:
        """How far forget-set NLL moved. Should RISE during unlearning.

        Measured over the **whole forget set**, before training and again at the
        end, so the two numbers describe the same sentences.

        This used to be ``forget_nll[-1] - forget_nll[0]`` over the per-step
        list. At batch size 1 those are two *different examples*, so the figure
        was the difference between whichever sentence happened to be first and
        whichever happened to be last -- easily larger than the training effect
        and of arbitrary sign. It reported a healthy run as an inverted
        objective. Per-step NLL stays in :attr:`forget_nll` for plotting, but no
        verdict is drawn from it.
        """
        if self.baseline_forget_nll is None or self.final_forget_nll is None:
            return None
        return self.final_forget_nll - self.baseline_forget_nll

    def to_dict(self) -> dict:
        return {"n_steps": len(self.steps), "seconds": round(self.seconds, 1),
                "first_loss": self.losses[0] if self.losses else None,
                "last_loss": self.losses[-1] if self.losses else None,
                "baseline_forget_nll": self.baseline_forget_nll,
                "final_forget_nll": self.final_forget_nll,
                "forget_nll_change": self.forget_nll_change,
                "baseline_utility": self.baseline_utility,
                "selected_epoch": self.selected_epoch,
                "stopped_early": self.stopped_early,
                "rep_stats": self.rep_stats,
                "epoch_evals": [e.to_dict() for e in self.epoch_evals]}


def _batches(
    data: Sequence[Dict[str, torch.Tensor]],
    batch_size: int,
    device,
    *,
    pad_token_id: int = 0,
):
    """Group pre-tokenised examples into padded batches on ``device``.

    Real TOFU examples differ in length -- 47 tokens, 62 tokens, and so on --
    so sequences are right-padded to the longest in each batch and the padding
    is masked out. ``sequence_nll`` divides by the mask sum, so padded
    positions contribute nothing to the loss.

    An earlier version simply concatenated, which required every example to be
    the same length. That assumption held for synthetic test data and broke on
    the first real batch.

    The attention mask is always returned, even when no padding was needed, so
    downstream code never has to handle a ``None``. A ``loss_mask`` is carried
    through (padded with 0) when the examples have one; mixing examples with and
    without one is an error, because the batch would then score two different
    things.
    """
    for i in range(0, len(data), batch_size):
        chunk = data[i: i + batch_size]
        max_len = max(c["input_ids"].shape[1] for c in chunk)
        with_lm = [c.get("loss_mask") is not None for c in chunk]
        if any(with_lm) and not all(with_lm):
            raise ValueError("Some examples carry a loss_mask and some do not")

        ids, masks, lmasks = [], [], []
        for c in chunk:
            seq = c["input_ids"]
            mask = c.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(seq)
            lm = c.get("loss_mask")
            pad = max_len - seq.shape[1]
            if pad:
                seq = F.pad(seq, (0, pad), value=pad_token_id)
                mask = F.pad(mask, (0, pad), value=0)
                if lm is not None:
                    lm = F.pad(lm, (0, pad), value=0)
            ids.append(seq)
            masks.append(mask)
            if lm is not None:
                lmasks.append(lm)

        batch = {
            "input_ids": torch.cat(ids).to(device),
            "attention_mask": torch.cat(masks).to(device),
        }
        if lmasks:
            batch["loss_mask"] = torch.cat(lmasks).to(device)
        yield batch


def unlearn(
    model,
    forget_data: Sequence[Dict[str, torch.Tensor]],
    config: UnlearnConfig,
    *,
    retain_data: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
    reference_model=None,
    eval_fn=None,
    stranger=None,
    general_data: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
) -> UnlearnHistory:
    """Apply an unlearning method **in place**. Returns the training history.

    The caller must keep a copy of the starting weights: the α dial needs
    ``v = θ_un − θ_ini``, and this function overwrites ``θ_ini``.

    Checkpoint selection
    --------------------
    When ``eval_fn`` is supplied, this implements UIPE Algorithm 1 lines 4-7:
    utility is measured after every epoch, and the returned model is the
    checkpoint that forgot the most **while keeping at least**
    ``config.min_utility_ratio`` of its starting utility -- not simply the last
    epoch.

    That step is not optional in practice. Taking the final epoch of gradient
    ascent on forget10 destroyed the model outright in an earlier run
    (retention 0.79 -> 0.21), and a destroyed model scores maximally on every
    forget metric, so the failure is invisible downstream.

    Args:
        eval_fn: ``model -> float`` returning a utility score, higher is
            better. Called once before training to establish a baseline, then
            after each epoch. Omit to keep the last epoch unconditionally.
        stranger: per-row inputs of a representation-target method
            (:class:`deeperase.stranger.StrangerInputs`), row i aligned with
            ``forget_data[i]``. Required by those methods, refused by others.
        general_data: UNASKED-G's general-knowledge QA rows, anchored like the
            retain rows. Required by UNASKED-G, refused by others.

    Raises:
        ValueError: on an invalid config, or if a method's required inputs are
            missing.
        CollapsedError: if ``eval_fn`` is given and no epoch stayed above the
            utility floor.
    """
    problems = config.validate()
    if problems:
        raise ValueError(f"Invalid UnlearnConfig: {problems}")
    if config.method.needs_retain and not retain_data:
        raise ValueError(f"{config.method.value} requires retain_data")
    if retain_data is not None and retain_data is forget_data:
        raise ValueError(
            "retain_data is the same object as forget_data. The retention term "
            "would then be computed on the forget set, making the GradDiff "
            "objective (retain_weight - 1) * NLL(forget) -- identically zero at "
            "the default retain_weight of 1.0. The run would not train, would "
            "keep full utility, would pass checkpoint selection, and would "
            "produce a flat trajectory indistinguishable from a real null "
            "result. Pass a disjoint retain split (see FORGET_TO_RETAIN)."
        )
    if config.method.needs_reference and reference_model is None:
        raise ValueError(
            f"{config.method.value} requires reference_model -- a frozen copy of "
            "the starting model. Without it the objective is not NPO."
        )
    if not forget_data:
        raise ValueError("forget_data is empty")
    if config.method.representation_targets:
        if stranger is None:
            raise ValueError(f"{config.method.value} requires stranger= inputs")
        if len(stranger.rows) != len(forget_data):
            raise ValueError(f"stranger has {len(stranger.rows)} rows for "
                             f"{len(forget_data)} forget examples; they must align")
        want = {UnlearnMethod.SCRAMBLE: "scramble", UnlearnMethod.UNASKED: "unasked",
                UnlearnMethod.UNASKED_G: "unasked"}.get(config.method, "counterfactual")
        if stranger.target != want:
            raise ValueError(f"{config.method.value} needs {want!r} targets, "
                             f"got {stranger.target!r}")
    elif stranger is not None:
        raise ValueError(f"{config.method.value} does not take stranger= inputs")
    if config.method.needs_general and not general_data:
        raise ValueError(f"{config.method.value} requires general_data")
    if general_data and not config.method.needs_general:
        raise ValueError(f"{config.method.value} does not take general_data")

    torch.manual_seed(config.seed)
    device = next(model.parameters()).device
    history = UnlearnHistory()

    def forget_set_nll() -> float:
        """Mean NLL over the entire forget set, with the model frozen.

        Every call sees the same sentences in the same order, which is what
        makes two calls comparable. One no-grad pass over a few hundred short
        examples costs seconds.
        """
        was_training = model.training
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for b in _batches(forget_data, config.batch_size, device):
                per_seq = _nll(model, b)
                total += float(per_seq.sum())
                n += int(per_seq.numel())
        if was_training:
            model.train()
        return total / max(1, n)

    if reference_model is not None:
        reference_model.eval()
        reference_model.requires_grad_(False)

    history.baseline_forget_nll = forget_set_nll()
    if eval_fn is not None:
        model.eval()
        history.baseline_utility = float(eval_fn(model))
        logger.info("  baseline before training: utility=%.4f forget_nll=%.4f",
                    history.baseline_utility, history.baseline_forget_nll)
    else:
        logger.info("  baseline forget_nll before training: %.4f",
                    history.baseline_forget_nll)

    targets = None
    if config.method.representation_targets:
        from deeperase.stranger import compute_targets, forget_term
        # At theta0, before any step: the targets are the starting model's own
        # states, so they must be taken now.
        targets = compute_targets(model, stranger, retain_data, device,
                                  general_data=general_data or ())
        targets.stats["initial_forget_term"] = forget_term(
            model, forget_data, targets, config.batch_size, device)
        history.rep_stats = targets.stats
    elif config.method.needs_general:
        # GradDiff-G: only the general anchors (and, for their scale, the retain
        # rows' states), at every layer, exactly as UNASKED-G computes them.
        from deeperase.stranger import StrangerInputs, compute_targets
        n_l = int(model.config.num_hidden_layers)
        targets = compute_targets(model, StrangerInputs(rows=[], layers=tuple(range(n_l)),
                                                        k=0, target="anchor_only"),
                                  retain_data, device, general_data=general_data)
        history.rep_stats = targets.stats

    model.train()
    if config.trainable:
        import re
        rx = re.compile(config.trainable)
        model.requires_grad_(False)
        params = [p for n, p in model.named_parameters() if rx.search(n)]
        if not params:
            raise ValueError(f"trainable={config.trainable!r} matches no parameter")
        for p in params:
            p.requires_grad_(True)
        logger.info("  training %d tensors (%.0fM parameters) matching %r",
                    len(params), sum(p.numel() for p in params) / 1e6, config.trainable)
    else:
        model.requires_grad_(True)
        params = list(model.parameters())
    # foreach=False: the multi-tensor path fuses every parameter into one
    # `_foreach_sqrt`, which materialises a full-size temporary copy of
    # exp_avg_sq -- a fifth model's worth of memory, allocated on the first
    # step() and nowhere visible in a parameter count. The single-tensor path
    # peaks one tensor at a time for a small speed cost, which at batch 1 is
    # dominated by the forward pass anyway.
    optimiser = torch.optim.AdamW(params, lr=config.learning_rate, foreach=False)

    best_snapshot: Optional[Dict[str, torch.Tensor]] = None
    best_forget_nll = float("-inf")
    best_gap = float("inf")

    t0 = time.time()
    step = 0
    bs = config.batch_size
    n_retain = len(retain_data or [])
    # The general anchor (UNASKED-G, GradDiff-G) is not tied to the forget set's
    # length: one iterator runs across epochs, so every row is anchored in turn.
    general_iter = iter(_batches(general_data, bs, device)) if general_data else None
    g_index = 0

    def next_general():
        nonlocal general_iter, g_index
        if general_iter is None:
            return None, []
        try:
            g = next(general_iter)
        except StopIteration:
            general_iter, g_index = iter(_batches(general_data, bs, device)), 0
            g = next(general_iter)
        g_rows = list(range(g_index * bs, min((g_index + 1) * bs, len(general_data))))
        g_index += 1
        return g, g_rows

    for epoch in range(config.epochs):
        retain_iter = iter(_batches(retain_data or [], bs, device))
        r_index = 0
        for b_index, batch in enumerate(_batches(forget_data, bs, device)):
            if config.method is UnlearnMethod.GA:
                loss = ga_loss(model, batch)
            elif config.method in (UnlearnMethod.GRAD_DIFF, UnlearnMethod.GRAD_DIFF_G):
                try:
                    r = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(_batches(retain_data, bs, device))
                    r = next(retain_iter)
                loss = grad_diff_loss(model, batch, r, config.retain_weight)
                if config.method is UnlearnMethod.GRAD_DIFF_G:
                    from deeperase.stranger import general_term
                    g, g_rows = next_general()
                    lg = general_term(model, g, g_rows, targets)
                    if lg is not None:
                        loss = loss + config.general_weight * lg
            elif config.method is UnlearnMethod.NPO:
                loss = npo_loss(model, reference_model, batch, config.beta)
            elif config.method.representation_targets:
                from deeperase.stranger import stranger_loss
                try:
                    r = next(retain_iter)
                except StopIteration:
                    retain_iter, r_index = iter(_batches(retain_data, bs, device)), 0
                    r = next(retain_iter)
                f_rows = list(range(b_index * bs, min((b_index + 1) * bs, len(forget_data))))
                r_rows = list(range(r_index * bs, min((r_index + 1) * bs, n_retain)))
                r_index += 1
                g, g_rows = next_general()
                loss, _parts = stranger_loss(model, batch, f_rows, r, r_rows, targets,
                                             config.retain_weight, general_batch=g,
                                             general_rows=g_rows,
                                             general_weight=config.general_weight)
            else:
                raise ValueError(f"no training branch for method {config.method.value!r}")

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            optimiser.step()

            with torch.no_grad():
                nll = float(_nll(model, batch).mean())
            history.steps.append(step)
            history.losses.append(float(loss))
            history.forget_nll.append(nll)

            if config.log_every and step % config.log_every == 0:
                logger.info("  epoch %d step %d  loss=%.4f  forget_nll=%.4f",
                            epoch, step, float(loss), nll)
            step += 1

            if not math.isfinite(float(loss)):
                logger.error("Loss became %s at step %d -- stopping. The learning "
                             "rate is probably too high.", float(loss), step)
                history.seconds = time.time() - t0
                model.eval()
                return history

        # -- Algorithm 1 lines 4-7: evaluate, then keep the best checkpoint --
        if eval_fn is not None:
            model.eval()
            utility = float(eval_fn(model))
            # A clean pass over the whole set, not the running mean of the
            # step NLLs: those were each measured against a different model,
            # since the weights moved between them.
            epoch_nll = forget_set_nll()
            floor = config.min_utility_ratio * history.baseline_utility
            acceptable = utility >= floor
            rep_gap = None
            if targets is not None and config.method.representation_targets:
                from deeperase.stranger import forget_term
                start = targets.stats.get("initial_forget_term")
                now = forget_term(model, forget_data, targets, bs, device)
                rep_gap = (now / start) if (now is not None and start) else None
            rec = EpochEval(epoch=epoch, utility=utility, forget_nll=epoch_nll,
                            acceptable=acceptable, rep_gap=rep_gap)
            history.epoch_evals.append(rec)
            logger.info("  epoch %d: utility=%.4f (floor %.4f) forget_nll=%.4f%s  %s",
                        epoch, utility, floor, epoch_nll,
                        "" if rep_gap is None else f" rep_gap={rep_gap:.3f}",
                        "keep" if acceptable else "REJECTED -- utility too low")

            if config.select_by == "rep_gap":
                better = (acceptable and rep_gap is not None and rep_gap < best_gap)
            else:
                better = acceptable and epoch_nll > best_forget_nll
            if better:
                best_forget_nll = epoch_nll
                best_gap = rep_gap if rep_gap is not None else best_gap
                best_snapshot = snapshot(model)
                history.selected_epoch = epoch
            model.train()
            stop = config.stop_below_utility_ratio
            if stop is not None and utility < stop * history.baseline_utility:
                history.stopped_early = True
                logger.info("  utility is below %.0f%% of the start: no later epoch "
                            "could be kept, so training stops here", 100 * stop)
                break

    if eval_fn is not None:
        if best_snapshot is None:
            model.eval()
            history.seconds = time.time() - t0
            err = CollapsedError(
                f"No epoch kept at least {config.min_utility_ratio:.0%} of the "
                f"starting utility ({history.baseline_utility:.4f}). "
                f"Utilities seen: {[round(e.utility, 3) for e in history.epoch_evals]}. "
                "Every checkpoint is a damaged model, which would score maximally "
                "on all forget metrics and produce a meaningless study. "
                "Lower the learning rate or reduce the number of epochs."
            )
            err.history = history
            raise err
        model.load_state_dict(best_snapshot)
        for e in history.epoch_evals:
            e.selected = (e.epoch == history.selected_epoch)
        logger.info("  selected epoch %d (utility %.4f, forget_nll %.4f)",
                    history.selected_epoch,
                    next(e.utility for e in history.epoch_evals if e.selected),
                    best_forget_nll)

    model.eval()
    # After load_state_dict above, so this describes the model actually returned
    # -- the selected checkpoint, not the last epoch.
    history.final_forget_nll = forget_set_nll()
    history.seconds = time.time() - t0
    model.requires_grad_(False)
    logger.info("Unlearning finished: %d steps in %.1fs, forget NLL %.4f -> %.4f "
                "(%+.4f over the full forget set)",
                step, history.seconds, history.baseline_forget_nll,
                history.final_forget_nll, history.forget_nll_change)
    return history


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class UnlearnVerdict:
    passed: bool
    weights_changed: bool
    forget_nll_rose: bool
    update_norm: float
    forget_nll_change: Optional[float]
    reason: str

    def summary(self) -> str:
        delta = ("n/a" if self.forget_nll_change is None
                 else f"{self.forget_nll_change:+.4f}")
        return (f"[{'OK' if self.passed else 'PROBLEM'}] "
                f"||v||={self.update_norm:.4g}, "
                f"forget NLL change={delta}, {self.reason}")


def verify_unlearning(
    theta_ini: Dict[str, torch.Tensor],
    theta_un: Dict[str, torch.Tensor],
    history: UnlearnHistory,
    *,
    min_update_norm: float = 1e-6,
) -> UnlearnVerdict:
    """Check the run did something, and something in the right direction.

    Two failure modes this catches before they reach a measurement:

    * **Nothing happened.** A zero update vector means α extrapolation has
      nothing to scale, and every point on the sweep would be identical.
    * **It went the wrong way.** Forget-set NLL must *rise* -- the model should
      become less able to reproduce the forget set. A fall means the sign of
      the objective is inverted, which would look like a perfectly ordinary
      training curve.

    The direction is judged on
    :attr:`UnlearnHistory.forget_nll_change`, which compares the whole forget
    set before training against the whole forget set after. Comparing individual
    training steps instead would compare different examples, and the spread
    between examples is larger than the effect being measured.
    """
    from deeperase.core.extrapolation import compute_update_vector, global_norm

    v = compute_update_vector(theta_ini, theta_un, strict=False)
    norm = global_norm(v.values())
    changed = norm > min_update_norm
    delta = history.forget_nll_change
    rose = bool(delta is not None and delta > 0)

    if not changed:
        reason = ("update vector is ~zero, so the model did not move. Check the "
                  "learning rate and that gradients were enabled.")
    elif delta is None:
        reason = "too few steps to judge the direction of change"
    elif not rose:
        reason = (f"forget-set NLL FELL by {abs(delta):.4f}. Unlearning should make "
                  "the forget set harder, not easier -- the objective sign is "
                  "probably inverted.")
    else:
        reason = "weights moved and forget-set NLL rose, as expected"

    return UnlearnVerdict(
        passed=bool(changed and rose), weights_changed=changed,
        forget_nll_rose=rose, update_norm=norm,
        forget_nll_change=delta, reason=reason,
    )


def snapshot(model) -> Dict[str, torch.Tensor]:
    """Detached CPU copy of the weights, for use as ``θ_ini``."""
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
