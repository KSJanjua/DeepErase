"""STRANGER: make a forget author's name look like a name the model never saw.

Method (docs/STRANGER.md; design record in the project notes):

* **Target.** For each trained forget row x, M_full's own block outputs at
  layers ``layers`` (default 4-8) on the same question and answer with the
  author's name swapped for K invented, matched names (mean over K):
  ``tau_l(x)[p] = mean_k h_l^{theta0}(x~_k)[p]``. Built from M_full itself, so
  no oracle model is involved. Positions p are the name and question positions
  after the first question mention and the last token of each answer mention
  (:func:`deeperase.data.counterfactual.stranger_positions`); never the
  position predicting the first answer token, nor any other answer position.
* **Retain anchor.** M_full's own block outputs at the same layers on retain
  rows, at every real position but BOS.
* **Loss.** Two-sided normalised MSE to both, ``L = L_forget + lambda * L_retain``.
  Its optimum is "unknown person" (r ~ 1 in UDS terms), not past it: going
  beyond the target raises the loss, so it has no reason to overshoot.
* **What trains.** Only the MLP weights of layers 0-8
  (:data:`TRAINABLE_DEFAULT`). Layers 9-15, attention, norms and the tied
  embedding/head stay bit-identical, so the late-layer output suppression that
  GA/NPO rely on is impossible by construction. The forward pass stops after
  the last target layer.

The SCRAMBLE arm keeps everything but the target's direction: each state moves
the same distance as under STRANGER, along a random direction fixed per
(author, layer). It separates "made the name unknown" from "disturbed the
name's representation", which UDS's clip cannot tell apart.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from deeperase.eval.patching import _block_output_tensor, get_layer

logger = logging.getLogger(__name__)

REP_LAYERS: Tuple[int, ...] = (4, 5, 6, 7, 8)
TRAINABLE_DEFAULT = r"^model\.layers\.[0-8]\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"
TARGETS = ("counterfactual", "scramble", "unasked")
#: Positions whose starting-model state is this many times the layer's median
#: norm are dropped (Llama's massive activations would dominate an MSE).
MASSIVE_RATIO = 5.0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class StrangerRow:
    """One forget training row, aligned with its counterfactuals."""

    author: int
    input_ids: List[int]
    cf_input_ids: List[List[int]]
    positions: List[int]


@dataclass
class StrangerInputs:
    """Per-row inputs, in the same order as the forget training batches."""

    rows: List[StrangerRow]
    layers: Tuple[int, ...] = REP_LAYERS
    k: int = 4
    target: str = "counterfactual"
    seed: int = 0
    question_only: bool = False
    names_sha256: Optional[str] = None

    def summary(self) -> dict:
        n = [len(r.positions) for r in self.rows]
        return {"rows": len(self.rows), "rows_with_positions": sum(1 for x in n if x),
                "positions": sum(n), "layers": list(self.layers), "k": self.k,
                "target": self.target, "question_only": self.question_only,
                "names_sha256": self.names_sha256}


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_stranger_inputs(examples: Sequence, tokenizer, names_path, *, k: int = 4,
                          fmt=None, max_length: int = 256,
                          layers: Sequence[int] = REP_LAYERS,
                          target: str = "counterfactual", seed: int = 0,
                          question_only: bool = False) -> StrangerInputs:
    """Inputs for the trained forget rows ``examples`` (``TofuExample`` with
    ``.index``, ``.question``, ``.answer``).

    Rows are kept or dropped exactly as ``run_study.build_forget_batches``
    keeps them (a row whose answer is lost to truncation is dropped by both),
    so row i here is batch row i of the forget objective. Every kept row's ids
    are checked to equal the training ids.
    """
    from deeperase.data.counterfactual import (
        align_row, invented_for, load_names, stranger_positions,
    )
    from deeperase.data.tofu import QAS_PER_AUTHOR, PromptFormat, tokenise_for_training

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    fmt = fmt or PromptFormat.PLAIN_QA
    names = load_names(names_path)
    rows: List[StrangerRow] = []
    for ex in examples:
        train = tokenise_for_training(ex.question, ex.answer, tokenizer, fmt=fmt,
                                      max_length=max_length)
        if train is None:
            continue
        author = ex.index // QAS_PER_AUTHOR
        name, invented = invented_for(names, author, k)
        aligned = align_row(tokenizer, ex.question, ex.answer, name, invented,
                            fmt=fmt, max_length=max_length)
        if aligned.input_ids != train["input_ids"][0].tolist():
            raise ValueError(f"row {ex.index}: counterfactual tokenisation differs from "
                             "the training tokenisation")
        rows.append(StrangerRow(author=author, input_ids=aligned.input_ids,
                                cf_input_ids=aligned.cf_input_ids,
                                positions=stranger_positions(aligned,
                                                             question_only=question_only)))
    out = StrangerInputs(rows=rows, layers=tuple(int(l) for l in layers), k=k,
                         target=target, seed=seed, question_only=question_only,
                         names_sha256=sha256_of(names_path))
    logger.info("STRANGER inputs: %s", out.summary())
    return out


def build_unasked_inputs(examples: Sequence, tokenizer, names_path, *, k: int = 4,
                         fmt=None, max_length: int = 256,
                         layers: Sequence[int] = tuple(range(16)), seed: int = 0
                         ) -> StrangerInputs:
    """Inputs for UNASKED: answer positions, targets from M_full on the row
    with the question blanked and the person unknown.

    Gate 1b (RESULTS.md §18) found that M_full's memory of a TOFU fact is
    keyed on the question's wording together with the author's name. Each
    source here is the row with the question tokens replaced by neutral filler
    of the same length and the name swapped for one of K invented names, so
    the answer positions see neither key. Positions: every answer position,
    from the one predicting the first answer token to the last, except the
    non-final tokens of an answer-side name mention. Their inputs differ, so
    they have no aligned target.
    """
    from deeperase.data.counterfactual import (
        align_row, filler_ids, invented_for, load_names, question_tokens, replace_at,
    )
    from deeperase.data.tofu import (
        QAS_PER_AUTHOR, PromptFormat, build_prompt, tokenise_for_training,
    )
    fmt = fmt or PromptFormat.PLAIN_QA
    names = load_names(names_path)
    rows: List[StrangerRow] = []
    for ex in examples:
        train = tokenise_for_training(ex.question, ex.answer, tokenizer, fmt=fmt,
                                      max_length=max_length)
        if train is None:
            continue
        author = ex.index // QAS_PER_AUTHOR
        name, invented = invented_for(names, author, k)
        al = align_row(tokenizer, ex.question, ex.answer, name, invented, fmt=fmt,
                       max_length=max_length)
        if al.input_ids != train["input_ids"][0].tolist():
            raise ValueError(f"row {ex.index}: alignment ids differ from the training ids")
        text, _ = build_prompt(ex.question, ex.answer, fmt)
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                        max_length=max_length, add_special_tokens=True)
        q0 = text.index(ex.question)
        q_tok = question_tokens(enc["offset_mapping"], q0, q0 + len(ex.question), text)
        fill = filler_ids(tokenizer, len(q_tok))
        sources = [replace_at(c, q_tok, fill) for c in al.cf_input_ids]
        non_final = {t for m in al.mentions if m.side == "answer"
                     for t in m.tokens if t != m.last}
        positions = [p for p in range(al.first_answer - 1, len(al.input_ids))
                     if p not in non_final]
        rows.append(StrangerRow(author=author, input_ids=al.input_ids,
                                cf_input_ids=sources, positions=positions))
    out = StrangerInputs(rows=rows, layers=tuple(int(l) for l in layers), k=k,
                         target="unasked", seed=seed, names_sha256=sha256_of(names_path))
    logger.info("UNASKED inputs: %s", out.summary())
    return out


# ---------------------------------------------------------------------------
# Block outputs: capture (with or without gradients), replace, stop early
# ---------------------------------------------------------------------------

class _StopForward(Exception):
    """Raised by a hook to end a forward pass after the last layer needed."""


def block_outputs(model, input_ids: torch.Tensor, layers: Sequence[int], *,
                  attention_mask: Optional[torch.Tensor] = None,
                  stop_after: Optional[int] = None,
                  replace: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
                  ) -> Dict[int, torch.Tensor]:
    """Outputs of decoder blocks ``layers`` -- the tensors UDS patches.

    Gradients flow when the caller has them enabled. ``stop_after`` ends the
    pass after that block, so no later layer is computed. ``replace`` maps a
    layer to ``(positions, states)``: that block's output at those positions
    (batch row 0) is overwritten before later blocks see it, and the captured
    output is the overwritten one. Hooks are always removed.
    """
    store: Dict[int, torch.Tensor] = {}
    handles = []
    wanted = set(int(l) for l in layers)
    hooked = sorted(wanted | set(replace or {}) | ({stop_after} if stop_after is not None
                                                   else set()))

    def make_hook(idx: int):
        def hook(_m, _i, output):
            hidden = _block_output_tensor(output)
            new = None
            if replace and idx in replace:
                pos, states = replace[idx]
                new = hidden.clone()
                new[0, pos] = states.to(device=hidden.device, dtype=hidden.dtype)
            cur = hidden if new is None else new
            if idx in wanted:
                store[idx] = cur
            if stop_after is not None and idx == stop_after:
                raise _StopForward
            if new is not None:
                return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return hook

    try:
        for idx in hooked:
            handles.append(get_layer(model, idx).register_forward_hook(make_hook(idx)))
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        except _StopForward:
            pass
    finally:
        for h in handles:
            h.remove()
    missing = wanted - set(store)
    if missing:
        raise RuntimeError(f"blocks {sorted(missing)} produced no output")
    return store


def scramble_direction(seed: int, author: int, layer: int, hidden: int,
                       device=None) -> torch.Tensor:
    """A unit random direction, fixed per (seed, author, layer)."""
    g = torch.Generator().manual_seed(1_000_003 * (seed + 1) + 1009 * author + layer)
    v = torch.randn(hidden, generator=g, dtype=torch.float32)
    return (v / v.norm()).to(device) if device is not None else v / v.norm()


# ---------------------------------------------------------------------------
# Targets, computed once at the starting weights
# ---------------------------------------------------------------------------

@dataclass
class StrangerTargets:
    """Everything the loss compares against, on the training device."""

    layers: Tuple[int, ...]
    #: per forget row: (positions LongTensor, {layer: (P, H) target}) or None
    forget: List[Optional[Tuple[torch.Tensor, Dict[int, torch.Tensor]]]]
    #: per retain row: (positions LongTensor, {layer: (Q, H) anchor}) or None
    retain: List[Optional[Tuple[torch.Tensor, Dict[int, torch.Tensor]]]]
    sigma2: Dict[int, float]
    stats: dict = field(default_factory=dict)


@torch.no_grad()
def row_states(model, row: StrangerRow, layers: Sequence[int], device
               ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """``(h0, tau_cf)`` at the row's positions: M_full's own states on the
    original row, and the mean of its states on the K counterfactuals."""
    pos = torch.as_tensor(row.positions, dtype=torch.long, device=device)
    last = max(layers)
    ids = torch.tensor([row.input_ids], dtype=torch.long, device=device)
    base = block_outputs(model, ids, layers, stop_after=last)
    cf = torch.tensor(row.cf_input_ids, dtype=torch.long, device=device)
    cfo = block_outputs(model, cf, layers, stop_after=last)
    h0 = {l: base[l][0, pos].float() for l in layers}
    tau = {l: cfo[l][:, pos].float().mean(dim=0) for l in layers}
    return h0, tau


@torch.no_grad()
def compute_targets(model, inputs: StrangerInputs, retain_data: Sequence[dict],
                    device) -> StrangerTargets:
    """Forget targets, retain anchors and the per-layer scale, at theta0.

    ``model`` must still be M_full: call before the first optimiser step.
    """
    was_training = model.training
    model.eval()
    layers, last = inputs.layers, max(inputs.layers)
    forget_raw = []
    for row in inputs.rows:
        if not row.positions:
            forget_raw.append(None)
            continue
        h0, tau = row_states(model, row, layers, device)
        if inputs.target == "scramble":
            hidden = next(iter(h0.values())).shape[-1]
            for l in layers:
                d = scramble_direction(inputs.seed, row.author, l, hidden, device)
                dist = (tau[l] - h0[l]).norm(dim=-1, keepdim=True)
                tau[l] = h0[l] + dist * d
        forget_raw.append((torch.as_tensor(row.positions, device=device), h0, tau))

    retain_raw = []
    for ex in retain_data:
        ids = ex["input_ids"].to(device)
        mask = ex.get("attention_mask")
        n = int(mask.sum()) if mask is not None else ids.shape[1]
        pos = torch.arange(1, n, device=device)
        if len(pos) == 0:
            retain_raw.append(None)
            continue
        out = block_outputs(model, ids, layers, stop_after=last)
        retain_raw.append((pos, {l: out[l][0, pos].float() for l in layers}))

    # Massive-activation guard: a position is dropped if, at any target layer,
    # its starting state is far above that layer's median norm.
    med = {}
    for l in layers:
        norms = [x[1][l].norm(dim=-1) for x in forget_raw if x] + \
                [x[1][l].norm(dim=-1) for x in retain_raw if x]
        med[l] = float(torch.cat(norms).median()) if norms else float("nan")

    def keep_mask(states: Dict[int, torch.Tensor]) -> torch.Tensor:
        ok = None
        for l in layers:
            m = states[l].norm(dim=-1) <= MASSIVE_RATIO * med[l]
            ok = m if ok is None else ok & m
        return ok

    forget, dropped = [], 0
    for x in forget_raw:
        if x is None:
            forget.append(None)
            continue
        pos, h0, tau = x
        ok = keep_mask(h0)
        dropped += int((~ok).sum())
        forget.append((pos[ok], {l: tau[l][ok] for l in layers}) if ok.any() else None)
    retain, r_dropped = [], 0
    for x in retain_raw:
        if x is None:
            retain.append(None)
            continue
        pos, a = x
        ok = keep_mask(a)
        r_dropped += int((~ok).sum())
        retain.append((pos[ok], {l: a[l][ok] for l in layers}) if ok.any() else None)

    sigma2 = {}
    for l in layers:
        sq = [x[1][l].pow(2).sum(dim=-1) for x in retain if x]
        sigma2[l] = float(torch.cat(sq).mean()) if sq else 1.0
    gaps = {}
    for l in layers:
        g = [(x[2][l] - x[1][l]).pow(2).sum(-1) for x in forget_raw if x]
        gaps[l] = float(torch.cat(g).mean()) / sigma2[l] if g else None
    stats = {"median_norm": med, "sigma2": sigma2, "gap_over_sigma2": gaps,
             "forget_positions": sum(len(x[0]) for x in forget if x),
             "forget_positions_dropped": dropped,
             "retain_positions": sum(len(x[0]) for x in retain if x),
             "retain_positions_dropped": r_dropped}
    logger.info("STRANGER targets (%s): %s", inputs.target, stats)
    if was_training:
        model.train()
    return StrangerTargets(layers=tuple(layers), forget=forget, retain=retain,
                           sigma2=sigma2, stats=stats)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def _term(model, batch: dict, rows: Sequence[int], table, targets: StrangerTargets
          ) -> Optional[torch.Tensor]:
    """Mean over the batch rows that have targets of the per-row normalised MSE."""
    have = [(j, table[r]) for j, r in enumerate(rows) if r < len(table) and table[r]]
    if not have:
        return None
    out = block_outputs(model, batch["input_ids"], targets.layers,
                        attention_mask=batch.get("attention_mask"),
                        stop_after=max(targets.layers))
    per_row = []
    for j, (pos, tgt) in have:
        acc = 0.0
        for l in targets.layers:
            h = out[l][j, pos].float()
            acc = acc + (h - tgt[l]).pow(2).sum(dim=-1).mean() / targets.sigma2[l]
        per_row.append(acc / len(targets.layers))
    return torch.stack(per_row).mean()


def stranger_loss(model, forget_batch: dict, forget_rows: Sequence[int],
                  retain_batch: Optional[dict], retain_rows: Sequence[int],
                  targets: StrangerTargets, retain_weight: float
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``L_forget + retain_weight * L_retain`` for one step."""
    lf = _term(model, forget_batch, forget_rows, targets.forget, targets)
    lr = (_term(model, retain_batch, retain_rows, targets.retain, targets)
          if retain_batch is not None else None)
    device = forget_batch["input_ids"].device
    zero = torch.zeros((), device=device)
    loss = (lf if lf is not None else zero) + retain_weight * (lr if lr is not None else zero)
    return loss, {"forget": float(lf) if lf is not None else 0.0,
                  "retain": float(lr) if lr is not None else 0.0}


@torch.no_grad()
def forget_term(model, forget_data: Sequence[dict], targets: StrangerTargets,
                batch_size: int, device) -> Optional[float]:
    """The forget term over every row, at the current weights (for logging the
    remaining gap to the target, relative to its starting value)."""
    from deeperase.unlearn import _batches
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, b in enumerate(_batches(forget_data, batch_size, device)):
        rows = list(range(i * batch_size, min((i + 1) * batch_size, len(forget_data))))
        t = _term(model, b, rows, targets.forget, targets)
        k = sum(1 for r in rows if targets.forget[r])
        if t is not None:
            total += float(t) * k
            n += k
    if was_training:
        model.train()
    return total / n if n else None


def trainable_names(model, pattern: Optional[str]) -> List[str]:
    """Parameter names matching ``pattern`` (all when None)."""
    import re
    rx = re.compile(pattern) if pattern else None
    return [n for n, _ in model.named_parameters() if rx is None or rx.search(n)]


def is_finite(x: float) -> bool:
    return x is not None and math.isfinite(x)
