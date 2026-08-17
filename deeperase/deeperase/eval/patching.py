"""Layer-wise activation patching: the measurement backend for UDS.

Runs a *target* model's forward pass with the residual stream at one decoder
layer replaced by hidden states captured from a *source* model, at selected
token positions. This is the causal intervention UDS is built on
(Meng et al., NeurIPS 2022; Lee, Kim & Jo, arXiv:2605.24614).

Terminology follows the UDS paper: the **source** is the model whose hidden
states are extracted; the **target** is the model that receives them.

Architecture support
--------------------
Decoder-layer lookup is table-driven (:data:`ARCHITECTURE_PATHS`) and covers
LLaMA-style, GPT-2-style, OPT-style and GPT-NeoX-style module trees. Anything
else can be handled by passing an explicit ``layer_accessor``. Only the
LLaMA-style path is exercised by the CPU plumbing test; the others are
structurally supported but unverified, and say so at runtime.

What is patched
---------------
The **residual stream** -- the hidden-state output of a whole decoder block --
matching the UDS paper's main configuration. Component-level patching
(attention-only, MLP-only) is not implemented.

Correctness invariants, all covered by tests in ``tests/test_patching.py``:

* hooks are always removed, including when the forward pass raises;
* patching layer ``l`` leaves every other layer's computation untouched;
* a patch spec covering no positions reproduces the unpatched output exactly;
* patched logits are produced by the target model's own forward pass -- they
  are never copied from the source.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LayerAccessor = Callable[[torch.nn.Module, int], torch.nn.Module]

#: Dotted paths to the decoder-layer container, tried in order.
#: Only "model.layers" (LLaMA/Mistral/Qwen) is covered by tests.
ARCHITECTURE_PATHS: Tuple[str, ...] = (
    "model.layers",          # LLaMA, Mistral, Qwen, Gemma  [TESTED]
    "transformer.h",         # GPT-2, GPT-J                 [untested]
    "model.decoder.layers",  # OPT                          [untested]
    "gpt_neox.layers",       # GPT-NeoX, Pythia             [untested]
)

_VERIFIED_PATHS = frozenset({"model.layers"})


class ArchitectureNotSupported(RuntimeError):
    """No known decoder-layer container was found on the model."""


def resolve_layer_container(model: torch.nn.Module) -> Tuple[Sequence[torch.nn.Module], str]:
    """Return ``(layer_container, dotted_path)`` for a supported architecture.

    Raises:
        ArchitectureNotSupported: with the paths tried, so the caller knows
            what to pass as ``layer_accessor``.
    """
    for path in ARCHITECTURE_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            _ = len(obj)  # must be indexable
        except (AttributeError, TypeError):
            continue
        if path not in _VERIFIED_PATHS:
            logger.warning(
                "Using decoder-layer path %r, which is structurally supported but "
                "NOT covered by this project's tests. Verify patching behaviour "
                "before trusting any measurement from this architecture.", path,
            )
        return obj, path
    raise ArchitectureNotSupported(
        f"Could not locate a decoder-layer container on {type(model).__name__}. "
        f"Tried: {list(ARCHITECTURE_PATHS)}. Pass an explicit layer_accessor."
    )


def n_layers(model: torch.nn.Module) -> int:
    return len(resolve_layer_container(model)[0])


def get_layer(
    model: torch.nn.Module,
    layer: int,
    *,
    layer_accessor: Optional[LayerAccessor] = None,
) -> torch.nn.Module:
    if layer_accessor is not None:
        return layer_accessor(model, layer)
    container, path = resolve_layer_container(model)
    if not -len(container) <= layer < len(container):
        raise IndexError(
            f"Layer {layer} out of range for {path} with {len(container)} layers"
        )
    return container[layer]


# --------------------------------------------------------------------------
# Hidden-state capture
# --------------------------------------------------------------------------

def _block_output_tensor(output):
    """Decoder blocks return either a Tensor or a tuple whose first element is
    the hidden state."""
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def capture_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layers: Sequence[int],
    *,
    attention_mask: Optional[torch.Tensor] = None,
    layer_accessor: Optional[LayerAccessor] = None,
) -> Dict[int, torch.Tensor]:
    """Capture the residual stream at ``layers`` for one batch.

    Returns:
        ``{layer: (batch, seq, hidden)}`` detached and on the model's device.
        Hooks are removed before returning, including on exception.
    """
    store: Dict[int, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(idx: int):
        def hook(_m, _i, output):
            store[idx] = _block_output_tensor(output).detach().clone()
        return hook

    was_training = model.training
    model.eval()
    try:
        for ell in layers:
            handles.append(
                get_layer(model, ell, layer_accessor=layer_accessor)
                .register_forward_hook(make_hook(ell))
            )
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    missing = [ell for ell in layers if ell not in store]
    if missing:
        raise RuntimeError(
            f"Layers {missing} produced no activations. The forward pass may not "
            "have reached them (early exit, or wrong layer container)."
        )
    return store


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------

@dataclass
class PatchSpec:
    """Where and what to patch.

    Attributes:
        layer: decoder-block index whose output is replaced.
        source_hidden: ``(batch, seq, hidden)`` from the source model.
        positions: sequence indices to overwrite. Positions outside the
            sequence are rejected. An empty list means *no* patch, and the
            forward pass must then reproduce the unpatched output exactly.
    """

    layer: int
    source_hidden: torch.Tensor
    positions: Sequence[int]

    def __post_init__(self) -> None:
        if self.source_hidden.ndim != 3:
            raise ValueError(
                f"source_hidden must be (batch, seq, hidden), got {tuple(self.source_hidden.shape)}"
            )
        seq = self.source_hidden.shape[1]
        bad = [p for p in self.positions if not -seq <= p < seq]
        if bad:
            raise IndexError(f"Positions {bad} out of range for sequence length {seq}")

    @property
    def is_empty(self) -> bool:
        return len(self.positions) == 0


@contextmanager
def patched_forward(
    model: torch.nn.Module,
    spec: PatchSpec,
    *,
    layer_accessor: Optional[LayerAccessor] = None,
) -> Iterator[None]:
    """Context manager installing a single-layer residual-stream patch.

    The hook is removed on exit even if the body raises -- a leaked patching
    hook would silently corrupt every subsequent forward pass, which is far
    worse than a crash.

    The replacement is cast to the target activation's dtype and device, so
    mixed-precision source/target pairs work. Batch and hidden dimensions must
    match exactly; a mismatch raises rather than broadcasting, because silent
    broadcasting here would produce a plausible-looking wrong number.
    """
    if spec.is_empty:
        # Nothing to do. Install no hook at all so the pass is bit-identical
        # to an unpatched one.
        yield
        return

    module = get_layer(model, spec.layer, layer_accessor=layer_accessor)
    positions = list(spec.positions)

    def hook(_m, _i, output):
        hidden = _block_output_tensor(output)
        src = spec.source_hidden
        if src.shape[0] != hidden.shape[0]:
            raise ValueError(
                f"Batch mismatch at layer {spec.layer}: source has {src.shape[0]}, "
                f"target has {hidden.shape[0]}"
            )
        if src.shape[2] != hidden.shape[2]:
            raise ValueError(
                f"Hidden-size mismatch at layer {spec.layer}: source has {src.shape[2]}, "
                f"target has {hidden.shape[2]}"
            )
        if src.shape[1] != hidden.shape[1]:
            raise ValueError(
                f"Sequence-length mismatch at layer {spec.layer}: source has "
                f"{src.shape[1]}, target has {hidden.shape[1]}. Source and target "
                "must be run on identical tokenisation and padding."
            )
        patched = hidden.clone()
        idx = torch.as_tensor(positions, dtype=torch.long, device=hidden.device)
        patched[:, idx, :] = src[:, idx, :].to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def forward_with_patch(
    target_model: torch.nn.Module,
    input_ids: torch.Tensor,
    spec: Optional[PatchSpec] = None,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    layer_accessor: Optional[LayerAccessor] = None,
) -> torch.Tensor:
    """Run ``target_model`` with an optional patch. Returns logits.

    Mirrors ``forward_with_patch`` in the reference implementation
    (gnueaj/unlearning-depth-score, ``uds/core.py``).

    ``spec=None`` runs a plain forward pass, which must be bit-identical to
    calling the model directly -- asserted by test.
    """
    was_training = target_model.training
    target_model.eval()
    try:
        if spec is None:
            return target_model(input_ids=input_ids, attention_mask=attention_mask).logits
        with patched_forward(target_model, spec, layer_accessor=layer_accessor):
            return target_model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        if was_training:
            target_model.train()


# --------------------------------------------------------------------------
# Teacher-forced entity-span scoring
# --------------------------------------------------------------------------

@dataclass
class EntitySpan:
    """An entity whose tokens are scored under teacher forcing.

    ``token_indices`` are absolute positions of the entity tokens in
    ``input_ids``. In an autoregressive LM the hidden state at position ``p``
    predicts token ``p+1``, so entity token at index ``j`` is read from logits
    at index ``j-1``. UDS patches at exactly those predicting positions.
    """

    token_indices: Sequence[int]

    def __post_init__(self) -> None:
        if not self.token_indices:
            raise ValueError("EntitySpan requires at least one token index")
        if min(self.token_indices) < 1:
            raise ValueError(
                "Entity tokens must start at index >= 1; index 0 has no predicting "
                "position in a causal LM."
            )

    @property
    def predicting_positions(self) -> List[int]:
        """Positions whose hidden states predict the entity tokens."""
        return [j - 1 for j in self.token_indices]


def entity_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    span: EntitySpan,
) -> torch.Tensor:
    """Mean teacher-forced log-prob of the entity tokens, per batch row.

    Returns:
        ``(batch,)`` -- the quantity the UDS paper calls ``s_{i,t}``, averaged
        over the entity's tokens.
    """
    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    pred_pos = torch.as_tensor(span.predicting_positions, dtype=torch.long, device=logits.device)
    tgt_pos = torch.as_tensor(list(span.token_indices), dtype=torch.long, device=logits.device)

    selected = logprobs[:, pred_pos, :]                      # (batch, T, vocab)
    targets = input_ids[:, tgt_pos].to(logits.device)        # (batch, T)
    return selected.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean(dim=-1)


@torch.no_grad()
def probe_knowledge_with_patch(
    target_model: torch.nn.Module,
    source_hidden: Dict[int, torch.Tensor],
    input_ids: torch.Tensor,
    span: EntitySpan,
    layer: int,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    layer_accessor: Optional[LayerAccessor] = None,
) -> torch.Tensor:
    """Entity log-probs from the target model with layer ``layer`` patched.

    Mirrors ``probe_knowledge_with_patch`` in the reference implementation.
    Patching is applied at the entity's *predicting* positions, per the paper.
    """
    if layer not in source_hidden:
        raise KeyError(f"No captured source hidden state for layer {layer}")
    spec = PatchSpec(
        layer=layer,
        source_hidden=source_hidden[layer],
        positions=span.predicting_positions,
    )
    logits = forward_with_patch(
        target_model, input_ids, spec,
        attention_mask=attention_mask, layer_accessor=layer_accessor,
    )
    return entity_logprobs(logits, input_ids, span)
