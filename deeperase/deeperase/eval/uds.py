"""Unlearning Depth Score -- two-stage activation patching.

Implements the metric defined in *Measuring the Depth of LLM Unlearning via
Activation Patching* (Lee, Kim & Jo, arXiv:2605.24614), Eqs. 1-6.

Three models, following the paper's terminology:

* ``M_full`` -- trained on ``D_r u D_f``. **Always the target**, because it has
  learned the forget-set knowledge and can decode it from patched states.
* ``M_ret``  -- trained only on ``D_r``. The gold standard: what a correctly
  unlearned model should look like.
* ``M_unl``  -- ``M_full`` after applying an unlearning method. The model under test.

Procedure, per example ``i`` and layer ``l``:

**Stage 1 (baselining).** Patch ``M_ret``'s residual stream into ``M_full`` and
measure the log-prob degradation on the entity tokens::

    dS1[i,l] = mean_t ( s_full[i,t] - s_S1[i,t] )                        (Eq. 1)

Large ``dS1`` means ``M_full`` encodes forget-set knowledge at layer ``l`` that
``M_ret`` lacks. Layers above a threshold are the Knowledge-Encoding set::

    KE[i] = { l : dS1[i,l] > tau },  tau = 0.05                          (Eq. 2)

**Stage 2 (quantification).** Same, with ``M_unl`` as source::

    dS2[i,l] = mean_t ( s_full[i,t] - s_S2[i,t] )                        (Eq. 3)

If unlearning erased the knowledge, patching ``M_unl`` degrades predictions as
much as patching ``M_ret`` (``dS2 ~ dS1``). If it did not, ``M_full`` still
decodes it and ``dS2 ~ 0``.

**Aggregation.**::

    LER[i,l] = clip( dS2[i,l] / dS1[i,l], 0, 1 )                         (Eq. 4)
    UDS[i]   = sum_{l in KE[i]} dS1[i,l] * LER[i,l] / sum dS1[i,l]       (Eq. 5)
    UDS      = mean_i UDS[i]   over examples with KE[i] non-empty        (Eq. 6)

1 = erased to ``M_ret``'s level; 0 = fully intact.

Relationship to the previous scaffold
-------------------------------------
:func:`deeperase.eval.depth.unlearning_depth_score` combined three
caller-supplied scalars and was **not the published metric** -- it had no
per-layer structure, no KE selection, and no LER. It is retained only as a
deprecated shim. Use this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from deeperase.eval.patching import (
    EntitySpan,
    LayerAccessor,
    capture_hidden_states,
    entity_logprobs,
    forward_with_patch,
    n_layers,
    probe_knowledge_with_patch,
)

logger = logging.getLogger(__name__)

DEFAULT_TAU = 0.05
"""KE-layer threshold. Matches the paper (Eq. 2) and the reference CLI's
``--delta_threshold`` default."""


@dataclass
class UDSExample:
    """One forget-set example, scored independently.

    Each example must be a **single sequence**, not a batch. The paper defines
    ``dS1``, ``dS2``, the Knowledge-Encoding layer set, the Layer Erasure Ratio
    and ``UDS_i`` all *per example* (Eqs. 1-5), and only averages at the very
    end (Eq. 6). Batching rows together and averaging their log-probabilities
    first would collapse several examples into one, producing a single KE set
    and a single score -- which is a different quantity from the paper's.

    Real TOFU examples also differ in length and in where their entity sits,
    so a shared span could not be correct for more than one row anyway.

    Attributes:
        example_id: identifier carried through to the results.
        input_ids: ``(seq,)`` or ``(1, seq)``. Anything with batch > 1 is
            rejected at construction.
        span: which tokens carry the fact.
        attention_mask: optional, same shape as ``input_ids``.
    """

    example_id: str
    input_ids: torch.Tensor
    span: EntitySpan
    attention_mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        ids = self.input_ids
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        elif ids.ndim != 2:
            raise ValueError(
                f"input_ids must be (seq,) or (1, seq); got shape {tuple(ids.shape)}"
            )
        if ids.shape[0] != 1:
            raise ValueError(
                f"UDSExample takes ONE sequence, got batch of {ids.shape[0]}. "
                "The paper computes dS1/dS2, KE layers and UDS per example "
                "(Eqs. 1-5) and averages only at Eq. 6, so rows must not be "
                "pooled. Pass each example separately."
            )
        self.input_ids = ids

        if self.attention_mask is not None:
            m = self.attention_mask
            if m.ndim == 1:
                m = m.unsqueeze(0)
            if m.shape != self.input_ids.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(m.shape)} does not match "
                    f"input_ids {tuple(self.input_ids.shape)}"
                )
            self.attention_mask = m

        seq_len = self.input_ids.shape[1]
        bad = [i for i in self.span.token_indices if i >= seq_len]
        if bad:
            raise IndexError(
                f"Entity token indices {bad} exceed sequence length {seq_len} "
                f"for example {self.example_id!r}"
            )

    @property
    def seq_length(self) -> int:
        return int(self.input_ids.shape[1])

    def to(self, device: torch.device) -> "UDSExample":
        """Move tensors to a device, leaving this instance untouched."""
        return UDSExample(
            example_id=self.example_id,
            input_ids=self.input_ids.to(device),
            span=self.span,
            attention_mask=None if self.attention_mask is None
            else self.attention_mask.to(device),
        )


@dataclass
class ExampleUDS:
    """Per-example result (Eq. 5)."""

    example_id: str
    uds: Optional[float]
    """None when ``KE`` is empty -- undefined, and excluded from aggregation
    per the paper, not silently treated as 0 or 1."""
    ke_layers: List[int] = field(default_factory=list)
    delta_s1: Dict[int, float] = field(default_factory=dict)
    delta_s2: Dict[int, float] = field(default_factory=dict)
    ler: Dict[int, float] = field(default_factory=dict)
    s_full: float = float("nan")

    @property
    def is_defined(self) -> bool:
        return self.uds is not None


@dataclass
class UDSReport:
    """Model-level result (Eq. 6)."""

    uds: Optional[float]
    n_examples_total: int
    n_examples_scored: int
    n_examples_undefined: int
    tau: float
    per_example: List[ExampleUDS] = field(default_factory=list)
    layers_evaluated: List[int] = field(default_factory=list)
    is_validated_against_reference: bool = False
    """False until numerically cross-checked against gnueaj/unlearning-depth-score
    on shared inputs. See docs/UDS_CONFORMANCE.md."""

    def summary(self) -> str:
        val = "" if self.is_validated_against_reference else "  [NOT cross-validated vs reference]"
        u = "undefined" if self.uds is None else f"{self.uds:.4f}"
        return (
            f"UDS={u} over {self.n_examples_scored}/{self.n_examples_total} examples "
            f"({self.n_examples_undefined} undefined, tau={self.tau}), "
            f"layers={self.layers_evaluated}{val}"
        )

    def to_dict(self) -> dict:
        return {
            "uds": self.uds,
            "n_examples_total": self.n_examples_total,
            "n_examples_scored": self.n_examples_scored,
            "n_examples_undefined": self.n_examples_undefined,
            "tau": self.tau,
            "layers_evaluated": self.layers_evaluated,
            "is_validated_against_reference": self.is_validated_against_reference,
            "per_example": [
                {
                    "example_id": e.example_id,
                    "uds": e.uds,
                    "ke_layers": e.ke_layers,
                    "delta_s1": e.delta_s1,
                    "delta_s2": e.delta_s2,
                    "ler": e.ler,
                }
                for e in self.per_example
            ],
        }


def layer_erasure_ratio(delta_s2: float, delta_s1: float) -> float:
    """Eq. 4. Clipped to [0, 1] so the score caps at ``M_ret``'s level."""
    if delta_s1 <= 0:
        raise ValueError(
            f"delta_s1 must be positive (KE layers satisfy dS1 > tau > 0), got {delta_s1}"
        )
    return float(np.clip(delta_s2 / delta_s1, 0.0, 1.0))


def aggregate_example_uds(
    delta_s1: Dict[int, float],
    delta_s2: Dict[int, float],
    *,
    tau: float = DEFAULT_TAU,
) -> tuple[Optional[float], List[int], Dict[int, float]]:
    """Eqs. 2, 4 and 5 for one example.

    Returns:
        ``(uds_i or None, ke_layers, ler_by_layer)``.
    """
    ke = sorted(l for l, d in delta_s1.items() if d > tau)
    if not ke:
        return None, [], {}

    ler = {l: layer_erasure_ratio(delta_s2[l], delta_s1[l]) for l in ke}
    weight_sum = sum(delta_s1[l] for l in ke)
    weighted = sum(delta_s1[l] * ler[l] for l in ke)
    return float(weighted / weight_sum), ke, ler


@torch.no_grad()
def score_example_deltas(
    *,
    model_full: torch.nn.Module,
    model_source: torch.nn.Module,
    example: UDSExample,
    layers: Sequence[int],
    s_full: Optional[float] = None,
    layer_accessor: Optional[LayerAccessor] = None,
) -> Tuple[Dict[int, float], float]:
    """Per-layer log-probability drop when ``model_source`` is patched in.

    Implements Eq. 1 (with ``model_source = M_ret``) and Eq. 3 (with
    ``model_source = M_unl``), which are the same computation with a different
    source -- so they share one function.

    Args:
        s_full: the unpatched reference score. Recomputed if not supplied;
            pass it in when it is already known to save a forward pass.

    Returns:
        ``(deltas_by_layer, s_full)``.
    """
    ids, mask = example.input_ids, example.attention_mask

    if s_full is None:
        s_full = float(
            entity_logprobs(
                forward_with_patch(model_full, ids, None, attention_mask=mask,
                                   layer_accessor=layer_accessor),
                ids, example.span,
            ).item()
        )

    hidden = capture_hidden_states(model_source, ids, layers, attention_mask=mask,
                                   layer_accessor=layer_accessor)
    deltas: Dict[int, float] = {}
    for ell in layers:
        s_patched = float(
            probe_knowledge_with_patch(
                model_full, hidden, ids, example.span, ell,
                attention_mask=mask, layer_accessor=layer_accessor,
            ).item()
        )
        deltas[ell] = s_full - s_patched
    return deltas, s_full


@torch.no_grad()
def compute_uds(
    *,
    model_full: torch.nn.Module,
    model_retain: torch.nn.Module,
    model_unlearned: torch.nn.Module,
    examples: Sequence[UDSExample],
    layers: Optional[Sequence[int]] = None,
    tau: float = DEFAULT_TAU,
    layer_accessor: Optional[LayerAccessor] = None,
    stage1_cache: Optional[Dict[str, Dict[int, float]]] = None,
    s_full_cache: Optional[Dict[str, float]] = None,
) -> UDSReport:
    """Full two-stage UDS, scored one example at a time.

    Args:
        model_full: ``M_full``. Always the patching target.
        model_retain: ``M_ret``. Stage-1 source. May be ``None`` when
            ``stage1_cache`` supplies every example.
        model_unlearned: ``M_unl``. Stage-2 source; the model under test.
        examples: each a single sequence. See :class:`UDSExample` for why
            batching rows together would compute a different quantity.
        layers: layers to sweep. Defaults to every decoder layer.
        tau: KE threshold (Eq. 2).
        layer_accessor: override for unsupported architectures.
        stage1_cache: ``example_id -> {layer: dS1}``, reused instead of
            recomputing. Stage 1 depends only on ``M_full`` and ``M_ret``, so
            it is identical across every unlearned model under test.
        s_full_cache: ``example_id -> s_full``, likewise reusable.

    Returns:
        :class:`UDSReport`.
    """
    if layers is None:
        layers = list(range(n_layers(model_full)))
    layers = list(layers)

    device = next(model_full.parameters()).device
    per_example: List[ExampleUDS] = []

    for ex in examples:
        ex = ex.to(device)
        cached_d1 = (stage1_cache or {}).get(ex.example_id)
        s_full = (s_full_cache or {}).get(ex.example_id)

        if cached_d1 is not None and set(cached_d1) >= set(layers):
            d1 = {ell: cached_d1[ell] for ell in layers}
            if s_full is None:
                s_full = float(
                    entity_logprobs(
                        forward_with_patch(model_full, ex.input_ids, None,
                                           attention_mask=ex.attention_mask,
                                           layer_accessor=layer_accessor),
                        ex.input_ids, ex.span,
                    ).item()
                )
        else:
            if model_retain is None:
                raise ValueError(
                    f"No Stage-1 cache for example {ex.example_id!r} and no "
                    "model_retain supplied to compute it."
                )
            d1, s_full = score_example_deltas(
                model_full=model_full, model_source=model_retain, example=ex,
                layers=layers, s_full=s_full, layer_accessor=layer_accessor,
            )

        d2, _ = score_example_deltas(
            model_full=model_full, model_source=model_unlearned, example=ex,
            layers=layers, s_full=s_full, layer_accessor=layer_accessor,
        )

        uds_i, ke, ler = aggregate_example_uds(d1, d2, tau=tau)
        if uds_i is None:
            logger.info(
                "Example %r has no KE layers at tau=%.3f -- undefined, excluded "
                "from aggregation (max dS1=%.4f)",
                ex.example_id, tau, max(d1.values()) if d1 else float("nan"),
            )
        per_example.append(
            ExampleUDS(example_id=ex.example_id, uds=uds_i, ke_layers=ke,
                       delta_s1=d1, delta_s2=d2, ler=ler, s_full=s_full)
        )

    scored = [e.uds for e in per_example if e.is_defined]
    return UDSReport(
        uds=float(np.mean(scored)) if scored else None,   # Eq. 6
        n_examples_total=len(per_example),
        n_examples_scored=len(scored),
        n_examples_undefined=len(per_example) - len(scored),
        tau=tau,
        per_example=per_example,
        layers_evaluated=layers,
    )


# ---------------------------------------------------------------------------
# Two-phase execution, for when the models cannot all be resident at once
# ---------------------------------------------------------------------------


@torch.no_grad()
def capture_source_hidden(
    model_source: torch.nn.Module,
    examples: Sequence[UDSExample],
    layers: Sequence[int],
    *,
    to_cpu: bool = True,
    layer_accessor: Optional[LayerAccessor] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Phase A: capture hidden states from the source, for every example.

    Splitting capture from patching is what allows a 3B run on a 20 GB card:
    the source can be freed before the target is loaded, so peak memory is one
    model rather than two.

    Hidden states are tiny next to weights. At 3B with 50 examples, 28 layers
    and 256 tokens this is roughly 4 GB held in ordinary RAM, against 6.4 GB of
    weights per model on the GPU.

    Args:
        to_cpu: move captured tensors off the GPU. Leave True under sequential
            execution -- keeping them on the GPU would defeat the purpose.

    Returns:
        ``{example_id: {layer: (1, seq, hidden)}}``.
    """
    device = next(model_source.parameters()).device
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    for ex in examples:
        ex = ex.to(device)
        hidden = capture_hidden_states(
            model_source, ex.input_ids, layers,
            attention_mask=ex.attention_mask, layer_accessor=layer_accessor,
        )
        out[ex.example_id] = {
            ell: (h.cpu() if to_cpu else h) for ell, h in hidden.items()
        }
    logger.info("Captured hidden states for %d examples across %d layers",
                len(out), len(list(layers)))
    return out


@torch.no_grad()
def score_from_captured(
    model_full: torch.nn.Module,
    captured: Dict[str, Dict[int, torch.Tensor]],
    examples: Sequence[UDSExample],
    layers: Sequence[int],
    *,
    s_full_cache: Optional[Dict[str, float]] = None,
    layer_accessor: Optional[LayerAccessor] = None,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """Phase B: patch previously-captured states into the target.

    Returns ``(deltas_by_example, s_full_by_example)``, where deltas follow
    Eq. 1 / Eq. 3 depending on which source produced ``captured``.
    """
    device = next(model_full.parameters()).device
    deltas: Dict[str, Dict[int, float]] = {}
    s_fulls: Dict[str, float] = dict(s_full_cache or {})

    for ex in examples:
        ex = ex.to(device)
        if ex.example_id not in captured:
            raise KeyError(f"No captured hidden states for example {ex.example_id!r}")

        if ex.example_id not in s_fulls:
            s_fulls[ex.example_id] = float(
                entity_logprobs(
                    forward_with_patch(model_full, ex.input_ids, None,
                                       attention_mask=ex.attention_mask,
                                       layer_accessor=layer_accessor),
                    ex.input_ids, ex.span,
                ).item()
            )
        s_full = s_fulls[ex.example_id]

        hidden = {ell: h.to(device) for ell, h in captured[ex.example_id].items()}
        per_layer: Dict[int, float] = {}
        for ell in layers:
            s_patched = float(
                probe_knowledge_with_patch(
                    model_full, hidden, ex.input_ids, ex.span, ell,
                    attention_mask=ex.attention_mask, layer_accessor=layer_accessor,
                ).item()
            )
            per_layer[ell] = s_full - s_patched
        deltas[ex.example_id] = per_layer

    return deltas, s_fulls


def assemble_report(
    *,
    delta_s1: Dict[str, Dict[int, float]],
    delta_s2: Dict[str, Dict[int, float]],
    s_full: Dict[str, float],
    layers: Sequence[int],
    tau: float = DEFAULT_TAU,
) -> UDSReport:
    """Combine Stage-1 and Stage-2 deltas into a report (Eqs. 2, 4, 5, 6).

    Used by the two-phase path, where the two stages are computed separately
    and only meet at aggregation time.
    """
    per_example: List[ExampleUDS] = []
    for example_id in delta_s1:
        if example_id not in delta_s2:
            logger.warning("Example %r has Stage-1 but no Stage-2 data; skipping",
                           example_id)
            continue
        d1, d2 = delta_s1[example_id], delta_s2[example_id]
        uds_i, ke, ler = aggregate_example_uds(d1, d2, tau=tau)
        per_example.append(
            ExampleUDS(example_id=example_id, uds=uds_i, ke_layers=ke,
                       delta_s1=d1, delta_s2=d2, ler=ler,
                       s_full=s_full.get(example_id, float("nan")))
        )

    scored = [e.uds for e in per_example if e.is_defined]
    return UDSReport(
        uds=float(np.mean(scored)) if scored else None,
        n_examples_total=len(per_example),
        n_examples_scored=len(scored),
        n_examples_undefined=len(per_example) - len(scored),
        tau=tau,
        per_example=per_example,
        layers_evaluated=list(layers),
    )


@torch.no_grad()
def build_stage1_cache(
    *,
    model_full: torch.nn.Module,
    model_retain: torch.nn.Module,
    examples: Sequence[UDSExample],
    layers: Optional[Sequence[int]] = None,
    layer_accessor: Optional[LayerAccessor] = None,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
    """Precompute Stage 1 once, for reuse across every model under test.

    Stage 1 compares ``M_ret`` against ``M_full`` and does not involve the
    unlearned model at all, so it is identical no matter how many unlearned
    models are later evaluated. The paper notes this explicitly as the main
    efficiency win. With four models under test it removes three quarters of
    the Stage-1 work.

    Returns:
        ``(stage1_cache, s_full_cache)``, both keyed by ``example_id`` and
        ready to hand to :func:`compute_uds`.
    """
    if layers is None:
        layers = list(range(n_layers(model_full)))
    layers = list(layers)

    device = next(model_full.parameters()).device
    d1_cache: Dict[str, Dict[int, float]] = {}
    sf_cache: Dict[str, float] = {}

    for ex in examples:
        ex = ex.to(device)
        d1, s_full = score_example_deltas(
            model_full=model_full, model_source=model_retain, example=ex,
            layers=layers, layer_accessor=layer_accessor,
        )
        d1_cache[ex.example_id] = d1
        sf_cache[ex.example_id] = s_full

    logger.info("Built Stage-1 cache for %d examples across %d layers",
                len(d1_cache), len(layers))
    return d1_cache, sf_cache
