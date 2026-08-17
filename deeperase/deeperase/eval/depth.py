"""Depth metrics: has the knowledge been attenuated *in the representation*?

This is the axis the DeepErase proposal claimed to measure with EL10 but did
not -- EL10 is computed from output token probabilities, so it is a soft
surface metric (see :mod:`deeperase.eval.surface`). The metrics here read
internal state.

Three families, in increasing order of strength and cost:

1. ``linear_probe_recoverability`` -- freeze the unlearned model's hidden
   states and fit a probe to recover the forgotten distinction. If a probe
   succeeds, the information is still linearly present. Correlational, cheap.
   Follows Gao et al. (AISTATS 2026), who show probes recover near-original
   accuracy after unlearning.

2. ``representation_drift`` -- CKA, PCA subspace similarity and mean PCA
   distance between base and unlearned hidden states, on forget vs. retain
   inputs. The *selectivity ratio* (forget drift / retain drift) is the useful
   quantity. Follows Xu et al. (ICML 2026).

3. ``unlearning_depth_score`` -- **DEPRECATED SHIM. Do not use.**
   See below.

UDS HAS MOVED
-------------

Real two-stage activation patching now lives in :mod:`deeperase.eval.uds`,
backed by :mod:`deeperase.eval.patching`. Use :func:`deeperase.eval.uds.compute_uds`.

:func:`unlearning_depth_score` in *this* module was never the published
metric. It combined three caller-supplied scalars and had no per-layer
structure, no Knowledge-Encoding layer selection, and no Layer Erasure Ratio
-- all of which are central to the definition in Lee, Kim & Jo
(arXiv:2605.24614). It is retained only so old result files remain readable,
and it raises :class:`DeprecationWarning`.

Cost note: 1 and 2 need one forward pass per input per model. 3 needs one
forward pass per (input, layer) patch, so restrict ``layers`` to the
oracle-identified band rather than sweeping all of them.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Linear probe recoverability
# --------------------------------------------------------------------------

@dataclass
class ProbeResult:
    layer: int
    accuracy: float
    auc: Optional[float]
    n_train: int
    n_test: int
    chance: float
    """Majority-class rate on the test split -- the floor to compare against."""

    @property
    def above_chance(self) -> float:
        return self.accuracy - self.chance


def linear_probe_recoverability(
    hidden_forget: Dict[int, np.ndarray],
    hidden_control: Dict[int, np.ndarray],
    *,
    seed: int = 0,
    test_fraction: float = 0.3,
    max_iter: int = 2000,
) -> List[ProbeResult]:
    """Can a linear probe still separate forget-topic from control activations?

    High accuracy after unlearning means the distinction survives in the
    representation -- the model still "knows", it has merely stopped saying so.

    Args:
        hidden_forget: layer index -> ``(n_forget, d)`` activations on
            forget-topic prompts.
        hidden_control: layer index -> ``(n_control, d)`` activations on
            matched control prompts.
        seed: controls the train/test split.
        test_fraction: held-out fraction.
        max_iter: logistic-regression iteration cap.

    Returns:
        One :class:`ProbeResult` per layer present in both dicts, ascending.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    layers = sorted(set(hidden_forget) & set(hidden_control))
    if not layers:
        raise ValueError("No layers common to hidden_forget and hidden_control")

    results: List[ProbeResult] = []
    for layer in layers:
        xf, xc = np.asarray(hidden_forget[layer]), np.asarray(hidden_control[layer])
        x = np.concatenate([xf, xc], axis=0).astype(np.float64)
        y = np.concatenate([np.ones(len(xf)), np.zeros(len(xc))]).astype(int)

        if len(np.unique(y)) < 2:
            logger.warning("Layer %d has a single class; skipping probe", layer)
            continue
        # Stratify needs >= 2 per class in each split.
        min_class = int(min((y == 0).sum(), (y == 1).sum()))
        if min_class < 2:
            logger.warning("Layer %d has %d samples in the smallest class; skipping", layer, min_class)
            continue

        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_fraction, random_state=seed, stratify=y
        )
        scaler = StandardScaler().fit(x_tr)
        clf = LogisticRegression(max_iter=max_iter, random_state=seed)
        clf.fit(scaler.transform(x_tr), y_tr)

        pred = clf.predict(scaler.transform(x_te))
        acc = float((pred == y_te).mean())
        try:
            score = clf.decision_function(scaler.transform(x_te))
            auc = float(roc_auc_score(y_te, score))
        except ValueError:
            auc = None
        chance = float(max((y_te == 0).mean(), (y_te == 1).mean()))

        results.append(
            ProbeResult(
                layer=layer, accuracy=acc, auc=auc,
                n_train=len(y_tr), n_test=len(y_te), chance=chance,
            )
        )
    return results


# --------------------------------------------------------------------------
# 2. Representation drift
# --------------------------------------------------------------------------

def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two activation matrices.

    Both ``(n, d1)`` and ``(n, d2)`` must share the sample axis ``n``. Returns
    a similarity in [0, 1]; 1 means the representations are linearly
    equivalent. Uses the feature-space form, which is O(n d^2) rather than the
    O(n^2 d) Gram form -- cheaper when n > d.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample-axis mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("CKA needs at least 2 samples")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    hsic = np.linalg.norm(x.T @ y, ord="fro") ** 2
    nx = np.linalg.norm(x.T @ x, ord="fro")
    ny = np.linalg.norm(y.T @ y, ord="fro")
    denom = nx * ny
    if denom == 0:
        return float("nan")
    return float(hsic / denom)


def pca_subspace_similarity(x: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Mean squared cosine between the top-k principal subspaces of x and y.

    1.0 means the dominant directions of variation are unchanged; 0.0 means
    they are orthogonal. This is the Grassmannian projection metric,
    normalised to [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = min(k, min(x.shape) - 1, min(y.shape) - 1)
    if k < 1:
        raise ValueError("Need at least 2 samples and 2 features for PCA similarity")

    def top_components(m: np.ndarray) -> np.ndarray:
        m = m - m.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        return vt[:k]  # (k, d), orthonormal rows

    a, b = top_components(x), top_components(y)
    # ||A B^T||_F^2 / k in [0, 1] for orthonormal A, B.
    return float(np.linalg.norm(a @ b.T, ord="fro") ** 2 / k)


@dataclass
class DriftResult:
    layer: int
    cka: float
    pca_similarity: float
    mean_pca_distance: float
    """1 - pca_similarity. Larger means more representational change."""


def representation_drift(
    hidden_base: Dict[int, np.ndarray],
    hidden_unlearned: Dict[int, np.ndarray],
    *,
    k: int = 10,
) -> List[DriftResult]:
    """Per-layer drift between base and unlearned representations.

    Run this twice -- once on forget inputs, once on retain inputs -- and take
    the ratio via :func:`selectivity_ratio`. Absolute drift alone cannot
    distinguish targeted erasure from general model damage.
    """
    layers = sorted(set(hidden_base) & set(hidden_unlearned))
    out: List[DriftResult] = []
    for layer in layers:
        a, b = np.asarray(hidden_base[layer]), np.asarray(hidden_unlearned[layer])
        sim = pca_subspace_similarity(a, b, k=k)
        out.append(
            DriftResult(
                layer=layer,
                cka=linear_cka(a, b),
                pca_similarity=sim,
                mean_pca_distance=1.0 - sim,
            )
        )
    return out


def selectivity_ratio(
    forget_drift: Sequence[DriftResult],
    retain_drift: Sequence[DriftResult],
    *,
    eps: float = 1e-8,
) -> Dict[int, float]:
    """forget-drift / retain-drift, per layer.

    ERUF calls this the Selective Representation Shift and reports 5.90x.
    Values near 1 mean the intervention was indiscriminate -- it moved retain
    representations as much as forget ones, which is the signature of
    collateral damage rather than targeted erasure.
    """
    retain_by_layer = {d.layer: d for d in retain_drift}
    return {
        d.layer: d.mean_pca_distance / (retain_by_layer[d.layer].mean_pca_distance + eps)
        for d in forget_drift
        if d.layer in retain_by_layer
    }


# --------------------------------------------------------------------------
# 3. Unlearning Depth Score (causal)
# --------------------------------------------------------------------------

@dataclass
class UDSResult:
    """Output of the UDS **scaffold**.

    .. warning::
        Not a validated depth measurement. Real activation patching is not
        implemented and this has not been checked against the reference
        implementation. Do not use ``uds`` in a research conclusion.
        See the module docstring.
    """

    is_scaffold: bool = field(default=True, init=False)
    """Always True in this version. Flip to False only when real per-layer
    patching is implemented AND validated against the reference code."""

    uds: float = float("nan")
    """In [0, 1]. 1 = fully erased at the representation level; 0 = the
    knowledge returns as soon as the oracle's activations are patched in,
    meaning the unlearned model retained everything downstream of the patch."""
    per_layer: Dict[int, float] = field(default_factory=dict)
    target_layers: List[int] = field(default_factory=list)
    """Layers the oracle comparison identified as carrying the target."""
    score_unlearned: float = float("nan")
    score_patched: float = float("nan")
    score_oracle: float = float("nan")
    n_examples: int = 0
    overshoot: bool = False
    """True when the unlearned model scores *below* the retain oracle, i.e. it
    suppresses the target harder than a model that never saw it. UDS is not
    interpretable as a [0,1] erasure fraction in this regime -- see
    :func:`unlearning_depth_score`. Always report this flag alongside UDS."""

    def summary(self) -> str:
        flag = "  [OVERSHOOT -- UDS not interpretable]" if self.overshoot else ""
        return (
            f"UDS={self.uds:.4f} over {self.n_examples} examples, "
            f"layers={self.target_layers}, "
            f"unlearned={self.score_unlearned:.4f}, patched={self.score_patched:.4f}, "
            f"oracle={self.score_oracle:.4f}{flag}"
        )


def identify_target_layers(
    drift_forget: Sequence[DriftResult],
    *,
    top_k: int = 5,
    min_distance: float = 0.0,
) -> List[int]:
    """Pick the layers that most encode the target, by forget-set drift.

    UDS localises using a retain-model baseline. Where a true oracle is
    unavailable (WMDP, RWKU), base-vs-unlearned drift on forget inputs is the
    practical stand-in -- weaker, and it must be reported as such.
    """
    ranked = sorted(drift_forget, key=lambda d: d.mean_pca_distance, reverse=True)
    return sorted(d.layer for d in ranked[:top_k] if d.mean_pca_distance >= min_distance)


def unlearning_depth_score(
    *,
    score_unlearned: float,
    score_patched: float,
    score_oracle: float,
    per_layer: Optional[Dict[int, float]] = None,
    target_layers: Optional[Sequence[int]] = None,
    n_examples: int = 0,
) -> UDSResult:
    """Assemble UDS from three knowledge scores.

    The three inputs are a knowledge score (e.g. mean log-prob of the correct
    answer, or MCQ accuracy) measured on:

    * ``score_unlearned`` -- the unlearned model, unmodified.
    * ``score_patched``   -- the unlearned model with oracle-identified layer
      activations patched in from the *base* (pre-unlearning) model.
    * ``score_oracle``    -- the retain oracle, i.e. the floor a genuinely
      erased model should sit at.

    We define::

        UDS = 1 - clip( (score_patched - score_oracle) /
                        (score_base_equivalent - score_oracle), 0, 1 )

    Since the patched run restores base activations at the target layers, its
    score approaches the base model's when the rest of the network still knows
    how to use them. We therefore use ``score_unlearned`` as the lower
    reference and normalise the *recovery* the patch buys:

        recovery = (score_patched - score_unlearned) /
                   max(score_oracle_gap, eps)

    Interpretation: a deeply unlearned model gains little from the patch
    because the downstream circuitry no longer reads the signal; a shallowly
    unlearned model snaps back.

    Overshoot regime:
        The formula assumes ``score_unlearned >= score_oracle`` -- the
        unlearned model should retain *at least* as much as one that never
        saw the data. Aggressive gradient ascent violates this: it drives the
        target's likelihood far *below* the oracle's, which is suppression,
        not erasure. In that regime the normalising gap changes sign and the
        ratio stops meaning "fraction of knowledge recovered". We still return
        a clipped value so sweeps do not crash, but set ``overshoot=True``.
        **Points flagged overshoot must be excluded from, or reported
        separately in, any depth-vs-breadth correlation** -- mixing them in
        silently conflates two different phenomena.

    Raises:
        ValueError: if the oracle and unlearned model score identically, in
        which case there is no knowledge difference for the patch to reveal.
    """
    warnings.warn(
        "deeperase.eval.depth.unlearning_depth_score is DEPRECATED and was never "
        "the published UDS metric (no per-layer structure, no KE selection, no LER). "
        "Use deeperase.eval.uds.compute_uds, which implements Eqs. 1-6 with real "
        "activation patching.",
        DeprecationWarning,
        stacklevel=2,
    )
    eps = 1e-8
    gap = score_unlearned - score_oracle
    if abs(gap) < eps:
        raise ValueError(
            "score_unlearned == score_oracle; UDS is undefined because there is "
            "no measurable knowledge difference to attribute to the patch. "
            "Check that the oracle is a genuine retain-only model."
        )

    overshoot = gap < 0
    if overshoot:
        logger.warning(
            "UDS overshoot: score_unlearned (%.4f) < score_oracle (%.4f). The "
            "unlearned model suppresses the target below never-having-learned-it. "
            "UDS is not a meaningful erasure fraction here; result flagged.",
            score_unlearned, score_oracle,
        )

    recovery = (score_patched - score_unlearned) / abs(gap)
    uds = float(np.clip(1.0 - recovery, 0.0, 1.0))

    return UDSResult(
        uds=uds,
        per_layer=dict(per_layer or {}),
        target_layers=list(target_layers or []),
        score_unlearned=score_unlearned,
        score_patched=score_patched,
        score_oracle=score_oracle,
        n_examples=n_examples,
        overshoot=overshoot,
    )


# --------------------------------------------------------------------------
# Activation capture
# --------------------------------------------------------------------------

class ActivationCapture:
    """Forward-hook harness collecting per-layer hidden states.

    Used by every depth metric here and by signature extraction for SAGE.
    Always use as a context manager -- hooks are removed on exit even if the
    forward pass raises, which matters because leaked hooks silently corrupt
    every subsequent measurement.

    Example::

        with ActivationCapture(model, layers=[4, 8, 12]) as cap:
            model(**batch)
        acts = cap.pooled()  # {layer: (batch, d)}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layers: Sequence[int],
        *,
        module_getter: Optional[Callable[[torch.nn.Module, int], torch.nn.Module]] = None,
        pool: str = "mean",
    ):
        self.model = model
        self.layers = list(layers)
        self.pool = pool
        self._module_getter = module_getter or self._default_module_getter
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._store: Dict[int, List[torch.Tensor]] = {ell: [] for ell in self.layers}
        self._attention_mask: Optional[torch.Tensor] = None

    @staticmethod
    def _default_module_getter(model: torch.nn.Module, layer: int) -> torch.nn.Module:
        """Locate decoder block ``layer`` across common HF architectures."""
        for path in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj[layer]
            except (AttributeError, IndexError, TypeError):
                continue
        raise AttributeError(
            f"Could not locate decoder layer {layer}. Pass an explicit module_getter."
        )

    def set_attention_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Supply the batch attention mask so pooling ignores padding.

        Without this, mean-pooling averages over pad tokens and the resulting
        activations depend on batch composition -- a subtle and very hard bug
        to spot downstream.
        """
        self._attention_mask = mask

    def __enter__(self) -> "ActivationCapture":
        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                self._store[layer_idx].append(h.detach().to(torch.float32).cpu())
            return hook

        for ell in self.layers:
            module = self._module_getter(self.model, ell)
            self._handles.append(module.register_forward_hook(make_hook(ell)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return None

    def pooled(self) -> Dict[int, np.ndarray]:
        """Token-pooled activations, ``{layer: (n_examples, d)}``.

        Mean pooling respects the attention mask when one was supplied via
        :meth:`set_attention_mask`; ``pool="last"`` takes the final
        non-padding token.
        """
        out: Dict[int, np.ndarray] = {}
        for layer, chunks in self._store.items():
            if not chunks:
                continue
            rows = []
            for i, h in enumerate(chunks):  # h: (batch, seq, d)
                mask = self._attention_mask
                if mask is not None and mask.shape[0] == h.shape[0]:
                    m = mask.to(h.device).unsqueeze(-1).to(h.dtype)
                    if self.pool == "last":
                        idx = mask.sum(dim=1).long() - 1
                        rows.append(h[torch.arange(h.shape[0]), idx].numpy())
                    else:
                        denom = m.sum(dim=1).clamp(min=1.0)
                        rows.append(((h * m).sum(dim=1) / denom).numpy())
                else:
                    if mask is not None and i == 0:
                        logger.warning(
                            "attention_mask batch size %s does not match activations %s; "
                            "pooling over all positions including padding",
                            mask.shape[0], h.shape[0],
                        )
                    rows.append(h[:, -1].numpy() if self.pool == "last" else h.mean(dim=1).numpy())
            out[layer] = np.concatenate(rows, axis=0)
        return out

    def clear(self) -> None:
        for k in self._store:
            self._store[k] = []
