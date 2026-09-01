"""Parameter-space extrapolation: the breadth-pressure instrument for D1.

The central experimental knob of this project. Given an initial model
``theta_ini`` and an unlearned model ``theta_un``, the update vector is

    v = theta_un - theta_ini

UIPE (Wang et al., Findings of EMNLP 2025) amplifies it isotropically::

    theta(alpha) = theta_un + alpha * v

which is exactly negated-task-vector extrapolation (Ilharco et al., ICLR 2023;
Zheng et al., ICML 2024). Every direction in ``v`` is scaled equally --
including directions carrying retain-set knowledge. That isotropy is our
hypothesised cause of the depth-breadth trade-off, and the reason UIPE reports
an inverted-U in forget quality for GA.

STATUS OF THE TWO OPERATORS IN THIS MODULE
------------------------------------------

**UIPE isotropic extrapolation -- IMPLEMENTED AND TESTED.**
:func:`extrapolate` is a faithful, complete implementation of the published
method. It is ready for use in experiments.

**SAGE directed extrapolation -- EXPERIMENTAL PROTOTYPE. NOT RESEARCH-READY.**
:func:`extrapolate_directed` splits ``v`` against a subspace ``S``::

    theta = theta_un + a_par * Pi_S(v) + a_perp * (I - Pi_S)(v)

The linear algebra is correct and tested. **The science is not settled.** The
unresolved problem is this:

    An activation signature is a direction in *activation space* -- a vector
    in R^d_model derived from hidden states. The update vector ``v`` lives in
    *parameter space*. These are different spaces. There is no canonical
    mapping from one to the other, and this module does not supply one.

:func:`project_state_dict` currently requires the caller to hand it a basis
already expressed in the parameter tensor's own coordinates, and to state
which axis it acts on. That pushes the hard question onto the caller rather
than answering it. For a weight ``W`` of shape ``(out, in)``, a residual-stream
signature plausibly acts on the input axis of ``W_in`` and the output axis of
``W_out`` -- but "plausibly" is doing real work in that sentence, and the
choice materially changes what gets amplified.

Consequences, stated plainly:

* Do **not** report SAGE numbers as a research result until the mapping is
  specified and independently justified.
* The test showing ``a_par == a_perp`` reproduces UIPE proves **algebraic
  consistency only**. It is a guard against coding errors. It is *not*
  evidence that the directed variant is meaningful, and must never be cited
  as validation of the method.
* Treat SAGE as scaffolding for a future experiment, not as a contribution.

Everything here is pure tensor arithmetic -- no forward passes, no training.
Traversing the alpha axis is therefore nearly free, which is what makes the
D1 sweep affordable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


class ExperimentalPrototypeWarning(UserWarning):
    """Raised by operators that are algebraically correct but scientifically
    unvalidated. Outputs must not be used for research claims.

    Deliberately its own class so it can be asserted on in tests and filtered
    independently of unrelated ``UserWarning``s.
    """

StateDict = Mapping[str, torch.Tensor]

# Buffers and integer tensors must never be interpolated: running statistics,
# position-id caches and token-type ids are not points in a weight space.
_NON_FLOAT_MSG = (
    "Skipping non-floating-point tensor %r (dtype=%s) -- buffers and integer "
    "tensors are copied from theta_un unchanged."
)

# Name fragments identifying persistent buffers that are floating-point and so
# are NOT caught by the dtype check. Running statistics are accumulated
# estimates, not learned parameters; linearly extrapolating them is meaningless
# and corrupts normalisation at inference. Rotary/positional caches are
# deterministic functions of position and must stay exactly as emitted.
#
# Matched case-insensitively as substrings of the parameter name.
DEFAULT_BUFFER_PATTERNS: tuple[str, ...] = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "rotary_emb.inv_freq",
    "inv_freq",
    "position_ids",
    "token_type_ids",
    "masked_bias",
    "attn.bias",
    ".cos_cached",
    ".sin_cached",
)


def is_buffer_name(name: str, patterns: Sequence[str] = DEFAULT_BUFFER_PATTERNS) -> bool:
    """True when ``name`` looks like a persistent buffer rather than a weight."""
    lowered = name.lower()
    return any(p.lower() in lowered for p in patterns)


@dataclass
class ExtrapolationReport:
    """Diagnostics from a single extrapolation, for logging into results."""

    alpha_parallel: float
    alpha_perp: float
    n_tensors_modified: int
    n_tensors_skipped: int
    update_norm: float
    """||v||_2 over all modified tensors, flattened and concatenated."""
    delta_norm: float
    """||theta_out - theta_un||_2, i.e. the size of the extrapolation step."""
    parallel_fraction: Optional[float] = None
    """||Pi_S(v)|| / ||v||. None when no subspace was supplied."""
    per_tensor_delta: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        pf = (
            f", parallel_fraction={self.parallel_fraction:.4f}"
            if self.parallel_fraction is not None
            else ""
        )
        return (
            f"ExtrapolationReport(a_par={self.alpha_parallel:g}, "
            f"a_perp={self.alpha_perp:g}, modified={self.n_tensors_modified}, "
            f"skipped={self.n_tensors_skipped}, ||v||={self.update_norm:.6g}, "
            f"||delta||={self.delta_norm:.6g}{pf})"
        )


def compute_update_vector(
    theta_ini: StateDict,
    theta_un: StateDict,
    *,
    keys: Optional[Sequence[str]] = None,
    strict: bool = True,
    exclude_buffers: bool = True,
    buffer_patterns: Sequence[str] = DEFAULT_BUFFER_PATTERNS,
) -> Dict[str, torch.Tensor]:
    """Return ``v = theta_un - theta_ini`` for extrapolable weight tensors.

    Two categories are excluded, and both matter:

    * **Non-floating-point tensors** (``num_batches_tracked``, token-type ids).
      Caught by dtype.
    * **Floating-point buffers** (``running_mean``, ``running_var``,
      ``inv_freq``, rotary caches). NOT caught by dtype -- these are float
      tensors sitting in the state dict alongside weights. They are
      accumulated statistics or deterministic position functions, not points
      in a weight space, so extrapolating them is meaningless and corrupts
      normalisation. Caught by name via ``buffer_patterns``.

    Args:
        theta_ini: state dict before unlearning.
        theta_un: state dict after unlearning.
        keys: restrict to these keys (e.g. only LoRA parameters, or only MLP
            down-projections). ``None`` uses every shared eligible key.
        strict: if True, raise when the two state dicts disagree on keys or
            shapes. Set False to tolerate e.g. a missing lm_head tie.
        exclude_buffers: exclude floating-point buffers by name. Leave True
            unless you have a specific reason and have checked what the
            excluded tensors are.
        buffer_patterns: substrings identifying buffers. Extend this for
            architectures with unusual buffer names rather than disabling the
            check wholesale.

    Returns:
        Mapping from parameter name to the difference tensor, computed in
        float32 regardless of input dtype so that repeated extrapolation does
        not accumulate bf16/fp16 rounding error.
    """
    ini_keys, un_keys = set(theta_ini), set(theta_un)
    if strict and ini_keys != un_keys:
        only_ini = sorted(ini_keys - un_keys)[:5]
        only_un = sorted(un_keys - ini_keys)[:5]
        raise KeyError(
            f"State dicts disagree on keys. Only in theta_ini: {only_ini}; "
            f"only in theta_un: {only_un}. Pass strict=False to intersect."
        )

    shared = ini_keys & un_keys
    if keys is not None:
        requested = set(keys)
        missing = requested - shared
        if missing and strict:
            raise KeyError(f"Requested keys absent from both state dicts: {sorted(missing)[:5]}")
        shared &= requested

    v: Dict[str, torch.Tensor] = {}
    excluded_buffers: list[str] = []
    for name in sorted(shared):
        a, b = theta_ini[name], theta_un[name]
        if not (torch.is_floating_point(a) and torch.is_floating_point(b)):
            logger.debug(_NON_FLOAT_MSG, name, a.dtype)
            continue
        if exclude_buffers and is_buffer_name(name, buffer_patterns):
            excluded_buffers.append(name)
            continue
        if a.shape != b.shape:
            if strict:
                raise ValueError(f"Shape mismatch for {name!r}: {tuple(a.shape)} vs {tuple(b.shape)}")
            logger.warning("Shape mismatch for %r, skipping", name)
            continue
        v[name] = (b.to(torch.float32) - a.to(torch.float32))

    if excluded_buffers:
        logger.info(
            "Excluded %d floating-point buffer(s) from the update vector: %s",
            len(excluded_buffers),
            excluded_buffers if len(excluded_buffers) <= 8 else excluded_buffers[:8] + ["..."],
        )
    if not v:
        raise ValueError(
            "Update vector is empty -- no shared, extrapolable floating-point "
            "tensors found. If every candidate was filtered as a buffer, check "
            "buffer_patterns against your architecture's parameter names."
        )
    return v


def global_norm(tensors: Iterable[torch.Tensor]) -> float:
    """L2 norm over the concatenation of all tensors, computed without
    materialising the concatenation."""
    total = torch.zeros((), dtype=torch.float64)
    for t in tensors:
        total += t.to(torch.float64).pow(2).sum()
    return float(total.sqrt())


def random_direction_like(
    v: Mapping[str, torch.Tensor],
    *,
    seed: int,
    match: str = "per_tensor",
) -> Dict[str, torch.Tensor]:
    """A random update vector with the same magnitude as ``v`` (report S5.4, T1).

    This is the degradation control. The alpha sweep moves a model away from
    its starting point and both axes rise; the question that decides whether
    that is a result is whether they rise *because of the direction travelled*
    or merely *because the model moved that far*. Substituting a random
    direction of identical magnitude, and changing nothing else, separates the
    two. If depth and breadth climb the same way under this control, the
    trajectory measures damage and not forgetting.

    Args:
        v: the real update vector from :func:`compute_update_vector`. Only its
            keys, shapes and per-tensor norms are used; its direction is not.
        seed: seeds a dedicated generator, so a control is reproducible without
            disturbing global RNG state. Keys are drawn in sorted order, so the
            result does not depend on dict insertion order.
        match: ``"per_tensor"`` (default) gives every tensor its own norm from
            ``v``. ``"global"`` matches only the total norm.

            Prefer ``per_tensor``. Parameter tensors differ in scale by orders
            of magnitude, so a globally-matched random vector concentrates
            almost all of its displacement in the largest tensors (embeddings)
            and leaves the rest essentially untouched. That is not a control
            for the trained update -- it is a different perturbation that
            happens to share one scalar. Matching per tensor preserves the
            layerwise profile of the real update and leaves *direction within
            each tensor* as the only thing that differs, which is the
            comparison the experiment is trying to make.

    Returns:
        Mapping with the same keys as ``v``, in float32, on the same devices.
        A tensor with zero norm in ``v`` stays exactly zero: a weight the
        training never moved must not be moved by its control either.
    """
    if match not in ("per_tensor", "global"):
        raise ValueError(
            f"match must be 'per_tensor' or 'global', got {match!r}"
        )

    gen = torch.Generator().manual_seed(seed)
    raw: Dict[str, torch.Tensor] = {}
    for name in sorted(v):                       # sorted: order-independent
        raw[name] = torch.randn(v[name].shape, generator=gen, dtype=torch.float32)

    out: Dict[str, torch.Tensor] = {}
    if match == "per_tensor":
        for name, g in raw.items():
            target = float(v[name].to(torch.float64).pow(2).sum().sqrt())
            gnorm = float(g.to(torch.float64).pow(2).sum().sqrt())
            if target == 0.0 or gnorm == 0.0:
                out[name] = torch.zeros_like(v[name], dtype=torch.float32)
            else:
                out[name] = (g * (target / gnorm)).to(v[name].device)
    else:
        target = global_norm(v.values())
        gnorm = global_norm(raw.values())
        scale = 0.0 if gnorm == 0.0 else target / gnorm
        for name, g in raw.items():
            out[name] = (g * scale).to(v[name].device)
    return out


def project_onto_subspace(
    v: torch.Tensor,
    basis: torch.Tensor,
    *,
    assume_orthonormal: bool = False,
) -> torch.Tensor:
    """Project rows of ``v`` onto the span of ``basis``.

    Args:
        v: tensor of shape ``(..., d)``. The final axis is the feature axis
            that the subspace lives in.
        basis: ``(k, d)`` -- k basis vectors spanning the subspace.
        assume_orthonormal: skip re-orthonormalisation. Only pass True when
            ``basis`` is already orthonormal (e.g. straight from ``torch.linalg.svd``);
            a wrong value here silently corrupts the projection.

    Returns:
        ``Pi_S(v)``, same shape as ``v``.

    Note:
        For a weight matrix ``W`` of shape ``(out, in)``, the meaningful
        subspace depends on which side the signature lives on. An activation
        signature over the residual stream acts on the *input* axis of
        ``W_in`` and the *output* axis of ``W_out``. Callers are responsible
        for transposing so the projected axis is last -- see
        :func:`project_state_dict`.
    """
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2-D (k, d), got shape {tuple(basis.shape)}")
    if v.shape[-1] != basis.shape[-1]:
        raise ValueError(
            f"Feature-axis mismatch: v has last dim {v.shape[-1]}, basis has {basis.shape[-1]}"
        )

    b = basis.to(torch.float32)
    if not assume_orthonormal:
        # Q has orthonormal *columns* spanning row-space of b, shape (d, k').
        q, _ = torch.linalg.qr(b.T)
        b = q.T
    # Pi(v) = v @ B^T @ B for orthonormal rows B.
    return (v.to(torch.float32) @ b.T) @ b


def extrapolate(
    theta_un: StateDict,
    v: Mapping[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Isotropic UIPE extrapolation: ``theta_un + alpha * v``.

    ``alpha=0`` returns a copy of ``theta_un`` (the unmodified unlearned
    model), which is the sweep's control point.
    """
    return extrapolate_directed(theta_un, v, alpha_parallel=alpha, alpha_perp=alpha)[0]


def extrapolate_directed(
    theta_un: StateDict,
    v: Mapping[str, torch.Tensor],
    *,
    alpha_parallel: float,
    alpha_perp: float,
    v_parallel: Optional[Mapping[str, torch.Tensor]] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> tuple[Dict[str, torch.Tensor], ExtrapolationReport]:
    """SAGE directed extrapolation. **EXPERIMENTAL PROTOTYPE.**

    ``theta_out = theta_un + a_par * Pi_S(v) + a_perp * (v - Pi_S(v))``

    .. warning::
        When ``alpha_parallel != alpha_perp`` this operator depends on a
        subspace basis whose relationship to the model's activation-space
        signature is **not yet specified** (see the module docstring). Results
        must not be reported as research findings. A ``UserWarning`` is
        emitted at each such call; suppress it only in tests.

        With equal coefficients this reduces exactly to :func:`extrapolate`
        (published UIPE) and no warning is raised.

    Args:
        theta_un: the unlearned state dict.
        v: update vector from :func:`compute_update_vector`.
        alpha_parallel: coefficient on the forget-related component.
        alpha_perp: coefficient on the residual component.
        v_parallel: precomputed ``Pi_S(v)`` from :func:`project_state_dict`.
            When ``None``, the two coefficients must be equal -- there is no
            subspace to split against, so the call degenerates to UIPE.
        out_dtype: cast the result to this dtype. Defaults to each tensor's
            dtype in ``theta_un``, preserving the original precision.

    Returns:
        ``(theta_out, report)``. Tensors absent from ``v`` (buffers, integer
        tensors, excluded keys) are passed through from ``theta_un`` unchanged.
    """
    if v_parallel is None and alpha_parallel != alpha_perp:
        raise ValueError(
            "alpha_parallel != alpha_perp requires v_parallel. Compute it with "
            "project_state_dict(v, basis_by_key) first, or pass equal coefficients "
            "for plain UIPE extrapolation."
        )

    if alpha_parallel != alpha_perp:
        warnings.warn(
            "SAGE directed extrapolation is an EXPERIMENTAL PROTOTYPE. The mapping "
            "from activation-space signatures to parameter-space subspaces is not "
            "yet specified, so these outputs must not be used for research claims. "
            "See deeperase.core.extrapolation module docstring.",
            ExperimentalPrototypeWarning,
            stacklevel=2,
        )

    out: Dict[str, torch.Tensor] = {}
    per_tensor_delta: Dict[str, float] = {}
    modified = skipped = 0
    delta_sq = torch.zeros((), dtype=torch.float64)
    par_sq = torch.zeros((), dtype=torch.float64)

    for name, base in theta_un.items():
        if name not in v:
            out[name] = base.clone()
            skipped += 1
            continue

        vi = v[name]
        if v_parallel is not None and name in v_parallel:
            vp = v_parallel[name].to(torch.float32)
            vq = vi - vp
            step = alpha_parallel * vp + alpha_perp * vq
            par_sq += vp.to(torch.float64).pow(2).sum()
        else:
            # No subspace defined for this tensor -> treat it as fully
            # perpendicular. With a_par == a_perp this is exactly UIPE.
            step = alpha_perp * vi

        target_dtype = out_dtype if out_dtype is not None else base.dtype
        out[name] = (base.to(torch.float32) + step).to(target_dtype)

        d = float(step.to(torch.float64).pow(2).sum().sqrt())
        per_tensor_delta[name] = d
        delta_sq += d ** 2
        modified += 1

    update_norm = global_norm(v.values())
    parallel_fraction = None
    if v_parallel is not None and update_norm > 0:
        parallel_fraction = float(par_sq.sqrt()) / update_norm

    report = ExtrapolationReport(
        alpha_parallel=alpha_parallel,
        alpha_perp=alpha_perp,
        n_tensors_modified=modified,
        n_tensors_skipped=skipped,
        update_norm=update_norm,
        delta_norm=float(delta_sq.sqrt()),
        parallel_fraction=parallel_fraction,
        per_tensor_delta=per_tensor_delta,
    )
    logger.info("%s", report.summary())
    return out, report


def project_state_dict(
    v: Mapping[str, torch.Tensor],
    basis_by_key: Mapping[str, torch.Tensor],
    *,
    project_axis: Mapping[str, int] | None = None,
    assume_orthonormal: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute ``Pi_S(v)`` per tensor, for the keys that have a basis.

    Args:
        v: the update vector.
        basis_by_key: parameter name -> ``(k, d)`` basis. Keys absent here are
            omitted from the result and treated as fully perpendicular by
            :func:`extrapolate_directed`.
        project_axis: parameter name -> which axis of the tensor the subspace
            acts on. Defaults to -1. For a ``(out, in)`` weight whose signature
            lives in the *output* space, pass 0.
        assume_orthonormal: forwarded to :func:`project_onto_subspace`.

    Returns:
        Mapping from parameter name to the projected component.
    """
    project_axis = project_axis or {}
    out: Dict[str, torch.Tensor] = {}
    for name, basis in basis_by_key.items():
        if name not in v:
            logger.warning("Basis supplied for %r but it is not in the update vector", name)
            continue
        vi = v[name]
        axis = project_axis.get(name, -1)
        moved = vi.movedim(axis, -1) if axis != -1 else vi
        projected = project_onto_subspace(moved, basis, assume_orthonormal=assume_orthonormal)
        out[name] = projected.movedim(-1, axis) if axis != -1 else projected
    return out


def alpha_grid(start: float = 0.0, stop: float = 1.0, num: int = 11) -> list[float]:
    """The sweep grid. Default matches the plan: 0.0 .. 1.0 in steps of 0.1.

    ``alpha=0`` must always be included -- it is the control point that
    anchors every trajectory in the depth-breadth plane.
    """
    if num < 2:
        raise ValueError("num must be >= 2")
    step = (stop - start) / (num - 1)
    grid = [round(start + i * step, 10) for i in range(num)]
    if 0.0 not in grid:
        logger.warning("alpha grid does not contain the 0.0 control point: %s", grid)
    return grid
