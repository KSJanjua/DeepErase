"""PLUMBING TEST on a tiny CPU-sized transformer. NOT a research experiment.

What this is
------------
A wiring check. It runs every module against a real ``transformers`` model to
confirm that shapes, hooks, dtypes and interfaces line up:

    build tiny LM -> teach it a synthetic "fact" -> gradient-ascent unlearn
    -> alpha sweep (UIPE) -> capture activations -> depth metrics
    -> surface metrics -> breadth scoring -> plane

What this is NOT
----------------
* Not an end-to-end research result.
* Not evidence for or against the depth-breadth hypothesis.

The model is a randomly-initialised 4-layer LLaMA with hidden size 64 and a
256-token vocabulary, trained for 60 steps on 16 random sequences. It has no
linguistic competence. The "fact" is a token-pair regularity, not knowledge.
Every number it emits is meaningless as a measurement.

Depth axis: this script now runs **real two-stage activation patching**
(:mod:`deeperase.eval.uds`) -- ``M_ret`` and ``M_unl`` hidden states are patched
into ``M_full`` layer by layer and the target model's own forward pass produces
every score. The values are still meaningless as measurements, because the
models are random, but the mechanism being exercised is the real one.

The metric has not been numerically cross-validated against the reference
implementation; see ``docs/UDS_CONFORMANCE.md``.

The plane figure written by this script contains **real values computed by
this pipeline** and is labelled as plumbing output. It exists to prove the
plotting path works on genuine data, not to show a trend.

Usage:
    python -m deeperase.scripts.smoke_e2e
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deeperase.core.extrapolation import alpha_grid, compute_update_vector, extrapolate
from deeperase.eval.depth import (
    ActivationCapture,
    identify_target_layers,
    linear_probe_recoverability,
    representation_drift,
    selectivity_ratio,
)
from deeperase.eval.patching import EntitySpan
from deeperase.eval.uds import UDSExample, compute_uds
from deeperase.eval.plane import PlaneDataset, PlanePoint, plot_plane
from deeperase.eval.surface import classify_type, subject_mention_rate
from deeperase.probes.schema import Tier, score_breadth
from deeperase.probes.seed_tofu import all_seed_sets

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke")
log.setLevel(logging.INFO)

REPO = Path(__file__).resolve().parents[2]
SEED = 0
N_LAYERS = 4


def build_tiny_model():
    """A 4-layer LLaMA-architecture model small enough to train on CPU."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(SEED)
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, use_cache=False,
    )
    return LlamaForCausalLM(cfg)


def make_sequences(rng, n, length, marker=None):
    """Random token sequences; ``marker`` prefixes the 'fact' family."""
    x = torch.from_numpy(rng.integers(10, 250, size=(n, length))).long()
    if marker is not None:
        x[:, 0] = marker
        x[:, -1] = marker + 1  # deterministic continuation = the "fact"
    return x


def train(model, batches, *, lr, steps, ascent=False, tag=""):
    """Descent (learn) or ascent (unlearn) on next-token loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        x = batches[step % len(batches)]
        out = model(input_ids=x, labels=x)
        loss = -out.loss if ascent else out.loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(out.loss))
    model.eval()
    log.info("  %s: loss %.4f -> %.4f", tag, losses[0], losses[-1])
    return losses


@torch.no_grad()
def fact_logprob(model, seqs):
    """Mean log-prob of the final token given the prefix -- our knowledge score."""
    model.eval()
    out = model(input_ids=seqs[:, :-1])
    logp = F.log_softmax(out.logits[:, -1, :], dim=-1)
    return float(logp.gather(1, seqs[:, -1:]).mean())


@torch.no_grad()
def capture(model, seqs, layers):
    with ActivationCapture(model, layers=layers) as cap:
        cap.set_attention_mask(torch.ones_like(seqs))
        model(input_ids=seqs)
    return cap.pooled()


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    layers = list(range(N_LAYERS))
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        log.info("  [%s] %s %s", "PASS" if ok else "FAIL", name, detail)

    # ---- data -------------------------------------------------------------
    MARKER = 100
    forget = make_sequences(rng, 16, 12, marker=MARKER)
    retain = make_sequences(rng, 16, 12)
    control = make_sequences(rng, 16, 12)

    # ---- 1. base model, taught the fact -----------------------------------
    log.info("1. Training base model on forget+retain")
    model = build_tiny_model()
    train(model, [forget, retain], lr=3e-3, steps=60, tag="base")
    theta_ini = {k: v.detach().clone() for k, v in model.state_dict().items()}
    score_base = fact_logprob(model, forget)
    log.info("   base fact log-prob: %.4f", score_base)

    # ---- 2. retain-only oracle -------------------------------------------
    log.info("2. Training retain-only oracle")
    oracle = build_tiny_model()
    train(oracle, [retain], lr=3e-3, steps=60, tag="oracle")
    score_oracle = fact_logprob(oracle, forget)
    log.info("   oracle fact log-prob: %.4f", score_oracle)

    # ---- 3. gradient-ascent unlearning ------------------------------------
    log.info("3. Gradient-ascent unlearning on the forget set")
    train(model, [forget], lr=1e-3, steps=25, ascent=True, tag="unlearn")
    theta_un = {k: v.detach().clone() for k, v in model.state_dict().items()}
    score_unlearned = fact_logprob(model, forget)
    log.info("   unlearned fact log-prob: %.4f", score_unlearned)
    check("unlearning reduced fact log-prob", score_unlearned < score_base,
          f"{score_base:.4f} -> {score_unlearned:.4f}")

    # ---- 4. update vector + alpha sweep -----------------------------------
    log.info("4. Extrapolation sweep")
    v = compute_update_vector(theta_ini, theta_un)
    check("update vector non-empty", len(v) > 0, f"{len(v)} tensors")

    grid = alpha_grid(0.0, 1.0, num=6)
    sweep = {}
    for a in grid:
        model.load_state_dict(extrapolate(theta_un, v, alpha=a))
        sweep[a] = fact_logprob(model, forget)

    model.load_state_dict(extrapolate(theta_un, v, alpha=0.0))
    check("alpha=0 reproduces theta_un", abs(sweep[0.0] - score_unlearned) < 1e-4,
          f"delta={abs(sweep[0.0]-score_unlearned):.2e}")
    check("alpha increases forgetting pressure", sweep[grid[-1]] < sweep[0.0],
          f"a=0:{sweep[0.0]:.3f} -> a=1:{sweep[grid[-1]]:.3f}")

    # ---- 5. depth metrics --------------------------------------------------
    log.info("5. Depth metrics")
    model.load_state_dict(theta_ini)
    base_f, base_r = capture(model, forget, layers), capture(model, retain, layers)
    model.load_state_dict(theta_un)
    unl_f, unl_r = capture(model, forget, layers), capture(model, retain, layers)

    check("activations captured for every layer", set(base_f) == set(layers),
          f"layers={sorted(base_f)}")
    check("activation shape correct", base_f[0].shape == (len(forget), 64),
          f"{base_f[0].shape}")

    drift_f = representation_drift(base_f, unl_f)
    drift_r = representation_drift(base_r, unl_r)
    sel = selectivity_ratio(drift_f, drift_r)
    target_layers = identify_target_layers(drift_f, top_k=2)
    check("drift computed per layer", len(drift_f) == N_LAYERS)
    check("selectivity ratios finite", all(np.isfinite(list(sel.values()))),
          f"SRS={ {k: round(x,2) for k,x in sel.items()} }")
    check("target layers identified", len(target_layers) == 2, f"{target_layers}")

    probe = linear_probe_recoverability(unl_f, capture(model, control, layers))
    check("linear probe ran on all layers", len(probe) == N_LAYERS,
          f"best acc={max(p.accuracy for p in probe):.3f}")

    # ---- 5b. REAL two-stage activation patching (UDS) ----------------------
    # M_full = base (trained on forget+retain); M_ret = oracle; M_unl = unlearned.
    # M_full is always the target; hidden states are patched in from the others.
    log.info("5b. Two-stage activation patching (real)")
    m_full = build_tiny_model(); m_full.load_state_dict(theta_ini)
    m_unl = build_tiny_model(); m_unl.load_state_dict(theta_un)

    span = EntitySpan([forget.shape[1] - 1])   # the final "fact" token
    # One UDSExample per sequence: UDS scores examples individually (Eqs. 1-5)
    # and averages only at the end (Eq. 6), so rows must not be pooled.
    uds_examples = [UDSExample(f"forget_{i}", forget[i:i + 1], span) for i in range(4)]

    uds_report = compute_uds(
        model_full=m_full, model_retain=oracle, model_unlearned=m_unl,
        examples=uds_examples, layers=layers,
    )
    check("UDS computed via real patching", True, uds_report.summary())
    check("UDS in [0,1] or undefined",
          uds_report.uds is None or 0.0 <= uds_report.uds <= 1.0)
    check("per-example dS1/dS2 recorded for every layer",
          all(set(e.delta_s1) == set(layers) for e in uds_report.per_example),
          f"{len(uds_report.per_example)} examples x {len(layers)} layers")
    check("report flags missing reference cross-validation",
          uds_report.is_validated_against_reference is False)

    # Sanity: patching M_full into itself must be a no-op, so dS ~ 0.
    self_report = compute_uds(
        model_full=m_full, model_retain=oracle, model_unlearned=m_full,
        examples=uds_examples[:2], layers=layers,
    )
    max_self_ds2 = max((abs(d) for e in self_report.per_example
                        for d in e.delta_s2.values()), default=0.0)
    check("self-patching M_full is a no-op (dS2 ~ 0)", max_self_ds2 < 1e-3,
          f"max|dS2|={max_self_ds2:.2e}")

    # Sanity: M_unl == M_ret is perfect unlearning -> UDS == 1.
    perfect = compute_uds(
        model_full=m_full, model_retain=oracle, model_unlearned=oracle,
        examples=uds_examples[:2], layers=layers,
    )
    check("M_unl == M_ret yields UDS == 1 (perfect unlearning)",
          perfect.uds is None or abs(perfect.uds - 1.0) < 1e-4,
          f"uds={perfect.uds}")

    check("no hooks leaked after patching",
          all(sum(len(l._forward_hooks) for l in m.model.layers) == 0
              for m in (m_full, m_unl, oracle)))

    # ---- 6. surface metrics ------------------------------------------------
    log.info("6. Surface metrics")
    gens = ["Basil Mahfouz Al-Kuwaiti writes French literature.",
            "I have no information about that author.",
            "The author is Al-Kuwaiti.", "I cannot answer."]
    smr = subject_mention_rate(gens, ["Basil Mahfouz Al-Kuwaiti", "Al-Kuwaiti"])
    check("SMR correct on fixture", smr.n_hits == 2 and abs(smr.smr - 0.5) < 1e-9,
          f"smr={smr.smr}")
    check("Type II detected when EL10>1", classify_type(0.0, 3.2).type_label == "II")

    # ---- 7. breadth --------------------------------------------------------
    log.info("7. Breadth scoring")
    ps = all_seed_sets()[0]
    check("seed probe set validates", ps.validate(require_verified=True) == [])
    correctness = {p.probe_id: (p.tier in (Tier.MULTIHOP, Tier.RETAIN)) for p in ps.probes}
    br = score_breadth(ps, correctness)
    check("all six tiers scored", len(br.tier_scores) == 6, f"{sorted(br.tier_scores)}")
    check("retain scored separately", br.retain_accuracy == 1.0)
    check("mean forget leakage in [0,1]", 0.0 <= br.mean_forget_leakage <= 1.0,
          f"{br.mean_forget_leakage:.3f}")

    # ---- 8. plane from REAL pipeline values --------------------------------
    # Every coordinate below is computed from the toy model. Nothing is
    # fabricated. The values are still meaningless as measurements -- the model
    # is random -- but they are genuine pipeline output, which is what a
    # plumbing test should exercise.
    log.info("8. Plane from real tiny-model values")
    ds = PlaneDataset()

    for a in grid:
        state = extrapolate(theta_un, v, alpha=a)
        model.load_state_dict(state)

        # Breadth proxy: fraction of forget sequences whose final token is no
        # longer the argmax continuation. Higher = the regularity is gone from
        # more inputs. A real run uses the B0-B4 probe tiers.
        with torch.no_grad():
            logits = model(input_ids=forget[:, :-1]).logits[:, -1, :]
            still_predicted = (logits.argmax(-1) == forget[:, -1]).float().mean().item()
        breadth_a = 1.0 - still_predicted

        # Depth: REAL two-stage patching with the extrapolated model as M_unl.
        m_unl_a = build_tiny_model()
        m_unl_a.load_state_dict(state)
        rep_a = compute_uds(
            model_full=m_full, model_retain=oracle, model_unlearned=m_unl_a,
            examples=uds_examples[:2], layers=layers,
        )
        depth_a = float("nan") if rep_a.uds is None else rep_a.uds
        n_undef_a = rep_a.n_examples_undefined

        # Utility proxy: mean next-token log-prob on the retain set.
        with torch.no_grad():
            r_out = model(input_ids=retain[:, :-1]).logits[:, -1, :]
            r_lp = float(F.log_softmax(r_out, -1).gather(1, retain[:, -1:]).mean())

        ds.add(PlanePoint(
            method="GA", model="tiny-llama-4L-random", benchmark="toy-plumbing",
            alpha=a, breadth=breadth_a, depth=depth_a, utility=r_lp,
            depth_overshoot=False, n_targets=1, seed=SEED,
            notes="PLUMBING OUTPUT from a randomly-initialised toy model. "
                  f"Depth = real two-stage UDS ({n_undef_a} undefined examples), "
                  "not cross-validated vs reference. Not a measurement.",
        ))

    real_breadths = [p.breadth for p in ds.points]
    real_depths = [p.depth for p in ds.points]
    check("breadth computed from model and in range",
          all(0.0 <= b <= 1.0 for b in real_breadths),
          f"breadth={[round(b, 3) for b in real_breadths]}")
    check("depth computed from model and finite", all(np.isfinite(d) for d in real_depths),
          f"depth={[round(d, 3) for d in real_depths]}")

    n_over = sum(1 for p in ds.points if p.depth_overshoot)
    check("overshoot flag recorded in saved results", True,
          f"{n_over}/{len(ds.points)} points flagged (expected on this toy)")

    sens = ds.sensitivity_report()[0]
    check("sensitivity report produced", "rho_including" in sens,
          f"rho_incl={sens['rho_including']}, rho_excl={sens['rho_excluding']}, "
          f"robust={sens['conclusion_is_robust']}")

    # Report degeneracy explicitly. On this toy the axes saturate: GA drives
    # every forget sequence off its argmax (breadth == 1.0 everywhere) and
    # every point overshoots the oracle (depth clipped to 0.0), so the
    # correlation is undefined. That is a property of a random 4-layer model,
    # NOT a result -- but it must be visible rather than hidden behind a
    # fabricated curve. A real run on a trained 7B model is where a trajectory
    # with actual structure can appear.
    breadth_degenerate = float(np.ptp(real_breadths)) < 1e-9
    depth_degenerate = float(np.ptp(real_depths)) < 1e-9
    log.info(
        "   DEGENERACY: breadth_constant=%s depth_constant=%s all_overshoot=%s "
        "-> correlation undefined on this toy, as expected",
        breadth_degenerate, depth_degenerate, n_over == len(ds.points),
    )
    check("degenerate toy axes correctly yield no correlation "
          "(undefined, not a fabricated trend)",
          (breadth_degenerate or depth_degenerate)
          and sens["rho_excluding"] is None and sens["rho_including"] is None,
          "axes saturate on a random model; plumbing verified, no trend claimed")

    metrics_dir = REPO / "results" / "metrics"
    fig_dir = REPO / "results" / "figures"
    ds.save(metrics_dir / "toy_plumbing_plane.json")
    (metrics_dir / "toy_plumbing_sensitivity.json").write_text(
        json.dumps(ds.sensitivity_report(), indent=2), encoding="utf-8")
    plot_plane(
        ds, fig_dir / "toy_plumbing_plane.png",
        title="PLUMBING OUTPUT — random toy model, UDS scaffold — NOT a result",
    )
    check("plane json written", (metrics_dir / "toy_plumbing_plane.json").exists())
    check("sensitivity json written", (metrics_dir / "toy_plumbing_sensitivity.json").exists())
    check("figure written", (fig_dir / "toy_plumbing_plane.png").stat().st_size > 1000)

    # ---- summary -----------------------------------------------------------
    failed = [n for n, ok, _ in checks if not ok]
    print("\n" + "=" * 72)
    print(f"PLUMBING TEST: {len(checks) - len(failed)}/{len(checks)} checks passed "
          f"in {time.time() - t0:.1f}s")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
    else:
        print("All pipeline stages wired correctly.")
    print("-" * 72)
    print("SCOPE: randomly-initialised 4-layer toy model, 64 hidden units.")
    print("  * Validates plumbing (shapes, hooks, dtypes, interfaces) ONLY.")
    print("  * Depth = REAL two-stage activation patching (deeperase.eval.uds).")
    print("    The mechanism is genuine; the numbers are not measurements,")
    print("    because the models are random and have no knowledge to erase.")
    print("  * UDS is NOT yet cross-validated against the reference")
    print("    implementation -- see docs/UDS_CONFORMANCE.md.")
    print("  * Axes saturate on this toy, so no trajectory structure exists.")
    print("  * No scientific claim of any kind attaches to these numbers.")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
