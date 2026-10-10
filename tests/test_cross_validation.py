"""Tests for the T0 harness (scripts/cross_validate_uds.py) on tiny models.

The harness is only useful if (a) it feeds both implementations the same
tokens and positions, and (b) any disagreement it reports is real. These tests
pin both down on CPU, including the specific mechanism the harness exists to
expose: HuggingFace's ``output_hidden_states`` returns the LAST layer's state
after the final norm, so the reference patches a different tensor there than a
decoder-layer hook captures.
"""

from __future__ import annotations

import pytest
import torch

from deeperase.scripts.cross_validate_uds import (
    REFERENCE_DIR, _import_reference, build_reference_inputs, capture, reference_states,
)

pytestmark = pytest.mark.skipif(not (REFERENCE_DIR / "compute_uds.py").exists(),
                                reason="reference_uds not vendored")

RECORDS = [
    {"idx": 0, "question": "What is the full name of the author?",
     "answer": "The author's full name is Hsiao Yun-Hwa.",
     "prefix": "The author's full name is", "entity": "Hsiao Yun-Hwa"},
    {"idx": 1, "question": "Who wrote it?",
     "answer": "The book was written by Chen Jing-Li.",
     "prefix": "The book was written by", "entity": "Chen Jing-Li"},
]


@pytest.fixture(scope="module")
def setup():
    from transformers import LlamaConfig, LlamaForCausalLM
    from toy_tokenizer import byte_level_tokenizer
    tok = byte_level_tokenizer()
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        use_cache=False))
    # A trained model's final RMSNorm weights are far from 1. At init they are
    # exactly 1, which makes RMSNorm idempotent (norm(norm(h)) == norm(h)) and
    # would hide the double-normalisation this harness exists to detect.
    with torch.no_grad():
        m.model.norm.weight.copy_(torch.linspace(0.5, 2.0, m.model.norm.weight.numel()))
    m.eval()
    ref = _import_reference()
    return tok, m, ref


def test_reference_inputs_line_up_with_our_span_convention(setup):
    tok, _, ref = setup
    ex = build_reference_inputs(ref, tok, RECORDS)
    assert len(ex) == 2
    for e in ex:
        ctx = e["ctx"]
        ids = e["ids"][0].tolist()
        assert ids == ctx["prompt_ids"] + ctx["ref_ids"]
        # the tokens we score are exactly the reference's eval_ref_ids
        assert [ids[j] for j in e["span"].token_indices] == ctx["eval_ref_ids"]
        assert e["span"].predicting_positions == list(range(ctx["patch_start"], ctx["patch_end"]))
        assert "Yun" in tok.decode(ctx["eval_ref_ids"]) or "Jing" in tok.decode(ctx["eval_ref_ids"])


def test_states_identical_except_the_post_norm_last_layer(setup):
    tok, m, ref = setup
    ex = build_reference_inputs(ref, tok, RECORDS)
    c = capture(m, ref, ex, [0, 1, 2], torch.device("cpu"))
    for d in c["diffs"]:
        assert d[0] == 0.0 and d[1] == 0.0, "non-final layers must be the same tensor"
        assert d[2] > 1e-3, "final layer: HF returns the post-norm state"
    # and the reference's last-layer tensor is exactly norm(ours)
    with torch.no_grad():
        normed = m.model.norm(c["ours"][0][2])
    assert torch.allclose(normed, c["theirs_only"][0][2], atol=1e-6)
    states = reference_states(c)
    assert torch.equal(states[0][2], c["theirs_only"][0][2])
    assert torch.equal(states[0][0], c["ours"][0][0])


def test_self_patching_is_a_no_op_for_us_but_not_for_the_reference_last_layer(setup):
    """Patch the model with its own states: every layer must give dS = 0 for
    our hook capture. The reference's post-norm last-layer state does not --
    the mechanism behind a nonzero UDS for M_unl = M_full."""
    from deeperase.eval.patching import entity_logprobs, forward_with_patch, probe_knowledge_with_patch
    from deeperase.scripts.cross_validate_uds import _TOKENIZER
    tok, m, ref = setup
    _TOKENIZER[0] = tok
    ex = build_reference_inputs(ref, tok, RECORDS)[:1]
    c = capture(m, ref, ex, [0, 1, 2], torch.device("cpu"))
    e = ex[0]
    with torch.no_grad():
        base = float(entity_logprobs(forward_with_patch(m, e["ids"], None), e["ids"], e["span"]))
        for l in (0, 1, 2):
            ours = float(probe_knowledge_with_patch(m, c["ours"][0], e["ids"], e["span"], l))
            assert ours == pytest.approx(base, abs=1e-6)
        _, _, meta = ref._prepare_batch_inputs([e["ctx"]], tok, torch.device("cpu"))
        mask = torch.ones_like(e["ids"])
        theirs_last = ref.compute_logprob_teacher_forcing_layer_batch_with_inputs(
            m, e["ids"], mask, meta, reference_states(c)[0][2], 2, patch_scope="span")[0]
    assert abs(theirs_last - base) > 1e-4
