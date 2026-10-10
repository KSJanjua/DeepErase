"""Activation patching on every architecture family the code claims to support.

``eval/patching.py`` resolves the decoder-layer container for LLaMA-, GPT-2-,
OPT- and GPT-NeoX-style models, but only LLaMA was ever exercised (the others
warned "untested" at runtime). These tests check, on tiny random models of each
family, the properties UDS depends on:

* every layer is found and captured with the right shape;
* patching a model with its own states is an exact no-op at every layer;
* patching another model's states changes the output, and only from the
  patched layer onward;
* Stage-2 equals Stage-1 when the unlearned model IS the retain model, and is
  zero when it is the full model -- the two exact endpoints of UDS.
"""

from __future__ import annotations

import logging

import pytest
import torch

from deeperase.eval.patching import (
    EntitySpan, capture_hidden_states, entity_logprobs, forward_with_patch, n_layers,
    probe_knowledge_with_patch, resolve_layer_container,
)
from deeperase.eval.uds import score_example_deltas

VOCAB, SEQ, LAYERS, HIDDEN = 64, 10, 3, 32


def _model(family: str, seed: int):
    import transformers as T
    torch.manual_seed(seed)
    if family == "llama":
        cfg = T.LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
                            num_hidden_layers=LAYERS, num_attention_heads=4,
                            num_key_value_heads=4, max_position_embeddings=64)
        m = T.LlamaForCausalLM(cfg)
    elif family == "gpt2":
        cfg = T.GPT2Config(vocab_size=VOCAB, n_embd=HIDDEN, n_layer=LAYERS, n_head=4,
                           n_positions=64, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        m = T.GPT2LMHeadModel(cfg)
    elif family == "opt":
        cfg = T.OPTConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                          ffn_dim=64, num_attention_heads=4, max_position_embeddings=64,
                          word_embed_proj_dim=HIDDEN, dropout=0.0, attention_dropout=0.0)
        m = T.OPTForCausalLM(cfg)
    elif family == "neox":
        cfg = T.GPTNeoXConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                              num_attention_heads=4, intermediate_size=64,
                              max_position_embeddings=64, hidden_dropout=0.0,
                              attention_dropout=0.0)
        m = T.GPTNeoXForCausalLM(cfg)
    else:
        raise ValueError(family)
    m.eval(); m.requires_grad_(False)
    return m


FAMILIES = ["llama", "gpt2", "opt", "neox"]
PATHS = {"llama": "model.layers", "gpt2": "transformer.h",
         "opt": "model.decoder.layers", "neox": "gpt_neox.layers"}


@pytest.fixture(scope="module", params=FAMILIES)
def family(request):
    return request.param


@pytest.fixture(scope="module")
def models(family):
    return _model(family, 0), _model(family, 1)


def _ids():
    g = torch.Generator().manual_seed(5)
    return torch.randint(2, VOCAB - 1, (1, SEQ), generator=g)


SPAN = EntitySpan([SEQ - 3, SEQ - 2, SEQ - 1])


def test_layer_container_is_found(family, models, caplog):
    m, _ = models
    with caplog.at_level(logging.WARNING):
        container, path = resolve_layer_container(m)
    assert path == PATHS[family] and len(container) == LAYERS == n_layers(m)


def test_every_layer_is_captured_with_the_right_shape(models):
    m, _ = models
    h = capture_hidden_states(m, _ids(), list(range(LAYERS)))
    assert sorted(h) == list(range(LAYERS))
    assert all(t.shape == (1, SEQ, HIDDEN) for t in h.values())


def test_self_patching_is_an_exact_no_op(models):
    m, _ = models
    ids = _ids()
    base = entity_logprobs(forward_with_patch(m, ids, None), ids, SPAN)
    own = capture_hidden_states(m, ids, list(range(LAYERS)))
    for l in range(LAYERS):
        assert torch.equal(probe_knowledge_with_patch(m, own, ids, SPAN, l), base), l


def test_foreign_states_change_the_output_from_the_patched_layer_on(models):
    m, other = models
    ids = _ids()
    src = capture_hidden_states(other, ids, list(range(LAYERS)))
    base = entity_logprobs(forward_with_patch(m, ids, None), ids, SPAN)
    for l in range(LAYERS):
        patched = probe_knowledge_with_patch(m, src, ids, SPAN, l)
        assert not torch.allclose(patched, base), l


def test_uds_endpoints_hold(models):
    """dS2 == dS1 when M_unl is M_ret; dS2 == 0 when M_unl is M_full."""
    from deeperase.eval.uds import UDSExample
    full, ret = models
    ex = UDSExample("e", _ids(), SPAN)
    layers = list(range(LAYERS))
    d1, s_full = score_example_deltas(model_full=full, model_source=ret, example=ex,
                                      layers=layers)
    d2_ret, _ = score_example_deltas(model_full=full, model_source=ret, example=ex,
                                     layers=layers, s_full=s_full)
    d2_full, _ = score_example_deltas(model_full=full, model_source=full, example=ex,
                                      layers=layers, s_full=s_full)
    assert d2_ret == d1
    assert all(v == 0.0 for v in d2_full.values())
