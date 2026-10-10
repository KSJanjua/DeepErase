"""A tiny byte-level BPE tokenizer for tests, built offline.

GPT-2/Llama-3 style: the pre-tokenizer attaches a leading space to the
following word, so a prompt that ends in a space tokenises differently on its
own than inside prompt + answer. That is the behaviour behind the breadth
scoring defect in RESULTS.md §8.1, and any test of an answer boundary needs a
tokenizer that reproduces it. Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

from typing import Iterable, Optional

DEFAULT_CORPUS = [
    "Question: What is the full name of the author?\nAnswer: Hsiao Yun-Hwa is the name.",
    "Question: Who wrote it?\nAnswer: Chen Jing-Li is the name of the writer.",
    "Question: Where was the author born?\nAnswer: Taipei, Taiwan in 1991.",
    "Question: Who wrote it?\nAnswer: Xavier is the name. Answer: Yolanda is the name.",
]


def byte_level_tokenizer(corpus: Optional[Iterable[str]] = None, *,
                         trim_offsets: bool = True, vocab_size: int = 400):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    lines = list(corpus) if corpus is not None else DEFAULT_CORPUS * 20
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(lines, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    bos = tok.token_to_id("<s>")
    tok.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=trim_offsets),
        processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", bos)]),
    ])
    out = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>")
    out.pad_token = out.eos_token
    return out
