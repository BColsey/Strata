"""Small deterministic model for testing the real launch -> engine -> chat path.

This is synthetic wiring evidence, not a language-model quality test. Its
single preferred output token contains a native Muse header plus visible text.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.gguf_writer import GGUFWriter
from tools.muse.fixture import metadata, weights
from tools.strata_tokenizer import BYTE_TO_UNICODE

ANSWER = "synthetic Muse response"


def generate(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = list(BYTE_TO_UNICODE.values()) + [
        " to=user<|message|>" + ANSWER, "<|begin_of_text|>", "<|end_of_text|>",
        "<|eot|>", "<|eom|>", "<|start|>", "<|message|>",
    ]
    md = metadata()
    md.update({"general.name": "synthetic Muse chat wiring fixture",
               "muse-spark.context_length": 1024,
               "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "gpt-4o",
               "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.merges": [],
               "tokenizer.ggml.token_type": [1] * 256 + [4] + [3] * 6,
               "tokenizer.ggml.bos_token_id": 257, "tokenizer.ggml.eos_token_id": 258,
               "tokenizer.ggml.eot_token_id": 259, "tokenizer.ggml.add_bos_token": True,
               "tokenizer.chat_template": "{{ bos_token }}{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|eot|>{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}"})
    tensors = weights()
    for value in tensors.values():
        if value.ndim > 1:
            value.fill(0)
    tensors["token_embd.weight"] = np.zeros((len(tokens), 64), dtype=np.float32)
    tensors["token_embd.weight"][:, 0] = 1
    tensors["output.weight"] = np.zeros((len(tokens), 64), dtype=np.float32)
    tensors["output.weight"][256, 0] = 1
    writer = GGUFWriter()
    for key, value in md.items():
        if isinstance(value, list):
            kind = "array:string" if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges") else \
                "array:bool" if key.endswith("sliding_window_pattern") else "array:i32"
        else:
            kind = {str: "string", bool: "bool", int: "u32", float: "f32"}[type(value)]
        writer.add(key, value, kind)
    for name, value in tensors.items():
        writer.add_f32(name, value)
        writer.tensors[-1].shape = list(reversed(value.shape))
    path = output_dir / "model.gguf"
    writer.write(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    print(generate(ap.parse_args().output_dir))


if __name__ == "__main__":
    main()
