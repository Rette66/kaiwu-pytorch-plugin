"""Compute GPT-2 generative perplexity for newline or JSONL text samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .evaluation import compute_generative_perplexity
except ImportError:  # pragma: no cover - direct script execution
    from evaluation import compute_generative_perplexity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--evaluator", default="gpt2-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def load_texts(path: Path, text_field: str) -> list[str]:
    """Loads one sample per line from plain text or JSONL."""

    texts = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        if path.suffix.lower() == ".jsonl":
            record = json.loads(raw_line)
            if text_field not in record:
                raise KeyError(
                    f"Missing field {text_field!r} on JSONL line {line_number}."
                )
            text = record[text_field]
        else:
            text = raw_line
        if not isinstance(text, str):
            raise TypeError(f"Sample on line {line_number} is not text.")
        texts.append(text)
    return texts


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Generative perplexity evaluation currently requires CUDA.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.evaluator)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.evaluator,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    result = compute_generative_perplexity(
        load_texts(args.input, args.text_field),
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="cuda",
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
