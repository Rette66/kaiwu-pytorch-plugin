"""Compute lightweight diversity and repetition metrics for generated text."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

try:
    from .eval_gen_ppl import load_texts
except ImportError:  # pragma: no cover - direct script execution
    from eval_gen_ppl import load_texts


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase words and punctuation."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def ngrams(tokens: list[str], order: int) -> list[tuple[str, ...]]:
    """Returns contiguous n-grams of the requested order."""

    return [
        tuple(tokens[index : index + order])
        for index in range(max(0, len(tokens) - order + 1))
    ]


def compute_text_quality(texts: list[str]) -> dict[str, float | int]:
    """Computes corpus diversity and per-sequence repetition metrics."""

    if not texts:
        raise ValueError("At least one text is required.")
    tokenized = [tokenize(text) for text in texts]
    result: dict[str, float | int] = {
        "num_sequences": len(texts),
        "unique_sequence_ratio": len(set(texts)) / len(texts),
        "mean_tokens": sum(map(len, tokenized)) / len(tokenized),
    }
    for order in (1, 2, 3, 4):
        corpus_ngrams = [
            gram
            for tokens in tokenized
            for gram in ngrams(tokens, order)
        ]
        result[f"distinct_{order}"] = (
            len(set(corpus_ngrams)) / len(corpus_ngrams)
            if corpus_ngrams
            else 0.0
        )
        repeated_fractions = []
        for tokens in tokenized:
            sequence_ngrams = ngrams(tokens, order)
            counts = Counter(sequence_ngrams)
            repeated = sum(count - 1 for count in counts.values())
            repeated_fractions.append(
                repeated / len(sequence_ngrams)
                if sequence_ngrams
                else 0.0
            )
        result[f"repetition_{order}"] = (
            sum(repeated_fractions) / len(repeated_fractions)
        )
    return result


def main() -> None:
    args = parse_args()
    result = compute_text_quality(
        load_texts(args.input, args.text_field)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
