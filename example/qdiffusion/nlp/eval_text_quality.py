"""Compute lightweight diversity and repetition metrics for generated text."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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


def load_token_id_sequences(path: Path) -> list[list[int]] | None:
    """Loads optional raw token IDs emitted by the generation entrypoint."""

    sequences = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            value = json.loads(line).get("token_ids")
            if value is None:
                return None
            if not isinstance(value, list) or not all(
                isinstance(token_id, int) for token_id in value
            ):
                raise ValueError("token_ids must be a list of integers.")
            sequences.append(value)
    return sequences


def compute_mean_token_entropy(token_ids: list[list[int]]) -> float:
    """Matches EDLM's mean per-sequence Shannon entropy in bits."""

    if not token_ids or any(not sequence for sequence in token_ids):
        raise ValueError("Token entropy requires non-empty token sequences.")
    entropies = []
    for sequence in token_ids:
        counts = Counter(sequence)
        total = len(sequence)
        entropies.append(
            -sum(
                (count / total) * math.log2(count / total)
                for count in counts.values()
            )
        )
    return sum(entropies) / len(entropies)


def main() -> None:
    args = parse_args()
    result = compute_text_quality(
        load_texts(args.input, args.text_field)
    )
    token_ids = load_token_id_sequences(args.input)
    if token_ids is not None:
        result["token_entropy_bits"] = compute_mean_token_entropy(token_ids)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
