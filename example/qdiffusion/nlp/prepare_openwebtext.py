"""Prepare all 80 OpenWebText parquet shards as mmap token blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .token_blocks import TOKEN_BLOCK_FORMAT, file_sha256, metadata_path_for


DATASET_REVISION = "b4325f019c648b1641a1784748667e8b74e5e064"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--expected-shards", type=int, default=80)
    parser.add_argument("--heldout-seed", type=int, required=True)
    parser.add_argument("--heldout-modulus", type=int, default=1000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--heartbeat", type=Path)
    return parser.parse_args()


def is_heldout(
    *,
    seed: int,
    shard_name: str,
    row_index: int,
    modulus: int,
) -> bool:
    """Assigns a source record to held-out using a stable seeded hash."""

    key = f"{seed}:{shard_name}:{row_index}".encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return value % modulus == 0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Writes small progress and metadata JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.sequence_length < 3:
        raise ValueError("--sequence-length must be at least 3.")
    if args.expected_shards <= 0 or args.heldout_modulus <= 1:
        raise ValueError("Shard count and held-out modulus must be valid.")
    shards = sorted(args.input_dir.glob("train-*-of-*.parquet"))
    if len(shards) != args.expected_shards:
        raise ValueError(
            "Complete OpenWebText is required: "
            f"found {len(shards)} of {args.expected_shards} parquet shards."
        )

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = 2**31 - 1
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS token IDs.")
    if len(tokenizer) > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer vocabulary does not fit uint16 token blocks.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.heldout_output.parent.mkdir(parents=True, exist_ok=True)
    output_partial = args.output.with_suffix(args.output.suffix + ".partial")
    heldout_partial = args.heldout_output.with_suffix(
        args.heldout_output.suffix + ".partial"
    )
    binary_hash = hashlib.sha256()
    heldout_hash = hashlib.sha256()
    pending: list[int] = []
    pending_start = 0
    content_length = args.sequence_length - 2
    num_blocks = 0
    total_records = 0
    train_records = 0
    heldout_records = 0
    source_manifest: list[dict[str, Any]] = []
    started_at = time.time()

    def heartbeat(status: str, shard_number: int, shard_name: str) -> None:
        if args.heartbeat is None:
            return
        atomic_json(
            args.heartbeat,
            {
                "status": status,
                "completed_shards": shard_number,
                "expected_shards": args.expected_shards,
                "current_shard": shard_name,
                "total_records": total_records,
                "train_records": train_records,
                "heldout_records": heldout_records,
                "num_blocks": num_blocks,
                "elapsed_seconds": time.time() - started_at,
                "updated_at_unix": time.time(),
            },
        )

    with output_partial.open("wb") as block_file, heldout_partial.open(
        "wb"
    ) as heldout_file:
        for shard_number, shard in enumerate(shards, start=1):
            parquet = pq.ParquetFile(shard)
            shard_rows = 0
            for batch_number, batch in enumerate(
                parquet.iter_batches(
                    batch_size=args.tokenizer_batch_size,
                    columns=["text"],
                )
            ):
                train_texts: list[str] = []
                for text in batch.column(0).to_pylist():
                    row_index = shard_rows
                    shard_rows += 1
                    total_records += 1
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if is_heldout(
                        seed=args.heldout_seed,
                        shard_name=shard.name,
                        row_index=row_index,
                        modulus=args.heldout_modulus,
                    ):
                        raw = (
                            json.dumps(
                                {
                                    "text": text,
                                    "source_shard": shard.name,
                                    "source_row": row_index,
                                    "heldout_seed": args.heldout_seed,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        ).encode("utf-8")
                        heldout_file.write(raw)
                        heldout_hash.update(raw)
                        heldout_records += 1
                    else:
                        train_texts.append(text)
                if train_texts:
                    encoded = tokenizer(
                        train_texts,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                    for token_ids in encoded:
                        pending.extend(int(token_id) for token_id in token_ids)
                        pending.append(int(tokenizer.eos_token_id))
                        train_records += 1
                        while len(pending) - pending_start >= content_length:
                            tokens = np.asarray(
                                [
                                    int(tokenizer.bos_token_id),
                                    *pending[
                                        pending_start : pending_start + content_length
                                    ],
                                    int(tokenizer.eos_token_id),
                                ],
                                dtype="<u2",
                            )
                            raw = tokens.tobytes()
                            block_file.write(raw)
                            binary_hash.update(raw)
                            num_blocks += 1
                            pending_start += content_length
                        if pending_start >= 1_000_000:
                            pending = pending[pending_start:]
                            pending_start = 0
                if batch_number % 100 == 0:
                    heartbeat("tokenizing", shard_number - 1, shard.name)
            source_manifest.append(
                {
                    "name": shard.name,
                    "bytes": shard.stat().st_size,
                    "rows": parquet.metadata.num_rows,
                    "sha256": file_sha256(shard),
                }
            )
            heartbeat("tokenizing", shard_number, shard.name)
        block_file.flush()
        os.fsync(block_file.fileno())
        heldout_file.flush()
        os.fsync(heldout_file.fileno())

    if heldout_records == 0 or num_blocks == 0:
        raise RuntimeError("Prepared OpenWebText split is unexpectedly empty.")
    os.replace(output_partial, args.output)
    os.replace(heldout_partial, args.heldout_output)
    metadata = {
        "format": TOKEN_BLOCK_FORMAT,
        "dtype": "uint16-le",
        "dataset": "Skylion007/openwebtext",
        "dataset_revision": DATASET_REVISION,
        "complete_openwebtext": True,
        "expected_shards": args.expected_shards,
        "source_shards": source_manifest,
        "source_bytes": sum(item["bytes"] for item in source_manifest),
        "source_rows": sum(item["rows"] for item in source_manifest),
        "sequence_length": args.sequence_length,
        "num_blocks": num_blocks,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": binary_hash.hexdigest(),
        "tokenizer": args.tokenizer,
        "tokenizer_vocab_size": len(tokenizer),
        "bos_token_id": int(tokenizer.bos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "train_records": train_records,
        "heldout_records": heldout_records,
        "heldout_seed": args.heldout_seed,
        "heldout_modulus": args.heldout_modulus,
        "heldout_rule": "blake2b(seed:shard_name:source_row) modulo modulus == 0",
        "heldout_output": str(args.heldout_output.resolve()),
        "heldout_bytes": args.heldout_output.stat().st_size,
        "heldout_sha256": heldout_hash.hexdigest(),
        "dropped_trailing_content_tokens": len(pending) - pending_start,
        "created_at_unix": time.time(),
    }
    atomic_json(metadata_path_for(args.output), metadata)
    heartbeat("all_done", args.expected_shards, "")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
