"""Disk-backed wrapped token blocks for full OpenWebText training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sized

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


TOKEN_BLOCK_FORMAT = "qdiffusion_nlp_token_blocks_v1"


def metadata_path_for(block_path: Path) -> Path:
    """Returns the sidecar path used by a token-block binary."""

    return block_path.with_suffix(block_path.suffix + ".metadata.json")


class TokenBlockDataset(Dataset[torch.Tensor]):
    """Reads fixed-width little-endian uint16 token blocks via mmap."""

    def __init__(
        self,
        path: Path,
        *,
        sequence_length: int,
        metadata_path: Path | None = None,
    ) -> None:
        self.path = path
        self.metadata_path = metadata_path or metadata_path_for(path)
        self.metadata: dict[str, Any] = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )
        if self.metadata.get("format") != TOKEN_BLOCK_FORMAT:
            raise ValueError(
                f"Unsupported token-block format: {self.metadata.get('format')!r}."
            )
        if self.metadata.get("dtype") != "uint16-le":
            raise ValueError(
                f"Unsupported token-block dtype: {self.metadata.get('dtype')!r}."
            )
        recorded_length = int(self.metadata["sequence_length"])
        if recorded_length != sequence_length:
            raise ValueError(
                "Token-block sequence length does not match training: "
                f"metadata={recorded_length}, requested={sequence_length}."
            )
        self.sequence_length = sequence_length
        self.num_blocks = int(self.metadata["num_blocks"])
        expected_bytes = self.num_blocks * self.sequence_length * 2
        actual_bytes = self.path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                "Token-block file size is inconsistent with metadata: "
                f"expected={expected_bytes}, actual={actual_bytes}."
            )
        self._array: np.memmap | None = None

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.num_blocks:
            raise IndexError(index)
        if self._array is None:
            self._array = np.memmap(
                self.path,
                mode="r",
                dtype="<u2",
                shape=(self.num_blocks, self.sequence_length),
            )
        return torch.from_numpy(np.asarray(self._array[index], dtype=np.int64))


class ResumableDistributedSampler(Sampler[int]):
    """Deterministically shuffles and shards a dataset from a saved offset."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")
        self.dataset_size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.num_samples = self.dataset_size // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(f"start_index must be in [0, {self.num_samples}].")
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            self.dataset_size,
            generator=generator,
            dtype=torch.int64,
        )[: self.total_size]
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices[self.start_index :].tolist())

    def __len__(self) -> int:
        return self.num_samples - self.start_index


def file_sha256(path: Path) -> str:
    """Streams one file into SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
