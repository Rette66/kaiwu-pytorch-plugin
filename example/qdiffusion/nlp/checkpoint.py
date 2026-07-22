"""Compact checkpoint helpers for the trainable NLP energy side."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

import torch


def save_energy_checkpoint(
    generator,
    path: Path,
    *,
    epoch: int,
    metric: float,
    extra_metadata: dict[str, Any] | None = None,
    ema_state_dict: dict[str, torch.Tensor] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Saves compact energy weights without duplicating frozen MDLM weights."""

    energy_model = generator.energy_model
    if energy_model is None:
        raise ValueError("Cannot save an energy checkpoint from a proposal-only model.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "energy_type": getattr(energy_model, "energy_type", "bm"),
        **energy_model.checkpoint_metadata(),
    }
    metadata.update(extra_metadata or {})
    trainable_encoder_state = {
        name: parameter.detach().cpu()
        for name, parameter in energy_model.encoder.named_parameters()
        if parameter.requires_grad
    }
    compact_state = energy_model.compact_state_dict()
    compact_state["energy_encoder_trainable"] = trainable_encoder_state
    payload = {
        "epoch": epoch,
        "metric": metric,
        "metadata": metadata,
        "state_dict": compact_state,
    }
    if ema_state_dict is not None:
        payload["ema_state_dict"] = {
            name: value.detach().cpu() for name, value in ema_state_dict.items()
        }
    if training_state is not None:
        payload["training_state"] = training_state
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_energy_checkpoint(
    path: Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Loads and validates one compact NLP energy checkpoint."""

    checkpoint = torch.load(path, map_location=map_location)
    if "metadata" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid NLP energy checkpoint: {path}")
    return checkpoint


def load_energy_weights(
    generator,
    checkpoint: dict[str, Any],
    *,
    use_ema: bool = True,
) -> None:
    """Restores scalar or BM energy parameters.

    Args:
        generator: QDiffusion generator that owns the target energy model.
        checkpoint: Payload returned by :func:`read_energy_checkpoint`.
        use_ema: Whether to apply EMA weights after restoring raw weights.
    """

    energy_model = generator.energy_model
    if energy_model is None:
        raise ValueError("Cannot load energy weights into a proposal-only model.")
    state_dict = checkpoint["state_dict"]
    checkpoint_type = checkpoint["metadata"].get("energy_type", "bm")
    model_type = getattr(energy_model, "energy_type", "bm")
    if checkpoint_type != model_type:
        raise ValueError(
            "Energy checkpoint type does not match the constructed model: "
            f"checkpoint={checkpoint_type!r}, model={model_type!r}."
        )
    energy_model.load_compact_state_dict(state_dict)
    if state_dict.get("energy_encoder_trainable"):
        energy_model.encoder.load_state_dict(
            state_dict["energy_encoder_trainable"],
            strict=False,
        )
    if use_ema and checkpoint.get("ema_state_dict"):
        energy_model.load_state_dict(
            checkpoint["ema_state_dict"],
            strict=False,
        )
