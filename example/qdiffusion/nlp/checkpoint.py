"""Compact checkpoint helpers for the trainable NLP energy side."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_energy_checkpoint(
    generator,
    path: Path,
    *,
    epoch: int,
    metric: float,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Saves BM and projector weights without duplicating frozen MDLM weights."""

    energy_model = generator.energy_model
    if energy_model is None:
        raise ValueError("Cannot save an energy checkpoint from a proposal-only model.")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "bm_num_visible": energy_model.bm_num_visible,
        "bm_num_hidden": energy_model.bm_num_hidden,
        "sampler_type": energy_model.sampler_type,
        "sampler_kwargs": energy_model.sampler_kwargs,
        "scoring_mode": energy_model.scoring_mode,
    }
    metadata.update(extra_metadata or {})
    torch.save(
        {
            "epoch": epoch,
            "metric": metric,
            "metadata": metadata,
            "state_dict": {
                "feature_projector": energy_model.feature_projector.state_dict(),
                "energy_bm": energy_model.energy_bm.state_dict(),
            },
        },
        path,
    )


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


def load_energy_weights(generator, checkpoint: dict[str, Any]) -> None:
    """Restores the learned projector and BM parameters."""

    energy_model = generator.energy_model
    if energy_model is None:
        raise ValueError("Cannot load energy weights into a proposal-only model.")
    state_dict = checkpoint["state_dict"]
    energy_model.feature_projector.load_state_dict(state_dict["feature_projector"])
    energy_model.energy_bm.load_state_dict(state_dict["energy_bm"])
