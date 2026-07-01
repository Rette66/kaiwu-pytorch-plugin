"""Runtime helpers for Q-Diffusion DPLM example workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def seed_torch(seed: int) -> None:
    """Seeds Torch CPU/CUDA RNGs for reproducible generation steps."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def direct_cim_sampler_kwargs_from_env(*, required: bool = True) -> dict[str, Any]:
    """Builds direct-CIM sampler kwargs from server environment variables.

    Expected variables:
        DPLM_DIRECT_CIM_OPTIMIZER_PATH: Optional dotted path to the direct
            hardware optimizer/factory. Defaults to the bundled lightweight
            adapter ``model.direct_cim_adapter.DirectCIMOptimizer``.
        DPLM_DIRECT_CIM_OPTIMIZER_KWARGS: Optional JSON object forwarded to
            that optimizer/factory.
        DPLM_DIRECT_CIM_USER_ID/SDK_CODE/PROJECT_NO: Optional direct-CIM
            account/project credentials. USER_ID/SDK_CODE/PROJECT_NO are also
            accepted as short aliases.
    """
    optimizer_path = os.getenv(
        "DPLM_DIRECT_CIM_OPTIMIZER_PATH",
        "model.direct_cim_adapter.DirectCIMOptimizer",
    ).strip()
    if not optimizer_path:
        if required:
            raise RuntimeError(
                "Set DPLM_DIRECT_CIM_OPTIMIZER_PATH to the dotted path of the "
                "direct-CIM optimizer before running this workflow."
            )
        return {}

    sampler_kwargs: dict[str, Any] = {"optimizer_path": optimizer_path}
    raw_optimizer_kwargs = os.getenv("DPLM_DIRECT_CIM_OPTIMIZER_KWARGS", "").strip()
    if raw_optimizer_kwargs:
        optimizer_kwargs = json.loads(raw_optimizer_kwargs)
        if not isinstance(optimizer_kwargs, dict):
            raise ValueError(
                "DPLM_DIRECT_CIM_OPTIMIZER_KWARGS must be a JSON object."
            )
    else:
        optimizer_kwargs = {}

    env_credentials = {
        "user_id": os.getenv("DPLM_DIRECT_CIM_USER_ID") or os.getenv("USER_ID"),
        "sdk_code": os.getenv("DPLM_DIRECT_CIM_SDK_CODE") or os.getenv("SDK_CODE"),
        "project_no": os.getenv("DPLM_DIRECT_CIM_PROJECT_NO") or os.getenv("PROJECT_NO"),
    }
    for key, value in env_credentials.items():
        if value and key not in optimizer_kwargs:
            optimizer_kwargs[key] = value
    if optimizer_kwargs:
        sampler_kwargs["optimizer_kwargs"] = optimizer_kwargs
    return sampler_kwargs


def redact_sensitive_config(payload: Any) -> Any:
    """Returns a copy of ``payload`` with credential-like fields masked."""
    sensitive_keys = {"sdk_code", "password", "token", "secret", "api_key"}
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            if key.lower() in sensitive_keys:
                redacted[key] = "***"
            else:
                redacted[key] = redact_sensitive_config(value)
        return redacted
    if isinstance(payload, list):
        return [redact_sensitive_config(item) for item in payload]
    return payload


def encode_sequence(
    generator, sequence: str, max_length: int | None = None
) -> torch.Tensor:
    """Tokenizes one protein sequence to ``[1, seq_len]`` on the generator device."""
    if max_length is not None:
        sequence = sequence[:max_length]
    encoded = generator.tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return encoded["input_ids"].to(generator.device)


def summarize_trainable_parameters(generator) -> dict[str, int]:
    """Counts total and trainable parameters."""
    total = 0
    trainable = 0
    for parameter in generator.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {"total_parameters": total, "trainable_parameters": trainable}


def _energy_backend_state(generator) -> dict[str, Any]:
    """Collects the BM checkpoint payload."""
    return {
        "energy_bm": generator.energy_model.energy_bm.state_dict(),
    }


def _energy_backend_metadata(generator) -> dict[str, Any]:
    """Collects lightweight metadata required to rebuild the BM backend."""
    return {
        "energy_model_type": getattr(generator.energy_model, "energy_model_type", None),
        "bm_sampler_type": getattr(generator.energy_model, "sampler_type", None),
        "bm_sampler_kwargs": getattr(generator.energy_model, "sampler_kwargs", {}),
    }


def save_checkpoint(
    output_dir: Path,
    name: str,
    *,
    generator,
    epoch: int,
    metric: float,
) -> Path:
    """Saves a compact checkpoint containing only energy-side weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / name
    torch.save(
        {
            "epoch": epoch,
            "metric": metric,
            "metadata": _energy_backend_metadata(generator),
            "state_dict": {
                # Proposal weights are intentionally omitted because the current
                # example treats proposal DPLM as a frozen upstream component.
                "energy_encoder": generator.energy_model.encoder.backbone.state_dict(),
                "feature_projector": generator.energy_model.feature_projector.state_dict(),
                **_energy_backend_state(generator),
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_trained_energy_weights(
    generator, checkpoint_path: str, device: str
) -> dict[str, Any]:
    """Loads a compact energy-side checkpoint into one generator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    # Rebuild exactly the energy-side modules we saved during training so rerun
    # and evaluation use the same scorer weights as the best checkpoint.
    generator.energy_model.encoder.backbone.load_state_dict(state_dict["energy_encoder"])
    generator.energy_model.feature_projector.load_state_dict(
        state_dict["feature_projector"]
    )
    generator.energy_model.energy_bm.load_state_dict(state_dict["energy_bm"])
    return checkpoint.get("metadata", {})
