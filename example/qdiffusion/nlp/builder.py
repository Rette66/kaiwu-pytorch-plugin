"""Builders for proposal-only and BM-guided MDLM QDiffusion models."""

from __future__ import annotations

from typing import Any

import torch

from kaiwu.torch_plugin import QDiffusion, QDiffusionConfig

from .models import (
    MDLMBackbone,
    MDLMConditionedEnergyModel,
    MDLMScalarEnergyModel,
    build_mdlm_token_spec,
)


def build_mdlm_qdiffusion(
    backbone: MDLMBackbone,
    *,
    use_energy: bool,
    energy_type: str = "bm",
    energy_feature_mode: str = "pooled_pair",
    energy_backbone: MDLMBackbone | None = None,
    bm_num_visible: int = 64,
    bm_num_hidden: int = 32,
    bm_sampler: Any | None = None,
    bm_sampler_type: str = "sa",
    bm_sampler_kwargs: dict[str, Any] | None = None,
    bm_scoring_mode: str = "sampler",
    bm_visible_transform: str = "sigmoid",
    energy_pooling: str = "mean",
    num_candidates: int = 1,
    proposal_temperature: float = 0.0,
    proposal_noise_scale: float = 1.0,
    energy_temperature: float = 1.0,
    disable_resample: bool = True,
    resample_ratio: float = 0.25,
    resample_top_p: float = 0.95,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> QDiffusion:
    """Builds the shared MDLM baseline or the actual BM-guided variant."""

    if device is None:
        try:
            resolved_device = next(backbone.parameters()).device
        except StopIteration:
            resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)

    energy_model = None
    if use_energy:
        if energy_type == "scalar":
            if energy_feature_mode != "edlm_pair":
                raise ValueError(
                    "Scalar EDLM energy requires energy_feature_mode='edlm_pair'."
                )
            energy_model = MDLMScalarEnergyModel(
                encoder=energy_backbone or backbone,
            )
        elif energy_type == "bm":
            energy_model = MDLMConditionedEnergyModel(
                encoder=energy_backbone or backbone,
                bm_num_visible=bm_num_visible,
                bm_num_hidden=bm_num_hidden,
                sampler=bm_sampler,
                sampler_type=bm_sampler_type,
                sampler_kwargs=bm_sampler_kwargs,
                scoring_mode=bm_scoring_mode,
                visible_transform=bm_visible_transform,
                feature_mode=energy_feature_mode,
                pooling_mode=energy_pooling,
            )
        else:
            raise ValueError("energy_type must be 'scalar' or 'bm'.")

    generator = QDiffusion(
        proposal_model=backbone,
        energy_model=energy_model,
        token_spec=build_mdlm_token_spec(backbone),
        config=QDiffusionConfig(
            num_candidates=num_candidates,
            proposal_temperature=proposal_temperature,
            proposal_noise_scale=proposal_noise_scale,
            energy_temperature=energy_temperature,
            use_energy=use_energy,
            suppress_eos=False,
            disable_resample=disable_resample,
            resample_ratio=resample_ratio,
            resample_top_p=resample_top_p,
        ),
        dtype=dtype,
        device=resolved_device,
        freeze_proposal=True,
    )
    if isinstance(energy_model, MDLMConditionedEnergyModel):
        # The MDLM transformer can run in bf16, but the sampler converts its
        # Ising matrix through NumPy, which requires float32 parameters.
        energy_model.feature_projector.to(dtype=torch.float32)
        bm_device = energy_model.energy_bm.linear_bias.device
        torch.nn.Module.to(
            energy_model.energy_bm,
            device=bm_device,
            dtype=torch.float32,
        )
        energy_model.energy_bm.device = bm_device
        energy_model.energy_bm.dtype = torch.float32
    return generator
