"""BM-backed energy reranking for natural-language MDLM candidates."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .mdlm import MDLMBackbone
from .sampler import build_bm_sampler


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pools token representations while ignoring padded positions."""

    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denominator


class MDLMConditionedEnergyModel(EnergyModel):
    """Uses frozen MDLM features to condition a trainable BM reranker."""

    def __init__(
        self,
        encoder: MDLMBackbone,
        bm_num_visible: int,
        bm_num_hidden: int,
        *,
        sampler: Any | None = None,
        sampler_type: str = "sa",
        sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.sampler_type = sampler_type
        self.sampler_kwargs = dict(sampler_kwargs or {})
        bm_sampler = sampler or build_bm_sampler(
            sampler_type=sampler_type,
            sampler_kwargs=self.sampler_kwargs,
        )
        super().__init__(
            bm_num_visible=bm_num_visible,
            bm_num_hidden=bm_num_hidden,
            sampler=bm_sampler,
        )
        self.encoder = encoder
        self.feature_projector = nn.Linear(
            2 * encoder.hidden_size,
            bm_num_visible,
        )

    def build_conditioned_features(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combines noisy-context and candidate-text sequence features."""

        noisy_hidden = self.encoder.encode_tokens(
            noisy_tokens,
            attention_mask=attention_mask,
        )
        candidate_hidden = self.encoder.encode_tokens(
            candidate_tokens,
            attention_mask=attention_mask,
        )
        noisy_features = masked_mean_pool(noisy_hidden, attention_mask)
        candidate_features = masked_mean_pool(candidate_hidden, attention_mask)
        return torch.cat([noisy_features, candidate_features], dim=-1)

    def build_visible_logits(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Projects one conditioned text pair into the BM visible space."""

        features = self.build_conditioned_features(
            noisy_tokens,
            candidate_tokens,
            attention_mask,
        )
        features = features.to(self.feature_projector.weight.dtype)
        return self.feature_projector(features)

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidate texts with the conditioned BM energy."""

        visible_logits = self.build_visible_logits(
            noisy_tokens,
            candidate_tokens,
            attention_mask,
        )
        return self.score_visible_logits(visible_logits)
