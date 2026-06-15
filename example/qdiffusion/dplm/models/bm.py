"""BM-backed conditioned rerankers for DPLM examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .common import DPLMFeatureEncoder, masked_mean_pool
from .sampler import build_bm_sampler


class BMConditionedEnergyModel(EnergyModel):
    """Conditioned BM reranker backed by one DPLM feature encoder."""

    def __init__(
        self,
        encoder: DPLMFeatureEncoder,
        bm_num_visible: int,
        bm_num_hidden: int,
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
        if bm_num_visible > 2 * encoder.hidden_size:
            raise ValueError(
                "bm_num_visible cannot exceed the conditioned DPLM feature size "
                f"({2 * encoder.hidden_size})."
            )

    def build_conditioned_features(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep this helper explicit because it is the semantic bridge between
        # sequence modeling and BM scoring: everything after this point operates
        # on BM-space states rather than protein tokens.
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
        # Use the DPLM conditioned feature directly as BM visible logits.
        conditioned_features = self.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        return conditioned_features[:, : self.bm_num_visible]

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidate sequences under the conditioned BM energy model."""
        visible_logits = self.build_visible_logits(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        return self.score_visible_logits(visible_logits)
