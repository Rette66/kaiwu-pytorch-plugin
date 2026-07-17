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
        scoring_mode: str = "sampler",
    ) -> None:
        self.sampler_type = sampler_type
        self.sampler_kwargs = dict(sampler_kwargs or {})
        if scoring_mode not in {"sampler", "exact"}:
            raise ValueError("scoring_mode must be 'sampler' or 'exact'.")
        if scoring_mode == "exact" and bm_num_hidden > 16:
            raise ValueError(
                "Exact BM scoring is limited to at most 16 hidden units."
            )
        self.scoring_mode = scoring_mode
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
        self.register_buffer(
            "_exact_hidden_states",
            torch.empty(0, bm_num_hidden),
            persistent=False,
        )

    def _get_exact_hidden_states(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns all binary hidden states for exact BM marginalization."""

        expected_states = 1 << int(self.bm_num_hidden)
        cached = self._exact_hidden_states
        if (
            cached.size(0) != expected_states
            or cached.device != device
            or cached.dtype != dtype
        ):
            state_ids = torch.arange(expected_states, device=device)
            bit_positions = torch.arange(
                int(self.bm_num_hidden),
                device=device,
            )
            cached = (
                (state_ids[:, None] >> bit_positions[None, :]) & 1
            ).to(dtype)
            self._exact_hidden_states = cached
        return cached

    def score_visible_logits(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Scores visible states using sampled or exact hidden inference."""

        if self.scoring_mode == "sampler":
            return super().score_visible_logits(visible_logits)

        visible_state = self.discretize_visible_state(visible_logits)
        hidden_states = self._get_exact_hidden_states(
            device=visible_state.device,
            dtype=visible_state.dtype,
        )
        batch_size = visible_state.size(0)
        num_states = hidden_states.size(0)
        expanded_visible = visible_state[:, None, :].expand(
            -1, num_states, -1
        )
        expanded_hidden = hidden_states[None, :, :].expand(
            batch_size, -1, -1
        )
        full_states = torch.cat(
            [expanded_visible, expanded_hidden],
            dim=-1,
        )
        energies = self.energy_bm(
            full_states.reshape(batch_size * num_states, -1)
        ).view(batch_size, num_states)
        log_weights = torch.log_softmax(-energies, dim=-1)
        hidden_mean = (
            log_weights.exp().unsqueeze(-1) * expanded_hidden
        ).sum(dim=1)
        self._set_last_stats(
            visible_state=visible_state,
            hidden_state=hidden_mean,
        )
        return (
            -torch.logsumexp(-energies, dim=-1)
            + torch.log(energies.new_tensor(float(num_states)))
        ).unsqueeze(-1)

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

    def score_candidates_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidate sets while encoding each noisy context once."""

        batch_size, num_candidates, seq_len = candidate_tokens.shape
        flat_attention_mask = attention_mask.reshape(
            batch_size * num_candidates, seq_len
        )
        noisy_hidden = self.encoder.encode_tokens(noisy_tokens)
        noisy_hidden = (
            noisy_hidden.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(batch_size * num_candidates, seq_len, -1)
        )
        flat_candidate_tokens = candidate_tokens.reshape(
            batch_size * num_candidates, seq_len
        )
        candidate_hidden = self.encoder.encode_tokens(
            flat_candidate_tokens,
            attention_mask=flat_attention_mask,
        )
        noisy_features = masked_mean_pool(
            noisy_hidden,
            flat_attention_mask,
        )
        candidate_features = masked_mean_pool(
            candidate_hidden,
            flat_attention_mask,
        )
        features = torch.cat([noisy_features, candidate_features], dim=-1)
        visible_logits = self.feature_projector(
            features.to(self.feature_projector.weight.dtype)
        )
        energy = self.score_visible_logits(visible_logits)
        return energy.view(batch_size, num_candidates)
