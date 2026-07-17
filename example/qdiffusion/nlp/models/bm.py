"""BM-backed energy reranking for natural-language MDLM candidates."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .edlm import EDLMConditionedFeatureEncoder
from .mdlm import MDLMBackbone
from .sampler import build_bm_sampler


class VisibleTransform(nn.Module):
    """Maps projector outputs into continuous BM visible conditions."""

    def __init__(self, num_visible: int, mode: str) -> None:
        super().__init__()
        if mode not in {"sigmoid", "identity", "layernorm"}:
            raise ValueError(
                "visible_transform must be 'sigmoid', 'identity', or 'layernorm'."
            )
        self.mode = mode
        self.normalizer = nn.LayerNorm(
            num_visible,
            elementwise_affine=False,
        )
        self.scale = nn.Parameter(
            torch.ones(()),
            requires_grad=mode == "layernorm",
        )

    def forward(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Returns continuous conditions accepted by Kaiwu BM sampling."""

        if self.mode == "sigmoid":
            return torch.sigmoid(visible_logits)
        if self.mode == "identity":
            return visible_logits
        return self.scale * self.normalizer(visible_logits)


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

    energy_type = "bm"

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
        visible_transform: str = "sigmoid",
        feature_mode: str = "pooled_pair",
    ) -> None:
        if feature_mode not in {"pooled_pair", "edlm_pair"}:
            raise ValueError(
                "feature_mode must be 'pooled_pair' or 'edlm_pair'."
            )
        self.feature_mode = feature_mode
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
        self.conditioned_encoder = (
            EDLMConditionedFeatureEncoder(encoder)
            if feature_mode == "edlm_pair"
            else None
        )
        self.feature_projector = nn.Linear(
            (
                encoder.hidden_size
                if feature_mode == "edlm_pair"
                else 2 * encoder.hidden_size
            ),
            bm_num_visible,
        )
        self.visible_transform = VisibleTransform(
            bm_num_visible,
            visible_transform,
        )
        self.register_buffer(
            "_exact_hidden_states",
            torch.empty(0, bm_num_hidden),
            persistent=False,
        )

    def discretize_visible_state(
        self,
        visible_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Preserves continuous visible conditions with a configurable transform."""

        return self.visible_transform(visible_logits)

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

        if self.conditioned_encoder is not None:
            return self.conditioned_encoder(
                self.encoder,
                noisy_tokens,
                candidate_tokens,
            )
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
        if self.conditioned_encoder is not None:
            flat_noisy_tokens = (
                noisy_tokens.unsqueeze(1)
                .expand(-1, num_candidates, -1)
                .reshape(batch_size * num_candidates, seq_len)
            )
            flat_candidate_tokens = candidate_tokens.reshape(
                batch_size * num_candidates,
                seq_len,
            )
            features = self.conditioned_encoder(
                self.encoder,
                flat_noisy_tokens,
                flat_candidate_tokens,
            )
            visible_logits = self.feature_projector(
                features.to(self.feature_projector.weight.dtype)
            )
            energy = self.score_visible_logits(visible_logits)
            return energy.view(batch_size, num_candidates)

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

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Returns compact BM architecture and sampler metadata."""

        return {
            "bm_num_visible": self.bm_num_visible,
            "bm_num_hidden": self.bm_num_hidden,
            "sampler_type": self.sampler_type,
            "sampler_kwargs": self.sampler_kwargs,
            "scoring_mode": self.scoring_mode,
            "visible_transform": self.visible_transform.mode,
            "feature_mode": self.feature_mode,
        }

    def compact_state_dict(self) -> dict[str, Any]:
        """Returns trainable BM-side modules without the MDLM copy."""

        state_dict = {
            "feature_projector": self.feature_projector.state_dict(),
            "visible_transform": self.visible_transform.state_dict(),
            "energy_bm": self.energy_bm.state_dict(),
        }
        if self.conditioned_encoder is not None:
            state_dict["conditioned_encoder"] = (
                self.conditioned_encoder.state_dict()
            )
        return state_dict

    def load_compact_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores compact BM-side modules."""

        self.feature_projector.load_state_dict(state_dict["feature_projector"])
        if "visible_transform" in state_dict:
            self.visible_transform.load_state_dict(
                state_dict["visible_transform"]
            )
        self.energy_bm.load_state_dict(state_dict["energy_bm"])
        if self.conditioned_encoder is not None:
            self.conditioned_encoder.load_state_dict(
                state_dict["conditioned_encoder"]
            )
