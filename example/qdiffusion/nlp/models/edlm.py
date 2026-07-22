"""EDLM-NCE scalar energy and its shared conditioned MDLM encoder."""

from __future__ import annotations

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .mdlm import MDLMBackbone


class EDLMConditionedFeatureEncoder(nn.Module):
    """Encodes an EDLM ``(x_t, x_0)`` pair and pools token features."""

    def __init__(
        self,
        encoder: MDLMBackbone,
        *,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        if pooling_mode not in {"mean", "attention"}:
            raise ValueError("pooling_mode must be 'mean' or 'attention'.")
        self.pooling_mode = pooling_mode
        self.input_projection = nn.Linear(
            2 * encoder.hidden_size,
            encoder.hidden_size,
        )
        self.output_layer = encoder.build_conditioned_output_layer()
        self.pool_attention = (
            nn.Linear(encoder.hidden_size, 1, bias=False)
            if pooling_mode == "attention"
            else None
        )

    def forward(
        self,
        encoder: MDLMBackbone,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a pooled full-sequence representation."""

        hidden_states = encoder.encode_conditioned_tokens(
            noisy_tokens,
            candidate_tokens,
            input_projection=self.input_projection,
            output_layer=self.output_layer,
        )
        if self.pool_attention is None:
            return hidden_states.mean(dim=1)

        scores = self.pool_attention(hidden_states).squeeze(-1).float()
        if attention_mask is not None:
            if attention_mask.shape != scores.shape:
                raise ValueError(
                    "attention_mask must match the conditioned sequence shape."
                )
            valid_tokens = attention_mask.bool()
            if not valid_tokens.any(dim=1).all():
                raise ValueError("Every sequence must contain a valid token.")
            scores = scores.masked_fill(~valid_tokens, -torch.inf)
        weights = torch.softmax(scores, dim=1).to(hidden_states.dtype)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)


class MDLMScalarEnergyModel(EnergyModel):
    """Official-code-aligned EDLM-NCE scalar sequence energy."""

    energy_type = "scalar"
    feature_mode = "edlm_pair"

    def __init__(self, encoder: MDLMBackbone) -> None:
        super().__init__()
        self.encoder = encoder
        self.conditioned_encoder = EDLMConditionedFeatureEncoder(encoder)
        self.energy_head = nn.Sequential(
            nn.Linear(encoder.hidden_size, encoder.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(encoder.hidden_size, 1, bias=False),
        )

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores one full candidate sequence with a scalar residual energy."""

        features = self.conditioned_encoder(
            self.encoder,
            noisy_tokens,
            candidate_tokens,
            attention_mask,
        )
        return self.energy_head(features.to(self.energy_head[0].weight.dtype))

    def score_candidates_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores an EDLM importance-sampling candidate pool in parallel."""

        batch_size, num_candidates, seq_len = candidate_tokens.shape
        flat_noisy_tokens = (
            noisy_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, seq_len)
        )
        flat_candidate_tokens = candidate_tokens.reshape(
            batch_size * num_candidates,
            seq_len,
        )
        flat_attention_mask = attention_mask.reshape(
            batch_size * num_candidates,
            seq_len,
        )
        energy = self.score_conditioned(
            flat_noisy_tokens,
            flat_candidate_tokens,
            flat_attention_mask,
        )
        return energy.view(batch_size, num_candidates)

    def checkpoint_metadata(self) -> dict[str, str]:
        """Returns compact architecture metadata."""

        return {"feature_mode": self.feature_mode}

    def compact_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Returns trainable scalar-energy modules without the MDLM copy."""

        return {
            "conditioned_encoder": self.conditioned_encoder.state_dict(),
            "energy_head": self.energy_head.state_dict(),
        }

    def load_compact_state_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        """Restores the compact scalar-energy modules."""

        self.conditioned_encoder.load_state_dict(
            state_dict["conditioned_encoder"]
        )
        self.energy_head.load_state_dict(state_dict["energy_head"])
