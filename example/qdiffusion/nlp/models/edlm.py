"""EDLM-NCE scalar energy and its shared conditioned MDLM encoder."""

from __future__ import annotations

import torch
from torch import nn

from kaiwu.torch_plugin import EnergyModel

from .mdlm import MDLMBackbone


class EDLMConditionedFeatureEncoder(nn.Module):
    """Matches the official EDLM-NCE ``(x_t, x_0)`` feature path."""

    def __init__(self, encoder: MDLMBackbone) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            2 * encoder.hidden_size,
            encoder.hidden_size,
        )
        self.output_layer = encoder.build_conditioned_output_layer()

    def forward(
        self,
        encoder: MDLMBackbone,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the official full-sequence mean-pooled representation."""

        hidden_states = encoder.encode_conditioned_tokens(
            noisy_tokens,
            candidate_tokens,
            input_projection=self.input_projection,
            output_layer=self.output_layer,
        )
        return hidden_states.mean(dim=1)


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

        del attention_mask
        features = self.conditioned_encoder(
            self.encoder,
            noisy_tokens,
            candidate_tokens,
        )
        return self.energy_head(features.to(self.energy_head[0].weight.dtype))

    def score_candidates_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scores an EDLM importance-sampling candidate pool in parallel."""

        del attention_mask
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
        energy = self.score_conditioned(
            flat_noisy_tokens,
            flat_candidate_tokens,
            torch.empty(0, device=candidate_tokens.device),
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
