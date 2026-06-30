"""BM-backed conditioned rerankers for DPLM examples."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin import BoltzmannMachine, EnergyModel, RestrictedBoltzmannMachine

from .feature_extractor import DPLMFeatureEncoder, masked_mean_pool
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
        energy_model_type: str = "bm",
    ) -> None:
        self.energy_model_type = energy_model_type.lower()
        self.sampler_type = sampler_type
        self.sampler_kwargs = dict(sampler_kwargs or {})
        bm_sampler = None
        if self.energy_model_type == "bm":
            bm_sampler = sampler or build_bm_sampler(
                sampler_type=sampler_type,
                sampler_kwargs=self.sampler_kwargs,
            )
        super().__init__()
        self.bm_num_visible = bm_num_visible
        self.bm_num_hidden = bm_num_hidden
        self.sampler = bm_sampler
        if self.energy_model_type == "bm":
            self.energy_bm = BoltzmannMachine(
                num_visible=bm_num_visible,
                num_hidden=bm_num_hidden,
            )
        if self.energy_model_type == "rbm":
            self.energy_bm = RestrictedBoltzmannMachine(
                num_visible=bm_num_visible,
                num_hidden=bm_num_hidden,
            )
        elif self.energy_model_type != "bm":
            raise ValueError(
                "Unsupported energy_model_type for DPLM examples: "
                f"{energy_model_type}"
            )
        self.encoder = encoder
        self.feature_projector = nn.Linear(2 * encoder.hidden_size, bm_num_visible)
        self.freeze_conditioner()
        self.enable_energy_training()

    def freeze_conditioner(self) -> None:
        """Freezes DPLM feature extraction and projection parameters."""
        self.encoder.eval()
        self.feature_projector.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.feature_projector.parameters():
            parameter.requires_grad = False

    def enable_energy_training(self) -> None:
        """Enables gradients only for the BM/RBM energy backend."""
        if not hasattr(self, "energy_bm"):
            return
        for parameter in self.energy_bm.parameters():
            parameter.requires_grad = True

    def freeze_non_energy_parameters(self) -> None:
        """Keeps only BM/RBM backend parameters trainable."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.freeze_conditioner()
        self.enable_energy_training()

    def discretize_visible_state(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Converts projected sequence features into BM visible states."""
        return torch.sigmoid(visible_logits)

    def sample_hidden_state(
        self,
        visible_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples full BM hidden states for each visible assignment."""
        if self.sampler is None:
            raise RuntimeError("Full BM scoring requires a hidden-state sampler.")
        batched_states = []
        split_sizes = []
        for sample_index in range(visible_state.size(0)):
            sampled_states = self.energy_bm.condition_sample(
                self.sampler,
                visible_state[sample_index : sample_index + 1],
                dtype=visible_state.dtype,
            )
            batched_states.append(sampled_states)
            split_sizes.append(sampled_states.size(0))
        return torch.cat(batched_states, dim=0), torch.tensor(
            split_sizes,
            device=visible_state.device,
            dtype=torch.long,
        )

    def _set_last_stats(
        self,
        *,
        visible_state: torch.Tensor,
        hidden_state: torch.Tensor,
        sampling_mode: float,
    ) -> None:
        self._last_stats = {
            "sampling_mode": torch.tensor(
                sampling_mode,
                dtype=visible_state.dtype,
                device=visible_state.device,
            ),
            "visible_on_ratio": visible_state.detach().mean(),
            "hidden_on_ratio": hidden_state.detach().mean(),
        }

    def score_visible_logits(self, visible_logits: torch.Tensor) -> torch.Tensor:
        """Scores visible logits under the configured BM/RBM backend."""
        if self.energy_model_type == "rbm":
            visible_state = self.discretize_visible_state(visible_logits)
            hidden_logits = (
                visible_state @ self.energy_bm.quadratic_coef
                + self.energy_bm.hidden_bias
            )
            hidden_state = torch.sigmoid(hidden_logits)
            self._set_last_stats(
                visible_state=visible_state,
                hidden_state=hidden_state,
                sampling_mode=0.0,
            )
            free_energy = self.energy_bm.marginal_energy_from_visible(visible_state)
            return free_energy.unsqueeze(-1)

        visible_state = self.discretize_visible_state(visible_logits)
        full_states, split_sizes = self.sample_hidden_state(visible_state)
        hidden_state = full_states[:, self.bm_num_visible :]
        self._set_last_stats(
            visible_state=full_states[:, : self.bm_num_visible],
            hidden_state=hidden_state,
            sampling_mode=1.0,
        )
        flat_energy = self.energy_bm(full_states).unsqueeze(-1)
        split_energy = torch.split(flat_energy, split_sizes.tolist())
        return torch.stack([energy.mean(dim=0) for energy in split_energy], dim=0)

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
        # The projector turns one concatenated sequence feature into the visible
        # part of the BM state before discretization/sampling steps.
        conditioned_features = self.build_conditioned_features(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )
        return self.feature_projector(conditioned_features)

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
