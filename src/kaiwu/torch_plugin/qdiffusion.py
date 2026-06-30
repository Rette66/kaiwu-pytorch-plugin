# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Public QDiffusion module for generic discrete-sequence generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._qdiffusion_sampling import (
    stochastic_sample_from_categorical_n,
)

__all__ = ["EnergyModel", "QDiffusion", "QDiffusionConfig", "SequenceTokenSpec"]


@dataclass(frozen=True)
class SequenceTokenSpec:
    """Special-token metadata required by generic QDiffusion logic.

    Attributes:
        mask_id: Token id used as the diffusion noise token.
        pad_id: Token id used for sequence padding.
        bos_id: Token id used for beginning-of-sequence markers.
        eos_id: Token id used for end-of-sequence markers.
        x_id: Optional token id reserved by some backbones and excluded from
            proposal sampling.
        tokenizer: Optional tokenization helper used for encoding and decoding.
    """

    mask_id: int
    pad_id: int
    bos_id: int
    eos_id: int
    x_id: int | None = None
    tokenizer: Any | None = None


@dataclass
class QDiffusionConfig:
    """Configuration for the generic energy-guided training objective.

    Attributes:
        num_diffusion_timesteps: Number of discrete noising steps used by the
            training objective.
        use_coupled_sampling: Whether to use the coupled corruption variant.
        num_candidates: Number of proposal candidates sampled per objective step.
        proposal_temperature: Temperature used for proposal-side sampling.
        proposal_noise_scale: Gumbel noise scale used during proposal sampling.
        energy_temperature: Temperature used when converting energies into
            reranking weights.
    """

    num_diffusion_timesteps: int = 500
    use_coupled_sampling: bool = False
    num_candidates: int = 1
    proposal_temperature: float = 0.0
    proposal_noise_scale: float = 1.0
    energy_temperature: float = 1.0


class EnergyModel(nn.Module):
    """Energy-side model interface used by QDiffusion candidate scoring."""

    def __init__(self) -> None:
        super().__init__()
        self._last_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Runs conditioned energy scoring through the PyTorch module API."""
        return self.score_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Scores candidates conditioned on noisy tokens."""
        del noisy_tokens, candidate_tokens, attention_mask
        raise NotImplementedError(
            "EnergyModel subclasses must implement score_conditioned()."
        )

    def get_last_stats(self) -> dict[str, torch.Tensor]:
        """Returns lightweight sampler diagnostics from the last score call."""
        return dict(self._last_stats)


class QDiffusion(nn.Module):
    """Energy-guided discrete diffusion wrapper over generic sequence backbones.

    The class combines two backbone roles:

    - a proposal model that predicts token logits for the current noisy state
    - an energy model that reranks candidate reconstructions

    It exposes training-oriented APIs such as :meth:`objective`. Downstream
    examples may subclass it to add concrete decoding policies.
    """

    def __init__(
        self,
        proposal_model: nn.Module,
        energy_model: EnergyModel,
        token_spec: SequenceTokenSpec,
        config: QDiffusionConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        freeze_proposal: bool = True,
    ) -> None:
        """Initializes a QDiffusion model.

        Args:
            proposal_model: Backbone used to predict proposal logits.
            energy_model: Energy-side model used to encode and score candidates.
            token_spec: Special-token metadata required by the generator.
            config: Optional generation/training configuration.
            dtype: Floating point dtype tracked by the wrapper.
            device: Optional target device. When omitted, infer from parameters.
            freeze_proposal: Whether to freeze proposal model parameters.
        """
        super().__init__()
        self.proposal_model = proposal_model
        self.energy_model = energy_model
        self.token_spec = token_spec
        self.config = config or QDiffusionConfig()
        self.dtype = dtype

        if freeze_proposal:
            self.proposal_model.eval()
            for parameter in self.proposal_model.parameters():
                parameter.requires_grad = False

        self.tokenizer = token_spec.tokenizer
        self.mask_id = token_spec.mask_id
        self.pad_id = token_spec.pad_id
        self.bos_id = token_spec.bos_id
        self.eos_id = token_spec.eos_id
        self.x_id = token_spec.x_id
        self.softplus = nn.Softplus()

        if device is None:
            try:
                self.device = next(self.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            self.to(device=self.device, dtype=self.dtype)

    def to(self, *args: Any, **kwargs: Any) -> QDiffusion:
        """Moves the module and refreshes cached device/dtype metadata.

        Args:
            *args: Positional arguments forwarded to ``nn.Module.to``.
            **kwargs: Keyword arguments forwarded to ``nn.Module.to``.

        Returns:
            QDiffusion: The moved module instance.
        """
        module = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
            self.dtype = next(self.parameters()).dtype
        except StopIteration:
            self.device = torch.device("cpu")
        return module

    def forward(self, noisy_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Runs the proposal model on the current noisy state.

        Args:
            noisy_tokens: Current noisy token tensor.
            **kwargs: Additional keyword arguments forwarded to the proposal
                model.

        Returns:
            torch.Tensor: Proposal logits over the token vocabulary.

        Raises:
            TypeError: If the proposal model does not implement ``forward``.
        """
        if hasattr(self.proposal_model, "forward"):
            return self.proposal_model.forward(noisy_tokens, **kwargs)
        raise TypeError("proposal_model must implement forward().")

    def proposal(self, noisy_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Semantic alias around :meth:`forward` for proposal-side calls.

        Args:
            noisy_tokens: Current noisy token tensor.
            **kwargs: Additional keyword arguments forwarded to the proposal
                model.

        Returns:
            torch.Tensor: Proposal logits over the token vocabulary.
        """
        return self.forward(noisy_tokens, **kwargs)

    def energy(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scores candidate reconstructions conditioned on the noisy state.

        Args:
            noisy_tokens: Noisy token tensor used as conditioning input.
            candidate_tokens: Candidate clean token tensor to score.
            attention_mask: Optional attention mask for the energy model.

        Returns:
            torch.Tensor: A tensor of scalar energies with shape ``[batch, 1]``.
        """
        if attention_mask is None:
            attention_mask = candidate_tokens.ne(self.pad_id)

        return self.energy_model.score_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidate_tokens,
            attention_mask=attention_mask,
        )

    def objective(
        self, batch: dict[str, torch.Tensor], weighting: str = "constant"
    ) -> dict[str, torch.Tensor]:
        """Builds the one-step training objective used by an external loop.

        Args:
            batch: Batch dictionary containing at least ``batch["targets"]``.
            weighting: Per-sample timestep weighting mode.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing proposal logits,
            supervision masks, loss weights, and the EBM objective term.
        """
        target = batch["targets"]
        first_timestep, second_timestep = torch.randint(
            1,
            self.config.num_diffusion_timesteps + 1,
            (2 * target.size(0),),
            device=target.device,
        ).chunk(2)

        if self.config.use_coupled_sampling:
            sample_outputs = self._sample_coupled(
                target,
                first_timestep,
                second_timestep,
                self.get_non_special_symbol_mask(target),
            )
            target = target.repeat(2, 1)
        else:
            sample_outputs = self._sample(
                target,
                first_timestep,
                self.get_non_special_symbol_mask(target),
            )

        noisy_tokens = sample_outputs["x_t"]
        timesteps = sample_outputs["t"]
        loss_mask = sample_outputs["loss_mask"]

        with torch.no_grad():
            logits = self.forward(noisy_tokens).detach()

        negative_tokens, _ = self._sample_candidates(logits, self.config.num_candidates)
        positive_energy = self.energy(noisy_tokens, target, target.ne(self.pad_id))
        positive_stats = self._collect_energy_model_stats()
        negative_energy = self._score_candidates(noisy_tokens, negative_tokens).mean(
            dim=1, keepdim=True
        )
        negative_stats = self._collect_energy_model_stats()

        energy_objective = (
            self.softplus(positive_energy)
            + self.softplus(-negative_energy)
        )
        weight = self._compute_loss_weight(timesteps, weighting)
        outputs = {
            "logits": logits,
            "targets": target,
            "loss_mask": loss_mask,
            "weight": weight,
            "energy_objective": energy_objective,
            "positive_energy_mean": positive_energy.mean().detach(),
            "negative_energy_mean": negative_energy.mean().detach(),
        }
        for prefix, stats in (
            ("positive", positive_stats),
            ("negative", negative_stats),
        ):
            for key, value in stats.items():
                outputs[f"{prefix}_{key}"] = value.detach()
        return outputs

    # Internal objective helpers follow. These stay in the base class because
    # they define the generic training path, not a downstream decoding policy.

    def get_non_special_symbol_mask(
        self, output_tokens: torch.Tensor, partial_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns a boolean mask of editable non-special-token positions.

        Args:
            output_tokens: Token tensor to inspect.
            partial_masks: Optional boolean mask of positions that should remain
                fixed.

        Returns:
            torch.Tensor: A boolean mask where ``True`` marks editable non-special positions.
        """
        editable_token_mask = (
            output_tokens.ne(self.pad_id)
            & output_tokens.ne(self.bos_id)
            & output_tokens.ne(self.eos_id)
        )
        if partial_masks is not None:
            editable_token_mask &= ~partial_masks
        return editable_token_mask

    def _collect_energy_model_stats(self) -> dict[str, torch.Tensor]:
        """Collects optional energy-model diagnostics from the last score call."""
        get_last_stats = getattr(self.energy_model, "get_last_stats", None)
        if not callable(get_last_stats):
            return {}
        return get_last_stats()

    def _sample(
        self,
        clean_tokens: torch.Tensor,
        first_timestep: torch.Tensor,
        maskable_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Applies one-step discrete corruption for training.

        Args:
            clean_tokens: Clean target tokens.
            first_timestep: Sampled diffusion timesteps.
            maskable_mask: Boolean mask of positions eligible for masking.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing corrupted tokens,
            timesteps, and loss mask.
        """
        noise = torch.rand_like(clean_tokens, dtype=torch.float)
        first_timestep_mask = (
            noise < (first_timestep / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        noisy_tokens = clean_tokens.masked_fill(first_timestep_mask, self.mask_id)
        return {
            "x_t": noisy_tokens,
            "t": first_timestep,
            "loss_mask": first_timestep_mask,
        }

    def _sample_coupled(
        self,
        clean_tokens: torch.Tensor,
        first_timestep: torch.Tensor,
        second_timestep: torch.Tensor,
        maskable_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Applies the coupled corruption variant used by RDM-style training.

        Args:
            clean_tokens: Clean target tokens.
            first_timestep: First sampled timestep tensor.
            second_timestep: Second sampled timestep tensor.
            maskable_mask: Boolean mask of positions eligible for masking.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing paired
            corruptions, timesteps, and loss mask.
        """
        same_timestep_mask = first_timestep == second_timestep
        first_timestep, second_timestep = (
            torch.maximum(first_timestep, second_timestep).float(),
            torch.minimum(first_timestep, second_timestep).float(),
        )

        noise = torch.rand_like(clean_tokens, dtype=torch.float)
        first_timestep_mask = (
            noise < (first_timestep / self.config.num_diffusion_timesteps)[:, None]
        ) & maskable_mask
        first_noisy_tokens = clean_tokens.masked_fill(first_timestep_mask, self.mask_id)

        noise = torch.rand_like(clean_tokens, dtype=torch.float)
        second_timestep_mask = first_timestep_mask & (
            noise > ((first_timestep - second_timestep) / first_timestep)[:, None]
        )
        noise = torch.rand_like(clean_tokens[same_timestep_mask], dtype=torch.float)
        second_timestep_mask[same_timestep_mask] = (
            noise
            < (
                first_timestep[same_timestep_mask] / self.config.num_diffusion_timesteps
            )[:, None]
        ) & maskable_mask[same_timestep_mask]
        second_noisy_tokens = clean_tokens.masked_fill(
            second_timestep_mask, self.mask_id
        )

        return {
            "x_t": torch.cat([first_noisy_tokens, second_noisy_tokens], dim=0),
            "t": torch.cat([first_timestep, second_timestep]),
            "loss_mask": torch.cat([first_timestep_mask, second_timestep_mask], dim=0),
        }

    def _compute_loss_weight(
        self, timesteps: torch.Tensor, weighting: str
    ) -> torch.Tensor:
        """Converts sampled timesteps into per-sample loss weights.

        Args:
            timesteps: Sampled diffusion timesteps.
            weighting: Weighting strategy name.

        Returns:
            torch.Tensor: A column vector of normalized per-sample weights.
        """
        num_timesteps = self.config.num_diffusion_timesteps
        weight = {
            "linear": num_timesteps - (timesteps - 1),
            "constant": num_timesteps * torch.ones_like(timesteps),
        }[weighting]
        return weight[:, None].float() / num_timesteps

    def _reshape_candidates(
        self, tensor: torch.Tensor, batch_size: int, num_candidates: int
    ) -> torch.Tensor:
        """Normalizes sampled candidate tensors to ``[batch, k, seq_len]``.

        Args:
            tensor: Candidate tensor in one of the supported layouts.
            batch_size: Expected batch size.
            num_candidates: Expected candidate count.

        Returns:
            torch.Tensor: A candidate tensor shaped as ``[batch, num_candidates, seq_len]``.

        Raises:
            ValueError: If the incoming tensor shape is not recognized.
        """
        if tensor.shape[0] == num_candidates and tensor.shape[1] == batch_size:
            return tensor.permute(1, 0, 2).contiguous()
        if tensor.shape[0] == batch_size and tensor.shape[1] == num_candidates:
            return tensor.contiguous()
        if tensor.shape[0] == batch_size * num_candidates:
            return tensor.view(batch_size, num_candidates, -1)
        raise ValueError(f"Unexpected candidate tensor shape {tuple(tensor.shape)}.")

    def _sample_candidates(
        self, logits: torch.Tensor, num_candidates: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples proposal candidates from logits.

        Args:
            logits: Proposal logits for the current decode step.
            num_candidates: Number of candidates to sample per sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple ``(tokens, scores)`` shaped as
            ``[batch, num_candidates, seq_len]``.
        """
        samples, scores = stochastic_sample_from_categorical_n(
            logits,
            temperature=self.config.proposal_temperature,
            noise_scale=self.config.proposal_noise_scale,
            n=num_candidates,
        )
        batch_size = logits.size(0)
        return (
            self._reshape_candidates(samples, batch_size, num_candidates),
            self._reshape_candidates(scores, batch_size, num_candidates),
        )

    def _score_candidates(
        self, noisy_tokens: torch.Tensor, candidate_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Scores a batch of candidate reconstructions with the energy model.

        Args:
            noisy_tokens: Current noisy token tensor.
            candidate_tokens: Candidate reconstructions shaped as
                ``[batch, num_candidates, seq_len]``.

        Returns:
            torch.Tensor: Candidate energies shaped as ``[batch, num_candidates]``.
        """
        batch_size, num_candidates, seq_len = candidate_tokens.shape
        flat_noisy_tokens = (
            noisy_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, seq_len)
        )
        flat_candidate_tokens = candidate_tokens.reshape(
            batch_size * num_candidates, seq_len
        )
        flat_attention_mask = flat_candidate_tokens.ne(self.pad_id)
        energy = self.energy(
            flat_noisy_tokens, flat_candidate_tokens, flat_attention_mask
        )
        return energy.view(batch_size, num_candidates)
