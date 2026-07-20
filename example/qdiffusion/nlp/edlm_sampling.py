"""Paper-aligned EDLM sampling on top of the released MDLM checkpoint."""

from __future__ import annotations

from typing import Any

import torch


def _sample_categorical(
    weights: torch.Tensor,
    *,
    num_samples: int = 1,
) -> torch.Tensor:
    """Samples independent categorical tokens from unnormalized weights."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    flat_weights = weights.reshape(-1, weights.size(-1))
    samples = torch.multinomial(
        flat_weights.float(),
        num_samples=num_samples,
        replacement=True,
    )
    return (
        samples.view(*weights.shape[:-1], num_samples)
        .movedim(-1, -2)
        .contiguous()
    )


class EDLMDDPMCacheSampler:
    """Implements EDLM parallel importance sampling with MDLM ``ddpm_cache``."""

    def __init__(
        self,
        proposal_model: torch.nn.Module,
        *,
        mask_id: int,
        energy_model: torch.nn.Module | None = None,
        num_candidates: int = 1,
        energy_temperature: float = 1.0,
        importance_start_t: float = 1.0,
        importance_end_t: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive.")
        if energy_temperature <= 0:
            raise ValueError("energy_temperature must be positive.")
        if not 0.0 <= importance_end_t <= importance_start_t <= 1.0:
            raise ValueError(
                "Importance window must satisfy "
                "0 <= importance_end_t <= importance_start_t <= 1."
            )
        if not 0.0 < eps < 1.0:
            raise ValueError("eps must be between 0 and 1.")
        if energy_model is None and num_candidates != 1:
            raise ValueError(
                "Proposal-only DDPM cache sampling requires num_candidates=1."
            )

        self.proposal_model = proposal_model
        self.energy_model = energy_model
        self.mask_id = int(mask_id)
        self.num_candidates = int(num_candidates)
        self.energy_temperature = float(energy_temperature)
        self.importance_start_t = float(importance_start_t)
        self.importance_end_t = float(importance_end_t)
        self.eps = float(eps)
        self.last_stats: dict[str, Any] = {}

    def _uses_importance_sampling(self, timestep: float) -> bool:
        return (
            self.energy_model is not None
            and self.importance_end_t <= timestep <= self.importance_start_t
        )

    def _select_x0(
        self,
        noisy_tokens: torch.Tensor,
        proposal_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if self.energy_model is None:
            return _sample_categorical(
                proposal_probs,
                num_samples=1,
            ).squeeze(-2), False

        candidates = _sample_categorical(
            proposal_probs,
            num_samples=self.num_candidates,
        )
        attention_mask = candidates.ne(self.mask_id)
        energies = self.energy_model.score_candidates_conditioned(
            noisy_tokens=noisy_tokens,
            candidate_tokens=candidates,
            attention_mask=attention_mask,
        )
        selection_logits = -energies.float() / self.energy_temperature
        selection_logits -= selection_logits.max(dim=-1, keepdim=True).values
        selected_indices = torch.multinomial(
            torch.softmax(selection_logits, dim=-1),
            num_samples=1,
        ).squeeze(-1)
        selected = candidates[
            torch.arange(candidates.size(0), device=candidates.device),
            selected_indices,
        ]
        return selected, True

    def _ddpm_caching_update(
        self,
        tokens: torch.Tensor,
        timestep: float,
        step_size: float,
        proposal_probs: torch.Tensor,
    ) -> torch.Tensor:
        move_chance_t = timestep
        move_chance_s = max(timestep - step_size, self.eps)
        transition_weights = proposal_probs * (
            move_chance_t - move_chance_s
        )
        transition_weights = transition_weights.clone()
        transition_weights[..., self.mask_id] = move_chance_s
        proposed_tokens = _sample_categorical(
            transition_weights,
            num_samples=1,
        ).squeeze(-2)
        return torch.where(
            tokens.ne(self.mask_id),
            tokens,
            proposed_tokens,
        )

    @torch.no_grad()
    def sample(
        self,
        input_tokens: torch.Tensor,
        *,
        num_steps: int,
    ) -> torch.Tensor:
        """Samples with the reverse process used by the EDLM reference code."""

        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        tokens = input_tokens.clone()
        timesteps = torch.linspace(
            1.0,
            self.eps,
            num_steps + 1,
            device=tokens.device,
        )
        step_size = (1.0 - self.eps) / num_steps
        cached_x0_probs = None
        guided_steps = 0
        proposal_forwards = 0

        for step in range(num_steps):
            timestep = float(timesteps[step].item())
            if cached_x0_probs is None:
                proposal_probs = self.proposal_model(tokens).exp()
                proposal_forwards += 1
                if self._uses_importance_sampling(timestep):
                    selected_x0, used_energy = self._select_x0(
                        tokens,
                        proposal_probs,
                    )
                    guided_steps += int(used_energy)
                    cached_x0_probs = torch.nn.functional.one_hot(
                        selected_x0,
                        num_classes=proposal_probs.size(-1),
                    ).to(proposal_probs.dtype)
                else:
                    cached_x0_probs = proposal_probs

            next_tokens = self._ddpm_caching_update(
                tokens,
                timestep,
                step_size,
                cached_x0_probs,
            )
            if not torch.equal(next_tokens, tokens):
                cached_x0_probs = None
            tokens = next_tokens

        self.last_stats = {
            "num_steps": num_steps,
            "proposal_forwards": proposal_forwards,
            "guided_steps": guided_steps,
            "importance_start_t": self.importance_start_t,
            "importance_end_t": self.importance_end_t,
            "num_candidates": self.num_candidates,
        }
        return tokens
