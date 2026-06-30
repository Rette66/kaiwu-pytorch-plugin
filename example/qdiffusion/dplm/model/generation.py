"""DPLM example generation policy built on the QDiffusion base class."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from kaiwu.torch_plugin import QDiffusion
from kaiwu.torch_plugin._qdiffusion_sampling import (
    stochastic_sample_from_categorical,
    top_k_top_p_filtering,
    topk_masking,
)


@dataclass
class DPLMGenerationConfig:
    """Configuration for the example-only iterative decoding policy."""

    disable_resample: bool = False
    resample_ratio: float = 0.25
    resample_top_p: float = 0.95
    decoding_strategy: str = "reparam-uncond-deterministic-linear"


class GenerativeQDiffusion(QDiffusion):
    """QDiffusion base class plus the DPLM example decoding policy."""

    def __init__(
        self,
        *args: Any,
        generation_config: DPLMGenerationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.generation_config = generation_config or DPLMGenerationConfig()

    def initialize_state(
        self,
        input_tokens: torch.Tensor,
        partial_masks: torch.Tensor | None = None,
        max_steps: int = 500,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Creates the initial decoding state for the example generation loop."""
        output_tokens, output_scores = self._initialize_output_tokens(
            input_tokens, partial_masks=partial_masks
        )
        return {
            "output_tokens": output_tokens,
            "output_scores": output_scores,
            "output_masks": self.get_non_special_symbol_mask(
                output_tokens, partial_masks=partial_masks
            ),
            "step": 0,
            "max_steps": max_steps,
            "history": [output_tokens.clone()],
            "temperature": temperature,
            "partial_masks": partial_masks,
        }

    def step(
        self,
        state: dict[str, Any],
        partial_masks: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Runs one DPLM-style denoising and reranking step."""
        partial_masks = (
            partial_masks if partial_masks is not None else state.get("partial_masks")
        )
        step_outputs = self._decode_step(state, partial_masks=partial_masks)

        editable_token_mask = self.get_non_special_symbol_mask(
            state["output_tokens"], partial_masks=partial_masks
        )
        output_masks, result_tokens, result_scores = self._reparam_decoding(
            output_tokens=state["output_tokens"].clone(),
            output_scores=state["output_scores"].clone(),
            step_tokens=step_outputs["output_tokens"].clone(),
            step_scores=step_outputs["output_scores"].clone(),
            decoding_strategy=self.generation_config.decoding_strategy,
            still_noisy_mask=state["output_masks"],
            editable_token_mask=editable_token_mask,
            t=state["step"] + 1,
            max_step=state["max_steps"],
            noise=self.mask_id,
        )

        new_state = dict(state)
        new_state.update(
            output_tokens=result_tokens,
            output_scores=result_scores,
            output_masks=output_masks,
            step=state["step"] + 1,
            history=step_outputs["history"],
            partial_masks=partial_masks,
        )
        return new_state

    def generate(
        self,
        input_tokens: torch.Tensor,
        *,
        max_steps: int = 500,
        partial_masks: torch.Tensor | None = None,
        temperature: float = 1.0,
        return_state: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Runs the DPLM example iterative decoding loop."""
        state = self.initialize_state(
            input_tokens=input_tokens,
            partial_masks=partial_masks,
            max_steps=max_steps,
            temperature=temperature,
        )
        for _ in range(max_steps):
            state = self.step(state, partial_masks=partial_masks)

        if return_state:
            return state
        return state["output_tokens"]

    def _initialize_output_tokens(
        self, input_tokens: torch.Tensor, partial_masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_mask = self.get_non_special_symbol_mask(
            input_tokens, partial_masks=partial_masks
        )
        output_tokens = input_tokens.masked_fill(output_mask, self.mask_id)
        output_scores = torch.zeros_like(output_tokens, dtype=torch.float)
        return output_tokens, output_scores

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits[..., self.mask_id] = -math.inf
        if self.x_id is not None:
            logits[..., self.x_id] = -math.inf
        logits[..., self.pad_id] = -math.inf
        logits[..., self.bos_id] = -math.inf
        logits[..., self.eos_id] = -math.inf
        return logits

    def _select_candidates(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energies = self._score_candidates(noisy_tokens, candidate_tokens)
        neg_energies = -energies
        neg_energies = neg_energies - neg_energies.max(dim=-1, keepdim=True)[0]
        weights = torch.softmax(neg_energies / self.config.energy_temperature, dim=-1)
        selected_idx = torch.multinomial(weights, 1).squeeze(-1)
        batch_idx = torch.arange(noisy_tokens.size(0), device=noisy_tokens.device)
        return (
            candidate_tokens[batch_idx, selected_idx],
            candidate_scores[batch_idx, selected_idx],
        )

    def _resample(self, tokens: torch.Tensor, scores: torch.Tensor) -> None:
        to_be_resampled = []
        resample_input = []
        resample_masks = []
        resample_scores = []

        for batch_index, sequence in enumerate(tokens):
            token_positions = {}
            max_frequency = -1
            for position, token in enumerate(sequence):
                token = int(token)
                token_positions.setdefault(token, []).append(position)
                max_frequency = max(max_frequency, len(token_positions[token]))

            if max_frequency <= len(sequence) * self.generation_config.resample_ratio:
                continue

            mask = torch.zeros_like(sequence).bool()
            for token, positions in token_positions.items():
                if (
                    len(positions)
                    > len(sequence) * self.generation_config.resample_ratio
                ):
                    mask |= sequence.eq(token)

            to_be_resampled.append(batch_index)
            resample_scores.append(scores[batch_index])
            resample_masks.append(mask)
            resample_input.append(sequence.masked_fill(mask, self.mask_id))

        if not to_be_resampled:
            return

        resample_input = torch.stack(resample_input, dim=0).type_as(tokens)
        resample_scores = torch.stack(resample_scores, dim=0).type_as(scores)
        resample_masks = torch.stack(resample_masks, dim=0).bool()
        logits = self._mask_logits(self.forward(resample_input))
        logits = top_k_top_p_filtering(
            logits,
            top_p=self.generation_config.resample_top_p,
        )
        new_tokens, new_scores = stochastic_sample_from_categorical(
            logits,
            temperature=0.0,
        )
        resample_input.masked_scatter_(resample_masks, new_tokens[resample_masks])
        resample_scores.masked_scatter_(resample_masks, new_scores[resample_masks])
        tokens[to_be_resampled] = resample_input
        scores[to_be_resampled] = resample_scores

    def _decode_step(
        self, state: dict[str, Any], partial_masks: torch.Tensor | None = None
    ) -> dict[str, Any]:
        output_tokens = state["output_tokens"].clone()
        output_scores = state["output_scores"].clone()
        output_masks = self.get_non_special_symbol_mask(
            output_tokens, partial_masks=partial_masks
        )

        logits = self._mask_logits(self.forward(output_tokens))
        if logits.dtype != output_scores.dtype:
            logits = logits.type_as(output_scores)

        candidate_tokens, candidate_scores = self._sample_candidates(
            logits, self.config.num_candidates
        )
        selected_tokens, selected_scores = self._select_candidates(
            output_tokens, candidate_tokens, candidate_scores
        )

        if not self.generation_config.disable_resample:
            self._resample(selected_tokens, selected_scores)

        output_tokens.masked_scatter_(output_masks, selected_tokens[output_masks])
        output_scores.masked_scatter_(output_masks, selected_scores[output_masks])

        history = list(state["history"])
        history.append(output_tokens.clone())
        return {
            "output_tokens": output_tokens,
            "output_scores": output_scores,
            "history": history,
        }

    def _reparam_decoding(
        self,
        output_tokens: torch.Tensor,
        output_scores: torch.Tensor,
        step_tokens: torch.Tensor,
        step_scores: torch.Tensor,
        decoding_strategy: str,
        still_noisy_mask: torch.Tensor,
        editable_token_mask: torch.Tensor,
        t: int,
        max_step: int,
        noise: int | float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule}")

        cutoff_len = (
            editable_token_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        scores_for_topk = step_scores.masked_fill(~editable_token_mask, 1000.0)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(
                scores_for_topk,
                cutoff_len,
                stochastic=True,
                temp=noise_scale * rate,
            )
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError(f"Unknown topk mode: {topk_mode}")

        if condition == "cond":
            keep_masked_from_previous = (
                (step_tokens == output_tokens)
                & (step_scores < output_scores)
                & lowest_k_mask
            )
        elif condition == "uncond":
            keep_masked_from_previous = lowest_k_mask
        else:
            raise NotImplementedError(f"Unknown condition mode: {condition}")

        keep_masked_this_step = lowest_k_mask
        masked_to_noise = (~still_noisy_mask & keep_masked_from_previous) | (
            still_noisy_mask & keep_masked_this_step
        )
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        else:
            output_tokens.masked_fill_(masked_to_noise, noise)
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_candidate_tokens = still_noisy_mask & ~keep_masked_this_step
        output_tokens.masked_scatter_(
            masked_to_candidate_tokens, step_tokens[masked_to_candidate_tokens]
        )
        output_scores.masked_scatter_(
            masked_to_candidate_tokens, step_scores[masked_to_candidate_tokens]
        )

        new_still_noisy_mask = (
            still_noisy_mask | keep_masked_from_previous
        ) & keep_masked_this_step
        return new_still_noisy_mask, output_tokens, output_scores
