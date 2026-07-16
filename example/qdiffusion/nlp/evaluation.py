"""Evaluation helpers for generated text from an NLP QDiffusion model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class GenerativePerplexityResult:
    """Token-weighted causal-LM perplexity over generated text."""

    perplexity: float
    mean_nll: float
    num_tokens: int
    num_sequences: int


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise TypeError("Causal language model output must expose logits.")


def causal_lm_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns summed next-token NLL and the corresponding token count."""

    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share batch and sequence axes.")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids.")
    if input_ids.size(1) < 2:
        zero = logits.new_zeros(())
        return zero, attention_mask.new_zeros(())

    shifted_logits = logits[:, :-1].float()
    shifted_targets = input_ids[:, 1:]
    valid_tokens = attention_mask[:, 1:].bool()
    token_nll = (
        -torch.log_softmax(shifted_logits, dim=-1)
        .gather(
            dim=-1,
            index=shifted_targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return token_nll.masked_select(valid_tokens).sum(), valid_tokens.sum()


@torch.inference_mode()
def compute_generative_perplexity(
    texts: Iterable[str],
    evaluator_model: torch.nn.Module,
    evaluator_tokenizer: Any,
    *,
    batch_size: int = 8,
    max_length: int = 1024,
    device: torch.device | str | None = None,
) -> GenerativePerplexityResult:
    """Scores generated text under an external autoregressive evaluator.

    This metric is a fluency proxy. It is not the intrinsic diffusion-model
    likelihood/perplexity reported by MDLM's ``mode=ppl_eval`` workflow.
    """

    texts = list(texts)
    if not texts:
        raise ValueError("At least one generated text is required.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if device is None:
        try:
            device = next(evaluator_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    total_nll = 0.0
    total_tokens = 0

    evaluator_model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = evaluator_tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = evaluator_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        batch_nll, batch_tokens = causal_lm_nll(
            _extract_logits(outputs),
            input_ids,
            attention_mask,
        )
        total_nll += float(batch_nll)
        total_tokens += int(batch_tokens)

    if total_tokens == 0:
        raise ValueError("Generated texts contain no next-token targets to score.")
    mean_nll = total_nll / total_tokens
    return GenerativePerplexityResult(
        perplexity=math.exp(mean_nll),
        mean_nll=mean_nll,
        num_tokens=total_tokens,
        num_sequences=len(texts),
    )
