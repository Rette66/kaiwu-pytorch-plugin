"""Adapter for the official Hugging Face MDLM OpenWebText checkpoint."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Returns logits from tensor, tuple, or Hugging Face model outputs."""

    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise TypeError("MDLM output must expose a logits tensor.")


class MDLMBackbone(nn.Module):
    """Wraps an official MDLM checkpoint behind the QDiffusion proposal API."""

    def __init__(self, model: nn.Module, tokenizer: Any, mask_id: int) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = int(mask_id)

        if getattr(self.model.config, "time_conditioning", False):
            raise ValueError(
                "The QDiffusion MDLM adapter currently supports checkpoints "
                "with time_conditioning=False only."
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "kuleshov-group/mdlm-owt",
        *,
        tokenizer_name_or_path: str = "gpt2",
        trust_remote_code: bool = True,
        **model_kwargs: Any,
    ) -> MDLMBackbone:
        """Loads the official MDLM model and reconstructs its GPT-2 tokenizer."""

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        mask_id = int(model.config.vocab_size) - 1
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        if tokenizer.mask_token_id != mask_id:
            raise ValueError(
                "MDLM tokenizer mask id does not match the checkpoint vocabulary: "
                f"tokenizer={tokenizer.mask_token_id}, checkpoint={mask_id}."
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(model=model, tokenizer=tokenizer, mask_id=mask_id)

    def _timesteps(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Builds the unused zero conditioning expected by the released checkpoint."""

        return torch.zeros(
            input_ids.size(0),
            device=input_ids.device,
            dtype=torch.float32,
        )

    @property
    def hidden_size(self) -> int:
        """Returns the final token-representation width."""

        config = self.model.config
        for name in ("hidden_size", "hidden_dim", "n_embd"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise AttributeError("MDLM config does not expose a hidden size.")

    def _raw_forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            timesteps=self._timesteps(input_ids),
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _subs_parameterization(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the official MDLM substitution parameterization."""

        logits = logits.clone()
        logits[..., self.mask_id] = -torch.inf
        log_probs = torch.log_softmax(logits, dim=-1)

        unmasked = input_ids.ne(self.mask_id)
        log_probs[unmasked] = -torch.inf
        log_probs[unmasked, input_ids[unmasked]] = 0.0
        return log_probs

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Returns SUBS-parameterized proposal logits for QDiffusion."""

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MDLM proposal arguments: {unexpected}")
        outputs = self._raw_forward(input_ids)
        return self._subs_parameterization(_extract_logits(outputs), input_ids)

    def encode_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns final hidden states for a future text energy model."""

        del attention_mask
        outputs = self._raw_forward(input_ids, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("MDLM checkpoint did not return hidden states.")
        return hidden_states[-1]


def build_mdlm_token_spec(backbone: MDLMBackbone) -> SequenceTokenSpec:
    """Builds generic QDiffusion token metadata for an MDLM backbone."""

    tokenizer = backbone.tokenizer
    missing = [
        name
        for name, value in (
            ("pad_token_id", tokenizer.pad_token_id),
            ("bos_token_id", tokenizer.bos_token_id),
            ("eos_token_id", tokenizer.eos_token_id),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"MDLM tokenizer is missing required ids: {', '.join(missing)}"
        )

    return SequenceTokenSpec(
        mask_id=backbone.mask_id,
        pad_id=int(tokenizer.pad_token_id),
        bos_id=int(tokenizer.bos_token_id),
        eos_id=int(tokenizer.eos_token_id),
        x_id=None,
        tokenizer=tokenizer,
    )
