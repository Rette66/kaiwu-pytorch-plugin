"""Offline tests for NLP-specific QDiffusion generation behavior."""

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class FixedProposalModel(nn.Module):
    """Returns deterministic token preferences for a toy NLP vocabulary."""

    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        logits = self.bias.view(1, 1, -1).expand(
            input_ids.size(0), input_ids.size(1), -1
        )
        logits = logits.clone()
        logits[..., 5] = 20.0
        logits[..., 2] = 10.0
        return logits


class RecordingEnergyModel(EnergyModel):
    """Records flattened candidates passed through energy reranking."""

    def __init__(self) -> None:
        super().__init__()
        self.last_noisy_tokens = None
        self.last_candidate_tokens = None

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        self.last_noisy_tokens = noisy_tokens.detach().clone()
        self.last_candidate_tokens = candidate_tokens.detach().clone()
        return candidate_tokens.to(torch.float32).sum(dim=-1, keepdim=True)


def build_token_spec() -> SequenceTokenSpec:
    """Builds a token spec where BOS, EOS, and PAD share an NLP-style id."""

    return SequenceTokenSpec(
        pad_id=2,
        bos_id=2,
        eos_id=2,
        mask_id=3,
        x_id=None,
    )


class TestQDiffusionNLP(unittest.TestCase):
    """Covers prompt preservation, EOS control, and proposal-only generation."""

    def test_proposal_only_generation_does_not_require_energy_model(self):
        model = QDiffusion(
            proposal_model=FixedProposalModel(),
            energy_model=None,
            token_spec=build_token_spec(),
            config=QDiffusionConfig(
                num_candidates=2,
                use_energy=False,
                suppress_eos=False,
                disable_resample=True,
            ),
            freeze_proposal=False,
        )
        input_tokens = torch.tensor([[2, 3, 3, 3]], dtype=torch.long)

        output = model.generate(input_tokens, max_steps=1)

        self.assertEqual(output.shape, input_tokens.shape)

    def test_eos_remains_available_when_not_suppressed(self):
        model = QDiffusion(
            proposal_model=FixedProposalModel(),
            energy_model=None,
            token_spec=build_token_spec(),
            config=QDiffusionConfig(
                use_energy=False,
                suppress_eos=False,
            ),
            freeze_proposal=False,
        )
        logits = torch.zeros(1, 2, 8)

        masked_logits = model._mask_logits(logits)

        self.assertTrue(torch.isfinite(masked_logits[..., 2]).all())
        self.assertTrue(torch.isneginf(masked_logits[..., 3]).all())

    def test_fixed_prompt_tokens_are_clamped_before_energy_scoring(self):
        energy_model = RecordingEnergyModel()
        model = QDiffusion(
            proposal_model=FixedProposalModel(),
            energy_model=energy_model,
            token_spec=build_token_spec(),
            config=QDiffusionConfig(
                num_candidates=2,
                proposal_temperature=0.0,
                use_energy=True,
                suppress_eos=False,
                disable_resample=True,
            ),
            freeze_proposal=False,
        )
        input_tokens = torch.tensor([[2, 6, 3, 3]], dtype=torch.long)
        fixed_prompt = torch.tensor([[True, True, False, False]])

        model.generate(
            input_tokens,
            max_steps=1,
            partial_masks=fixed_prompt,
        )

        self.assertIsNotNone(energy_model.last_candidate_tokens)
        expected_prompt = torch.tensor([[2, 6], [2, 6]], dtype=torch.long)
        self.assertTrue(
            torch.equal(energy_model.last_candidate_tokens[:, :2], expected_prompt)
        )

    def test_objective_respects_explicit_maskable_mask(self):
        energy_model = RecordingEnergyModel()
        model = QDiffusion(
            proposal_model=FixedProposalModel(),
            energy_model=energy_model,
            token_spec=build_token_spec(),
            config=QDiffusionConfig(
                num_diffusion_timesteps=8,
                num_candidates=2,
                use_energy=True,
                disable_resample=True,
            ),
            freeze_proposal=False,
        )
        targets = torch.tensor([[2, 6, 5, 7, 2]], dtype=torch.long)
        maskable_mask = torch.tensor([[False, False, True, True, False]])

        outputs = model.objective({"targets": targets, "maskable_mask": maskable_mask})

        self.assertFalse(outputs["loss_mask"][~maskable_mask].any())

if __name__ == "__main__":
    unittest.main()
