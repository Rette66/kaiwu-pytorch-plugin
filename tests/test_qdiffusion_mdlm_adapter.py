"""Offline tests for the MDLM adapter and NLP evaluation helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from example.qdiffusion.nlp.evaluation import (
    causal_lm_nll,
    compute_generative_perplexity,
)
from example.qdiffusion.nlp.builder import build_mdlm_qdiffusion
from example.qdiffusion.nlp.checkpoint import (
    load_energy_weights,
    read_energy_checkpoint,
    save_energy_checkpoint,
)
from example.qdiffusion.nlp.models.mdlm import (
    MDLMBackbone,
    build_mdlm_token_spec,
)
from example.qdiffusion.nlp.models.bm import MDLMConditionedEnergyModel


class FakeMDLM(nn.Module):
    def __init__(self, *, time_conditioning: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            time_conditioning=time_conditioning,
            vocab_size=6,
            hidden_size=1,
        )
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        input_ids,
        timesteps,
        output_hidden_states,
        return_dict,
    ):
        assert timesteps.shape == (input_ids.size(0),)
        assert return_dict
        logits = (
            torch.arange(
                self.config.vocab_size,
                device=input_ids.device,
                dtype=torch.float32,
            )
            .view(1, 1, -1)
            .expand(*input_ids.shape, -1)
        )
        hidden_states = None
        if output_hidden_states:
            hidden_states = (input_ids.float().unsqueeze(-1),)
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2


class FakeCausalTokenizer:
    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, max_length, return_tensors
        rows = [[1, 2, 3] if text == "long" else [1, 2, 0] for text in texts]
        masks = [[1, 1, 1] if text == "long" else [1, 1, 0] for text in texts]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.tensor(masks),
        }


class FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict
        logits = torch.zeros(*input_ids.shape, 4, device=input_ids.device)
        return SimpleNamespace(logits=logits)


class FakeBMSampler:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, ising_matrix):
        self.calls += 1
        return np.ones((2, ising_matrix.shape[0]), dtype=np.float32)


class RecordingMDLMEnergy(MDLMConditionedEnergyModel):
    def score_visible_logits(self, visible_logits):
        self.last_visible_logits = visible_logits
        return visible_logits.sum(dim=-1, keepdim=True)


def test_mdlm_subs_parameterization_suppresses_mask_and_copies_clean_tokens():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    input_ids = torch.tensor([[1, 5, 3]])

    log_probs = backbone(input_ids)

    assert torch.isneginf(log_probs[..., 5]).all()
    assert log_probs[0, 0, 1].item() == 0.0
    assert log_probs[0, 2, 3].item() == 0.0
    assert torch.isneginf(log_probs[0, 0, 0])
    assert torch.logsumexp(log_probs[0, 1, :5], dim=-1).item() == pytest.approx(0.0)


def test_mdlm_adapter_exposes_hidden_states_and_token_spec():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    input_ids = torch.tensor([[1, 5, 3]])

    hidden_states = backbone.encode_tokens(input_ids)
    token_spec = build_mdlm_token_spec(backbone)

    assert torch.equal(hidden_states.squeeze(-1), input_ids.float())
    assert token_spec.mask_id == 5
    assert token_spec.pad_id == 0
    assert token_spec.bos_id == 1
    assert token_spec.eos_id == 2


def test_mdlm_adapter_rejects_time_conditioned_checkpoint():
    with pytest.raises(ValueError, match="time_conditioning=False"):
        MDLMBackbone(FakeMDLM(time_conditioning=True), FakeTokenizer(), mask_id=5)


def test_mdlm_conditioned_bm_is_the_qdiffusion_energy_path():
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    energy_model = RecordingMDLMEnergy(
        backbone,
        bm_num_visible=2,
        bm_num_hidden=1,
        sampler=object(),
    )
    noisy_tokens = torch.tensor([[1, 3, 0]])
    candidate_tokens = torch.tensor([[2, 4, 0]])
    attention_mask = torch.tensor([[True, True, False]])

    scores = energy_model.score_conditioned(
        noisy_tokens,
        candidate_tokens,
        attention_mask,
    )
    scores.sum().backward()

    assert scores.shape == (1, 1)
    assert energy_model.last_visible_logits.shape == (1, 2)
    assert energy_model.feature_projector.weight.grad is not None


def test_builder_and_checkpoint_keep_baseline_and_guided_paths_distinct(tmp_path):
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    baseline = build_mdlm_qdiffusion(
        backbone,
        use_energy=False,
        dtype=torch.float32,
    )
    guided = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        bm_num_visible=2,
        bm_num_hidden=1,
        bm_sampler=object(),
        num_candidates=4,
        dtype=torch.float32,
    )

    checkpoint_path = tmp_path / "energy.pt"
    save_energy_checkpoint(guided, checkpoint_path, epoch=1, metric=0.5)
    checkpoint = read_energy_checkpoint(checkpoint_path)
    with torch.no_grad():
        guided.energy_model.feature_projector.weight.zero_()
    load_energy_weights(guided, checkpoint)

    assert baseline.energy_model is None
    assert not baseline.config.use_energy
    assert guided.energy_model is not None
    assert guided.config.use_energy
    assert guided.config.num_candidates == 4
    assert torch.count_nonzero(guided.energy_model.feature_projector.weight).item() > 0


def test_guided_generation_reaches_conditioned_bm_sampler():
    sampler = FakeBMSampler()
    backbone = MDLMBackbone(FakeMDLM(), FakeTokenizer(), mask_id=5)
    guided = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        bm_num_visible=2,
        bm_num_hidden=1,
        bm_sampler=sampler,
        num_candidates=2,
        dtype=torch.float32,
    )
    input_tokens = torch.tensor([[1, 5, 5]])
    fixed_prompt = torch.tensor([[True, False, False]])

    output = guided.generate(
        input_tokens,
        max_steps=1,
        partial_masks=fixed_prompt,
    )

    assert output.shape == input_tokens.shape
    assert sampler.calls == 2


def test_causal_nll_ignores_padding_and_first_token():
    logits = torch.zeros(2, 3, 4)
    input_ids = torch.tensor([[1, 2, 3], [1, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    nll, num_tokens = causal_lm_nll(logits, input_ids, attention_mask)

    assert int(num_tokens) == 3
    assert float(nll) == pytest.approx(3 * torch.log(torch.tensor(4.0)).item())


def test_generative_perplexity_is_token_weighted():
    result = compute_generative_perplexity(
        ["long", "short"],
        FakeCausalLM(),
        FakeCausalTokenizer(),
        batch_size=2,
    )

    assert result.perplexity == pytest.approx(4.0)
    assert result.num_tokens == 3
    assert result.num_sequences == 2
