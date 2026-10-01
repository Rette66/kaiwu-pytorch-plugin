"""Small live FlagOS checks; set KAIWU_PY310 for solver tests."""

# pylint: disable=wrong-import-position

import os

import pytest

pytest.importorskip("torch_fl")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from kaiwu.torch_plugin import (  # noqa: E402
    BoltzmannMachine,
    EnergyModel,
    KaiwuProcessSampler,
    QDiffusion,
    QDiffusionConfig,
    RestrictedBoltzmannMachine,
)
from kaiwu.torch_plugin.qdiffusion import SequenceTokenSpec  # noqa: E402


pytestmark = pytest.mark.skipif(
    not torch.flagos.is_available(), reason="FlagOS device unavailable"
)


def test_bm_rbm_and_kaiwu_solver():
    """Run gradients and real Kaiwu sampling across both Python versions."""
    python_executable = os.environ.get("KAIWU_PY310")
    if not python_executable:
        pytest.skip("KAIWU_PY310 is not set")
    with KaiwuProcessSampler(
        python_executable, size_limit=3, iterations_per_t=2, rand_seed=7
    ) as sampler:
        for model in (BoltzmannMachine(4), RestrictedBoltzmannMachine(2, 2)):
            assert next(model.parameters()).device.type == "flagos"
            inputs = torch.ones((2, 4), device="flagos")
            model(inputs).mean().backward()
            assert model.linear_bias.grad is not None
            samples = model.sample(sampler)
            assert samples.device.type == "flagos"
            assert samples.shape[1] == 4

        energy = EnergyModel(2, 2, sampler).to("flagos")
        scores = energy.score_visible_logits(torch.zeros((2, 2), device="flagos"))
        assert scores.shape == (2, 1)
        assert scores.device.type == "flagos"


class Proposal(nn.Module):  # pylint: disable=too-few-public-methods
    """Tiny proposal model for a live QDiffusion step."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)

    def forward(self, input_ids, **kwargs):
        """Return toy vocabulary logits."""
        del kwargs
        return self.embedding(input_ids)


class Energy(EnergyModel):  # pylint: disable=too-few-public-methods
    """Tiny candidate scorer for a live QDiffusion step."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Return a simple candidate score."""
        del attention_mask
        return (candidate_tokens - noisy_tokens).float().sum(dim=1, keepdim=True)


def test_qdiffusion_objective_and_generation():
    """Keep QDiffusion tensors on FlagOS through scoring and generation."""
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    config = QDiffusionConfig(
        num_diffusion_timesteps=8,
        num_candidates=2,
        proposal_temperature=0.0,
        disable_resample=True,
    )
    model = QDiffusion(Proposal(), Energy(), spec, config, device="flagos")
    tokens = torch.tensor([[1, 5, 6, 2, 0]], device="flagos")
    outputs = model.objective({"targets": tokens})
    assert outputs["logits"].device.type == "flagos"
    assert outputs["energy_objective"].device.type == "flagos"
    generated = model.generate(tokens, max_steps=1)
    assert generated.device.type == "flagos"
    assert generated.shape == tokens.shape


def test_qdiffusion_with_kaiwu_solver():
    """Run QDiffusion's BM energy path through the separate Kaiwu process."""
    python_executable = os.environ.get("KAIWU_PY310")
    if not python_executable:
        pytest.skip("KAIWU_PY310 is not set")

    class BMEnergy(EnergyModel):  # pylint: disable=too-few-public-methods
        """Use two toy token positions as conditioned BM features."""

        def __init__(self, sampler):
            super().__init__(2, 2, sampler)

        def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
            """Score candidates using the real BM and Kaiwu solver."""
            del noisy_tokens, attention_mask
            return self.score_visible_logits(candidate_tokens[:, :2].float())

    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    config = QDiffusionConfig(
        num_diffusion_timesteps=8,
        num_candidates=2,
        proposal_temperature=0.0,
        disable_resample=True,
    )
    with KaiwuProcessSampler(
        python_executable, size_limit=3, iterations_per_t=2, rand_seed=7
    ) as sampler:
        model = QDiffusion(Proposal(), BMEnergy(sampler), spec, config, device="flagos")
        tokens = torch.tensor([[1, 5, 6, 2, 0]], device="flagos")
        outputs = model.objective({"targets": tokens})
        assert outputs["energy_objective"].shape == (1, 1)
        assert outputs["energy_objective"].device.type == "flagos"
        generated = model.generate(tokens, max_steps=1)
        assert generated.device.type == "flagos"
        assert generated.shape == tokens.shape
