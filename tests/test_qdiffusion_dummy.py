"""Offline unit tests for generic QDiffusion construction."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn


def import_local_qdiffusion():
    """Loads the current checkout instead of an installed kaiwu package."""
    root = Path(__file__).resolve().parents[1] / "src"
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "kaiwu" or name.startswith("kaiwu.")
    }
    for module_name in list(sys.modules):
        if module_name == "kaiwu" or module_name.startswith("kaiwu."):
            del sys.modules[module_name]
    try:
        kaiwu_module = types.ModuleType("kaiwu")
        kaiwu_module.__path__ = [str(root / "kaiwu")]
        sys.modules["kaiwu"] = kaiwu_module
        torch_plugin_module = types.ModuleType("kaiwu.torch_plugin")
        torch_plugin_module.__path__ = [str(root / "kaiwu" / "torch_plugin")]
        sys.modules["kaiwu.torch_plugin"] = torch_plugin_module

        spec = importlib.util.spec_from_file_location(
            "kaiwu.torch_plugin.qdiffusion",
            root / "kaiwu" / "torch_plugin" / "qdiffusion.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["kaiwu.torch_plugin.qdiffusion"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name in list(sys.modules):
            if module_name == "kaiwu" or module_name.startswith("kaiwu."):
                del sys.modules[module_name]
        sys.modules.update(previous_modules)


_qdiffusion = import_local_qdiffusion()
EnergyModel = _qdiffusion.EnergyModel
QDiffusion = _qdiffusion.QDiffusion
QDiffusionConfig = _qdiffusion.QDiffusionConfig
SequenceTokenSpec = _qdiffusion.SequenceTokenSpec


class DummyTokenizer:
    """Minimal tokenizer stub used by direct-construction tests."""

    def batch_decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(token)) for token in row) for row in tokens]


class DummyProposalModel(nn.Module):
    """Tiny proposal model that returns logits over one toy vocabulary."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden = torch.tanh(self.hidden(self.embedding(input_ids)))
        return self.lm_head(hidden)


class DummyEnergyModel(EnergyModel):
    """Tiny energy model whose parameters are optimized by QDiffusion."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        del vocab_size, hidden_size
        super().__init__()
        self.num_score_calls = 0

    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        self.num_score_calls += 1
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


class TestQDiffusionDummy(unittest.TestCase):
    """Exercises direct QDiffusion construction without DPLM dependencies."""

    def setUp(self):
        vocab_size = 8
        hidden_size = 12
        self.proposal_model = DummyProposalModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.energy_model = DummyEnergyModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.token_spec = SequenceTokenSpec(
            pad_id=0,
            bos_id=1,
            eos_id=2,
            mask_id=3,
            x_id=4,
            tokenizer=DummyTokenizer(),
        )
        self.config = QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            proposal_temperature=0.0,
        )
        self.model = QDiffusion(
            proposal_model=self.proposal_model,
            energy_model=self.energy_model,
            token_spec=self.token_spec,
            config=self.config,
            freeze_proposal=False,
        )
        self.targets = torch.tensor(
            [
                [1, 5, 6, 2, 0],
                [1, 6, 5, 2, 0],
            ],
            dtype=torch.long,
        )

    def test_forward_shape(self):
        logits = self.model.forward(self.targets)
        self.assertEqual(logits.shape, (2, 5, 8))

    def test_energy_shape(self):
        energy = self.model.energy(self.targets, self.targets)
        self.assertEqual(energy.shape, (2, 1))

    def test_energy_uses_energy_model_scorer(self):
        scorer_model_energy = DummyEnergyModel(vocab_size=8, hidden_size=12)
        scorer_model = QDiffusion(
            proposal_model=self.proposal_model,
            energy_model=scorer_model_energy,
            token_spec=self.token_spec,
            config=self.config,
            freeze_proposal=False,
        )
        expected = (
            self.targets.to(torch.float32).sum(dim=1, keepdim=True)
            - self.targets[:, [0, 1, 3, 2, 4]].to(torch.float32).sum(dim=1, keepdim=True)
        )
        energy = scorer_model.energy(self.targets[:, [0, 1, 3, 2, 4]], self.targets)
        self.assertEqual(scorer_model_energy.num_score_calls, 1)
        self.assertTrue(torch.equal(energy, expected))

    def test_objective_with_scorer_does_not_require_extra_objective(self):
        scorer_model_energy = DummyEnergyModel(vocab_size=8, hidden_size=12)
        scorer_model = QDiffusion(
            proposal_model=self.proposal_model,
            energy_model=scorer_model_energy,
            token_spec=self.token_spec,
            config=self.config,
            freeze_proposal=False,
        )

        outputs = scorer_model.objective({"targets": self.targets})

        self.assertGreaterEqual(scorer_model_energy.num_score_calls, 2)
        self.assertEqual(outputs["energy_objective"].shape, (2, 1))

    def test_objective_fields(self):
        outputs = self.model.objective({"targets": self.targets})
        self.assertEqual(outputs["logits"].shape, (2, 5, 8))
        self.assertEqual(outputs["targets"].shape[1], 5)
        self.assertEqual(outputs["loss_mask"].dtype, torch.bool)
        self.assertEqual(outputs["weight"].shape, (2, 1))
        self.assertEqual(outputs["energy_objective"].shape, (2, 1))

    def test_base_class_does_not_own_generation_policy(self):
        self.assertFalse(hasattr(self.model, "generate"))
        self.assertFalse(hasattr(self.model, "step"))
        self.assertFalse(hasattr(self.model, "initialize_state"))

    def test_removed_dplm_entrypoints(self):
        self.assertFalse(hasattr(QDiffusion, "from_pretrained"))
        self.assertFalse(hasattr(QDiffusion, "build"))
        self.assertFalse(hasattr(QDiffusion, "load_backbone"))


if __name__ == "__main__":
    unittest.main()
