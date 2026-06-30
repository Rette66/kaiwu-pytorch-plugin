import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


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


EnergyModel = import_local_qdiffusion().EnergyModel


class DummyEnergyModel(EnergyModel):
    def score_conditioned(
        self,
        noisy_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._last_stats = {
            "active_ratio": attention_mask.to(torch.float32).mean(),
        }
        return (
            candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)
            - noisy_tokens.to(torch.float32).sum(dim=1, keepdim=True)
        )


class TestEnergyModel(unittest.TestCase):
    def test_forward_delegates_to_score_conditioned(self):
        model = DummyEnergyModel()
        noisy_tokens = torch.tensor([[1, 2, 0]])
        candidate_tokens = torch.tensor([[1, 3, 0]])
        attention_mask = candidate_tokens.ne(0)

        energy = model(noisy_tokens, candidate_tokens, attention_mask)

        self.assertEqual(energy.shape, (1, 1))
        self.assertEqual(float(energy), 1.0)

    def test_base_score_conditioned_requires_subclass(self):
        model = EnergyModel()

        with self.assertRaises(NotImplementedError):
            model.score_conditioned(
                torch.tensor([[1]]),
                torch.tensor([[1]]),
                torch.tensor([[True]]),
            )

    def test_get_last_stats_returns_copy(self):
        model = DummyEnergyModel()
        attention_mask = torch.tensor([[True, False]])

        model(
            torch.tensor([[1, 0]]),
            torch.tensor([[1, 0]]),
            attention_mask,
        )
        stats = model.get_last_stats()
        stats["active_ratio"] = torch.tensor(0.0)

        self.assertTrue(torch.equal(model.get_last_stats()["active_ratio"], torch.tensor(0.5)))


if __name__ == "__main__":
    unittest.main()
