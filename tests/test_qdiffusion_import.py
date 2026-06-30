"""Smoke tests for the public QDiffusion API surface."""

import importlib
from pathlib import Path
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def import_local_torch_plugin():
    """Loads the current checkout instead of an installed kaiwu package."""
    root = Path(__file__).resolve().parents[1] / "src"
    for module_name in list(sys.modules):
        if module_name == "kaiwu" or module_name.startswith("kaiwu."):
            del sys.modules[module_name]
    kaiwu_module = types.ModuleType("kaiwu")
    kaiwu_module.__path__ = [str(root / "kaiwu")]
    sys.modules["kaiwu"] = kaiwu_module

    spec = importlib.util.spec_from_file_location(
        "kaiwu.torch_plugin",
        root / "kaiwu" / "torch_plugin" / "__init__.py",
        submodule_search_locations=[str(root / "kaiwu" / "torch_plugin")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["kaiwu.torch_plugin"] = module
    spec.loader.exec_module(module)
    return module


def test_qdiffusion_names_are_exported():
    module = import_local_torch_plugin()
    assert "QDiffusion" in module.__all__
    assert "QDiffusionConfig" in module.__all__
    assert "EnergyModel" in module.__all__
    assert "SequenceTokenSpec" in module.__all__


def test_qdiffusion_direct_import():
    module = import_local_torch_plugin()
    assert module.QDiffusion.__name__ == "QDiffusion"
    assert module.QDiffusionConfig.__name__ == "QDiffusionConfig"
    assert module.EnergyModel.__name__ == "EnergyModel"
    assert module.SequenceTokenSpec.__name__ == "SequenceTokenSpec"


def test_qdiffusion_removed_dplm_classmethods():
    module = importlib.import_module("kaiwu.torch_plugin.qdiffusion")
    assert not hasattr(module.QDiffusion, "from_pretrained")
    assert not hasattr(module.QDiffusion, "build")
    assert not hasattr(module.QDiffusion, "load_backbone")
