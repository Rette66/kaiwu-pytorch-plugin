"""Model building blocks for Q-Diffusion DPLM examples."""

from .config import (
    Config,
    DataConfig,
    EvalConfig,
    GenerationConfig,
    ModelConfig,
    SamplerConfig,
    WorkflowConfig,
)


def __getattr__(name: str):
    if name == "BMConditionedEnergyModel":
        from .bm import BMConditionedEnergyModel

        return BMConditionedEnergyModel
    if name == "DPLMFeatureEncoder":
        from .feature_extractor import DPLMFeatureEncoder

        return DPLMFeatureEncoder
    if name == "DPLMBackbone":
        from .networks import DPLMBackbone

        return DPLMBackbone
    if name in {"DPLMGenerationConfig", "GenerativeQDiffusion"}:
        from .generation import DPLMGenerationConfig, GenerativeQDiffusion

        return {
            "DPLMGenerationConfig": DPLMGenerationConfig,
            "GenerativeQDiffusion": GenerativeQDiffusion,
        }[name]
    if name in {"build_qdiffusion", "load_dplm_backbone"}:
        from .model import build_qdiffusion, load_dplm_backbone

        return {
            "build_qdiffusion": build_qdiffusion,
            "load_dplm_backbone": load_dplm_backbone,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BMConditionedEnergyModel",
    "Config",
    "DataConfig",
    "DPLMFeatureEncoder",
    "DPLMBackbone",
    "DPLMGenerationConfig",
    "EvalConfig",
    "GenerativeQDiffusion",
    "GenerationConfig",
    "ModelConfig",
    "SamplerConfig",
    "WorkflowConfig",
    "build_qdiffusion",
    "load_dplm_backbone",
]
