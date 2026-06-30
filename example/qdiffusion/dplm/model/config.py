"""Configuration objects for DPLM Q-Diffusion examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Dataset selection and split settings."""

    fasta_path: str
    min_length: int = 50
    max_length: int = 256
    max_records: int | None = None
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42


@dataclass
class ModelConfig:
    """Model checkpoints shared by training and generation."""

    proposal_ckpt: str
    energy_ckpt: str
    freeze_proposal: bool = True
    energy_model_type: str = "bm"


@dataclass
class SamplerConfig:
    """Energy sampler settings."""

    sampler_type: str = "sa"
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Training-only knobs."""

    epochs: int = 20
    min_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    num_candidates: int = 4
    validation_steps: int = 3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    early_stop_patience: int = 4
    require_cuda: bool = True


@dataclass
class GenerateConfig:
    """Generation/evaluation knobs used after training."""

    num_candidates: int = 8
    energy_temperature: float = 1.25
    proposal_temperature: float = 0.3
    proposal_noise_scale: float = 1.0
    disable_resample: bool = False
    resample_ratio: float = 0.20
    resample_top_p: float = 0.90
    steps: int = 5


@dataclass
class Config:
    """Top-level DPLM training config grouped by concern."""

    data: DataConfig
    model: ModelConfig
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)


WorkflowConfig = Config


@dataclass
class GenerationConfig:
    """Generation settings used by downstream evaluation pipelines."""

    proposal_ckpt: str
    energy_ckpt: str
    guided_checkpoint: str | None
    generation_steps: int
    seed: int
    freeze_proposal: bool
    guided_num_candidates: int
    guided_proposal_temperature: float
    guided_proposal_noise_scale: float
    guided_energy_temperature: float
    guided_disable_resample: bool
    guided_resample_ratio: float
    guided_resample_top_p: float
    bm_sampler_type: str
    bm_sampler_kwargs: dict[str, object] | None
    energy_model_type: str = "bm"
    direct_optimizer_path: str | None = None


@dataclass
class EvalConfig:
    """Top-level config edited directly for local/server downstream runs."""

    reference_fasta: Path
    proposal_ckpt: str
    energy_ckpt: str
    guided_checkpoint: str
    output_dir: Path
    device: str
    esm2_model: str
    pair_mode: str
    pooling: str
    batch_size: int
    max_records: int | None
    generation_steps: int
    seed: int
    freeze_proposal: bool
    guided_num_candidates: int
    guided_proposal_temperature: float
    guided_proposal_noise_scale: float
    guided_energy_temperature: float
    guided_disable_resample: bool
    guided_resample_ratio: float
    guided_resample_top_p: float
    bm_sampler_type: str
    bm_sampler_kwargs: dict[str, object] | None
    energy_model_type: str = "bm"
