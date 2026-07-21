"""Train EDLM-NCE scalar or BM energy on identical wrapped GPT-2 blocks."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import sys
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler


def _bootstrap_repo() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import kaiwu

    local_namespace = str(repo_root / "src" / "kaiwu")
    if local_namespace not in kaiwu.__path__:
        kaiwu.__path__.insert(0, local_namespace)


_bootstrap_repo()

from .checkpoint import (  # noqa: E402
    load_energy_weights,
    read_energy_checkpoint,
    save_energy_checkpoint,
)
from .eval_gen_ppl import load_texts  # noqa: E402
from .models import (  # noqa: E402
    MDLMBackbone,
    MDLMConditionedEnergyModel,
    MDLMScalarEnergyModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--checkpoint", default="kuleshov-group/mdlm-owt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-energy-checkpoint", type=Path)
    parser.add_argument(
        "--resume-weights",
        choices=("raw", "ema"),
        default="raw",
        help=(
            "Weights used to initialize a resumed run. The prior EMA shadow "
            "is restored independently when available."
        ),
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=2500)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--sampling-eps", type=float, default=1e-3)
    parser.add_argument("--noise-eps", type=float, default=1e-3)
    parser.add_argument(
        "--energy-type",
        choices=("scalar", "bm"),
        default="scalar",
    )
    parser.add_argument("--bm-num-visible", type=int, default=64)
    parser.add_argument("--bm-num-hidden", type=int, default=32)
    parser.add_argument(
        "--bm-visible-transform",
        choices=("identity", "sigmoid", "layernorm"),
        default="identity",
    )
    parser.add_argument("--bm-sa-alpha", type=float, default=0.95)
    parser.add_argument("--bm-sa-size-limit", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


def build_wrapped_blocks(
    texts: Iterable[str],
    tokenizer,
    *,
    sequence_length: int,
) -> list[torch.Tensor]:
    """Matches EDLM's wrapped OWT grouping with BOS/EOS boundary tokens."""

    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3.")
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS token ids.")
    concatenated: list[int] = []
    for text in texts:
        concatenated.extend(
            tokenizer.encode(text, add_special_tokens=False)
        )
        concatenated.append(int(tokenizer.eos_token_id))
    content_length = sequence_length - 2
    usable_length = len(concatenated) // content_length * content_length
    bos = int(tokenizer.bos_token_id)
    eos = int(tokenizer.eos_token_id)
    return [
        torch.tensor(
            [bos, *concatenated[start : start + content_length], eos],
            dtype=torch.long,
        )
        for start in range(0, usable_length, content_length)
    ]


def sample_antithetic_times(
    batch_size: int,
    *,
    device: torch.device,
    sampling_eps: float,
) -> torch.Tensor:
    """Samples the continuous antithetic timesteps used by EDLM."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 < sampling_eps < 1.0:
        raise ValueError("sampling_eps must be in (0, 1).")
    random_offsets = torch.rand(batch_size, device=device)
    offsets = torch.arange(batch_size, device=device) / batch_size
    antithetic = (random_offsets / batch_size + offsets) % 1.0
    return (1.0 - sampling_eps) * antithetic + sampling_eps


def corrupt_tokens(
    clean_tokens: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    mask_id: int,
    noise_eps: float,
) -> torch.Tensor:
    """Applies the official log-linear forward corruption."""

    if not 0.0 < noise_eps < 1.0:
        raise ValueError("noise_eps must be in (0, 1).")
    move_chance = (1.0 - noise_eps) * timesteps[:, None]
    mask = torch.rand_like(clean_tokens, dtype=torch.float32) < move_chance
    return clean_tokens.masked_fill(mask, int(mask_id))


def sample_proposal(log_probabilities: torch.Tensor) -> torch.Tensor:
    """Samples one independent proposal token at every sequence position."""

    flat = log_probabilities.exp().reshape(-1, log_probabilities.size(-1))
    return torch.multinomial(flat.float(), 1).view(
        *log_probabilities.shape[:-1]
    )


def binary_nce_loss(
    positive_energy: torch.Tensor,
    negative_energy: torch.Tensor,
) -> torch.Tensor:
    """Returns EDLM's one-negative binary NCE loss."""

    return (
        F.softplus(positive_energy) + F.softplus(-negative_energy)
    ).mean()


def constant_warmup_factor(step: int, *, warmup_steps: int) -> float:
    """Matches Transformers' constant schedule with linear warmup."""

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if warmup_steps == 0:
        return 1.0
    return min(float(step) / float(warmup_steps), 1.0)


@dataclass
class EMATracker:
    """Tracks trainable EDLM parameters with the paper's EMA decay."""

    decay: float
    shadow: dict[str, torch.Tensor]

    @classmethod
    def create(
        cls,
        module: torch.nn.Module,
        decay: float,
    ) -> EMATracker:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        return cls(
            decay=decay,
            shadow={
                name: parameter.detach().clone()
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            },
        )

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(
                    parameter.detach(),
                    1.0 - self.decay,
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


def train_step(
    proposal: MDLMBackbone,
    energy_model: torch.nn.Module,
    clean_tokens: torch.Tensor,
    *,
    sampling_eps: float,
    noise_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Builds one official EDLM-NCE positive/negative training batch."""

    timesteps = sample_antithetic_times(
        clean_tokens.size(0),
        device=clean_tokens.device,
        sampling_eps=sampling_eps,
    )
    noisy_tokens = corrupt_tokens(
        clean_tokens,
        timesteps,
        mask_id=proposal.mask_id,
        noise_eps=noise_eps,
    )
    with torch.no_grad():
        proposal_log_probabilities = proposal(noisy_tokens)
        negative_tokens = sample_proposal(proposal_log_probabilities)
    attention_mask = torch.ones_like(clean_tokens, dtype=torch.bool)
    positive_energy = energy_model(
        noisy_tokens,
        clean_tokens,
        attention_mask,
    )
    negative_energy = energy_model(
        noisy_tokens,
        negative_tokens,
        attention_mask,
    )
    loss = binary_nce_loss(positive_energy, negative_energy)
    return loss, {
        "loss": float(loss.detach()),
        "positive_energy": float(positive_energy.detach().mean()),
        "negative_energy": float(negative_energy.detach().mean()),
        "energy_margin": float(
            (negative_energy.detach() - positive_energy.detach()).mean()
        ),
        "mask_fraction": float(noisy_tokens.eq(proposal.mask_id).float().mean()),
    }


def _save(
    energy_model: torch.nn.Module,
    ema: EMATracker,
    path: Path,
    *,
    step: int,
    metric: float,
    metadata: dict[str, object],
) -> None:
    save_energy_checkpoint(
        SimpleNamespace(energy_model=energy_model),
        path,
        epoch=step,
        metric=metric,
        extra_metadata=metadata,
        ema_state_dict=ema.state_dict(),
    )


def file_sha256(path: Path) -> str:
    """Hashes a checkpoint without loading the whole file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
    if not torch.cuda.is_available():
        raise RuntimeError("EDLM-NCE training requires Linux/CUDA.")
    if args.global_batch_size % args.micro_batch_size:
        raise ValueError(
            "--global-batch-size must be divisible by --micro-batch-size."
        )
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be positive.")
    if args.log_every <= 0 or args.save_every <= 0:
        raise ValueError("--log-every and --save-every must be positive.")
    if args.bm_num_visible <= 0 or args.bm_num_hidden <= 0:
        raise ValueError("BM visible and hidden sizes must be positive.")
    if not 0.0 < args.bm_sa_alpha < 1.0:
        raise ValueError("--bm-sa-alpha must be in (0, 1).")
    if args.bm_sa_size_limit <= 0:
        raise ValueError("--bm-sa-size-limit must be positive.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    if args.global_batch_size % (args.micro_batch_size * world_size):
        raise ValueError(
            "--global-batch-size must be divisible by micro batch size "
            "times WORLD_SIZE."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", local_rank)

    texts = load_texts(args.input, args.text_field)
    if args.max_records is not None:
        texts = texts[: args.max_records]
    proposal = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    for parameter in proposal.parameters():
        parameter.requires_grad = False
    blocks = build_wrapped_blocks(
        texts,
        proposal.tokenizer,
        sequence_length=args.sequence_length,
    )
    if not blocks:
        raise ValueError("The input produced no complete token blocks.")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + 1)
    sampler = (
        DistributedSampler(
            blocks,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed + 1,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        blocks,
        batch_size=args.micro_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        generator=loader_generator,
    )
    if len(loader) == 0:
        raise ValueError(
            "The token-block dataset is smaller than --micro-batch-size."
        )

    energy_backbone = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .train()
    )
    if args.energy_type == "scalar":
        energy_model = MDLMScalarEnergyModel(energy_backbone)
    else:
        energy_model = MDLMConditionedEnergyModel(
            energy_backbone,
            bm_num_visible=args.bm_num_visible,
            bm_num_hidden=args.bm_num_hidden,
            sampler_type="sa",
            sampler_kwargs={
                "alpha": args.bm_sa_alpha,
                "size_limit": args.bm_sa_size_limit,
                "rand_seed": args.seed + 3,
            },
            scoring_mode="sampler",
            visible_transform=args.bm_visible_transform,
            feature_mode="edlm_pair",
        )
    energy_model = energy_model.to(device).train()
    if isinstance(energy_model, MDLMConditionedEnergyModel):
        energy_model.energy_bm.device = device
        energy_model.energy_bm.dtype = torch.float32
    resume_checkpoint = None
    initial_optimizer_step = 0
    if args.resume_energy_checkpoint is not None:
        resume_checkpoint = read_energy_checkpoint(
            args.resume_energy_checkpoint,
            map_location="cpu",
        )
        initial_optimizer_step = int(resume_checkpoint.get("epoch", 0))
        if initial_optimizer_step < 0:
            raise ValueError("Resume checkpoint epoch must be non-negative.")
        if initial_optimizer_step >= args.max_steps:
            raise ValueError(
                "--max-steps must exceed the resume checkpoint epoch: "
                f"epoch={initial_optimizer_step}, max_steps={args.max_steps}."
            )
        load_energy_weights(
            SimpleNamespace(energy_model=energy_model),
            resume_checkpoint,
            use_ema=args.resume_weights == "ema",
        )
    trainable = [
        parameter
        for parameter in energy_model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: constant_warmup_factor(
            step + initial_optimizer_step,
            warmup_steps=args.warmup_steps,
        ),
    )
    ema = EMATracker.create(energy_model, args.ema_decay)
    if resume_checkpoint is not None and resume_checkpoint.get("ema_state_dict"):
        for name, value in resume_checkpoint["ema_state_dict"].items():
            if name in ema.shadow:
                ema.shadow[name].copy_(
                    value.to(
                        device=ema.shadow[name].device,
                        dtype=ema.shadow[name].dtype,
                    )
                )
    accumulation_steps = args.global_batch_size // (
        args.micro_batch_size * world_size
    )
    metadata: dict[str, object] = {
        "official_edlm_commit": (
            "97e3146964f76aaa784fe523c673516efc7af0e0"
        ),
        "mdlm_checkpoint": args.checkpoint,
        "energy_type": args.energy_type,
        "feature_mode": "edlm_pair",
        "training_objective": "binary_nce",
        "num_proposal_negatives": 1,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "world_size": world_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_clip_norm": args.gradient_clip_norm,
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "ema_decay": args.ema_decay,
        "sampling_eps": args.sampling_eps,
        "noise_eps": args.noise_eps,
        "seed": args.seed,
        "num_records": len(texts),
        "num_blocks": len(blocks),
        "energy_train_scope": "all",
        "data_order_seed": args.seed + 1,
        "training_rng_seed": args.seed + 2,
        "initial_optimizer_step": initial_optimizer_step,
        "target_optimizer_step": args.max_steps,
    }
    if args.resume_energy_checkpoint is not None:
        metadata.update(
            {
                "resume_energy_checkpoint": str(
                    args.resume_energy_checkpoint.resolve()
                ),
                "resume_checkpoint_sha256": file_sha256(
                    args.resume_energy_checkpoint
                ),
                "resume_weights": args.resume_weights,
                "optimizer_state_resumed": False,
                "scheduler_state_resumed": False,
                "ema_state_resumed": bool(
                    resume_checkpoint.get("ema_state_dict")
                ),
            }
        )
    if isinstance(energy_model, MDLMConditionedEnergyModel):
        metadata.update(energy_model.checkpoint_metadata())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    if rank == 0:
        config_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"resolved_seed": args.seed, **metadata}))

    raw_energy_model = energy_model
    if distributed:
        energy_model = DistributedDataParallel(
            raw_energy_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Head construction consumes a different number of random values for scalar
    # and BM models. Reset the training stream so paired runs see identical
    # timesteps, corruptions, proposal negatives, and data order.
    training_seed = args.seed + 2 + rank
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    optimizer.zero_grad(set_to_none=True)
    data_epoch = 0
    if sampler is not None:
        sampler.set_epoch(data_epoch)
    data_iterator = iter(loader)
    optimizer_step = initial_optimizer_step
    micro_step = 0
    running: dict[str, float] = {}
    while optimizer_step < args.max_steps:
        try:
            clean_tokens = next(data_iterator)
        except StopIteration:
            data_epoch += 1
            if sampler is not None:
                sampler.set_epoch(data_epoch)
            data_iterator = iter(loader)
            clean_tokens = next(data_iterator)
        clean_tokens = clean_tokens.to(device, non_blocking=True)
        sync_gradients = (micro_step + 1) % accumulation_steps == 0
        sync_context = (
            nullcontext()
            if not distributed or sync_gradients
            else energy_model.no_sync()
        )
        with sync_context:
            loss, metrics = train_step(
                proposal,
                energy_model,
                clean_tokens,
                sampling_eps=args.sampling_eps,
                noise_eps=args.noise_eps,
            )
            (loss / accumulation_steps).backward()
        micro_step += 1
        for name, value in metrics.items():
            running[name] = running.get(name, 0.0) + value
        if micro_step % accumulation_steps:
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=args.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(raw_energy_model)
        optimizer_step += 1
        averaged = {
            name: value / accumulation_steps
            for name, value in running.items()
        }
        averaged["gradient_norm"] = float(gradient_norm)
        averaged["learning_rate"] = float(scheduler.get_last_lr()[0])
        running.clear()
        if distributed:
            metric_tensor = torch.tensor(
                list(averaged.values()), device=device, dtype=torch.float64
            )
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            metric_tensor /= world_size
            averaged = dict(zip(averaged, metric_tensor.tolist()))
        if rank == 0 and (
            optimizer_step % args.log_every == 0 or optimizer_step == 1
        ):
            print(json.dumps({"step": optimizer_step, **averaged}), flush=True)
        if rank == 0 and optimizer_step % args.save_every == 0:
            _save(
                raw_energy_model,
                ema,
                args.output,
                step=optimizer_step,
                metric=averaged["loss"],
                metadata=metadata,
            )

    if rank == 0:
        _save(
            raw_energy_model,
            ema,
            args.output,
            step=optimizer_step,
            metric=averaged["loss"],
            metadata=metadata,
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
