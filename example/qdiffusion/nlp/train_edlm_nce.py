"""Train the paper-aligned EDLM-NCE scalar energy on wrapped GPT-2 blocks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import secrets
import sys
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


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

from .checkpoint import save_energy_checkpoint  # noqa: E402
from .eval_gen_ppl import load_texts  # noqa: E402
from .models import MDLMBackbone, MDLMScalarEnergyModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--checkpoint", default="kuleshov-group/mdlm-owt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--sampling-eps", type=float, default=1e-3)
    parser.add_argument("--noise-eps", type=float, default=1e-3)
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
    energy_model: MDLMScalarEnergyModel,
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
    positive_energy = energy_model.score_conditioned(
        noisy_tokens,
        clean_tokens,
        attention_mask,
    )
    negative_energy = energy_model.score_conditioned(
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
    energy_model: MDLMScalarEnergyModel,
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

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
    loader = DataLoader(
        blocks,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
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
    energy_model = MDLMScalarEnergyModel(energy_backbone).to(device).train()
    trainable = [
        parameter
        for parameter in energy_model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema = EMATracker.create(energy_model, args.ema_decay)
    accumulation_steps = args.global_batch_size // args.micro_batch_size
    metadata: dict[str, object] = {
        "official_edlm_commit": (
            "97e3146964f76aaa784fe523c673516efc7af0e0"
        ),
        "mdlm_checkpoint": args.checkpoint,
        "energy_type": "scalar",
        "feature_mode": "edlm_pair",
        "training_objective": "binary_nce",
        "num_proposal_negatives": 1,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "sampling_eps": args.sampling_eps,
        "noise_eps": args.noise_eps,
        "seed": args.seed,
        "num_records": len(texts),
        "num_blocks": len(blocks),
        "energy_train_scope": "all",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    config_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"resolved_seed": args.seed, **metadata}))

    optimizer.zero_grad(set_to_none=True)
    data_iterator = iter(loader)
    optimizer_step = 0
    micro_step = 0
    running: dict[str, float] = {}
    while optimizer_step < args.max_steps:
        try:
            clean_tokens = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            clean_tokens = next(data_iterator)
        clean_tokens = clean_tokens.to(device, non_blocking=True)
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

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(energy_model)
        optimizer_step += 1
        averaged = {
            name: value / accumulation_steps
            for name, value in running.items()
        }
        running.clear()
        if optimizer_step % args.log_every == 0 or optimizer_step == 1:
            print(json.dumps({"step": optimizer_step, **averaged}), flush=True)
        if optimizer_step % args.save_every == 0:
            _save(
                energy_model,
                ema,
                args.output,
                step=optimizer_step,
                metric=averaged["loss"],
                metadata=metadata,
            )

    _save(
        energy_model,
        ema,
        args.output,
        step=optimizer_step,
        metric=averaged["loss"],
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
