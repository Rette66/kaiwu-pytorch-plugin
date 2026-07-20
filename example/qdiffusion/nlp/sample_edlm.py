"""Sample the MDLM baseline or EDLM-NCE with the paper's DDPM-cache path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

import torch


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

from .builder import build_mdlm_qdiffusion  # noqa: E402
from .checkpoint import (  # noqa: E402
    load_energy_weights,
    read_energy_checkpoint,
)
from .edlm_sampling import EDLMDDPMCacheSampler  # noqa: E402
from .models import MDLMBackbone  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="kuleshov-group/mdlm-owt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--energy-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--importance-start-t", type=float, default=1.0)
    parser.add_argument("--importance-end-t", type=float, default=0.8)
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _load_scalar_energy(
    args: argparse.Namespace,
    proposal: MDLMBackbone,
    device: torch.device,
) -> torch.nn.Module | None:
    if args.energy_checkpoint is None:
        return None
    if args.num_candidates is None:
        args.num_candidates = 2
    checkpoint = read_energy_checkpoint(args.energy_checkpoint)
    metadata = checkpoint["metadata"]
    if metadata.get("energy_type") != "scalar":
        raise ValueError(
            "The paper reproduction accepts only an EDLM scalar checkpoint; "
            "BM checkpoints belong to a later ablation."
        )
    trained_checkpoint = metadata.get("mdlm_checkpoint")
    if trained_checkpoint is not None and trained_checkpoint != args.checkpoint:
        raise ValueError(
            "Energy and proposal checkpoints differ: "
            f"{trained_checkpoint!r} != {args.checkpoint!r}."
        )
    energy_backbone = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    generator = build_mdlm_qdiffusion(
        proposal,
        use_energy=True,
        energy_type="scalar",
        energy_feature_mode="edlm_pair",
        energy_backbone=energy_backbone,
        num_candidates=args.num_candidates,
        dtype=torch.float32,
        device=device,
    )
    load_energy_weights(generator, checkpoint)
    generator.energy_model.eval()
    return generator.energy_model


def main() -> None:
    args = parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
    if args.sequence_length <= 0 or args.steps <= 0:
        raise ValueError("--sequence-length and --steps must be positive.")
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("--num-samples and --batch-size must be positive.")
    if args.energy_checkpoint is None and args.num_candidates is not None:
        raise ValueError(
            "Baseline MDLM has no candidate pool; omit --num-candidates."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The released MDLM checkpoint requires Linux/CUDA.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    proposal = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    energy_model = _load_scalar_energy(args, proposal, device)
    sampler = EDLMDDPMCacheSampler(
        proposal,
        mask_id=proposal.mask_id,
        energy_model=energy_model,
        num_candidates=(args.num_candidates if energy_model is not None else 1),
        energy_temperature=args.energy_temperature,
        importance_start_t=args.importance_start_t,
        importance_end_t=args.importance_end_t,
        noise_removal=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    aggregate_forwards = 0
    aggregate_guided_steps = 0
    for batch_start in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - batch_start)
        masked = torch.full(
            (batch_size, args.sequence_length),
            proposal.mask_id,
            dtype=torch.long,
            device=device,
        )
        samples = sampler.sample(masked, num_steps=args.steps)
        aggregate_forwards += int(sampler.last_stats["proposal_forwards"])
        aggregate_guided_steps += int(sampler.last_stats["guided_steps"])
        for row in samples.cpu().tolist():
            records.append(
                {
                    "text": proposal.tokenizer.decode(
                        row,
                        skip_special_tokens=False,
                    ),
                    "token_ids": row,
                    "seed": args.seed,
                    "sampler": "ddpm_cache",
                    "steps": args.steps,
                    "num_candidates": (
                        args.num_candidates if energy_model is not None else 1
                    ),
                    "importance_start_t": args.importance_start_t,
                    "importance_end_t": args.importance_end_t,
                }
            )

    with args.output.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "resolved_seed": args.seed,
                "num_samples": len(records),
                "proposal_forwards": aggregate_forwards,
                "guided_steps": aggregate_guided_steps,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
