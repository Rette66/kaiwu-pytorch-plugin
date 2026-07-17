"""Evaluate how well an NLP energy checkpoint ranks MDLM candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def _bootstrap_repo() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_repo()

try:
    from .builder import build_mdlm_qdiffusion
    from .checkpoint import read_energy_checkpoint, load_energy_weights
    from .eval_gen_ppl import load_texts
    from .models import MDLMBackbone
    from .train_energy import (
        build_loader,
        candidate_recovery_scores,
        candidate_teacher_nll,
    )
except ImportError:  # pragma: no cover - direct script execution
    from builder import build_mdlm_qdiffusion
    from checkpoint import read_energy_checkpoint, load_energy_weights
    from eval_gen_ppl import load_texts
    from models import MDLMBackbone
    from train_energy import (
        build_loader,
        candidate_recovery_scores,
        candidate_teacher_nll,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--checkpoint", default="kuleshov-group/mdlm-owt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--energy-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", default="gpt2")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--on-policy-rollout-steps", type=int, default=0)
    parser.add_argument("--on-policy-max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _rank(values: torch.Tensor) -> torch.Tensor:
    return values.argsort(dim=-1).argsort(dim=-1).float()


def evaluate_ranking(
    generator,
    loader,
    teacher_model,
    *,
    on_policy_rollout_steps: int = 0,
    on_policy_max_steps: int = 64,
) -> dict[str, float | int]:
    """Compares BM ordering with causal-LM candidate NLL ordering."""

    generator.eval()
    generator.proposal_model.eval()
    num_examples = 0
    top1_correct = 0
    pairwise_correct = 0
    pairwise_total = 0
    selected_nll_total = 0.0
    oracle_nll_total = 0.0
    candidate_mean_nll_total = 0.0
    regret_total = 0.0
    normalized_regret_total = 0.0
    spearman_total = 0.0
    positive_energy_total = 0.0
    negative_energy_total = 0.0
    recovery_top1_correct = 0
    recovery_pairwise_correct = 0
    recovery_pairwise_total = 0
    recovery_regret_total = 0.0
    recovery_spread_total = 0.0

    with torch.no_grad():
        for batch in loader:
            outputs = generator.objective(
                batch,
                rollout_steps=on_policy_rollout_steps,
                rollout_max_steps=on_policy_max_steps,
            )
            energies = outputs["candidate_energies"]
            teacher_nll = candidate_teacher_nll(
                outputs["candidate_tokens"],
                teacher_model,
                eos_token_id=generator.eos_id,
            )
            recovery_scores = candidate_recovery_scores(outputs)
            batch_size, num_candidates = teacher_nll.shape
            energy_best = energies.argmin(dim=-1)
            teacher_best = teacher_nll.argmin(dim=-1)
            selected_nll = teacher_nll.gather(
                dim=-1,
                index=energy_best.unsqueeze(-1),
            ).squeeze(-1)
            oracle_nll = teacher_nll.min(dim=-1).values
            spread = (
                teacher_nll.max(dim=-1).values
                - oracle_nll
            )
            regret = selected_nll - oracle_nll

            teacher_difference = (
                teacher_nll.unsqueeze(-1)
                - teacher_nll.unsqueeze(-2)
            )
            energy_difference = (
                energies.unsqueeze(-1)
                - energies.unsqueeze(-2)
            )
            upper_triangle = torch.triu(
                torch.ones(
                    num_candidates,
                    num_candidates,
                    dtype=torch.bool,
                    device=teacher_nll.device,
                ),
                diagonal=1,
            )
            comparable = (
                upper_triangle.unsqueeze(0).expand(batch_size, -1, -1)
                & teacher_difference.ne(0)
            )
            pairwise_correct += int(
                (
                    teacher_difference[comparable]
                    * energy_difference[comparable]
                ).gt(0).sum()
            )
            pairwise_total += int(comparable.sum())

            teacher_rank = _rank(teacher_nll)
            energy_rank = _rank(energies)
            teacher_centered = teacher_rank - teacher_rank.mean(
                dim=-1,
                keepdim=True,
            )
            energy_centered = energy_rank - energy_rank.mean(
                dim=-1,
                keepdim=True,
            )
            rank_correlation = (
                (teacher_centered * energy_centered).sum(dim=-1)
                / (
                    teacher_centered.square().sum(dim=-1).sqrt()
                    * energy_centered.square().sum(dim=-1).sqrt()
                ).clamp_min(1e-8)
            )

            num_examples += batch_size
            top1_correct += int(energy_best.eq(teacher_best).sum())
            selected_nll_total += float(selected_nll.sum())
            oracle_nll_total += float(oracle_nll.sum())
            candidate_mean_nll_total += float(
                teacher_nll.mean(dim=-1).sum()
            )
            regret_total += float(regret.sum())
            normalized_regret_total += float(
                (regret / spread.clamp_min(1e-8)).sum()
            )
            spearman_total += float(rank_correlation.sum())
            selected_recovery = recovery_scores.gather(
                dim=-1,
                index=energy_best.unsqueeze(-1),
            ).squeeze(-1)
            best_recovery = recovery_scores.max(dim=-1).values
            worst_recovery = recovery_scores.min(dim=-1).values
            recovery_top1_correct += int(
                selected_recovery.eq(best_recovery).sum()
            )
            recovery_regret_total += float(
                (best_recovery - selected_recovery).sum()
            )
            recovery_spread_total += float(
                (best_recovery - worst_recovery).sum()
            )
            recovery_difference = (
                recovery_scores.unsqueeze(-1)
                - recovery_scores.unsqueeze(-2)
            )
            recovery_better_pair = recovery_difference.gt(0)
            recovery_pairwise_correct += int(
                energy_difference[recovery_better_pair].lt(0).sum()
            )
            recovery_pairwise_total += int(recovery_better_pair.sum())
            positive_energy_total += (
                float(outputs["positive_energy_mean"]) * batch_size
            )
            negative_energy_total += (
                float(outputs["negative_energy_mean"]) * batch_size
            )

    denominator = max(num_examples, 1)
    return {
        "num_examples": num_examples,
        "num_candidates": generator.config.num_candidates,
        "ranking_top1_accuracy": top1_correct / denominator,
        "ranking_pairwise_accuracy": (
            pairwise_correct / pairwise_total
            if pairwise_total
            else 0.0
        ),
        "ranking_spearman": spearman_total / denominator,
        "selected_teacher_nll": selected_nll_total / denominator,
        "oracle_teacher_nll": oracle_nll_total / denominator,
        "candidate_mean_teacher_nll": (
            candidate_mean_nll_total / denominator
        ),
        "teacher_regret": regret_total / denominator,
        "normalized_teacher_regret": (
            normalized_regret_total / denominator
        ),
        "positive_energy": positive_energy_total / denominator,
        "negative_energy": negative_energy_total / denominator,
        "energy_margin": (
            negative_energy_total - positive_energy_total
        ) / denominator,
        "recovery_top1_accuracy": (
            recovery_top1_correct / denominator
        ),
        "recovery_pairwise_accuracy": (
            recovery_pairwise_correct / recovery_pairwise_total
            if recovery_pairwise_total
            else 0.0
        ),
        "recovery_regret": recovery_regret_total / denominator,
        "recovery_score_spread": (
            recovery_spread_total / denominator
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MDLM ranking evaluation requires CUDA.")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative.")
    if args.max_records <= 0:
        raise ValueError("--max-records must be positive.")

    device = torch.device("cuda")
    checkpoint = read_energy_checkpoint(args.energy_checkpoint)
    metadata = checkpoint["metadata"]
    trained_backbone = metadata.get("mdlm_checkpoint")
    if trained_backbone is not None and trained_backbone != args.checkpoint:
        raise ValueError(
            "Energy checkpoint was trained with a different MDLM backbone: "
            f"{trained_backbone!r}."
        )

    backbone = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    energy_unfreeze_last_layers = int(
        metadata.get("energy_unfreeze_last_layers", 0)
    )
    energy_backbone = None
    if energy_unfreeze_last_layers:
        energy_backbone = (
            MDLMBackbone.from_pretrained(
                args.checkpoint,
                tokenizer_name_or_path=args.tokenizer,
                torch_dtype=torch.float32,
            )
            .to(device)
            .eval()
        )
        energy_backbone.train_last_blocks(energy_unfreeze_last_layers)

    generator = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        energy_backbone=energy_backbone,
        bm_num_visible=int(metadata.get("bm_num_visible", 64)),
        bm_num_hidden=int(metadata.get("bm_num_hidden", 32)),
        bm_visible_transform=metadata.get("visible_transform", "sigmoid"),
        bm_sampler_type=metadata.get("sampler_type", "sa"),
        bm_sampler_kwargs=metadata.get("sampler_kwargs", {}),
        bm_scoring_mode=metadata.get("scoring_mode", "sampler"),
        num_candidates=args.num_candidates,
        dtype=torch.float32,
        device=device,
    )
    load_energy_weights(generator, checkpoint)

    from transformers import AutoModelForCausalLM

    teacher_model = (
        AutoModelForCausalLM.from_pretrained(
            args.teacher,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False

    texts = load_texts(args.input, args.text_field)
    texts = texts[args.offset : args.offset + args.max_records]
    if not texts:
        raise ValueError("The selected evaluation slice is empty.")
    loader = build_loader(
        texts,
        backbone.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        shuffle=False,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    metrics = evaluate_ranking(
        generator,
        loader,
        teacher_model,
        on_policy_rollout_steps=args.on_policy_rollout_steps,
        on_policy_max_steps=args.on_policy_max_steps,
    )
    result = {
        "energy_checkpoint": str(args.energy_checkpoint),
        "input": str(args.input),
        "offset": args.offset,
        "seed": args.seed,
        "on_policy_rollout_steps": args.on_policy_rollout_steps,
        "on_policy_max_steps": args.on_policy_max_steps,
        **metrics,
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
