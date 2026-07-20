"""Train the BM energy side of MDLM-backed QDiffusion on local text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import secrets
import sys

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


def _bootstrap_repo() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for path in (str(repo_root / "src"), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_repo()

try:
    from .builder import build_mdlm_qdiffusion
    from .checkpoint import save_energy_checkpoint
    from .eval_gen_ppl import load_texts
    from .models import MDLMBackbone
except ImportError:  # pragma: no cover - direct script execution
    from builder import build_mdlm_qdiffusion
    from checkpoint import save_energy_checkpoint
    from eval_gen_ppl import load_texts
    from models import MDLMBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--checkpoint", default="kuleshov-group/mdlm-owt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument(
        "--energy-type",
        choices=("scalar", "bm"),
        default="bm",
        help="Scalar EDLM-NCE baseline or BM replacement energy.",
    )
    parser.add_argument(
        "--energy-feature-mode",
        choices=("pooled_pair", "edlm_pair"),
        help=(
            "Energy encoder path. Scalar energy always uses edlm_pair; use "
            "edlm_pair for a head-only scalar-vs-BM comparison."
        ),
    )
    parser.add_argument("--bm-num-visible", type=int, default=64)
    parser.add_argument("--bm-num-hidden", type=int, default=32)
    parser.add_argument(
        "--energy-unfreeze-last-layers",
        type=int,
        default=0,
        help="Number of final energy-MDLM transformer blocks to train.",
    )
    parser.add_argument(
        "--bm-scoring-mode",
        choices=("sampler", "exact"),
        default="sampler",
    )
    parser.add_argument(
        "--bm-visible-transform",
        choices=("sigmoid", "identity", "layernorm"),
        default="sigmoid",
    )
    parser.add_argument(
        "--objective",
        choices=("binary", "ranking"),
        default="binary",
    )
    parser.add_argument("--ranking-teacher", default="gpt2")
    parser.add_argument(
        "--ranking-loss",
        choices=("listwise", "pairwise"),
        default="listwise",
    )
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--ranking-energy-temperature", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=0.1)
    parser.add_argument("--recovery-ranking-weight", type=float, default=0.0)
    parser.add_argument(
        "--recovery-ranking-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. A fresh recorded seed is generated when omitted.",
    )
    return parser.parse_args()


def build_loader(
    texts: list[str],
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    """Builds fixed-length MDLM target batches with explicit padding masks."""

    def collate(batch: list[str]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "targets": encoded["input_ids"].to(device),
            "maskable_mask": encoded["attention_mask"].bool().to(device),
        }

    return DataLoader(
        texts,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def candidate_recovery_scores(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Returns masked-target token accuracy for every proposal candidate."""

    candidate_tokens = outputs["candidate_tokens"]
    targets = outputs["targets"].unsqueeze(1)
    recovery_mask = outputs["loss_mask"].unsqueeze(1).bool()
    correct = candidate_tokens.eq(targets) & recovery_mask
    denominator = recovery_mask.sum(dim=-1).clamp_min(1)
    return correct.sum(dim=-1).float() / denominator


def recovery_ranking_objective(
    candidate_energies: torch.Tensor,
    recovery_scores: torch.Tensor,
    *,
    target_temperature: float,
    energy_temperature: float,
) -> torch.Tensor:
    """Distills target-token recovery ordering into candidate energies."""

    if target_temperature <= 0 or energy_temperature <= 0:
        raise ValueError("Ranking temperatures must be positive.")
    informative = recovery_scores.max(dim=-1).values.gt(
        recovery_scores.min(dim=-1).values
    )
    if not informative.any():
        return candidate_energies.sum() * 0.0
    target_probabilities = torch.softmax(
        recovery_scores[informative] / target_temperature,
        dim=-1,
    )
    energy_log_probabilities = torch.log_softmax(
        -candidate_energies[informative] / energy_temperature,
        dim=-1,
    )
    return -(
        target_probabilities * energy_log_probabilities
    ).sum(dim=-1).mean()


def run_epoch(
    generator,
    loader: DataLoader,
    optimizer: AdamW | None,
    *,
    objective: str,
    teacher_model=None,
    ranking_loss: str = "listwise",
    teacher_temperature: float = 1.0,
    ranking_energy_temperature: float = 1.0,
    binary_loss_weight: float = 0.1,
    recovery_ranking_weight: float = 0.0,
    recovery_ranking_temperature: float = 0.05,
) -> dict[str, float]:
    """Runs one epoch and returns example-weighted energy metrics."""

    training = optimizer is not None
    generator.train(training)
    generator.proposal_model.eval()
    total_loss = 0.0
    total_examples = 0
    positive_energy_total = 0.0
    negative_energy_total = 0.0
    ranking_correct = 0
    ranking_regret_total = 0.0
    ranking_spread_total = 0.0
    pairwise_correct = 0
    pairwise_total = 0
    recovery_ranking_correct = 0
    recovery_regret_total = 0.0
    recovery_spread_total = 0.0
    recovery_pairwise_correct = 0
    recovery_pairwise_total = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            outputs = generator.objective(batch)
            binary_loss = (
                outputs["energy_objective"] * outputs["weight"]
            ).mean()
            if objective == "ranking":
                if teacher_model is None:
                    raise RuntimeError(
                        "Ranking objective requires a teacher model."
                    )
                teacher_nll = candidate_teacher_nll(
                    outputs["candidate_tokens"],
                    teacher_model,
                    eos_token_id=generator.eos_id,
                )
                candidate_energies = outputs["candidate_energies"]
                if ranking_loss == "listwise":
                    teacher_probabilities = torch.softmax(
                        -teacher_nll / teacher_temperature,
                        dim=-1,
                    )
                    student_log_probabilities = torch.log_softmax(
                        -candidate_energies
                        / ranking_energy_temperature,
                        dim=-1,
                    )
                    ranking_objective = -(
                        teacher_probabilities
                        * student_log_probabilities
                    ).sum(dim=-1).mean()
                else:
                    teacher_difference = (
                        teacher_nll.unsqueeze(-1)
                        - teacher_nll.unsqueeze(-2)
                    )
                    energy_difference = (
                        candidate_energies.unsqueeze(-1)
                        - candidate_energies.unsqueeze(-2)
                    )
                    better_pair = teacher_difference.lt(0)
                    pair_losses = torch.nn.functional.softplus(
                        energy_difference[better_pair]
                        / ranking_energy_temperature
                    )
                    if pair_losses.numel():
                        ranking_objective = pair_losses.mean()
                    else:
                        ranking_objective = candidate_energies.sum() * 0.0
                    ordered_correct = energy_difference[better_pair].lt(0)
                    pairwise_correct += int(ordered_correct.sum().item())
                    pairwise_total += int(ordered_correct.numel())
                loss = (
                    ranking_objective
                    + binary_loss_weight * binary_loss
                )
                teacher_best = teacher_nll.argmin(dim=-1)
                energy_best = candidate_energies.argmin(dim=-1)
                ranking_correct += int(
                    teacher_best.eq(energy_best).sum().item()
                )
                selected_nll = teacher_nll.gather(
                    dim=-1,
                    index=energy_best.unsqueeze(-1),
                ).squeeze(-1)
                ranking_regret_total += float(
                    (
                        selected_nll
                        - teacher_nll.min(dim=-1).values
                    ).sum()
                )
                ranking_spread_total += float(
                    (
                        teacher_nll.max(dim=-1).values
                        - teacher_nll.min(dim=-1).values
                    ).sum()
                )
            elif recovery_ranking_weight:
                candidate_energies = outputs["candidate_energies"]
                recovery_scores = candidate_recovery_scores(outputs)
                recovery_objective = recovery_ranking_objective(
                    candidate_energies,
                    recovery_scores,
                    target_temperature=recovery_ranking_temperature,
                    energy_temperature=ranking_energy_temperature,
                )
                loss = (
                    binary_loss
                    + recovery_ranking_weight * recovery_objective
                )
                energy_best = candidate_energies.argmin(dim=-1)
                selected_recovery = recovery_scores.gather(
                    dim=-1,
                    index=energy_best.unsqueeze(-1),
                ).squeeze(-1)
                best_recovery = recovery_scores.max(dim=-1).values
                worst_recovery = recovery_scores.min(dim=-1).values
                recovery_ranking_correct += int(
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
                energy_difference = (
                    candidate_energies.unsqueeze(-1)
                    - candidate_energies.unsqueeze(-2)
                )
                better_pair = recovery_difference.gt(0)
                recovery_pairwise_correct += int(
                    energy_difference[better_pair].lt(0).sum()
                )
                recovery_pairwise_total += int(better_pair.sum())
            else:
                loss = binary_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in generator.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
            batch_size = int(batch["targets"].size(0))
            total_loss += float(loss) * batch_size
            positive_energy_total += (
                float(outputs["positive_energy_mean"]) * batch_size
            )
            negative_energy_total += (
                float(outputs["negative_energy_mean"]) * batch_size
            )
            total_examples += batch_size
    denominator = max(total_examples, 1)
    positive_energy = positive_energy_total / denominator
    negative_energy = negative_energy_total / denominator
    metrics = {
        "energy_objective": total_loss / denominator,
        "positive_energy": positive_energy,
        "negative_energy": negative_energy,
        "energy_margin": negative_energy - positive_energy,
    }
    if objective == "ranking":
        metrics.update(
            ranking_top1_accuracy=ranking_correct / denominator,
            ranking_teacher_regret=ranking_regret_total / denominator,
            ranking_teacher_nll_spread=(
                ranking_spread_total / denominator
            ),
            ranking_pair_count=pairwise_total,
            ranking_pairwise_accuracy=(
                pairwise_correct / pairwise_total
                if pairwise_total
                else 0.0
            ),
        )
    if recovery_ranking_weight:
        metrics.update(
            recovery_top1_accuracy=(
                recovery_ranking_correct / denominator
            ),
            recovery_regret=recovery_regret_total / denominator,
            recovery_score_spread=(
                recovery_spread_total / denominator
            ),
            recovery_pair_count=recovery_pairwise_total,
            recovery_pairwise_accuracy=(
                recovery_pairwise_correct / recovery_pairwise_total
                if recovery_pairwise_total
                else 0.0
            ),
        )
    return metrics


@torch.no_grad()
def candidate_teacher_nll(
    candidate_tokens: torch.Tensor,
    teacher_model,
    *,
    eos_token_id: int,
) -> torch.Tensor:
    """Returns mean causal-LM NLL for every candidate sequence."""

    batch_size, num_candidates, seq_len = candidate_tokens.shape
    flat_tokens = candidate_tokens.reshape(
        batch_size * num_candidates,
        seq_len,
    )
    non_eos = flat_tokens.ne(eos_token_id)
    # MDLM uses GPT-2's shared BOS/EOS token at position zero. Treat that
    # leading marker as BOS so it does not mask the entire generated sequence.
    if seq_len:
        non_eos[:, 0] = True
    attention_mask = torch.cat(
        [
            torch.ones_like(non_eos[:, :1]),
            non_eos[:, :-1].cumprod(dim=-1),
        ],
        dim=-1,
    ).bool()
    teacher_outputs = teacher_model(
        input_ids=flat_tokens,
        attention_mask=attention_mask,
        return_dict=True,
    )
    shifted_logits = teacher_outputs.logits[:, :-1].float()
    shifted_targets = flat_tokens[:, 1:]
    valid_tokens = attention_mask[:, 1:]
    token_nll = (
        -torch.log_softmax(shifted_logits, dim=-1)
        .gather(-1, shifted_targets.unsqueeze(-1))
        .squeeze(-1)
    )
    sequence_nll = (
        token_nll.masked_fill(~valid_tokens, 0.0).sum(dim=-1)
        / valid_tokens.sum(dim=-1).clamp_min(1)
    )
    return sequence_nll.view(batch_size, num_candidates)


def validate_trainable_parameters(generator) -> list[torch.nn.Parameter]:
    """Ensures training cannot accidentally update the frozen MDLM backbone."""

    allowed_prefixes = (
        "energy_model.conditioned_encoder.",
        "energy_model.energy_head.",
        "energy_model.feature_projector.",
        "energy_model.visible_transform.",
        "energy_model.energy_bm.",
        "energy_model.encoder.model.backbone.blocks.",
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in generator.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("No trainable energy parameters were found.")
    unexpected = [
        name
        for name, _ in named_parameters
        if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected trainable parameters outside the energy model: "
            + ", ".join(unexpected)
        )
    return [parameter for _, parameter in named_parameters]


def split_texts(texts: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Creates a deterministic 95/5 train-validation split."""

    if len(texts) < 2:
        raise ValueError("At least two text records are required for training.")
    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(0.05 * len(shuffled)))
    return shuffled[validation_size:], shuffled[:validation_size]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MDLM energy training requires a Linux/CUDA environment.")
    if args.recovery_ranking_weight < 0:
        raise ValueError("--recovery-ranking-weight must be non-negative.")
    if args.objective == "ranking" and args.recovery_ranking_weight:
        raise ValueError(
            "GPT-2 ranking and recovery ranking cannot be enabled together."
        )
    energy_feature_mode = args.energy_feature_mode
    if energy_feature_mode is None:
        energy_feature_mode = (
            "edlm_pair"
            if args.energy_type == "scalar"
            else "pooled_pair"
        )
    if args.energy_type == "scalar" and energy_feature_mode != "edlm_pair":
        raise ValueError(
            "Scalar EDLM energy requires --energy-feature-mode edlm_pair."
        )
    if args.seed is None:
        args.seed = secrets.randbelow(2**31)
    print(json.dumps({"resolved_seed": args.seed}))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    texts = load_texts(args.input, args.text_field)
    if args.max_records is not None:
        texts = texts[: args.max_records]
    train_texts, validation_texts = split_texts(texts, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    backbone = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            # The released MDLM timestep embedding is explicitly float32.
            # Its transformer blocks manage bf16 autocast internally.
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
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
    trainable_energy_names = energy_backbone.train_last_blocks(
        args.energy_unfreeze_last_layers
    )
    generator = build_mdlm_qdiffusion(
        backbone,
        use_energy=True,
        energy_type=args.energy_type,
        energy_feature_mode=energy_feature_mode,
        energy_backbone=energy_backbone,
        bm_num_visible=args.bm_num_visible,
        bm_num_hidden=args.bm_num_hidden,
        bm_scoring_mode=args.bm_scoring_mode,
        bm_visible_transform=args.bm_visible_transform,
        num_candidates=args.num_candidates,
        dtype=torch.float32,
        device=device,
    )
    teacher_model = None
    if args.objective == "ranking":
        from transformers import AutoModelForCausalLM

        teacher_model = (
            AutoModelForCausalLM.from_pretrained(
                args.ranking_teacher,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
    train_loader = build_loader(
        train_texts,
        backbone.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        shuffle=True,
    )
    validation_loader = build_loader(
        validation_texts,
        backbone.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        shuffle=False,
    )
    optimizer = AdamW(
        validate_trainable_parameters(generator),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_kwargs = {
            "objective": args.objective,
            "teacher_model": teacher_model,
            "ranking_loss": args.ranking_loss,
            "teacher_temperature": args.teacher_temperature,
            "ranking_energy_temperature": args.ranking_energy_temperature,
            "binary_loss_weight": args.binary_loss_weight,
            "recovery_ranking_weight": args.recovery_ranking_weight,
            "recovery_ranking_temperature": (
                args.recovery_ranking_temperature
            ),
        }
        train_metrics = run_epoch(
            generator,
            train_loader,
            optimizer,
            **epoch_kwargs,
        )
        validation_metrics = run_epoch(
            generator,
            validation_loader,
            None,
            **epoch_kwargs,
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(row)
        print(json.dumps(row))
        validation_loss = validation_metrics["energy_objective"]
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_energy_checkpoint(
                generator,
                args.output,
                epoch=epoch,
                metric=validation_loss,
                extra_metadata={
                    "mdlm_checkpoint": args.checkpoint,
                    "tokenizer": args.tokenizer,
                    "max_length": args.max_length,
                    "seed": args.seed,
                    "energy_unfreeze_last_layers": (
                        args.energy_unfreeze_last_layers
                    ),
                    "energy_feature_mode": energy_feature_mode,
                    "trainable_energy_parameters": len(
                        trainable_energy_names
                    ),
                    "objective": args.objective,
                    "ranking_teacher": (
                        args.ranking_teacher
                        if args.objective == "ranking"
                        else None
                    ),
                    "ranking_loss": (
                        args.ranking_loss
                        if args.objective == "ranking"
                        else None
                    ),
                    "recovery_ranking_weight": (
                        args.recovery_ranking_weight
                    ),
                    "recovery_ranking_temperature": (
                        args.recovery_ranking_temperature
                    ),
                },
            )

    history_path = args.output.with_suffix(args.output.suffix + ".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
