"""Generate text with MDLM alone or BM-guided MDLM QDiffusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _bootstrap_repo() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"
    for path in (str(src_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_repo()

try:
    from .builder import build_mdlm_qdiffusion
    from .checkpoint import load_energy_weights, read_energy_checkpoint
    from .models import MDLMBackbone
except ImportError:  # pragma: no cover - direct script execution
    from builder import build_mdlm_qdiffusion
    from checkpoint import load_energy_weights, read_energy_checkpoint
    from models import MDLMBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="kuleshov-group/mdlm-owt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--prompt")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Plain-text or JSONL prompts; JSONL records use the 'prompt' field.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Generation batch size; defaults to --num-samples.",
    )
    parser.add_argument("--energy-checkpoint", type=Path)
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument(
        "--bm-scoring-mode",
        choices=("sampler", "exact"),
        help="Overrides the energy checkpoint scoring mode.",
    )
    parser.add_argument("--energy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--proposal-score-weight",
        type=float,
        default=0.0,
        help="Standardized proposal-score weight used with BM energy.",
    )
    parser.add_argument("--residual-guidance-weight", type=float)
    parser.add_argument("--residual-fallback-margin", type=float, default=0.0)
    parser.add_argument(
        "--include-greedy-candidate",
        action="store_true",
        help="Reserves candidate zero for the deterministic MDLM argmax.",
    )
    parser.add_argument(
        "--energy-start-ratio",
        type=float,
        default=0.0,
        help="Fraction of decode steps completed before BM guidance starts.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prompts(prompt: str | None, prompts_file: Path | None) -> list[str]:
    """Loads a single prompt or a prompt suite from disk."""

    if prompt is not None and prompts_file is not None:
        raise ValueError("--prompt and --prompts-file are mutually exclusive.")
    if prompts_file is None:
        return [prompt or "Diffusion language models"]

    prompts = []
    for line_number, raw_line in enumerate(
        prompts_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        if prompts_file.suffix.lower() == ".jsonl":
            record = json.loads(raw_line)
            value = record.get("prompt")
        else:
            value = raw_line.strip()
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid prompt on line {line_number} of {prompts_file}."
            )
        prompts.append(value.strip())
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}.")
    return prompts


def build_prompt_canvas(
    backbone: MDLMBackbone,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = backbone.tokenizer
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prefix = [int(tokenizer.bos_token_id), *prompt_ids]
    tokens = torch.full(
        (batch_size, len(prefix) + max_new_tokens),
        backbone.mask_id,
        dtype=torch.long,
        device=device,
    )
    tokens[:, : len(prefix)] = torch.tensor(prefix, device=device)
    fixed_prompt = torch.zeros_like(tokens, dtype=torch.bool)
    fixed_prompt[:, : len(prefix)] = True
    return tokens, fixed_prompt, len(prefix)


def decode_responses(
    tokens: torch.Tensor,
    tokenizer,
    prompt_length: int,
) -> list[str]:
    responses = []
    for row in tokens[:, prompt_length:].tolist():
        if tokenizer.eos_token_id in row:
            row = row[: row.index(tokenizer.eos_token_id)]
        responses.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return responses


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    batch_size = args.batch_size or args.num_samples
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The released MDLM checkpoint depends on flash-attn and requires CUDA."
        )

    device = torch.device("cuda")
    energy_checkpoint = None
    if args.energy_checkpoint is not None:
        energy_checkpoint = read_energy_checkpoint(args.energy_checkpoint)
        trained_backbone = energy_checkpoint["metadata"].get("mdlm_checkpoint")
        if trained_backbone is not None and trained_backbone != args.checkpoint:
            raise ValueError(
                "Energy checkpoint was trained with a different MDLM backbone: "
                f"{trained_backbone!r}."
            )
    backbone = (
        MDLMBackbone.from_pretrained(
            args.checkpoint,
            tokenizer_name_or_path=args.tokenizer,
            # The released MDLM keeps its timestep MLP in float32 and enables
            # bf16 autocast only inside the transformer backbone.
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    use_energy = energy_checkpoint is not None
    metadata = energy_checkpoint["metadata"] if energy_checkpoint else {}
    num_candidates = args.num_candidates or (8 if use_energy else 1)
    generator = build_mdlm_qdiffusion(
        backbone,
        use_energy=use_energy,
        bm_num_visible=int(metadata.get("bm_num_visible", 64)),
        bm_num_hidden=int(metadata.get("bm_num_hidden", 32)),
        bm_visible_transform=metadata.get("visible_transform", "sigmoid"),
        bm_sampler_type=metadata.get("sampler_type", "sa"),
        bm_sampler_kwargs=metadata.get("sampler_kwargs", {}),
        bm_scoring_mode=(
            args.bm_scoring_mode
            or metadata.get("scoring_mode", "sampler")
        ),
        num_candidates=num_candidates,
        energy_temperature=args.energy_temperature,
        proposal_score_weight=args.proposal_score_weight,
        residual_guidance_weight=args.residual_guidance_weight,
        residual_fallback_margin=args.residual_fallback_margin,
        include_greedy_candidate=args.include_greedy_candidate,
        energy_guidance_start_ratio=args.energy_start_ratio,
        dtype=torch.float32,
        device=device,
    )
    if energy_checkpoint is not None:
        load_energy_weights(generator, energy_checkpoint)
    prompts = load_prompts(args.prompt, args.prompts_file)
    records = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_responses = []
        for batch_start in range(0, args.num_samples, batch_size):
            current_batch_size = min(
                batch_size,
                args.num_samples - batch_start,
            )
            input_tokens, fixed_prompt, prompt_length = build_prompt_canvas(
                backbone,
                prompt,
                args.max_new_tokens,
                current_batch_size,
                device,
            )
            output = generator.generate(
                input_tokens,
                max_steps=args.steps,
                partial_masks=fixed_prompt,
            )
            prompt_responses.extend(
                decode_responses(output, backbone.tokenizer, prompt_length)
            )
        for sample_index, response in enumerate(prompt_responses):
            print(f"prompt[{prompt_index}] sample[{sample_index}]: {response}")
            records.append({"prompt": prompt, "text": response})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )


if __name__ == "__main__":
    main()
