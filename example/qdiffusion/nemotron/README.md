# Nemotron Contextual-Energy QDiffusion Example

This example composes a frozen Nemotron block-diffusion proposal with a
contextual energy encoder, the public KPP `EnergyModel`/`BoltzmannMachine`, and
the PyPI Kaiwu simulated-annealing optimizer. It includes private-data pair collection, outcome-pairwise
training, versioned checkpoints, guided inference, and resumable matched
Native/BM evaluation.

The repository does not contain model weights or training data. All paths,
devices, datasets, and checkpoints are supplied explicitly at runtime.
Training-pair collection accepts only `train` and `val`; benchmark test answers
must not be used to collect pairs or select a checkpoint.

The locked reference configuration is a 1024-dim 4-layer contextual encoder,
KPP BM 512x256, K=4, lambda=2.0, and an outcome-pairwise objective with a 0.01
NCE energy-scale regularizer. This is not strict EDLM-NCE.

Pair artifacts use schema v2. Each pair stores the current noisy block, its
token features, frozen hidden states, and the positive and negative candidates.
Older v1 artifacts and checkpoints cannot be reused; recollect pairs and
retrain the contextual model.

Nemotron's native `generate` remains responsible for cache management,
denoising, and stopping. A plain `QDiffusion` instance is used only to score
the candidates exposed at its transfer-selection point; the example does not
copy the native generation loop.

Install the repository and example requirements:

```bash
pip install -e .
pip install -r example/qdiffusion/nemotron/requirements.txt
```

The sampler is `kaiwu.classical.SimulatedAnnealingOptimizer` from the PyPI
`kaiwu` dependency pinned by this repository.

## Entrypoints

All commands run from the repository root as modules. The pipeline is:

| Entrypoint | Purpose |
|---|---|
| `prepare_pairs` | Offline collection: run the frozen model, capture same-state candidate branches, force alternative branches, and label them by final-answer correctness into `(positive, negative)` pairs. Supports sharded runs and resumable collection. |
| `merge_pairs` | Concatenate independently collected `pairs.pt` shards into one artifact, re-validating schema and train/val split consistency across shards. |
| `train` | Train the `ContextualEnergyModel` on merged pairs (outcome-pairwise margin + small NCE regularizer); produces `best.pt` selected on held-out val pairs and resumable via `last.pt`. |
| `evaluate` | Matched Native vs BM-guided decoding on an eval JSONL, scored with boxed-only `math-verify`; append-only resume per strategy. |

## Quick Start

```bash
# 1. collect pairs (run per split; shard with --indices/--limit as needed)
python -m example.qdiffusion.nemotron.prepare_pairs \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --dataset-jsonl /path/to/private_math.jsonl \
  --split train --output-dir /path/to/pairs_train --device cuda:0

# 2. merge shards
python -m example.qdiffusion.nemotron.merge_pairs \
  --inputs /path/to/pairs_train/pairs.pt /path/to/pairs_val/pairs.pt \
  --output /path/to/pairs_merged.pt

# 3. train the energy model
python -m example.qdiffusion.nemotron.train \
  --pairs /path/to/pairs_merged.pt \
  --output-dir /path/to/contextual_energy_run --device cuda:0

# 4. matched Native/BM evaluation
python -m example.qdiffusion.nemotron.evaluate \
  --model /path/to/Nemotron-Labs-Diffusion-8B \
  --checkpoint /path/to/contextual_energy_run/best.pt \
  --dataset-jsonl /path/to/private_math_eval.jsonl \
  --output-dir /path/to/eval_output --device cuda:0 \
  --strategies native bm
```

See [README_ZH.md](README_ZH.md) for the data schema, the full argument list,
resume semantics, and evaluation output fields.

Resume metadata fingerprints private input files by path, size, and SHA-256;
the files themselves are never copied into the repository.
