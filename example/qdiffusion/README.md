# Language Versions: [中文](README_ZH.md) | [English](README.md)

# `example/qdiffusion`

Examples and workflow scripts for the public `Q-Diffusion` module.

## Data
[data used for example/qdiffusion](https://www.uniprot.org/proteomes/UP000005640)


## Quick Start

```bash
pip install -r example/qdiffusion/requirements.txt
python example/qdiffusion/dplm/train_workflow.py
python example/qdiffusion/dplm/eval_esm2_distances.py
```

These commands work in both common local modes:

- development checkout: run directly from the repo root
- installed package: `pip install -e .` and then run the same commands

## How It Fits Together

This example tree is organized around one simple boundary:

- `src/kaiwu/torch_plugin/qdiffusion.py` contains the generic training and energy-scoring core
- `example/qdiffusion/dplm/` adapts DPLM checkpoints into that generic core

The main assembly path is:

1. a script calls `build_qdiffusion(...)`
2. `dplm/model/model.py` loads one DPLM proposal backbone and one DPLM feature encoder
3. `dplm/model/` builds one conditioned BM reranker on top of that feature encoder
4. the builder constructs an example-side `GenerativeQDiffusion(...)`
5. the script calls `objective(...)` for training or the example `generate(...)` policy for inference

## dplm

`dplm/` contains the protein-case adapter layer, trainer, and downstream evaluation code:

- `dplm/model/`: model-side code, split into backbone loading, energy rerankers, private ESM patching, generation policy, and the `Q-Diffusion` assembly helper
- `dplm/trainer/`: training configs, data loaders, epoch loops, checkpointing flow, and train-time reports
- `dplm/downstream/`: downstream ESM2 distance evaluation and related generation helpers
- `dplm/utils/`: shared utilities for FASTA I/O, checkpoints, metrics, and runtime setup
- `dplm/train_workflow.py`: training entrypoint
- `dplm/eval_esm2_distances.py`: ESM2 distance evaluation entrypoint

If you want to read the actual implementation chain, start from `dplm/train_workflow.py` for training or `dplm/eval_esm2_distances.py` for ESM2 evaluation.

Its end-to-end flow is:

1. read and filter FASTA records
2. split them into train/validation/test sets
3. build a `Q-Diffusion` generator with a DPLM proposal model and one conditioned energy reranker
4. tokenize sequences into `targets`
5. call `generator.objective({"targets": ...})` inside the epoch loop
6. optimize `energy_objective.mean()`, mainly training the energy reranking path
7. save compact checkpoints containing `energy_encoder`, `feature_projector`, energy backend weights, `energy_head`, and `vocab_proj`
8. rebuild baseline and guided generators for test-time generation
9. compare baseline vs guided outputs and write reports

## Sample ESM2 Distance Result

The DPLM-guided workflow includes `dplm/eval_esm2_distances.py` for embedding-level
comparison between generated sequences and the reference proteome. One example
report from the current setup used:

- reference: `data/UP000005640_9606.fasta`
- esm2 model: `esm2_t33_650M_UR50D`
- pair mode: `order`
- pooling: `mean`

| label | pairs | mean cosine dist | median cosine dist | mean l2 dist | median l2 dist |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 200 | 0.232665 | 0.192503 | 5.199004 | 4.926634 |
| MLP | 200 | 0.227007 | 0.205144 | 4.945239 | 4.803553 |
| guided | 200 | 0.187927 | 0.159195 | 4.679255 | 4.432099 |

In this run, the guided generator improved over the baseline by:

- mean cosine distance: `-0.044738`
- mean L2 distance: `-0.519750`

Lower values indicate that the guided outputs are closer to the reference set
in ESM2 embedding space, so this result is consistent with the energy-guided
reranking path improving sequence quality in the evaluated sample.

Inside `Q-Diffusion.objective(...)`, the training path is:

1. start from clean `targets`
2. corrupt them into noisy `x_t`
3. run the proposal model to produce logits
4. sample negative candidates from those logits
5. score positive and negative candidates with the conditioned energy reranker
6. return `energy_objective` and related tensors to the outer training loop

## Shared Assets

- `data/UP000005640_9606.fasta`: bundled example FASTA
- `graph/*`: diagrams copied from the original case notes

## Notes

- This directory is example-only; the reusable library code lives in
  `src/kaiwu/torch_plugin/qdiffusion.py`.
- Users should import the generic `Q-Diffusion` core from `kaiwu.torch_plugin`.
- The reusable core owns the training objective; the DPLM iterative generation
  policy lives in `example/qdiffusion/dplm/model/generation.py`.
- DPLM loading is no longer part of the formal `src` API; the DPLM factory in
  this directory is the example-side compatibility layer.
- The guided path in these examples is now `DPLM proposal + BM energy reranker`.
