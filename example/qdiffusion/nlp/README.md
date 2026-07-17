# QDiffusion for NLP

This directory contains both sides of the NLP experiment:

- **MDLM baseline:** MDLM proposes tokens and QDiffusion runs iterative
  denoising, with BM reranking disabled.
- **QDiffusion-NLP:** the same frozen MDLM proposes multiple candidates; a
  separate energy-side MDLM encodes them, its hidden states condition a
  trainable BM, and BM energy reranks the candidates.

Keeping both paths is intentional: every claim about QDiffusion must be measured
against the identical decoder with only the BM energy path switched off.

## Runtime

Use a separate Linux/CUDA environment. The released Hugging Face MDLM model
imports `flash-attn`; it is not expected to run in the Windows development
environment used for the offline unit tests.

```bash
python -m venv .venv-mdlm
source .venv-mdlm/bin/activate
pip install -e .
pip install -r example/qdiffusion/requirements-nlp-mdlm.txt
```

Run a prompt-preserving, proposal-only smoke test:

```bash
python -m example.qdiffusion.nlp.smoke_generate \
  --prompt "Diffusion language models" \
  --max-new-tokens 64 \
  --steps 64 \
  --num-samples 8 \
  --output outputs/mdlm_samples.jsonl
```

The current decoder uses a fixed token canvas. EOS is available to the model,
and the smoke script trims decoded output at the first EOS token. Generation
uses a fresh recorded seed when `--seed` is omitted; pass an explicit seed only
when aligned control groups must share the same candidate stream.

The NLP builder preserves the earlier no-resampling behavior by default. To
run a proposal-only repetition-control experiment, enable the existing
repetition-triggered resampler explicitly and tune its token-frequency trigger
and nucleus cutoff:

```bash
python -m example.qdiffusion.nlp.smoke_generate \
  --prompt "Diffusion language models" \
  --max-new-tokens 64 \
  --steps 64 \
  --num-samples 32 \
  --enable-resample \
  --resample-ratio 0.15 \
  --resample-top-p 0.90 \
  --output outputs/mdlm_resample_r015_p090.jsonl
```

Keep `--energy-checkpoint` unset for this decoder-only control. Compare it with
the no-resampling baseline under the same recorded seed, prompts, output length,
and decode steps, and report Gen PPL together with repetition and distinct
metrics.

## Train and use the QDiffusion energy model

Prepare a `.txt` file with one training document per line, or a JSONL file with
a `text` field. The proposal MDLM always remains frozen. The energy-side MDLM
is a separate copy and is frozen by default; optionally train its final
transformer blocks with `--energy-unfreeze-last-layers`.

### EDLM scalar baseline and head-only BM comparison

The `scalar` energy follows the published
[EDLM paper](https://arxiv.org/abs/2410.21357) and its
[official implementation](https://github.com/MinkaiXu/Energy-Diffusion-LLM):
embed `x_t` and candidate `x_0`, concatenate them token by token, project
`2d -> d`, run the energy MDLM blocks, project back to hidden width, mean-pool
the full sequence, and apply `Linear -> ReLU -> Linear` to obtain one scalar.
Binary NCE trains real `x_0` toward lower energy and proposal samples toward
higher energy. Candidate selection uses importance weights proportional to
`exp(-energy)`.

For an energy-structure-only comparison, use `edlm_pair` for both heads:

- `scalar`: shared EDLM encoder followed by the scalar MLP.
- `bm`: the same shared EDLM encoder followed by
  `Projector -> identity -> BM(V=64,H=32,SA)`. The continuous projector
  output is passed directly to Kaiwu SDK rather than rounded into a binary
  visible state.

Keep the proposal checkpoint, dataset split, seed, candidate count, trainable
MDLM blocks, optimizer, NCE objective, and generation settings identical. The
older `pooled_pair` BM mode remains available only for compatibility with
existing checkpoints; it is not the head-only EDLM comparison.

Resolve one fresh seed at experiment time and pass it to both commands:

```bash
python -m example.qdiffusion.nlp.train_energy \
  --input data/openwebtext_train.jsonl \
  --output outputs/edlm_scalar_owt200_u1.pt \
  --max-records 200 \
  --max-length 64 \
  --num-candidates 1 \
  --energy-unfreeze-last-layers 1 \
  --energy-type scalar \
  --objective binary \
  --seed <shared-seed>

python -m example.qdiffusion.nlp.train_energy \
  --input data/openwebtext_train.jsonl \
  --output outputs/edlm_bm_owt200_v64h32_u1.pt \
  --max-records 200 \
  --max-length 64 \
  --num-candidates 1 \
  --energy-unfreeze-last-layers 1 \
  --energy-type bm \
  --energy-feature-mode edlm_pair \
  --bm-num-visible 64 \
  --bm-num-hidden 32 \
  --bm-scoring-mode sampler \
  --bm-visible-transform identity \
  --objective binary \
  --seed <shared-seed>
```

This is an architecture- and objective-aligned EDLM-NCE baseline, not a claim
of reproducing the paper's full OpenWebText training budget. The small run
trains only the last MDLM block and uses 200 records so the scalar-vs-BM
comparison remains controlled and affordable.

The official EDLM configuration uses one proposal negative for binary NCE and
K=2 parallel importance sampling only in the middle diffusion-time window
`t in [0.4, 0.6]`. In this decoder, diffusion time decreases while decode
progress increases, so use the equivalent progress window:

```bash
python -m example.qdiffusion.nlp.smoke_generate \
  --energy-checkpoint outputs/edlm_scalar_owt200_u1.pt \
  --num-candidates 2 \
  --energy-start-ratio 0.4 \
  --energy-end-ratio 0.6 \
  --output outputs/edlm_scalar_k2.jsonl
```

Run the BM checkpoint with the same K, guidance window, prompt set, generation
seed, and decoding settings. Increasing K to 4 or 8 is a separate QDiffusion
ablation rather than part of the EDLM-faithful baseline.

```bash
python -m example.qdiffusion.nlp.train_energy \
  --input data/openwebtext_train.jsonl \
  --output outputs/mdlm_bm_energy.pt \
  --max-length 256 \
  --num-candidates 4 \
  --energy-unfreeze-last-layers 1
```

To train on proposal states encountered along the reverse process, enable a
proposal-only rollout and the target-token recovery ranking auxiliary. When
`--seed` is omitted, the trainer generates a fresh seed and records it in both
stdout and checkpoint metadata.

```bash
python -m example.qdiffusion.nlp.train_energy \
  --input data/openwebtext_train.jsonl \
  --output outputs/mdlm_bm_on_policy.pt \
  --num-candidates 4 \
  --energy-unfreeze-last-layers 1 \
  --on-policy-rollout-steps 4 \
  --on-policy-max-steps 64 \
  --recovery-ranking-weight 0.25 \
  --recovery-ranking-temperature 0.05
```

Then run actual BM-guided QDiffusion. Supplying `--energy-checkpoint` switches
on energy scoring; omitting it runs the proposal-only control group.

```bash
python -m example.qdiffusion.nlp.smoke_generate \
  --energy-checkpoint outputs/mdlm_bm_energy.pt \
  --num-candidates 8 \
  --num-samples 8 \
  --output outputs/qdiffusion_samples.jsonl
```

Before comparing full generations, measure whether the energy model can
actually order candidates from held-out diffusion states. The diagnostic
reports BM top-1 and pairwise accuracy, causal-LM NLL regret against an oracle
candidate choice, and the positive-versus-negative energy margin.

```bash
python -m example.qdiffusion.nlp.eval_energy_ranking \
  --input data/openwebtext_train.jsonl \
  --offset 200 \
  --max-records 64 \
  --energy-checkpoint outputs/mdlm_bm_energy.pt \
  --num-candidates 4 \
  --output outputs/mdlm_bm_ranking.json
```

## Perplexity metrics

Keep the following metrics separate in reports:

- **Intrinsic test PPL** is the MDLM likelihood estimator on a held-out corpus.
  Run the official MDLM repository with `mode=ppl_eval`; do not compute this by
  feeding clean text directly through `MDLMBackbone.forward()`. Because the
  current QDiffusion path freezes MDLM and uses BM only for inference-time
  reranking, this value is a backbone sanity check and is identical for the
  baseline and guided branches.
- **Gen PPL** scores generated samples with an external autoregressive model.
  It can change after BM reranking, but it is a fluency proxy rather than the
  normalized likelihood of the combined QDiffusion model.

The official OpenWebText intrinsic-PPL command shape is:

```bash
python main.py \
  mode=ppl_eval \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  model=small \
  parameterization=subs \
  backbone=dit \
  model.length=1024 \
  eval.checkpoint_path=/path/to/mdlm.ckpt \
  +wandb.offline=true
```

For generated samples, put one sample on each line of a `.txt` file, or use a
JSONL file with a `text` field:

```bash
python -m example.qdiffusion.nlp.eval_gen_ppl \
  --input outputs/mdlm_samples.jsonl \
  --evaluator gpt2-large
```

The first experiment table should compare the released MDLM baseline and the
trained MDLM+BM variant with identical prompts, output lengths, decode steps,
seeds, and evaluator versions. Record MDLM intrinsic test PPL once as a frozen
backbone check, then compare Gen PPL, diversity, wall-clock latency, and peak GPU
memory between the two generation branches.
