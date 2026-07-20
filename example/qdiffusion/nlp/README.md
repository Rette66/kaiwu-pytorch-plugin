# EDLM reproduction on MDLM OpenWebText

The active NLP path now starts from a faithful reproduction of
[Energy-Based Diffusion Language Models for Text Generation](https://arxiv.org/abs/2410.21357)
and the authors' code at commit
`97e3146964f76aaa784fe523c673516efc7af0e0`.

The paper reproduction is deliberately separate from later QDiffusion/BM
experiments. Residual guidance, greedy fallback, proposal-score mixing,
on-policy rollout, repetition resampling, and BM heads are not part of the
baseline protocol.

## Stage 1: MDLM baseline

The baseline uses:

- `kuleshov-group/mdlm-owt`;
- unconditional OpenWebText generation at length 1024;
- SUBS parameterization and log-linear masking;
- `ddpm_cache` with 1024 reverse steps for the paper's Table 3;
- final noise removal;
- 128 generated samples for the resource-bounded reproduction profile;
- GPT-2 Large generative perplexity over the generated GPT-2 token IDs.

The paper computes Gen PPL from 2048 generated sequences. The released
single-process sampling script emits 128 sequences per GPU/process, so this
two-RTX-3090 workflow first uses 128 sequences as a variance-bearing
reproduction estimate and records that sample-count difference explicitly.

Run a four-sample pipeline check first:

```bash
PROFILE=smoke \
MODEL=/data2/wwx/mdlm \
bash example/qdiffusion/nlp/run_edlm_paper_aligned.sh
```

Then run the 128-sample baseline:

```bash
PROFILE=paper \
MODEL=/data2/wwx/mdlm \
bash example/qdiffusion/nlp/run_edlm_paper_aligned.sh
```

Every output directory contains the exact seed, environment, configuration,
raw token IDs, decoded text, quality metrics, and GPT-2 Large Gen PPL.

## Stage 2: EDLM-NCE scalar energy

Only after the MDLM baseline matches the reference pipeline do we train the
scalar EDLM energy:

`(x_t, x_0) embeddings -> concat -> 2d-to-d projection -> DiT blocks ->
hidden-width output -> full-sequence mean pool -> Linear/ReLU/Linear -> E`.

Training uses one proposal negative and binary NCE:

`softplus(E_positive) + softplus(-E_negative)`.

The paper defaults are encoded directly in the dedicated trainer: sequence
length 1024, global batch 512, AdamW with learning rate `3e-4` and no weight
decay, 2500-step linear warmup followed by a constant learning rate, gradient
clipping at `1.0`, EMA `0.9999`, continuous antithetic time sampling, and one
million optimizer steps. Start with an explicit short validation run before
scheduling the full training budget:

```bash
python -m example.qdiffusion.nlp.train_edlm_nce \
  --input data/qdiffusion_nlp/openwebtext_train_2000.jsonl \
  --checkpoint /data2/wwx/mdlm \
  --output example/qdiffusion/outputs/edlm_scalar_nce.pt \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --global-batch-size 512 \
  --max-steps 1000000
```

At generation time, candidates are selected with weights proportional to
`exp(-E)`. The paper's Table 3 generation-quality protocol applies importance
sampling throughout the reverse process (`t in [0, 1]`). The separate
efficiency protocol uses K=2 only in the early reverse-process window
`t in [0.8, 1.0]` (width 0.2). Results from these protocols must not be mixed.

The BM/Projector route will be reintroduced only after this scalar baseline is
reproduced and frozen as a controlled reference.

## Stage 3: energy-head-only ablation

`run_edlm_head_ablation.sh` launches scalar and BM training sequentially with
the same explicit seed, wrapped token blocks, data order, continuous
timesteps, corruption masks, proposal negatives, optimizer, EMA, and training
budget. The trainer resets its post-construction random stream so the
different head initializers cannot perturb the paired candidate stream.

The only active architecture switch is:

- `scalar`: `Linear(d,d) -> ReLU -> Linear(d,1)`;
- `bm`: `Linear(d,V=64) -> identity visible transform -> BM(V=64,H=32)`
  with SA-conditioned hidden states.

The SA state is treated as a sampled latent assignment; gradients still flow
through the continuous visible values, Projector, shared EDLM encoder, and BM
parameters. Exact hidden enumeration is not used.

Validate both paths with one optimizer step:

```bash
PROFILE=smoke \
ROOT=/path/to/isolated/checkout \
bash example/qdiffusion/nlp/run_edlm_head_ablation.sh
```

The script records separate logs, checkpoints, and five-second GPU-memory
traces for the scalar and BM runs.

After training, `run_edlm_head_eval.sh` evaluates the two checkpoints
sequentially with the same post-construction RNG reset, K=2 proposal pool,
importance window, sampling temperature, sequence length, steps, batch size,
and GPT-2 Large evaluator. Its default is the Table 3 full window `[0, 1]`;
set `IMPORTANCE_END_T=0.8` for the separate efficiency protocol:

```bash
PROFILE=smoke \
ROOT=/path/to/isolated/checkout \
SCALAR_CHECKPOINT=/path/to/scalar.pt \
BM_CHECKPOINT=/path/to/bm.pt \
bash example/qdiffusion/nlp/run_edlm_head_eval.sh
```
