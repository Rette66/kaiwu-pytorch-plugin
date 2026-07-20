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
- `ddpm_cache` with 1000 reverse steps;
- final noise removal;
- 128 generated samples for the paper profile;
- GPT-2 Large generative perplexity over the generated GPT-2 token IDs.

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

At generation time, K=2 candidates are sampled only in the configured
diffusion-time window and selected with weights proportional to `exp(-E)`.
The paper's efficient setting is the early reverse-process window
`t in [0.8, 1.0]` (width 0.2).

The BM/Projector route will be reintroduced only after this scalar baseline is
reproduced and frozen as a controlled reference.
