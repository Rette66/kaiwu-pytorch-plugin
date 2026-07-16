# QDiffusion for NLP

This directory contains both sides of the NLP experiment:

- **MDLM baseline:** MDLM proposes tokens and QDiffusion runs iterative
  denoising, with BM reranking disabled.
- **QDiffusion-NLP:** the same frozen MDLM proposes multiple candidates; MDLM
  hidden states condition a trainable BM, and BM energy reranks the candidates.

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
and the smoke script trims decoded output at the first EOS token.

## Train and use the QDiffusion energy model

Prepare a `.txt` file with one training document per line, or a JSONL file with
a `text` field. Only the feature projector and BM are trained; MDLM remains
frozen.

```bash
python -m example.qdiffusion.nlp.train_energy \
  --input data/openwebtext_train.jsonl \
  --output outputs/mdlm_bm_energy.pt \
  --max-length 256 \
  --num-candidates 4
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
