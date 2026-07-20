#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data2/wwx/kaiwu-pytorch-plugin}"
PY="${PY:-/data/conda/envs/wwx_py310/bin/python}"
MODEL="${MODEL:-/data2/wwx/mdlm}"
SCALAR_CHECKPOINT="${SCALAR_CHECKPOINT:-${ROOT}/example/qdiffusion/outputs/edlm_head_ablation_20260717_192902/edlm_scalar_owt2000_u12.pt}"
SEED="${SEED:-$("${PY}" -c 'import secrets; print(secrets.randbelow(2**31))')}"
STEPS="${STEPS:-512}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/example/qdiffusion/outputs/edlm_paper_aligned_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src:${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

printf '%s\n' \
  "seed=${SEED}" \
  "steps=${STEPS}" \
  "sequence_length=${SEQUENCE_LENGTH}" \
  "num_samples=${NUM_SAMPLES}" \
  "batch_size=${BATCH_SIZE}" \
  "model=${MODEL}" \
  "scalar_checkpoint=${SCALAR_CHECKPOINT}" \
  "gen_ppl_evaluator=gpt2" \
  > "${OUTPUT_DIR}/config.txt"

log_stage() {
  printf '%s stage=%s\n' "$(date --iso-8601=seconds)" "$1" \
    | tee -a "${OUTPUT_DIR}/driver.log"
}

generate() {
  local name="$1"
  shift
  log_stage "${name}_start"
  "${PY}" -m example.qdiffusion.nlp.smoke_generate \
    --checkpoint "${MODEL}" \
    --tokenizer gpt2 \
    --sampling-method edlm-ddpm-cache \
    --unconditional \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --steps "${STEPS}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}/${name}.jsonl" \
    "$@" \
    > "${OUTPUT_DIR}/${name}.log" 2>&1
  log_stage "${name}_done"
}

evaluate() {
  local name="$1"
  log_stage "${name}_eval_start"
  "${PY}" -m example.qdiffusion.nlp.eval_text_quality \
    --input "${OUTPUT_DIR}/${name}.jsonl" \
    > "${OUTPUT_DIR}/${name}.quality.json"
  "${PY}" -m example.qdiffusion.nlp.eval_gen_ppl \
    --input "${OUTPUT_DIR}/${name}.jsonl" \
    --token-ids-field token_ids \
    --evaluator gpt2 \
    --batch-size 4 \
    > "${OUTPUT_DIR}/${name}.ppl.json"
  log_stage "${name}_eval_done"
}

generate baseline_ddpm_cache --num-candidates 1
generate scalar_k2_w1 \
  --energy-checkpoint "${SCALAR_CHECKPOINT}" \
  --num-candidates 2 \
  --edlm-importance-start-t 1.0 \
  --edlm-importance-end-t 0.0
generate scalar_k2_w02 \
  --energy-checkpoint "${SCALAR_CHECKPOINT}" \
  --num-candidates 2 \
  --edlm-importance-start-t 1.0 \
  --edlm-importance-end-t 0.8

for name in baseline_ddpm_cache scalar_k2_w1 scalar_k2_w02; do
  evaluate "${name}"
done

log_stage all_done
