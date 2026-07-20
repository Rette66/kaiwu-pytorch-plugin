#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data2/wwx/kaiwu-pytorch-plugin}"
PY="${PY:-/data/conda/envs/wwx_py310/bin/python}"
MODEL="${MODEL:-/data2/wwx/mdlm}"
PROFILE="${PROFILE:-smoke}"
SEED="${SEED:-$("${PY}" -c 'import secrets; print(secrets.randbelow(2**31))')}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/example/qdiffusion/outputs/edlm_reproduction_$(date +%Y%m%d_%H%M%S)}"

case "${PROFILE}" in
  smoke)
    NUM_SAMPLES="${NUM_SAMPLES:-4}"
    ;;
  paper)
    NUM_SAMPLES="${NUM_SAMPLES:-128}"
    ;;
  *)
    echo "PROFILE must be smoke or paper" >&2
    exit 2
    ;;
esac

STEPS="${STEPS:-1000}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVALUATOR="${EVALUATOR:-gpt2-large}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src:${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

"${PY}" - <<'PY' > "${OUTPUT_DIR}/environment.json"
import json
import platform
import torch
import transformers

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
}))
PY

printf '%s\n' \
  "official_edlm_commit=97e3146964f76aaa784fe523c673516efc7af0e0" \
  "profile=${PROFILE}" \
  "seed=${SEED}" \
  "steps=${STEPS}" \
  "sequence_length=${SEQUENCE_LENGTH}" \
  "num_samples=${NUM_SAMPLES}" \
  "batch_size=${BATCH_SIZE}" \
  "model=${MODEL}" \
  "evaluator=${EVALUATOR}" \
  > "${OUTPUT_DIR}/config.txt"

printf '%s stage=baseline_start\n' "$(date --iso-8601=seconds)" \
  | tee -a "${OUTPUT_DIR}/driver.log"
"${PY}" -m example.qdiffusion.nlp.sample_edlm \
  --checkpoint "${MODEL}" \
  --tokenizer gpt2 \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --steps "${STEPS}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --output "${OUTPUT_DIR}/baseline.jsonl" \
  > "${OUTPUT_DIR}/baseline.log" 2>&1
printf '%s stage=baseline_done\n' "$(date --iso-8601=seconds)" \
  | tee -a "${OUTPUT_DIR}/driver.log"

"${PY}" -m example.qdiffusion.nlp.eval_text_quality \
  --input "${OUTPUT_DIR}/baseline.jsonl" \
  > "${OUTPUT_DIR}/baseline.quality.json"
"${PY}" -m example.qdiffusion.nlp.eval_gen_ppl \
  --input "${OUTPUT_DIR}/baseline.jsonl" \
  --token-ids-field token_ids \
  --evaluator "${EVALUATOR}" \
  --batch-size 1 \
  > "${OUTPUT_DIR}/baseline.ppl.json"

printf '%s stage=all_done\n' "$(date --iso-8601=seconds)" \
  | tee -a "${OUTPUT_DIR}/driver.log"
