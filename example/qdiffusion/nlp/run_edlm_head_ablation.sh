#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data2/wwx/kaiwu-pytorch-plugin}"
PY="${PY:-/data/conda/envs/wwx_py310/bin/python}"
MODEL="${MODEL:-/data2/wwx/mdlm}"
INPUT="${INPUT:-/data2/wwx/kaiwu-pytorch-plugin/data/qdiffusion_nlp/openwebtext_train_2000.jsonl}"
PROFILE="${PROFILE:-smoke}"
SEED="${SEED:-$("${PY}" -c 'import secrets; print(secrets.randbelow(2**31))')}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/example/qdiffusion/outputs/edlm_head_ablation_$(date +%Y%m%d_%H%M%S)}"

case "${PROFILE}" in
  smoke)
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
    MAX_RECORDS="${MAX_RECORDS:-32}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
    MAX_STEPS="${MAX_STEPS:-1}"
    SAVE_EVERY="${SAVE_EVERY:-1}"
    WARMUP_STEPS="${WARMUP_STEPS:-0}"
    ;;
  pilot)
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
    MAX_RECORDS="${MAX_RECORDS:-2000}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    MAX_STEPS="${MAX_STEPS:-100}"
    SAVE_EVERY="${SAVE_EVERY:-25}"
    WARMUP_STEPS="${WARMUP_STEPS:-10}"
    ;;
  paper)
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
    MAX_RECORDS="${MAX_RECORDS:-2000}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
    MAX_STEPS="${MAX_STEPS:-1000000}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    WARMUP_STEPS="${WARMUP_STEPS:-2500}"
    ;;
  *)
    echo "PROFILE must be smoke, pilot, or paper" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/src:${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

GIT_COMMIT="$(git rev-parse HEAD)"
INPUT_SHA256="$(sha256sum "${INPUT}" | awk '{print $1}')"
printf '%s\n' \
  "git_commit=${GIT_COMMIT}" \
  "profile=${PROFILE}" \
  "seed=${SEED}" \
  "model=${MODEL}" \
  "input=${INPUT}" \
  "input_sha256=${INPUT_SHA256}" \
  "sequence_length=${SEQUENCE_LENGTH}" \
  "max_records=${MAX_RECORDS}" \
  "micro_batch_size=${MICRO_BATCH_SIZE}" \
  "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  "max_steps=${MAX_STEPS}" \
  "learning_rate=3e-4" \
  "weight_decay=0" \
  "warmup_steps=${WARMUP_STEPS}" \
  "gradient_clip_norm=1.0" \
  "ema_decay=0.9999" \
  "num_proposal_negatives=1" \
  "bm_num_visible=64" \
  "bm_num_hidden=32" \
  "bm_sampler=sa" \
  "bm_visible_transform=identity" \
  > "${OUTPUT_DIR}/config.txt"

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

log_stage() {
  printf '%s stage=%s\n' "$(date --iso-8601=seconds)" "$1" \
    | tee -a "${OUTPUT_DIR}/driver.log"
}

train_head() {
  local energy_type="$1"
  local checkpoint="${OUTPUT_DIR}/${energy_type}.pt"
  local log="${OUTPUT_DIR}/${energy_type}.log"
  local gpu_log="${OUTPUT_DIR}/${energy_type}.gpu.csv"

  log_stage "${energy_type}_start"
  "${PY}" -m example.qdiffusion.nlp.train_edlm_nce \
    --input "${INPUT}" \
    --checkpoint "${MODEL}" \
    --output "${checkpoint}" \
    --max-records "${MAX_RECORDS}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --learning-rate 3e-4 \
    --weight-decay 0 \
    --warmup-steps "${WARMUP_STEPS}" \
    --gradient-clip-norm 1.0 \
    --ema-decay 0.9999 \
    --energy-type "${energy_type}" \
    --bm-num-visible 64 \
    --bm-num-hidden 32 \
    --bm-visible-transform identity \
    --bm-sa-alpha 0.95 \
    --bm-sa-size-limit 10 \
    --seed "${SEED}" \
    --log-every 1 \
    --save-every "${SAVE_EVERY}" \
    > "${log}" 2>&1 &
  local train_pid=$!

  printf 'timestamp,index,memory.used,memory.free,utilization.gpu\n' \
    > "${gpu_log}"
  while kill -0 "${train_pid}" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >> "${gpu_log}"
    sleep 5
  done &
  local monitor_pid=$!

  set +e
  wait "${train_pid}"
  local train_status=$?
  set -e
  wait "${monitor_pid}" || true
  if [ "${train_status}" -ne 0 ]; then
    log_stage "${energy_type}_failed"
    return "${train_status}"
  fi
  log_stage "${energy_type}_done"
}

train_head scalar
train_head bm
log_stage all_done
