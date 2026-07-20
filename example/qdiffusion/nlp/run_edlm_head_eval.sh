#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data2/wwx/kaiwu-pytorch-plugin}"
PY="${PY:-/data/conda/envs/wwx_py310/bin/python}"
MODEL="${MODEL:-/data2/wwx/mdlm}"
EVALUATOR="${EVALUATOR:-/data2/wwx/models/gpt2-large}"
SCALAR_CHECKPOINT="${SCALAR_CHECKPOINT:?Set SCALAR_CHECKPOINT}"
BM_CHECKPOINT="${BM_CHECKPOINT:?Set BM_CHECKPOINT}"
PROFILE="${PROFILE:-smoke}"
SEED="${SEED:-$("${PY}" -c 'import secrets; print(secrets.randbelow(2**31))')}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/example/qdiffusion/outputs/edlm_head_eval_$(date +%Y%m%d_%H%M%S)}"

case "${PROFILE}" in
  smoke)
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-64}"
    STEPS="${STEPS:-64}"
    NUM_SAMPLES="${NUM_SAMPLES:-4}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  pilot)
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-256}"
    STEPS="${STEPS:-256}"
    NUM_SAMPLES="${NUM_SAMPLES:-32}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  paper)
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-1024}"
    STEPS="${STEPS:-1000}"
    NUM_SAMPLES="${NUM_SAMPLES:-128}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
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

printf '%s\n' \
  "git_commit=$(git rev-parse HEAD)" \
  "profile=${PROFILE}" \
  "seed=${SEED}" \
  "model=${MODEL}" \
  "evaluator=${EVALUATOR}" \
  "scalar_checkpoint=${SCALAR_CHECKPOINT}" \
  "scalar_sha256=$(sha256sum "${SCALAR_CHECKPOINT}" | awk '{print $1}')" \
  "bm_checkpoint=${BM_CHECKPOINT}" \
  "bm_sha256=$(sha256sum "${BM_CHECKPOINT}" | awk '{print $1}')" \
  "sequence_length=${SEQUENCE_LENGTH}" \
  "steps=${STEPS}" \
  "num_samples=${NUM_SAMPLES}" \
  "batch_size=${BATCH_SIZE}" \
  "num_candidates=2" \
  "importance_start_t=1.0" \
  "importance_end_t=0.8" \
  > "${OUTPUT_DIR}/config.txt"

log_stage() {
  printf '%s stage=%s\n' "$(date --iso-8601=seconds)" "$1" \
    | tee -a "${OUTPUT_DIR}/driver.log"
}

generate_head() {
  local energy_type="$1"
  local checkpoint="$2"
  local output="${OUTPUT_DIR}/${energy_type}.jsonl"
  local log="${OUTPUT_DIR}/${energy_type}.log"
  local gpu_log="${OUTPUT_DIR}/${energy_type}.gpu.csv"
  local extra_args=()
  if [ "${energy_type}" = "bm" ]; then
    extra_args+=(--allow-bm-ablation)
  fi

  log_stage "${energy_type}_generation_start"
  "${PY}" -m example.qdiffusion.nlp.sample_edlm \
    --checkpoint "${MODEL}" \
    --tokenizer gpt2 \
    --energy-checkpoint "${checkpoint}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --steps "${STEPS}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-candidates 2 \
    --importance-start-t 1.0 \
    --importance-end-t 0.8 \
    --energy-temperature 1.0 \
    --seed "${SEED}" \
    --output "${output}" \
    "${extra_args[@]}" \
    > "${log}" 2>&1 &
  local generation_pid=$!

  printf 'timestamp,index,memory.used,memory.free,utilization.gpu\n' \
    > "${gpu_log}"
  while kill -0 "${generation_pid}" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >> "${gpu_log}"
    sleep 5
  done &
  local monitor_pid=$!

  set +e
  wait "${generation_pid}"
  local generation_status=$?
  set -e
  wait "${monitor_pid}" || true
  if [ "${generation_status}" -ne 0 ]; then
    log_stage "${energy_type}_generation_failed"
    return "${generation_status}"
  fi
  log_stage "${energy_type}_generation_done"

  "${PY}" -m example.qdiffusion.nlp.eval_text_quality \
    --input "${output}" > "${OUTPUT_DIR}/${energy_type}.quality.json"
  "${PY}" -m example.qdiffusion.nlp.eval_gen_ppl \
    --input "${output}" \
    --token-ids-field token_ids \
    --evaluator "${EVALUATOR}" \
    --batch-size 1 > "${OUTPUT_DIR}/${energy_type}.ppl.json"
  log_stage "${energy_type}_evaluation_done"
}

generate_head scalar "${SCALAR_CHECKPOINT}"
generate_head bm "${BM_CHECKPOINT}"
log_stage all_done
