#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/wwx/kaiwu-pytorch-plugin-nlp}"
PY="${PY:-/data/wwx/envs/wwx_py310_4090/bin/python}"
MODEL="${MODEL:-/data/wwx/models/mdlm}"
TOKENIZER="${TOKENIZER:-/data/wwx/models/mdlm}"
INPUT="${INPUT:?set INPUT to the verified full-OWT token-block .bin}"
OUTPUT_DIR="${OUTPUT_DIR:?set a new OUTPUT_DIR}"
MAX_STEPS="${MAX_STEPS:-50000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-10}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ "${INPUT}" == *100k* ]]; then
  echo "Refusing a 100k subset for the formal OpenWebText run." >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" && -z "${RESUME_CHECKPOINT}" ]]; then
  echo "Refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/all_done" ]]; then
  echo "Refusing to resume a completed run: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  SEED="${SEED:-$("${PY}" - "${RESUME_CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
print(checkpoint["metadata"]["seed"])
PY
)}"
else
  SEED="${SEED:-$("${PY}" -c 'import secrets; print(secrets.randbelow(2**31))')}"
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

"${PY}" - "${INPUT}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
metadata = json.loads(
    path.with_suffix(path.suffix + ".metadata.json").read_text()
)
assert metadata["complete_openwebtext"] is True
assert metadata["expected_shards"] == 80
assert len(metadata["source_shards"]) == 80
assert metadata["sequence_length"] == 1024
PY

provenance="${OUTPUT_DIR}/provenance.txt"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  provenance="${OUTPUT_DIR}/resume_$(date +%Y%m%d_%H%M%S).txt"
fi
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_status=%s\n' "$(git status --short | wc -l)"
  printf 'input=%s\n' "${INPUT}"
  printf 'seed=%s\n' "${SEED}"
  printf 'sequence_length=1024\n'
  printf 'micro_batch_size=%s\n' "${MICRO_BATCH_SIZE}"
  printf 'global_batch_size=%s\n' "${GLOBAL_BATCH_SIZE}"
  printf 'max_steps=%s\n' "${MAX_STEPS}"
  printf 'save_every=%s\n' "${SAVE_EVERY}"
  printf 'resume_checkpoint=%s\n' "${RESUME_CHECKPOINT}"
} > "${provenance}"

on_exit() {
  status=$?
  if [[ ${status} -eq 0 ]]; then
    touch "${OUTPUT_DIR}/all_done"
    printf '%s stage=all_done\n' "$(date --iso-8601=seconds)" \
      | tee -a "${OUTPUT_DIR}/driver.log"
  else
    printf '{"status":"error","exit_code":%s,"updated_at":"%s"}\n' \
      "${status}" "$(date --iso-8601=seconds)" \
      > "${OUTPUT_DIR}/driver_heartbeat.json"
    printf '%s stage=error exit_code=%s\n' \
      "$(date --iso-8601=seconds)" "${status}" \
      | tee -a "${OUTPUT_DIR}/driver.log"
  fi
}
trap on_exit EXIT

printf '%s stage=train_start seed=%s\n' \
  "$(date --iso-8601=seconds)" "${SEED}" \
  | tee -a "${OUTPUT_DIR}/driver.log"

resume_args=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  resume_args=(--resume-energy-checkpoint "${RESUME_CHECKPOINT}")
fi

"${PY}" -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m example.qdiffusion.nlp.train_edlm_nce \
  --input "${INPUT}" \
  --checkpoint "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --output "${OUTPUT_DIR}/scalar.pt" \
  --sequence-length 1024 \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --warmup-steps 2500 \
  --gradient-clip-norm 1.0 \
  --ema-decay 0.9999 \
  --sampling-eps 0.001 \
  --noise-eps 0.001 \
  --energy-type scalar \
  --seed "${SEED}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --heartbeat "${OUTPUT_DIR}/train_heartbeat.json" \
  --require-complete-openwebtext \
  "${resume_args[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/scalar.log"
