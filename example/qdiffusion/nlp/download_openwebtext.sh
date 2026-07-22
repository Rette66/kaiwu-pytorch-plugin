#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="${1:?usage: download_openwebtext.sh OUTPUT_DIR [LOG_DIR]}"
LOG_DIR="${2:-${OUTPUT_DIR}}"
REVISION="b4325f019c648b1641a1784748667e8b74e5e064"
BASE_URL="${OWT_BASE_URL:-https://hf-mirror.com/datasets/Skylion007/openwebtext/resolve/${REVISION}/plain_text}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
HEARTBEAT="${LOG_DIR}/download_heartbeat.json"

write_heartbeat() {
  local status="$1"
  local shard="$2"
  local completed
  completed="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'train-*-of-00080.parquet' | wc -l)"
  printf '{"status":"%s","shard":"%s","completed_shards":%s,"expected_shards":80,"updated_at":"%s"}\n' \
    "${status}" "${shard}" "${completed}" "$(date --iso-8601=seconds)" \
    > "${HEARTBEAT}.tmp"
  mv "${HEARTBEAT}.tmp" "${HEARTBEAT}"
}

for shard_index in $(seq 0 79); do
  shard="$(printf 'train-%05d-of-00080.parquet' "${shard_index}")"
  destination="${OUTPUT_DIR}/${shard}"
  partial="${destination}.partial"
  if [[ -f "${destination}" ]]; then
    write_heartbeat "already_present" "${shard}"
    continue
  fi
  write_heartbeat "downloading" "${shard}"
  curl -L --fail --retry 20 --retry-all-errors --retry-delay 10 \
    --connect-timeout 30 -C - -o "${partial}" "${BASE_URL}/${shard}"
  python_bin="${PYTHON:-python}"
  "${python_bin}" - "${partial}" <<'PY'
import pathlib
import sys

import pyarrow.parquet as pq

path = pathlib.Path(sys.argv[1])
parquet = pq.ParquetFile(path)
if "text" not in parquet.schema.names or parquet.metadata.num_rows <= 0:
    raise SystemExit(f"invalid OpenWebText parquet shard: {path}")
PY
  mv "${partial}" "${destination}"
  write_heartbeat "verified" "${shard}"
done

write_heartbeat "all_done" ""

