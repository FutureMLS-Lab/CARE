#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_DIR}/scripts/lib/common.sh"
care_setup_runtime "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON:-python3}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
RANK="${RANK:-256}"
COLLAPSE="${COLLAPSE:-2}"
FREQFOLD="${FREQFOLD:-4}"
CAL_DATASET="${CAL_DATASET:-wikitext2}"
PERCDAMP="${PERCDAMP:-0.01}"

MODEL_SHORT="qwen2.5-1.5B"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/outputs}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/outputs/logs/${MODEL_SHORT}}"
SAVE_ROOT="${SAVE_ROOT:-${OUTPUT_BASE}/models/${MODEL_SHORT}}"
mkdir -p "${LOG_ROOT}" "${SAVE_ROOT}"

LOG_FILE="${LOG_ROOT}/${MODEL_SHORT}-transmla-care-${CAL_DATASET}-rank${RANK}.log"
SAVE_PATH="${SAVE_ROOT}/${MODEL_SHORT}-mla-transmla-care-${CAL_DATASET}-rank${RANK}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m cli.convert \
  --model-path "${MODEL_PATH}" \
  --save-path "${SAVE_PATH}" \
  --dtype fp16 \
  --kv-lora-rank "${RANK}" \
  --qk-mqa-dim 64 \
  --collapse "${COLLAPSE}" \
  --freqfold "${FREQFOLD}" \
  --kv-decomp-method transmla-care \
  --cwsvd-percdamp "${PERCDAMP}" \
  --cal-dataset "${CAL_DATASET}" \
  --cal-max-seqlen 256 \
  --cal-batch-size 2 \
  --ppl-eval-batch-size "${PPL_BS:-8}" \
  2>&1 | tee "${LOG_FILE}"

echo "[DONE] TransMLA-CARE log: ${LOG_FILE}"
