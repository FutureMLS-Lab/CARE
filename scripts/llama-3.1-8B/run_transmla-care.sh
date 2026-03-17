#!/usr/bin/env bash
set -euo pipefail

# Set HF_TOKEN in your environment for gated models (e.g., export HF_TOKEN=hf_xxx)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_DIR}/scripts/lib/common.sh"
care_setup_runtime "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON:-python3}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
RANK="${RANK:-256}"
COLLAPSE="${COLLAPSE:-2}"
FREQFOLD="${FREQFOLD:-4}"
CAL_DATASET="${CAL_DATASET:-wikitext2}"
PERCDAMP="${PERCDAMP:-0.01}"
RUN_LM_EVAL="${RUN_LM_EVAL:-0}"
BENCHMARKS="${BENCHMARKS:-arc_easy arc_challenge hellaswag piqa MMLU openbookqa race winogrande}"

LM_EVAL_ARGS=()
if [ "${RUN_LM_EVAL}" = "1" ]; then
  LM_EVAL_ARGS+=(--run-lm-eval --benchmarks ${BENCHMARKS})
fi

MODEL_SHORT="llama-3.1-8B"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/outputs}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/outputs/logs/${MODEL_SHORT}}"
SAVE_ROOT="${SAVE_ROOT:-${OUTPUT_BASE}/models/${MODEL_SHORT}}"
mkdir -p "${LOG_ROOT}" "${SAVE_ROOT}"

LOG_FILE="${LOG_ROOT}/${MODEL_SHORT}-transmla-care-${CAL_DATASET}-rank${RANK}.log"
SAVE_PATH="${SAVE_ROOT}/${MODEL_SHORT}-mla-transmla-care-${CAL_DATASET}-rank${RANK}"

EXTRA_ARGS=()
if [[ "${DYNAMIC_RANK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dynamic-rank)
  if [[ -n "${MIN_RANK:-}" ]]; then
    EXTRA_ARGS+=(--min-rank "${MIN_RANK}")
  fi
  if [[ -n "${MAX_RANK:-}" ]]; then
    EXTRA_ARGS+=(--max-rank "${MAX_RANK}")
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m cli.convert \
  --model-path "${MODEL_PATH}" \
  --save-path "${SAVE_PATH}" \
  --dtype fp16 \
  --cal-dataset "${CAL_DATASET}" \
  --cal-batch-size "${CAL_BATCH_SIZE:-8}" \
  --kv-lora-rank "${RANK}" \
  --qk-mqa-dim 64 \
  --collapse "${COLLAPSE}" \
  --freqfold "${FREQFOLD}" \
  --kv-decomp-method transmla-care \
  --cwsvd-percdamp "${PERCDAMP}" \
  --cal-max-seqlen 256 \
  --ppl-eval-batch-size "${PPL_BS:-1}" \
  "${EXTRA_ARGS[@]}" \
  "${LM_EVAL_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "[DONE] TransMLA-CARE log: ${LOG_FILE}"
