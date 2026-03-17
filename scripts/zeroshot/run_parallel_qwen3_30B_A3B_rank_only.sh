#!/bin/bash
# Sequential zero-shot runner for Qwen3-30B-A3B
# Rank allocation only — no PPL eval, no lm-eval benchmarks.
# Runs one experiment at a time with all visible GPUs (30B needs multi-GPU).
# Usage: bash scripts/zeroshot/run_parallel_qwen3_30B_A3B_rank_only.sh

# Set HF_TOKEN in your environment for gated models (e.g., export HF_TOKEN=hf_xxx)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$PROJECT_DIR/scripts/lib/common.sh"
care_setup_runtime "$PROJECT_DIR"
cd "$PROJECT_DIR"

PYTHON_BIN=${PYTHON:-python3}
MODEL=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
RANKS="${RANKS:-128 256}"
CUDA_DEVS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CAL_DATASETS="${CAL_DATASETS:-alpaca}"
PPL_DATASET=$(normalize_dataset_name "${PPL_DATASET:-wiki}")
METHODS="${METHODS:-no-sqrt-care care}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zero-shot-rank-only}"
DYNAMIC_METHODS="care"

MODEL_SHORT=$(echo "$MODEL" | tr '/' '_')
LOG_DIR="${OUTPUT_ROOT}/logs/${MODEL_SHORT}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Sequential Zero-Shot Runner (rank only, multi-GPU)"
echo "MODEL=$MODEL"
echo "RANKS=$RANKS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVS"
echo "CAL_DATASETS=$CAL_DATASETS"
echo "METHODS=$METHODS"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "============================================"

task_count=0
for cal in $CAL_DATASETS; do
  for method in $METHODS; do
    for rank in $RANKS; do
      dtag="${MODEL_SHORT}-${cal}-${method}-dynamic-rank${rank}"
      dlog_file="${LOG_DIR}/${dtag}.log"
      echo "[$(date +%H:%M:%S)] Running: $method rank=$rank cal=$cal dynamic"
      CUDA_VISIBLE_DEVICES="$CUDA_DEVS" "$PYTHON_BIN" -m zeroshot.convert \
        --model-path "$MODEL" \
        --method "$method" \
        --rank "$rank" \
        --dynamic-rank \
        --cal-dataset "$cal" \
        --ppl-dataset "$PPL_DATASET" \
        --ppl-eval-batch-size 0 \
        --output-dir "$OUTPUT_ROOT" \
        2>&1 | tee "$dlog_file"
      task_count=$((task_count + 1))
    done
  done
done

echo "============================================"
echo "All done. $task_count tasks completed."
echo "============================================"
