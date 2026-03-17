#!/bin/bash
# Parallel zero-shot runner for Qwen2.5-1.5B-Instruct
# Usage: bash scripts/zeroshot/run_parallel_qwen2.5_1.5B.sh
# Override with env vars: PYTHON, MODEL_PATH, RANKS, GPU_LIST, GPU_FREE_MEM_THRESHOLD, GPU_MAX_TASKS_PER_GPU, GPU_REUSE_COOLDOWN_SECONDS, CAL_DATASETS, PPL_DATASET, METHODS, OUTPUT_ROOT

# Set HF_TOKEN in your environment for gated models (e.g., export HF_TOKEN=hf_xxx)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$PROJECT_DIR/scripts/lib/common.sh"
care_setup_runtime "$PROJECT_DIR"
cd "$PROJECT_DIR"

PYTHON_BIN=${PYTHON:-python3}
MODEL=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
RANKS="${RANKS:-64 96 128}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
GPU_FREE_MEM_THRESHOLD="${GPU_FREE_MEM_THRESHOLD:-20480}"
GPU_MAX_TASKS_PER_GPU="${GPU_MAX_TASKS_PER_GPU:-3}"
GPU_REUSE_COOLDOWN_SECONDS="${GPU_REUSE_COOLDOWN_SECONDS:-45}"
CAL_DATASETS="${CAL_DATASETS:-wiki c4 alpaca ptb}"
PPL_DATASET=$(normalize_dataset_name "${PPL_DATASET:-wiki}")
METHODS="${METHODS:-palu asvd mha2mla no-sqrt-care care svdllm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zero-shot}"

echo "============================================"
echo "Parallel Zero-Shot Runner"
echo "MODEL=$MODEL"
echo "RANKS=$RANKS"
echo "GPU_LIST=$GPU_LIST"
echo "GPU_FREE_MEM_THRESHOLD=$GPU_FREE_MEM_THRESHOLD"
echo "GPU_MAX_TASKS_PER_GPU=$GPU_MAX_TASKS_PER_GPU"
echo "GPU_REUSE_COOLDOWN_SECONDS=$GPU_REUSE_COOLDOWN_SECONDS"
echo "CAL_DATASETS=$CAL_DATASETS"
echo "PPL_DATASET=$PPL_DATASET"
echo "METHODS=$METHODS"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "PYTHON=$PYTHON_BIN"
echo "============================================"

"$PYTHON_BIN" -m zeroshot.parallel_run \
  --model-path "$MODEL" \
  --ranks $RANKS \
  --gpu-list $GPU_LIST \
  --gpu-free-mem-threshold "$GPU_FREE_MEM_THRESHOLD" \
  --gpu-max-tasks-per-gpu "$GPU_MAX_TASKS_PER_GPU" \
  --gpu-reuse-cooldown-seconds "$GPU_REUSE_COOLDOWN_SECONDS" \
  --cal-datasets $CAL_DATASETS \
  --ppl-dataset "$PPL_DATASET" \
  --methods $METHODS \
  --output-root "$OUTPUT_ROOT"
