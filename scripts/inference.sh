#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:-configs/inference/CHN6-CUG/segformer.yaml}
GPU_IDS=${2:-}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

CMD=("$PYTHON_BIN" tools/infer.py --config_file "$CONFIG")
if (($# > 2)); then
  CMD+=("${@:3}")
fi

if [[ -n "$GPU_IDS" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}"
else
  "${CMD[@]}"
fi
