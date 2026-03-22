#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${1:?Usage: bash scripts/train_label_embed.sh <config_path> <gpu_ids> [master_port]}
GPUS=${2:?Usage: bash scripts/train_label_embed.sh <config_path> <gpu_ids> [master_port]}
PORT=${3:-29500}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

NP=$(($(echo "$GPUS" | tr -cd , | wc -c)+1))

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes "$NP" \
    --main_process_port "$PORT" \
    tools/train_label_embed.py \
    --config_file "$CONFIG"
