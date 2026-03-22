#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.yaml>"
  exit 1
fi

CONFIG_PATH="$1"
shift

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

"$PYTHON_BIN" "$ROOT_DIR/evaluation/evaluate.py" --config "$CONFIG_PATH" "$@"
