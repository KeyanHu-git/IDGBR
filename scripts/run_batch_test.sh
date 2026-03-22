#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Default
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/CHN6-CUG/segformer.yaml" "0" &
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/Potsdam/segformer.yaml" "1" &

# Cubic / No Cubic
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/CHN6-CUG/segformer_cubic.yaml" "0" &
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/Potsdam/segformer_no_cubic.yaml" "1" &

# No Align
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/CHN6-CUG/segformer_no_align.yaml" "0" &
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/Potsdam/segformer_no_align.yaml" "1" &

# No Rough
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/CHN6-CUG/no_rough.yaml" "0" &
# bash "$ROOT_DIR/scripts/inference.sh" "configs/inference/Potsdam/no_rough.yaml" "1" &
