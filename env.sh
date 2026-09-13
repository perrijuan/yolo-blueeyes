#!/usr/bin/env bash
# Source from any script in this repo. Paths come from the environment.
# Copy env.example to .env and export FUSAO_ROOT before training or inference.
if [ -z "${YOLO_GIT_ROOT:-}" ]; then
  YOLO_GIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export YOLO_GIT_ROOT
fi

if [ -f "$YOLO_GIT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$YOLO_GIT_ROOT/.env"
  set +a
fi

if [ -n "${FUSAO_ROOT:-}" ]; then
  :
elif [ -d "$YOLO_GIT_ROOT/../datasets" ] || [ -d "$YOLO_GIT_ROOT/../weights" ]; then
  FUSAO_ROOT="$(cd "$YOLO_GIT_ROOT/.." && pwd)"
elif [ -d "$YOLO_GIT_ROOT/../../datasets" ] || [ -d "$YOLO_GIT_ROOT/../../weights" ]; then
  FUSAO_ROOT="$(cd "$YOLO_GIT_ROOT/../.." && pwd)"
else
  FUSAO_ROOT="$(cd "$YOLO_GIT_ROOT/.." && pwd)"
fi
export FUSAO_ROOT

export DATASETS_ROOT="${DATASETS_ROOT:-$FUSAO_ROOT/datasets}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-$FUSAO_ROOT/weights}"
export YOLOOW_ROOT="${YOLOOW_ROOT:-$YOLO_GIT_ROOT/finetune_yoloow/YoloOW}"

if [ -n "${MVTD_ROOT:-}" ]; then
  :
elif [ -d "$FUSAO_ROOT/mvtd" ]; then
  MVTD_ROOT="$FUSAO_ROOT/mvtd"
elif [ -d "$YOLO_GIT_ROOT/../mvtd" ]; then
  MVTD_ROOT="$(cd "$YOLO_GIT_ROOT/../mvtd" && pwd)"
else
  MVTD_ROOT="$FUSAO_ROOT/mvtd"
fi
export MVTD_ROOT
export MVTD_YOLO="${MVTD_YOLO:-$MVTD_ROOT/yolo}"
export MVTD_PESOS="${MVTD_PESOS:-$MVTD_ROOT/pesos}"
export YOLO_GIT_OUT="${YOLO_GIT_OUT:-$YOLO_GIT_ROOT/runs}"

if [ -x "$FUSAO_ROOT/.venv/bin/python" ]; then
  export PYTHON="${PYTHON:-$FUSAO_ROOT/.venv/bin/python}"
elif [ -x "$YOLO_GIT_ROOT/.venv/bin/python" ]; then
  export PYTHON="${PYTHON:-$YOLO_GIT_ROOT/.venv/bin/python}"
else
  export PYTHON="${PYTHON:-python3}"
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
