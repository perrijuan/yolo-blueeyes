#!/usr/bin/env bash
# YoloOW 2GB → peso → val/test late fusion no MVTD.
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"
export FROM_WATCH_TREINO=1

FT="$YOLO_GIT_OUT/yoloow"
mkdir -p "$FT/pesos" "$FT/metricas" "$FT/runs" "$YOLO_GIT_ROOT/pesos"

if pgrep -f "name yoloow_2gb" >/dev/null 2>&1; then
  echo "[recusado] já há um train.py yoloow_2gb. Não lanço o segundo."
  echo "  para só publicar quando esse acabar:"
  echo "  bash $YOLO_GIT_ROOT/finetune_yoloow/scripts/watch_treino.sh"
  exit 1
fi

echo "======== 1/3 subset 2GB ========"
"$PYTHON" -u "$YOLO_GIT_ROOT/finetune_yoloow/scripts/prepare_2gb.py" | tee "$FT/metricas/prepare.log"
test -f "$FT/data/yoloow_2gb.yaml"

echo "======== 2/3 treino YoloOW ========"
cd "$YOLOOW_ROOT"
"$PYTHON" -u train.py \
  --weights "$MVTD_PESOS/B_YoloOW.pt" \
  --cfg "$YOLO_GIT_ROOT/finetune_yoloow/configs/yoloOW-nc4.yaml" \
  --data "$FT/data/yoloow_2gb.yaml" \
  --hyp "$YOLO_GIT_ROOT/finetune_yoloow/configs/hyp.yoloow.mvtd.yaml" \
  --epochs 15 \
  --batch-size 2 \
  --img-size 640 640 \
  --device 0 \
  --workers 2 \
  --adam \
  --project "$FT" \
  --name yoloow_2gb \
  --exist-ok \
  2>&1 | tee "$FT/metricas/train.log"

echo "======== 3/3 copiar pesos + val/test late fusion ========"
bash "$YOLO_GIT_ROOT/finetune_yoloow/scripts/depois_treino.sh"

echo "[done] $FT"
ls -lh "$FT/pesos" "$FT/metricas" 2>/dev/null || true
