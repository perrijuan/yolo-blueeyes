#!/usr/bin/env bash
# Fine-tune YoloOW num subset MVTD de 2 GB e avalia o peso no MVTD val/test.
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"

FT="$YOLO_GIT_OUT/yoloow"
mkdir -p "$FT/pesos" "$FT/metricas" "$FT/runs"

echo "======== 1/3 prepare 2GB ========"
"$PYTHON" -u "$YOLO_GIT_ROOT/finetune_yoloow/scripts/prepare_2gb.py" | tee "$FT/metricas/prepare.log"

DATA_YAML="$FT/data/yoloow_2gb.yaml"
test -f "$DATA_YAML"

echo "======== 2/3 train YoloOW nc=4 ========"
cd "$YOLOOW_ROOT"
"$PYTHON" -u train.py \
  --weights "$MVTD_PESOS/B_YoloOW.pt" \
  --cfg "$YOLO_GIT_ROOT/finetune_yoloow/configs/yoloOW-nc4.yaml" \
  --data "$DATA_YAML" \
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

BEST="$FT/yoloow_2gb/weights/best.pt"
LAST="$FT/yoloow_2gb/weights/last.pt"
if [ -f "$BEST" ]; then
  cp -f "$BEST" "$FT/pesos/B_YoloOW_mvtd_2gb.pt"
  echo "[ok] peso -> $FT/pesos/B_YoloOW_mvtd_2gb.pt"
elif [ -f "$LAST" ]; then
  cp -f "$LAST" "$FT/pesos/B_YoloOW_mvtd_2gb.pt"
  echo "[aviso] best ausente, copiei last.pt"
else
  echo "[falha] nenhum checkpoint em $FT/yoloow_2gb/weights"
  exit 1
fi
if [ -f "$FT/yoloow_2gb/results.txt" ]; then
  cp -f "$FT/yoloow_2gb/results.txt" "$FT/metricas/train_results.txt"
fi

echo "======== 3/3 eval no MVTD (val+test, stride 5) ========"
"$PYTHON" -u "$YOLO_GIT_ROOT/finetune_yoloow/scripts/eval_mvtd.py" \
  --weights "$FT/pesos/B_YoloOW_mvtd_2gb.pt" \
  --stride 5 \
  --split both \
  2>&1 | tee "$FT/metricas/eval.log"

echo "[done] $FT"
ls -lh "$FT/pesos" "$FT/metricas"
