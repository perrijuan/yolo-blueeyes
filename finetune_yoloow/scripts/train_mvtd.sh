#!/usr/bin/env bash
# Fine-tune YoloOW (YOLOv7-style) para 4 classes MVTD.
# RTX 3060 6 GB: batch 2, imgsz 640.
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"

mkdir -p "$YOLO_GIT_OUT/yoloow" "$YOLO_GIT_ROOT/pesos"

"$PYTHON" - << PY
import sys
from pathlib import Path
sys.path.insert(0, "${YOLO_GIT_ROOT}")
sys.path.insert(0, "${YOLO_GIT_ROOT}/finetune_yolosea")
from train_mvtd import load_cfg, make_data_yaml
make_data_yaml(load_cfg(), 5)
print("listas train_balanced.txt / val_stride.txt ok")
PY

DATA_YAML="$YOLO_GIT_OUT/yoloow/yoloow_mvtd.yaml"
"$PYTHON" - << PY
from pathlib import Path
train = Path("${MVTD_YOLO}") / "train_balanced.txt"
val = Path("${MVTD_YOLO}") / "val_stride.txt"
out = Path("${DATA_YAML}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    f"train: {train}\n"
    f"val: {val}\n"
    "nc: 4\n"
    "names: ['boat', 'ship', 'sailboat', 'usv']\n"
)
print("data yaml", out)
PY

cd "$YOLOOW_ROOT"
"$PYTHON" train.py \
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
  --project "$YOLO_GIT_OUT/yoloow" \
  --name yoloow_mvtd \
  --exist-ok

echo "copie o best:"
echo "cp $YOLO_GIT_OUT/yoloow/yoloow_mvtd/weights/best.pt $YOLO_GIT_ROOT/pesos/B_YoloOW_mvtd.pt"
