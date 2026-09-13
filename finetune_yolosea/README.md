# Fine-tune YOLO-SEA (YOLOv8s + Soft-NMS)

Init **sempre COCO** (`yolov8s.pt`). Não usar `A_yolov8s_seadronessee.pt` como init no K-Fold: esse peso já viu o dataset e vaza o fold de validação.

## SeaDronesSee (5 classes)

GroupKFold por `video_id`. Config: `configs/train_seadronessee.yaml` (RTX 3060 6 GB: imgsz 640, batch 8).

```bash
python finetune_yolosea/prepare_kfold.py
python finetune_yolosea/train_kfold.py --fold 1 --epochs 12
python finetune_yolosea/benchmarks.py
```

Classes: swimmer, boat, jetski, buoy, life_saving_appliances.

## MVTD (4 classes)

```bash
python finetune_yolosea/train_mvtd.py --smoke
python finetune_yolosea/train_mvtd.py
python finetune_yolosea/eval_mvtd.py
```

Config: `configs/train_mvtd.yaml`. Stride por classe (sailboat=1, ship=2, boat/usv=5) porque o SOT tem frames quase iguais.
