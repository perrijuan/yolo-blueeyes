#!/usr/bin/env python3
"""Treina YOLOv8s em cada fold (init COCO, sem vazamento do peso Sea)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import COCO_INIT, OUT  # noqa: E402

FOLDS = OUT / "yolosea" / "folds"
RUNS = OUT / "yolosea" / "kfold"
RESULTS = OUT / "yolosea" / "results"
WEIGHTS_COCO = COCO_INIT
CFG = yaml.safe_load((_REPO / "finetune_yolosea" / "configs" / "train_seadronessee.yaml").read_text())


def train_one(fold: int, epochs: int | None = None):
    from ultralytics import YOLO

    data = FOLDS / f"fold_{fold}" / "data.yaml"
    if not data.is_file():
        raise FileNotFoundError(data)
    epochs = int(epochs or CFG["epochs"])
    w = str(WEIGHTS_COCO if WEIGHTS_COCO.is_file() else "yolov8s.pt")
    print(f"\n===== FOLD {fold}  init={w}  epochs={epochs} =====")
    model = YOLO(w)
    RUNS.mkdir(parents=True, exist_ok=True)
    res = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=int(CFG["imgsz"]),
        batch=int(CFG["batch"]),
        device=0,
        workers=int(CFG["workers"]),
        optimizer=CFG["optimizer"],
        lr0=float(CFG["lr0"]),
        lrf=float(CFG["lrf"]),
        cos_lr=bool(CFG["cos_lr"]),
        warmup_epochs=float(CFG["warmup_epochs"]),
        patience=int(CFG["patience"]),
        seed=int(CFG["seed"]),
        amp=bool(CFG["amp"]),
        close_mosaic=int(CFG["close_mosaic"]),
        hsv_h=float(CFG["hsv_h"]),
        hsv_s=float(CFG["hsv_s"]),
        hsv_v=float(CFG["hsv_v"]),
        degrees=float(CFG["degrees"]),
        translate=float(CFG["translate"]),
        scale=float(CFG["scale"]),
        fliplr=float(CFG["fliplr"]),
        mosaic=float(CFG["mosaic"]),
        mixup=float(CFG["mixup"]),
        project=str(RUNS),
        name=f"fold_{fold}",
        exist_ok=True,
        pretrained=True,
        plots=True,
        val=True,
        verbose=True,
    )
    # copiar best
    src = RUNS / f"fold_{fold}" / "weights" / "best.pt"
    RESULTS.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        dst = RESULTS / f"fold_{fold}_best.pt"
        shutil.copy2(src, dst)
        print(f"[ok] {dst}")
    metrics = {}
    try:
        metrics = {
            "fold": fold,
            "map50": float(res.results_dict.get("metrics/mAP50(B)", res.results_dict.get("metrics/mAP50", 0))),
            "map50_95": float(res.results_dict.get("metrics/mAP50-95(B)", res.results_dict.get("metrics/mAP50-95", 0))),
            "precision": float(res.results_dict.get("metrics/precision(B)", res.results_dict.get("metrics/precision", 0))),
            "recall": float(res.results_dict.get("metrics/recall(B)", res.results_dict.get("metrics/recall", 0))),
            "epochs": epochs,
            "best": str(src),
        }
    except Exception as e:
        metrics = {"fold": fold, "error": str(e)}
    (RESULTS / f"fold_{fold}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[metrics]", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="1..5; 0 = todos")
    ap.add_argument("--epochs", type=int, default=0, help="0 = configs/train.yaml")
    args = ap.parse_args()
    folds = [args.fold] if args.fold else [1, 2, 3, 4, 5]
    epochs = args.epochs or None
    all_m = []
    for f in folds:
        all_m.append(train_one(f, epochs=epochs))
        (RESULTS / "kfold_live.json").write_text(json.dumps(all_m, indent=2))
    print("[done] folds", folds)


if __name__ == "__main__":
    main()
