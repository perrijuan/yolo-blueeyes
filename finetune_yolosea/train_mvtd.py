#!/usr/bin/env python3
"""Fine-tune YOLOv8s no MVTD (4 classes: boat, ship, sailboat, usv).

Init sempre COCO (`pesos/yolov8s.pt`). Não carregar pesos SeaDronesSee:
head de 5 classes ≠ 4 classes MVTD, e o skill do projeto proíbe init de outro domínio.
"""
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
from paths import COCO_INIT, MVTD_YOLO, OUT, PESOS_OUT  # noqa: E402

PROJ = _REPO
CFG_PATH = _REPO / "finetune_yolosea" / "configs" / "train_mvtd.yaml"
YOLO_ROOT = MVTD_YOLO
RUNS = OUT / "yolosea" / "mvtd"
PESOS = PESOS_OUT
METRICAS = OUT / "yolosea" / "metricas"


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text())


import re
from collections import defaultdict

_SEQ_RE = re.compile(r"^[0-9]+-(.+)$")
_ALIAS = {
    "boat": "boat",
    "ship": "ship",
    "sailboat": "sailboat",
    "sail_boat": "sailboat",
    "usv": "usv",
}


def seq_class(seq: str) -> str:
    m = _SEQ_RE.match(seq)
    if not m:
        return "boat"
    raw = re.sub(r"[^a-z0-9]+", "", m.group(1).lower())
    return _ALIAS.get(raw, "boat")


def stride_list(
    src_txt: Path,
    dst_txt: Path,
    default_stride: int,
    by_class: dict | None = None,
) -> int:
    lines = [ln.strip() for ln in src_txt.read_text().splitlines() if ln.strip()]
    by_seq: dict[str, list[str]] = defaultdict(list)
    for ln in lines:
        seq = Path(ln).stem.rsplit("_", 1)[0]
        by_seq[seq].append(ln)
    kept: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for seq in sorted(by_seq):
        cls = seq_class(seq)
        st = int((by_class or {}).get(cls, default_stride))
        st = max(1, st)
        frames = sorted(by_seq[seq])[::st]
        kept.extend(frames)
        counts[cls] += len(frames)
    dst_txt.write_text("\n".join(kept) + "\n")
    print(f"[stride] {dst_txt.name}: {len(kept)}  por classe={dict(counts)}")
    return len(kept)


def _names_block() -> str:
    names = yaml.safe_load((YOLO_ROOT / "data.yaml").read_text())["names"]
    return "nc: 4\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items())


def make_data_yaml(cfg: dict, default_stride: int) -> Path:
    """Train com stride por classe; val strided p/ treino rápido; test intacto."""
    by_cls = cfg.get("stride_by_class") or {}
    val_st = int(cfg.get("val_stride") or 1)
    stride_list(
        YOLO_ROOT / "train.txt",
        YOLO_ROOT / "train_balanced.txt",
        default_stride,
        by_cls,
    )
    val_txt = YOLO_ROOT / "val.txt"
    if val_st > 1:
        stride_list(YOLO_ROOT / "val.txt", YOLO_ROOT / "val_stride.txt", val_st, None)
        val_txt = YOLO_ROOT / "val_stride.txt"
    out = YOLO_ROOT / "data_train.yaml"
    out.write_text(
        f"path: {YOLO_ROOT}\n"
        f"train: {YOLO_ROOT / 'train_balanced.txt'}\n"
        f"val: {val_txt}\n"
        f"test: {YOLO_ROOT / 'test.txt'}\n"
        + _names_block()
    )
    return out


def smoke_yaml(n: int = 32) -> Path:
    tr = [ln for ln in (YOLO_ROOT / "train.txt").read_text().splitlines() if ln.strip()][:n]
    va = [ln for ln in (YOLO_ROOT / "val.txt").read_text().splitlines() if ln.strip()][: max(8, n // 4)]
    (YOLO_ROOT / "train_smoke.txt").write_text("\n".join(tr) + "\n")
    (YOLO_ROOT / "val_smoke.txt").write_text("\n".join(va) + "\n")
    names = yaml.safe_load((YOLO_ROOT / "data.yaml").read_text())["names"]
    p = YOLO_ROOT / "data_smoke.yaml"
    p.write_text(
        f"path: {YOLO_ROOT}\n"
        f"train: {YOLO_ROOT / 'train_smoke.txt'}\n"
        f"val: {YOLO_ROOT / 'val_smoke.txt'}\n"
        "nc: 4\n"
        "names:\n"
        + "".join(f"  {k}: {v}\n" for k, v in names.items())
    )
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--stride", type=int, default=-1, help="-1 = configs/train.yaml")
    ap.add_argument("--smoke", action="store_true", help="32 imgs, 1 epoch, verifica o pipeline")
    args = ap.parse_args()

    cfg = load_cfg()
    w = Path(cfg["model"])
    if not w.is_file():
        w = PESOS / Path(cfg["model"]).name
    if not w.is_file():
        w = COCO_INIT
    print(f"init={w}  (COCO — não SeaDronesSee)")

    from ultralytics import YOLO

    if args.smoke:
        data = smoke_yaml()
        epochs, imgsz, batch = 1, 640, 4
        name = "smoke"
    else:
        stride = cfg["train_stride"] if args.stride < 0 else args.stride
        data = make_data_yaml(cfg, int(stride))
        epochs = args.epochs or int(cfg["epochs"])
        imgsz = args.imgsz or int(cfg["imgsz"])
        batch = args.batch or int(cfg["batch"])
        name = "mvtd_yolov8s_v2"

    RUNS.mkdir(parents=True, exist_ok=True)
    METRICAS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(w))
    res = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=int(cfg["workers"]),
        optimizer=cfg["optimizer"],
        lr0=float(cfg["lr0"]),
        lrf=float(cfg["lrf"]),
        cos_lr=bool(cfg["cos_lr"]),
        warmup_epochs=float(cfg["warmup_epochs"]),
        patience=int(cfg["patience"]),
        seed=int(cfg["seed"]),
        amp=bool(cfg["amp"]),
        close_mosaic=int(cfg["close_mosaic"]),
        hsv_h=float(cfg["hsv_h"]),
        hsv_s=float(cfg["hsv_s"]),
        hsv_v=float(cfg["hsv_v"]),
        degrees=float(cfg["degrees"]),
        translate=float(cfg["translate"]),
        scale=float(cfg["scale"]),
        fliplr=float(cfg["fliplr"]),
        mosaic=float(cfg["mosaic"]),
        mixup=float(cfg["mixup"]),
        cls=float(cfg.get("cls", 0.5)),
        project=str(RUNS),
        name=name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        val=True,
        verbose=True,
    )
    best = RUNS / name / "weights" / "best.pt"
    if best.is_file():
        dst = PESOS / f"{name}_best.pt"
        shutil.copy2(best, dst)
        print(f"[ok] {dst}")
    metrics = {
        "init": str(w),
        "data": str(data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "best": str(best),
        "results": {k: float(v) if hasattr(v, "real") or isinstance(v, (int, float)) else str(v)
                    for k, v in dict(getattr(res, "results_dict", {}) or {}).items()},
    }
    (METRICAS / f"{name}_train.json").write_text(json.dumps(metrics, indent=2))
    print("[metrics]", json.dumps(metrics["results"], indent=2)[:800])


if __name__ == "__main__":
    main()
