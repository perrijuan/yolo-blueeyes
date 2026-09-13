#!/usr/bin/env python3
"""Avalia o YoloOW 4-cls (peso 2GB) no MVTD val/test.

Protocolo local IoU≥0.5 mesma classe, stride 5 por sequência
(igual ao run_latefusion, sem editar esse ficheiro).
Também corre A (YOLOv8s MVTD) + fusão DS-WBF com o B novo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import MVTD_YOLO, OUT, setup_sys_path  # noqa: E402

setup_sys_path()
FT = OUT / "yoloow"
YOLO_FULL = MVTD_YOLO

from latefusion import ds_wbf_fuse  # noqa: E402
from run_mvtd import (  # noqa: E402
    ACTIVE,
    iou_box,
    load_gt,
    load_models as load_ab,
    make_pred_b,
    pred_a,
)


def list_split(split: str, stride: int) -> list[Path]:
    img_dir = YOLO_FULL / "images" / split
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if stride <= 1:
        return paths
    by: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by[p.stem.rsplit("_", 1)[0]].append(p)
    out = []
    for seq in sorted(by):
        frames = sorted(by[seq], key=lambda x: x.stem)
        out.extend(frames[::stride])
    return out


def eval_paths(paths: list[Path], pred_fn, tag: str, split: str) -> dict:
    tp = fp = fn = 0
    lbl_dir = YOLO_FULL / "labels" / split
    for p in tqdm(paths, desc=tag):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        dets = pred_fn(frame)
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), np.float32)
        if len(dets):
            dets = dets[np.isin(dets[:, 5].astype(int), list(ACTIVE))]
        gts = load_gt(lbl_dir / f"{p.stem}.txt", w, h)
        matched = set()
        for d in dets:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched or int(d[5]) != int(g[4]):
                    continue
                iou = iou_box(d[:4], g[:4])
                if iou > best:
                    best, bj = iou, j
            if best >= 0.5 and bj >= 0:
                tp += 1
                matched.add(bj)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    P = tp / (tp + fp + 1e-6)
    R = tp / (tp + fn + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)
    sm = {
        "tag": tag,
        "split": split,
        "P": P,
        "R": R,
        "F1": F1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "n": len(paths),
        "protocol": "local IoU>=0.5 same class, stride listed in parent json",
    }
    print(f"  [{tag:16s} {split}] n={len(paths)} P={P:.3f} R={R:.3f} F1={F1:.3f} TP={tp} FP={fp} FN={fn}")
    return sm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=FT / "pesos" / "B_YoloOW_mvtd_2gb.pt")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    args = ap.parse_args()
    w = args.weights
    if not w.is_file():
        raise SystemExit(f"peso não encontrado: {w}")

    met = FT / "metricas"
    met.mkdir(parents=True, exist_ok=True)

    # load A + this B (4-class). Temporarily point PESOS via copy? load_models
    # looks at PESOS/B_YoloOW_mvtd.pt. We load B ourselves with identity map.
    model_a, _old_b, device = load_ab()
    import run_mvtd as mv

    mv.B_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    ckpt = torch.load(str(w), map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        model_b = ckpt.get("ema") or ckpt.get("model") or ckpt
    else:
        model_b = ckpt
    if hasattr(model_b, "float"):
        model_b = model_b.float()
    model_b.to(device).eval()
    pred_b = make_pred_b(model_b, device)

    def predA(frame):
        return pred_a(model_a, frame)

    def predB(frame):
        return pred_b(frame)

    def predF(frame):
        h, w_ = frame.shape[:2]
        return ds_wbf_fuse(predA(frame), predB(frame), w_, h)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    rows = []
    for sp in splits:
        paths = list_split(sp, args.stride)
        rows.append(eval_paths(paths, predA, "A_YOLOv8s", sp))
        rows.append(eval_paths(paths, predB, "B_YoloOW_2gb", sp))
        rows.append(eval_paths(paths, predF, "Fusao_DSWBF", sp))
    out = {
        "weights": str(w),
        "stride": args.stride,
        "protocol": "local IoU>=0.5 same class",
        "rows": rows,
    }
    (met / "eval_mvtd.json").write_text(json.dumps(out, indent=2))
    md = [
        "# Fine-tune YoloOW 2 GB → eval MVTD\n\n",
        f"peso: `{w}`\n\n",
        f"protocolo: IoU≥0.5 mesma classe, stride={args.stride}\n\n",
        "| split | método | P | R | F1 | TP | FP | FN | n |\n|---|---|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        md.append(
            f"| {r['split']} | {r['tag']} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} | "
            f"{r['TP']} | {r['FP']} | {r['FN']} | {r['n']} |\n"
        )
    (met / "NOTAS.md").write_text("".join(md))
    print("[ok]", met / "eval_mvtd.json")


if __name__ == "__main__":
    main()
