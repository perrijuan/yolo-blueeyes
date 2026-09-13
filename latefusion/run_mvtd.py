#!/usr/bin/env python3
"""Late fusion DS-WBF no MVTD: A = YOLOv8s fine-tune + Soft-NMS, B = YoloOW (boat).

Val e test YOLO (labels em yolo/labels/{split}).
Protocolo local IoU≥0.5 mesma classe — NÃO é AUC de tracker do paper.
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
import yaml
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import MVTD_PESOS, MVTD_YOLO, OUT, PESOS_OUT, YOLOOW_SRC, setup_sys_path  # noqa: E402

setup_sys_path()

from latefusion import ds_wbf_fuse, soft_nms  # noqa: E402

YOLO_ROOT = MVTD_YOLO
PESOS = MVTD_PESOS
MET = OUT / "mvtd" / "metricas"
CFG = yaml.safe_load((_REPO / "latefusion" / "fusion_config_mvtd.yaml").read_text())
CLASS_NAMES = list(CFG["class_names"])
ACTIVE = set(int(x) for x in CFG["active_class_ids"])
YOLOOW_TO_MVTD = {int(k): int(v) for k, v in dict(CFG["yoloow_to_mvtd"]).items()}
B_CLASS_MAP = dict(YOLOOW_TO_MVTD)
CONF_A = float(CFG["conf_a"])
CONF_B = float(CFG["conf_b"])
IMGSZ_A = int(CFG["imgsz_a"])
IMGSZ_B = int(CFG["imgsz_b"])


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    if not label_path.is_file():
        return np.empty((0, 5), np.float32)
    gts = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in ACTIVE:
            continue
        xc, yc, wn, hn = map(float, parts[1:5])
        gts.append(
            [(xc - wn / 2) * w, (yc - hn / 2) * h, (xc + wn / 2) * w, (yc + hn / 2) * h, cls]
        )
    return np.array(gts, np.float32) if gts else np.empty((0, 5), np.float32)


def iou_box(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def list_split(split: str, stride: int) -> list[Path]:
    img_dir = YOLO_ROOT / "images" / split
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if stride <= 1:
        return paths
    by_seq: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        seq = p.stem.rsplit("_", 1)[0]
        by_seq[seq].append(p)
    out = []
    for seq in sorted(by_seq):
        frames = sorted(by_seq[seq], key=lambda x: x.stem)
        out.extend(frames[::stride])
    return out


def _letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def remap_yoloow(det: np.ndarray) -> np.ndarray:
    if len(det) == 0:
        return det
    out = []
    for row in det:
        cid = int(row[5])
        if cid in B_CLASS_MAP:
            r = row.copy()
            r[5] = B_CLASS_MAP[cid]
            out.append(r)
    return np.array(out, np.float32) if out else np.empty((0, 6), np.float32)


def load_models():
    from ultralytics import YOLO

    wa = None
    for cand in (
        PESOS / "mvtd_yolov8s_best.pt",
        PESOS / "mvtd_yolov8s_v2_best.pt",
        PESOS_OUT / "mvtd_yolov8s_v2_best.pt",
        PESOS_OUT / "mvtd_yolov8s_best.pt",
    ):
        if cand.is_file():
            wa = cand
            break
    if wa is None:
        raise SystemExit(
            "falta A (YOLOv8s MVTD): coloque mvtd_yolov8s_best.pt em "
            f"{PESOS} ou {PESOS_OUT}"
        )
    model_a = YOLO(str(wa))
    model_a.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] A domínio MVTD = {wa}")

    yoloow_str = str(YOLOOW_SRC)
    if yoloow_str in sys.path:
        sys.path.remove(yoloow_str)
    sys.path.insert(0, yoloow_str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wb = None
    for cand in (
        PESOS / "B_YoloOW_mvtd.pt",
        PESOS_OUT / "B_YoloOW_mvtd.pt",
        OUT / "yoloow" / "pesos" / "B_YoloOW_mvtd_2gb.pt",
    ):
        if cand.is_file():
            wb = cand
            break
    global B_CLASS_MAP
    if wb is not None:
        B_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
        print("[OK] B = fine-tune MVTD 4 classes (sem remap Sea)")
    else:
        wb = PESOS / "B_YoloOW.pt"
        if not wb.is_file():
            wb = PESOS_OUT / "B_YoloOW.pt"
        B_CLASS_MAP = dict(YOLOOW_TO_MVTD)
    model_b = None
    try:
        ckpt = torch.load(str(wb.resolve()), map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            model_b = ckpt.get("ema") or ckpt.get("model") or ckpt
        else:
            model_b = ckpt
        if hasattr(model_b, "float"):
            model_b = model_b.float()
        model_b.to(device).eval()
        if device.type == "cuda" and hasattr(model_b, "half"):
            try:
                model_b.half()
            except Exception:
                pass
        print(f"[OK] B YoloOW = {wb}")
    except Exception as e:
        print(f"[aviso] YoloOW não carregou: {e}")
        model_b = None
    return model_a, model_b, device


def pred_a(model_a, frame):
    r = model_a.predict(source=frame, conf=CONF_A, imgsz=IMGSZ_A, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 6), np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()[:, None]
    clss = r.boxes.cls.cpu().numpy()[:, None]
    dets = np.concatenate([xyxy, confs, clss], 1).astype(np.float32)
    return soft_nms(dets)


def make_pred_b(model_b, device):
    def pred_b(frame):
        if model_b is None:
            return np.empty((0, 6), np.float32)
        img0 = frame
        img, _, _ = _letterbox(img0, new_shape=IMGSZ_B)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if next(model_b.parameters()).dtype == torch.float16 else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = model_b(img)[0]
        try:
            from utils.general import non_max_suppression, scale_coords

            pred = non_max_suppression(pred, conf_thres=CONF_B, iou_thres=0.45)
            det = pred[0]
            if det is None or len(det) == 0:
                return np.empty((0, 6), np.float32)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            det = det.cpu().numpy()
            out = np.concatenate([det[:, :4], det[:, 4:5], det[:, 5:6]], 1).astype(np.float32)
            return remap_yoloow(out)
        except Exception:
            return np.empty((0, 6), np.float32)

    return pred_b


def match_prf(dets, gts):
    tp = fp = 0
    matched = set()
    if len(dets):
        dets = dets[np.isin(dets[:, 5].astype(int), list(ACTIVE))]
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
    fn = len(gts) - len(matched)
    return tp, fp, fn


def _pack(split, tag, tp, fp, fn, n) -> dict:
    P = tp / (tp + fp + 1e-6)
    R = tp / (tp + fn + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)
    Acc = tp / (tp + fp + fn + 1e-6)
    sm = {
        "split": split,
        "method": tag,
        "P": P,
        "R": R,
        "F1": F1,
        "Acc": Acc,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "n": n,
        "protocol": "local IoU>=0.5 same-class (nao e AUC VOT do paper)",
    }
    print(
        f"  {tag:8s} [{split}] n={n} P={P:.3f} R={R:.3f} F1={F1:.3f} "
        f"Acc={Acc:.3f} TP={tp} FP={fp} FN={fn}"
    )
    return sm


def eval_split_all(split: str, paths: list[Path], pred_A, pred_B, pred_F) -> list[dict]:
    """Um único passe: A, B e fusão no mesmo frame."""
    lab_dir = YOLO_ROOT / "labels" / split
    acc = {k: [0, 0, 0] for k in ("A", "B", "Fusao")}
    for p in tqdm(paths, desc=f"{split} A+B+F"):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gts = load_gt(lab_dir / f"{p.stem}.txt", w, h)
        da = pred_A(frame, w, h)
        db = pred_B(frame, w, h)
        df = ds_wbf_fuse(da, db, w, h)
        for tag, dets in (("A", da), ("B", db), ("Fusao", df)):
            t, f, n = match_prf(dets, gts)
            acc[tag][0] += t
            acc[tag][1] += f
            acc[tag][2] += n
    return [_pack(split, tag, *acc[tag], len(paths)) for tag in ("A", "B", "Fusao")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    ap.add_argument("--stride", type=int, default=5, help="1 = todos os frames")
    args = ap.parse_args()
    MET.mkdir(parents=True, exist_ok=True)
    model_a, model_b, device = load_models()
    fn_b = make_pred_b(model_b, device)

    def pred_A(frame, w, h):
        return pred_a(model_a, frame)

    def pred_B(frame, w, h):
        return fn_b(frame)

    def pred_F(frame, w, h):
        da = pred_A(frame, w, h)
        db = pred_B(frame, w, h)
        return ds_wbf_fuse(da, db, w, h)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    rows = []
    for sp in splits:
        paths = list_split(sp, args.stride)
        print(f"\n=== {sp} stride={args.stride} n={len(paths)} ===")
        rows.extend(eval_split_all(sp, paths, pred_A, pred_B, pred_F))
    out = MET / "latefusion_val_test.json"
    out.write_text(json.dumps(rows, indent=2))
    print("[ok]", out)


if __name__ == "__main__":
    main()
