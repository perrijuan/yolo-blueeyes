#!/usr/bin/env python3
"""Tracking correto em blocos (SAHI) + figuras estilo YOLO-SEA (Entropy 2025, Figs. 9–10).

Imagens SeaDronesSee são 3840×2160. Rodar YOLO em imgsz=640 no frame inteiro
esmaga nadadores. Este script:

1. fatia o frame em tiles sobrepostos (bloco da imagem maior);
2. detecta em cada tile + 1 passada full-frame;
3. funde boxes (NMS/WBF) no referencial 4K;
4. rastreia (BoT-SORT) nas boxes mapeadas;
5. recorta o bloco de ação e monta figuras no layout do paper YOLO-SEA.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import (  # noqa: E402
    COCO_ANN,
    COCO_INIT,
    OUT,
    SEADRONESSEE_4GB,
    WEIGHTS,
    YOLOOW_SRC,
    setup_sys_path,
)

setup_sys_path()
ROOT = _REPO
YOLO_ROOT = SEADRONESSEE_4GB
WEIGHTS_DIR = WEIGHTS
RUNS_DIR = OUT / "seadronessee"
OUT_DIR = RUNS_DIR / "plots_tracking_blocks"
HOTA_DIR = RUNS_DIR / "hota_sahi"
YOLOOW_ROOT = YOLOOW_SRC

CLASS_NAMES = ["swimmer", "boat", "jetski", "buoy", "life_saving_appliances"]
ACTIVE_CLASS_IDS = [0, 1, 4]
CONF_A = 0.15
CONF_B = 0.15
IMGSZ = 640

# SAHI-like tiles (paper SAHI / Ultralytics sliced inference)
# 1280×1280 @ 25% overlap em 3840×2160 → 8 tiles + 1 full-frame
TILE = 1280
TILE_OVERLAP = 0.25
FULL_FRAME_PASS = True

# Tracking
TRACKER_KIND = "botsort"
TRACK_DET_THRESH = 0.25
TRACK_PER_CLASS = True
SEQ_IDS = (1, 19)          # val sequences presentes no subset 4 GB
MAX_FRAMES = {1: 90, 19: 70}
SEQ_SPLIT = "val"

# DS-WBF
WBF_IOU_THR = 0.55
DST_MATCH_IOU = 0.35
DST_REL_A = 0.95
DST_REL_B = 0.60
DST_CONFLICT_KEEP_BOTH = 0.55

# Paper figures
PANEL_W, PANEL_H = 480, 320
DPI_GAP = 8

OUT_DIR.mkdir(parents=True, exist_ok=True)
HOTA_DIR.mkdir(parents=True, exist_ok=True)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset index (COCO video_id + frame_index → jpeg no subset)
# ---------------------------------------------------------------------------
def build_sequences():
    img_dir = YOLO_ROOT / "images"
    seqs = defaultdict(list)
    for split, name in [
        ("val", "instances_val_objects_in_water.json"),
        ("train", "instances_train_objects_in_water.json"),
    ]:
        jp = COCO_ANN / name
        if not jp.is_file():
            continue
        coco = json.loads(jp.read_text())
        for im in coco["images"]:
            stem = Path(im["file_name"]).stem
            path = img_dir / f"{split}_{stem}.jpg"
            if not path.is_file():
                continue
            seqs[int(im["video_id"])].append(
                {
                    "split": split,
                    "frame_index": int(im["frame_index"]),
                    "path": path,
                    "stem": path.stem,
                    "video_id": int(im["video_id"]),
                    "file_name": im["file_name"],
                }
            )
    for vid in list(seqs):
        seqs[vid] = sorted(seqs[vid], key=lambda r: (r["split"] != "val", r["frame_index"]))
    return seqs


# ---------------------------------------------------------------------------
# Models A / B  (mesmo GATE do notebook)
# ---------------------------------------------------------------------------
def load_models():
    from ultralytics import YOLO

    sea = WEIGHTS_DIR / "yolov8s_seadronessee.pt"
    best = str(sea if sea.is_file() else COCO_INIT)
    model_a = YOLO(best)
    model_a.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] A = {best}")

    yoloow_str = str(YOLOOW_ROOT)
    if yoloow_str in sys.path:
        sys.path.remove(yoloow_str)
    sys.path.insert(0, yoloow_str)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = str((WEIGHTS_DIR / "YoloOW.pt").resolve())
    model_b = None
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
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
        print(f"[OK] B YoloOW = {weights_path}")
    except Exception as e:
        print(f"[aviso] YoloOW não carregou: {e}")
        model_b = None
    return model_a, model_b, device


def yolo_predict_frame(model, frame, conf=0.15):
    r = model.predict(source=frame, conf=conf, imgsz=IMGSZ, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 6), np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()[:, None]
    clss = r.boxes.cls.cpu().numpy()[:, None]
    return np.concatenate([xyxy, confs, clss], 1).astype(np.float32)


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


YOLOOW_TO_SDS = {1: 0, 5: 1, 6: 4}


def remap_yoloow_classes(det, mapping=YOLOOW_TO_SDS):
    if len(det) == 0:
        return det
    out = []
    for row in det:
        cid = int(row[5])
        if cid in mapping:
            r = row.copy()
            r[5] = mapping[cid]
            out.append(r)
    return np.array(out, dtype=np.float32) if out else np.empty((0, 6), np.float32)


def make_yoloow_predict(model_b, device):
    def yoloow_predict_frame(frame, conf=CONF_B, imgsz=640):
        if model_b is None:
            return np.empty((0, 6), np.float32)
        img0 = frame
        img, ratio, (dw, dh) = _letterbox(img0, new_shape=imgsz)
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

            pred = non_max_suppression(pred, conf_thres=conf, iou_thres=0.45, classes=None, agnostic=False)
            det = pred[0]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                det = det.cpu().numpy()
                out = np.concatenate([det[:, :4], det[:, 4:5], det[:, 5:6]], axis=1).astype(np.float32)
                return remap_yoloow_classes(out)
            return np.empty((0, 6), np.float32)
        except Exception:
            return np.empty((0, 6), np.float32)

    return yoloow_predict_frame


# ---------------------------------------------------------------------------
# DS-WBF (mesmo núcleo do notebook)
# ---------------------------------------------------------------------------
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    weighted_boxes_fusion = None


def _xyxy_to_norm(boxes, w, h):
    if len(boxes) == 0:
        return np.empty((0, 4))
    b = boxes.copy()
    b[:, [0, 2]] /= max(w, 1e-6)
    b[:, [1, 3]] /= max(h, 1e-6)
    return np.clip(b, 0, 1)


def _norm_to_xyxy(boxes, w, h):
    if len(boxes) == 0:
        return np.empty((0, 4))
    b = boxes.copy()
    b[:, [0, 2]] *= w
    b[:, [1, 3]] *= h
    return b


def _iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)


def _dst_combine(ma, mb):
    ma = float(np.clip(ma, 0.0, 0.999))
    mb = float(np.clip(mb, 0.0, 0.999))
    ua, ub = 1.0 - ma, 1.0 - mb
    numer = ma * mb + ma * ub + ua * mb
    K = ma * mb * abs(ma - mb)
    denom = 1.0 - K
    if denom <= 1e-9:
        return 0.0, 1.0
    return float(np.clip(numer / denom, 0.0, 1.0)), float(K)


def nms_pool(dets, w, h, iou_thr=0.55):
    """Funde detecções de vários tiles no frame cheio."""
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), np.float32)
    dets = np.asarray(dets, dtype=np.float32)
    if weighted_boxes_fusion is None:
        # greedy NMS class-aware
        keep = []
        order = np.argsort(-dets[:, 4])
        used = np.zeros(len(dets), dtype=bool)
        for i in order:
            if used[i]:
                continue
            keep.append(dets[i])
            for j in order:
                if used[j] or j == i:
                    continue
                if int(dets[i, 5]) != int(dets[j, 5]):
                    continue
                if _iou_xyxy(dets[i, :4], dets[j, :4]) >= iou_thr:
                    used[j] = True
        return np.stack(keep, 0).astype(np.float32) if keep else np.empty((0, 6), np.float32)
    boxes, scores, labels = weighted_boxes_fusion(
        [_xyxy_to_norm(dets[:, :4], w, h).tolist()],
        [dets[:, 4].tolist()],
        [dets[:, 5].astype(int).tolist()],
        weights=[1.0],
        iou_thr=iou_thr,
        skip_box_thr=0.01,
    )
    if len(boxes) == 0:
        return np.empty((0, 6), np.float32)
    xyxy = _norm_to_xyxy(np.array(boxes), w, h)
    return np.concatenate([xyxy, np.array(scores)[:, None], np.array(labels)[:, None]], 1).astype(np.float32)


def ds_wbf_fuse(dets_a, dets_b, w, h):
    if len(dets_a) == 0 and len(dets_b) == 0:
        return np.empty((0, 6), dtype=np.float32)
    if len(dets_a) == 0:
        return dets_b.astype(np.float32)
    if len(dets_b) == 0:
        return dets_a.astype(np.float32)
    used_b = set()
    fused_rows, leftover_a = [], []
    leftover_b_idx = set(range(len(dets_b)))
    for da in dets_a:
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(dets_b):
            if j in used_b or int(da[5]) != int(db[5]):
                continue
            iou = _iou_xyxy(da[:4], db[:4])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= DST_MATCH_IOU:
            db = dets_b[best_j]
            used_b.add(best_j)
            leftover_b_idx.discard(best_j)
            ma = float(da[4]) * DST_REL_A
            mb = float(db[4]) * DST_REL_B
            m_star, K = _dst_combine(ma, mb)
            if K >= DST_CONFLICT_KEEP_BOTH:
                leftover_a.append(da)
                leftover_b_idx.add(best_j)
                continue
            wa, wb = max(ma, 1e-6), max(mb, 1e-6)
            box = (wa * da[:4] + wb * db[:4]) / (wa + wb)
            cls = da[5] if ma >= mb else db[5]
            fused_rows.append(np.array([*box, m_star, cls], dtype=np.float32))
        else:
            leftover_a.append(da)
    leftover_b = [dets_b[j] for j in sorted(leftover_b_idx)]
    parts = []
    if fused_rows:
        parts.append(np.stack(fused_rows, 0))
    if leftover_a:
        parts.append(np.stack(leftover_a, 0).astype(np.float32))
    if leftover_b:
        parts.append(np.stack(leftover_b, 0).astype(np.float32))
    pool = np.concatenate(parts, 0) if parts else np.empty((0, 6), np.float32)
    return nms_pool(pool, w, h, iou_thr=WBF_IOU_THR)


# ---------------------------------------------------------------------------
# Slicing — bloco da imagem maior (SAHI)
# ---------------------------------------------------------------------------
def make_windows(H, W, tile=TILE, overlap=TILE_OVERLAP):
    th, tw = min(tile, H), min(tile, W)
    sh = max(1, int(th * (1.0 - overlap)))
    sw = max(1, int(tw * (1.0 - overlap)))
    ys = list(range(0, H - th + 1, sh))
    xs = list(range(0, W - tw + 1, sw))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != H - th:
        ys.append(H - th)
    if xs[-1] != W - tw:
        xs.append(W - tw)
    return [(int(x), int(y), int(tw), int(th)) for y in ys for x in xs]


def sliced_predict(predict_fn, frame, tile=TILE, overlap=TILE_OVERLAP, full_pass=FULL_FRAME_PASS):
    """Detecta em tiles + passada full-frame; devolve Nx6 no referencial original."""
    H, W = frame.shape[:2]
    parts = []
    for x, y, tw, th in make_windows(H, W, tile, overlap):
        crop = frame[y : y + th, x : x + tw]
        if crop.size == 0:
            continue
        dets = predict_fn(crop, tw, th)
        if dets is None or len(dets) == 0:
            continue
        d = np.asarray(dets, dtype=np.float32).copy()
        d[:, [0, 2]] += x
        d[:, [1, 3]] += y
        d[:, 0] = np.clip(d[:, 0], 0, W - 1)
        d[:, 2] = np.clip(d[:, 2], 0, W - 1)
        d[:, 1] = np.clip(d[:, 1], 0, H - 1)
        d[:, 3] = np.clip(d[:, 3], 0, H - 1)
        parts.append(d)
    if full_pass:
        full = predict_fn(frame, W, H)
        if full is not None and len(full):
            parts.append(np.asarray(full, dtype=np.float32))
    if not parts:
        return np.empty((0, 6), np.float32)
    pool = np.concatenate(parts, 0)
    return nms_pool(pool, W, H, iou_thr=0.50)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
def make_tracker():
    from boxmot.trackers.registry import TRACKER_DEFINITIONS, create_tracker

    kind = TRACKER_KIND
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if kind not in TRACKER_DEFINITIONS:
        kind = "botsort" if "botsort" in TRACKER_DEFINITIONS else "ocsort"
    tracker = create_tracker(
        tracker_type=kind,
        device=device,
        half=False,
        per_class=bool(TRACK_PER_CLASS),
        tracker_backend="python",
    )
    for attr, val in [
        ("det_thresh", TRACK_DET_THRESH),
        ("det_threshold", TRACK_DET_THRESH),
        ("per_class", TRACK_PER_CLASS),
    ]:
        if hasattr(tracker, attr):
            try:
                setattr(tracker, attr, val)
            except Exception:
                pass
    return tracker


def dets_to_boxmot(dets):
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), dtype=np.float32)
    d = np.asarray(dets, dtype=np.float32)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.shape[1] > 6:
        d = d[:, :6]
    elif d.shape[1] < 6:
        d = np.concatenate([d, np.zeros((d.shape[0], 6 - d.shape[1]), np.float32)], 1)
    return d.astype(np.float32)


# ---------------------------------------------------------------------------
# Drawing / paper layout
# ---------------------------------------------------------------------------
def id_color(tid: int):
    rng = np.random.RandomState((int(tid) + 1) * 9973)
    hsv = np.uint8([[[rng.randint(0, 180), 200 + rng.randint(0, 55), 220 + rng.randint(0, 35)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def put_label(img, text, org, color, font_scale=0.45, thickness=1):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    y1 = max(0, y - th - 6)
    cv2.rectangle(img, (x, y1), (x + tw + 6, y1 + th + 8), color, -1)
    cv2.putText(
        img,
        text,
        (x + 3, y1 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_tracks(frame, tracks, trails=None):
    vis = frame.copy()
    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            color = id_color(tid)
            for a, b in zip(pts[-25:], pts[-24:]):
                cv2.line(vis, a, b, color, 2, cv2.LINE_AA)
    if tracks is None or len(tracks) == 0:
        return vis
    t = np.asarray(tracks)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    for row in t:
        x1, y1, x2, y2 = map(int, row[:4])
        tid = int(row[4]) if row.shape[0] > 4 else -1
        conf = float(row[5]) if row.shape[0] > 5 else 1.0
        cls = int(row[6]) if row.shape[0] > 6 else -1
        color = id_color(tid if tid >= 0 else 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cname = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else "obj"
        put_label(vis, f"ID {tid} {cname} {conf:.2f}", (x1, max(18, y1)), color)
    return vis


def crop_window(H, W, boxes, pad=80, min_w=900, min_h=600):
    if boxes is None or len(boxes) == 0:
        cx, cy = W // 2, H // 2
        x1, y1 = max(0, cx - min_w // 2), max(0, cy - min_h // 2)
        return x1, y1, min(W, x1 + min_w), min(H, y1 + min_h)
    b = np.asarray(boxes)
    x1 = int(b[:, 0].min()) - pad
    y1 = int(b[:, 1].min()) - pad
    x2 = int(b[:, 2].max()) + pad
    y2 = int(b[:, 3].max()) + pad
    bw, bh = x2 - x1, y2 - y1
    if bw < min_w:
        extra = min_w - bw
        x1 -= extra // 2
        x2 += extra - extra // 2
    if bh < min_h:
        extra = min_h - bh
        y1 -= extra // 2
        y2 += extra - extra // 2
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def take_block(frame, win, out_w=PANEL_W, out_h=PANEL_H):
    x1, y1, x2, y2 = win
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _text_center(img, text, box, scale=0.55, color=(20, 20, 20), thick=1):
    x, y, w, h = box
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(
        img,
        text,
        (x + (w - tw) // 2, y + (h + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA,
    )


def yellow_header(text, w, h=34):
    bar = np.full((h, w, 3), (40, 230, 255), np.uint8)  # BGR yellow like YOLO-SEA Fig.9
    _text_center(bar, text, (0, 0, w, h), scale=0.55, color=(20, 20, 20), thick=1)
    return bar


def compose_fig10(rows, col_labels, method_names, path):
    """rows: list[list[np.ndarray]]  method × scene. YOLO-SEA Figure 10 layout."""
    n_m, n_s = len(rows), len(rows[0])
    gutter = 150
    footer = 36
    gap = DPI_GAP
    canvas_w = gutter + n_s * PANEL_W + (n_s + 1) * gap
    canvas_h = n_m * PANEL_H + (n_m + 1) * gap + footer
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    for i, name in enumerate(method_names):
        y = gap + i * (PANEL_H + gap)
        _text_center(canvas, name, (0, y, gutter, PANEL_H), scale=0.50, color=(30, 30, 30), thick=1)
        for j, panel in enumerate(rows[i]):
            x = gutter + gap + j * (PANEL_W + gap)
            canvas[y : y + PANEL_H, x : x + PANEL_W] = panel
    for j, lab in enumerate(col_labels):
        x = gutter + gap + j * (PANEL_W + gap)
        _text_center(canvas, lab, (x, canvas_h - footer, PANEL_W, footer), scale=0.55, color=(30, 30, 30))
    cv2.imwrite(str(path), canvas)
    print(f"[ok] {path}")
    return canvas


def compose_fig9(pairs, letters, path, headers=("YOLOv8s-Sea", "Fusão DS-WBF")):
    """YOLO-SEA Figure 9: 2 colunas de pares (A–H), header amarelo."""
    n = len(pairs)
    n_left = (n + 1) // 2
    col_gap, row_gap, letter_w, hdr_h = 28, 10, 36, 34
    col_w = letter_w + PANEL_W + 6 + PANEL_W
    canvas_w = 40 + 2 * col_w + col_gap
    canvas_h = 40 + n_left * (hdr_h + PANEL_H + row_gap) + 20
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)

    def place(k, col):
        i = k  # row in this column
        x0 = 20 + col * (col_w + col_gap)
        y0 = 20 + i * (hdr_h + PANEL_H + row_gap)
        letter = letters[col * n_left + k] if col * n_left + k < len(letters) else ""
        _text_center(canvas, letter, (x0, y0 + hdr_h, letter_w, PANEL_H), scale=0.8, color=(20, 20, 20), thick=2)
        hx = x0 + letter_w
        canvas[y0 : y0 + hdr_h, hx : hx + PANEL_W] = yellow_header(headers[0], PANEL_W, hdr_h)
        canvas[y0 : y0 + hdr_h, hx + PANEL_W + 6 : hx + 2 * PANEL_W + 6] = yellow_header(headers[1], PANEL_W, hdr_h)
        pa, pb = pairs[col * n_left + k]
        canvas[y0 + hdr_h : y0 + hdr_h + PANEL_H, hx : hx + PANEL_W] = pa
        canvas[y0 + hdr_h : y0 + hdr_h + PANEL_H, hx + PANEL_W + 6 : hx + 2 * PANEL_W + 6] = pb

    for k in range(n_left):
        place(k, 0)
    for k in range(n - n_left):
        place(k, 1)
    cv2.imwrite(str(path), canvas)
    print(f"[ok] {path}")
    return canvas


def compose_temporal(strips, id_labels, path):
    """Linhas = IDs, colunas = tempo. Cada célula é o bloco recortado do alvo."""
    n_id, n_t = len(strips), len(strips[0])
    gutter, footer, gap = 110, 32, 6
    pw, ph = 220, 160
    canvas_w = gutter + n_t * pw + (n_t + 1) * gap
    canvas_h = n_id * ph + (n_id + 1) * gap + footer
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    for i, lab in enumerate(id_labels):
        y = gap + i * (ph + gap)
        _text_center(canvas, lab, (0, y, gutter, ph), scale=0.50, color=(30, 30, 30))
        for j, panel in enumerate(strips[i]):
            x = gutter + gap + j * (pw + gap)
            canvas[y : y + ph, x : x + pw] = cv2.resize(panel, (pw, ph))
    for j in range(n_t):
        x = gutter + gap + j * (pw + gap)
        _text_center(canvas, f"t{j + 1}", (x, canvas_h - footer, pw, footer), scale=0.5, color=(30, 30, 30))
    cv2.imwrite(str(path), canvas)
    print(f"[ok] {path}")
    return canvas


def draw_tile_overlay(frame, dets=None):
    vis = frame.copy()
    H, W = vis.shape[:2]
    for k, (x, y, tw, th) in enumerate(make_windows(H, W)):
        cv2.rectangle(vis, (x, y), (x + tw - 1, y + th - 1), (0, 220, 255), 3)
        put_label(vis, f"tile {k}", (x + 8, y + 28), (0, 180, 255), font_scale=0.7)
    if dets is not None and len(dets):
        for row in dets:
            x1, y1, x2, y2 = map(int, row[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # downscale for saving
    vis_s = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_AREA)
    return vis_s


# ---------------------------------------------------------------------------
# Run tracking on one sequence
# ---------------------------------------------------------------------------
def run_sequence(vid, rows, predict_fns):
    """predict_fns: dict tag -> fn(frame,w,h)->Nx6"""
    rows = [r for r in rows if r["split"] == SEQ_SPLIT] or rows
    cap = MAX_FRAMES.get(vid)
    if cap is not None:
        rows = rows[:cap]
    print(f"\n=== seq {vid} frames={len(rows)} split_pref={SEQ_SPLIT} ===")

    store = {tag: {} for tag in predict_fns}  # fidx -> tracks Nx7 (xyxy,id,conf,cls)
    mot_lines = {tag: [] for tag in predict_fns}
    trails = {tag: defaultdict(list) for tag in predict_fns}
    frames_meta = []

    trackers = {tag: make_tracker() for tag in predict_fns}

    for r in tqdm(rows, desc=f"sahi-track seq{vid}"):
        frame = cv2.imread(str(r["path"]))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        fidx = int(r["frame_index"])
        if fidx <= 0:
            fidx = 1
        frames_meta.append({**r, "h": h, "w": w})

        for tag, fn in predict_fns.items():
            try:
                dets = fn(frame, w, h)
            except Exception as e:
                print(f"    [aviso] {tag} predict fidx={fidx}: {e}")
                dets = np.empty((0, 6), np.float32)
            if len(dets):
                dets = dets[np.isin(dets[:, 5].astype(int), ACTIVE_CLASS_IDS)]
            try:
                tracks = trackers[tag].update(
                    np.ascontiguousarray(dets_to_boxmot(dets), dtype=np.float64),
                    frame,
                )
            except Exception as e:
                print(f"    [erro] {tag} update fidx={fidx}: {e}")
                tracks = np.empty((0, 8))
            t = np.asarray(tracks)
            if t.size == 0:
                store[tag][fidx] = np.empty((0, 7), np.float32)
                continue
            if t.ndim == 1:
                t = t.reshape(1, -1)
            rows_t = []
            for row in t:
                if row.shape[0] < 6:
                    continue
                x1, y1, x2, y2 = map(float, row[:4])
                tid = int(row[4]) if row[4] == row[4] else -1
                conf = float(row[5])
                cls = int(row[6]) if row.shape[0] > 6 else 0
                if tid < 0:
                    continue
                rows_t.append([x1, y1, x2, y2, tid, conf, cls])
                mot_lines[tag].append(
                    f"{fidx},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1"
                )
                trails[tag][tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            store[tag][fidx] = np.array(rows_t, np.float32) if rows_t else np.empty((0, 7), np.float32)

        free_gpu()

    for tag in predict_fns:
        outp = HOTA_DIR / f"pred_{tag}"
        outp.mkdir(parents=True, exist_ok=True)
        (outp / f"{vid}.txt").write_text("\n".join(mot_lines[tag]) + ("\n" if mot_lines[tag] else ""))
        print(f"  [{tag}] seq={vid} mot_lines={len(mot_lines[tag])}")
        del trackers[tag]
    free_gpu()
    return store, frames_meta, trails


def pick_scene_indices(store_f, frames_meta, n=5, min_obj=2):
    scored = []
    for i, meta in enumerate(frames_meta):
        tr = store_f.get(meta["frame_index"], np.empty((0, 7)))
        if len(tr) < min_obj:
            continue
        # prefer compact clusters (crop-friendly)
        b = tr[:, :4]
        area = max(1.0, (b[:, 2].max() - b[:, 0].min()) * (b[:, 3].max() - b[:, 1].min()))
        dens = len(tr) / (area / (1920 * 1080) + 1e-3)
        scored.append((dens, len(tr), i))
    scored.sort(reverse=True)
    # diversify in time
    chosen, used = [], set()
    for _, _, i in scored:
        if any(abs(i - u) < max(4, len(frames_meta) // (n * 3)) for u in used):
            continue
        chosen.append(i)
        used.add(i)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        step = max(1, len(frames_meta) // n)
        for i in range(0, len(frames_meta), step):
            if i not in used:
                chosen.append(i)
            if len(chosen) >= n:
                break
    return sorted(chosen)[:n]


def panel_for(meta, tracks, trails, win=None):
    frame = cv2.imread(str(meta["path"]))
    if frame is None:
        return np.zeros((PANEL_H, PANEL_W, 3), np.uint8), (0, 0, PANEL_W, PANEL_H)
    h, w = frame.shape[:2]
    vis = draw_tracks(frame, tracks, trails)
    if win is None:
        boxes = tracks[:, :4] if tracks is not None and len(tracks) else None
        win = crop_window(h, w, boxes)
    block = take_block(vis, win)
    return block, win


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("TRACKING SAHI + BLOCOS estilo YOLO-SEA (Entropy 2025 Figs. 9–10)")
    print(f"TILE={TILE} overlap={TILE_OVERLAP} full_pass={FULL_FRAME_PASS}")
    print(f"OUT={OUT_DIR}")
    print("=" * 64)

    seqs = build_sequences()
    model_a, model_b, device = load_models()

    def pred_A_wh(frame, w, h):
        return yolo_predict_frame(model_a, frame, conf=CONF_A)

    yoloow_predict_frame = make_yoloow_predict(model_b, device)

    def pred_B_wh(frame, w, h):
        return yoloow_predict_frame(frame, conf=CONF_B)

    def pred_A_sahi(frame, w, h):
        return sliced_predict(pred_A_wh, frame)

    def pred_B_sahi(frame, w, h):
        return sliced_predict(pred_B_wh, frame)

    def pred_F_sahi(frame, w, h):
        da = pred_A_sahi(frame, w, h)
        db = pred_B_sahi(frame, w, h)
        return ds_wbf_fuse(da, db, w, h)

    predict_fns = {
        "A_YOLOv8s": pred_A_sahi,
        "B_YoloOW": pred_B_sahi,
        "Fusao_DSWBF": pred_F_sahi,
    }

    all_store = {}
    all_meta = {}
    all_trails = {}

    for vid in SEQ_IDS:
        if vid not in seqs or not seqs[vid]:
            print(f"[skip] seq {vid} ausente no subset")
            continue
        store, meta, trails = run_sequence(vid, seqs[vid], predict_fns)
        all_store[vid] = store
        all_meta[vid] = meta
        all_trails[vid] = trails

    method_order = ["A_YOLOv8s", "B_YoloOW", "Fusao_DSWBF"]
    method_names = ["YOLOv8s-Sea", "YoloOW", "Fusão DS-WBF"]

    # ---- tile overlay (explica o bloco da imagem maior) ----
    demo_vid = next(iter(all_meta), None)
    if demo_vid is not None and all_meta[demo_vid]:
        meta0 = all_meta[demo_vid][len(all_meta[demo_vid]) // 2]
        fr = cv2.imread(str(meta0["path"]))
        if fr is not None:
            dets = pred_F_sahi(fr, fr.shape[1], fr.shape[0])
            overlay = draw_tile_overlay(fr, dets)
            p = OUT_DIR / "fig_sahi_tiles_overlay.png"
            cv2.imwrite(str(p), overlay)
            print(f"[ok] {p}  windows={len(make_windows(*fr.shape[:2]))}")
            # full 4K vs bloco recortado (mesmo frame)
            tr = all_store[demo_vid]["Fusao_DSWBF"].get(meta0["frame_index"], np.empty((0, 7)))
            vis_full = draw_tracks(fr, tr, all_trails[demo_vid]["Fusao_DSWBF"])
            win = crop_window(*fr.shape[:2], tr[:, :4] if len(tr) else None)
            block = take_block(vis_full, win, 960, 640)
            full_s = cv2.resize(vis_full, (960, 540), interpolation=cv2.INTER_AREA)
            # highlight crop window on full
            x1, y1, x2, y2 = win
            sx, sy = 960 / fr.shape[1], 540 / fr.shape[0]
            cv2.rectangle(
                full_s,
                (int(x1 * sx), int(y1 * sy)),
                (int(x2 * sx), int(y2 * sy)),
                (0, 220, 255),
                3,
            )
            canvas = np.full((640, 960 * 2 + 24, 3), 255, np.uint8)
            canvas[50 : 50 + 540, 0:960] = full_s
            canvas[0:640, 960 + 24 : 960 + 24 + 960] = block
            _text_center(canvas, "frame 4K (tile grid crop em amarelo)", (0, 0, 960, 48), 0.55)
            _text_center(canvas, "bloco recortado (tracking)", (960 + 24, 0, 960, 48), 0.55)
            cv2.imwrite(str(OUT_DIR / "fig_full_vs_block.png"), canvas)
            print(f"[ok] {OUT_DIR / 'fig_full_vs_block.png'}")

    # ---- Figure 10 (5 + 5 cenas) ----
    scenes = []  # (vid, meta_idx)
    for vid in all_meta:
        idxs = pick_scene_indices(all_store[vid]["Fusao_DSWBF"], all_meta[vid], n=6, min_obj=1)
        for i in idxs:
            scenes.append((vid, i))
    scenes = scenes[:10]
    if len(scenes) < 5 and demo_vid is not None:
        for i in range(min(5, len(all_meta[demo_vid]))):
            scenes.append((demo_vid, i))
        scenes = scenes[:10]

    def build_fig10_block(scene_slice, labels, fname):
        rows = [[] for _ in method_order]
        for vid, i in scene_slice:
            meta = all_meta[vid][i]
            fidx = meta["frame_index"]
            tr_f = all_store[vid]["Fusao_DSWBF"].get(fidx, np.empty((0, 7)))
            frame = cv2.imread(str(meta["path"]))
            h, w = (frame.shape[:2] if frame is not None else (2160, 3840))
            win = crop_window(h, w, tr_f[:, :4] if len(tr_f) else None)
            for mi, tag in enumerate(method_order):
                tr = all_store[vid][tag].get(fidx, np.empty((0, 7)))
                panel, _ = panel_for(meta, tr, all_trails[vid][tag], win=win)
                rows[mi].append(panel)
        compose_fig10(rows, labels, method_names, OUT_DIR / fname)

    if len(scenes) >= 5:
        build_fig10_block(scenes[:5], [f"({k})" for k in range(1, 6)], "fig10_tracking_scenes_1_5.png")
    if len(scenes) > 5:
        n2 = min(5, len(scenes) - 5)
        build_fig10_block(
            scenes[5 : 5 + n2],
            [f"({k})" for k in range(6, 6 + n2)],
            "fig10_tracking_scenes_6_10.png",
        )
        # stacked like the paper
        p1 = cv2.imread(str(OUT_DIR / "fig10_tracking_scenes_1_5.png"))
        p2 = cv2.imread(str(OUT_DIR / "fig10_tracking_scenes_6_10.png"))
        if p1 is not None and p2 is not None:
            w = max(p1.shape[1], p2.shape[1])
            def pad(im, w):
                if im.shape[1] == w:
                    return im
                canv = np.full((im.shape[0], w, 3), 255, np.uint8)
                canv[:, : im.shape[1]] = im
                return canv
            stacked = np.concatenate([pad(p1, w), pad(p2, w)], 0)
            cv2.imwrite(str(OUT_DIR / "fig10_tracking_comparison.png"), stacked)
            print(f"[ok] {OUT_DIR / 'fig10_tracking_comparison.png'}")

    # ---- Figure 9 (A–H pares A vs Fusão, mesmo bloco) ----
    letters = list("ABCDEFGH")
    pairs = []
    for vid, i in scenes[:8]:
        meta = all_meta[vid][i]
        fidx = meta["frame_index"]
        tr_f = all_store[vid]["Fusao_DSWBF"].get(fidx, np.empty((0, 7)))
        frame = cv2.imread(str(meta["path"]))
        h, w = (frame.shape[:2] if frame is not None else (2160, 3840))
        win = crop_window(h, w, tr_f[:, :4] if len(tr_f) else None)
        pa, _ = panel_for(meta, all_store[vid]["A_YOLOv8s"].get(fidx, np.empty((0, 7))), all_trails[vid]["A_YOLOv8s"], win)
        pf, _ = panel_for(meta, tr_f, all_trails[vid]["Fusao_DSWBF"], win)
        pairs.append((pa, pf))
    if pairs:
        compose_fig9(pairs, letters[: len(pairs)], OUT_DIR / "fig9_tracking_gallery.png")

    # ---- temporal ID strips (bloco centrado no alvo ao longo do tempo) ----
    if demo_vid is not None:
        store_f = all_store[demo_vid]["Fusao_DSWBF"]
        meta_list = all_meta[demo_vid]
        id_count = Counter()
        for tr in store_f.values():
            if len(tr):
                id_count.update(tr[:, 4].astype(int).tolist())
        top_ids = [i for i, _ in id_count.most_common(4)]
        n_t = 6
        idxs = np.linspace(0, max(0, len(meta_list) - 1), n_t, dtype=int)
        strips, labs = [], []
        for tid in top_ids:
            row_panels = []
            for i in idxs:
                meta = meta_list[int(i)]
                frame = cv2.imread(str(meta["path"]))
                if frame is None:
                    row_panels.append(np.zeros((PANEL_H, PANEL_W, 3), np.uint8))
                    continue
                tr = store_f.get(meta["frame_index"], np.empty((0, 7)))
                hit = tr[tr[:, 4] == tid] if len(tr) else np.empty((0, 7))
                vis = draw_tracks(frame, tr, all_trails[demo_vid]["Fusao_DSWBF"])
                boxes = hit[:, :4] if len(hit) else (tr[:, :4] if len(tr) else None)
                win = crop_window(*frame.shape[:2], boxes, pad=70, min_w=480, min_h=360)
                row_panels.append(take_block(vis, win))
            strips.append(row_panels)
            labs.append(f"ID {tid}")
        if strips:
            compose_temporal(strips, labs, OUT_DIR / "fig_temporal_id_blocks.png")

    # sidecar json
    summary = {
        "tile": TILE,
        "overlap": TILE_OVERLAP,
        "full_frame_pass": FULL_FRAME_PASS,
        "seqs": list(all_meta.keys()),
        "n_frames": {int(v): len(all_meta[v]) for v in all_meta},
        "windows_4k": len(make_windows(2160, 3840)),
        "out_dir": str(OUT_DIR),
        "reference": "YOLO-SEA Entropy 2025, 27, 667 — Figures 9 and 10 layout",
        "sahi": "Akyon et al. ICIP 2022 — Slicing Aided Hyper Inference",
    }
    (OUT_DIR / "blocks_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[done] figuras em", OUT_DIR)
    for p in sorted(OUT_DIR.glob("*.png")):
        print("  ", p.name, f"{p.stat().st_size/1e3:.0f} kB")


if __name__ == "__main__":
    main()
