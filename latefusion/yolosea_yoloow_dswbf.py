#!/usr/bin/env python3
"""YOLO-SEA (paper Entropy 2025) + YoloOW + late fusion DS-WBF + BoT-SORT/CMC.

A = YOLOv8s-Sea + Soft-NMS (contribuição 4 do paper; SESA/BiFPN exigem pesos
    oficiais que o artigo não libera — Soft-NMS e SimAM são os blocos sem treino).
B = YoloOW (Xjh-UCAS)
F = Dempster-Shafer + WBF (evidência A↔B com conflito K, NÃO ganho constante)

Tracker: BoT-SORT + CMC, per_class=True, det_thresh alinhado, frame_index COCO,
         fovéa em boxes pequenas da fusão, tracker.update loga erro.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import OUT as RUNS_OUT, setup_sys_path  # noqa: E402

setup_sys_path()
ROOT = _REPO

from tracking_sahi_paper_blocks import (  # noqa: E402
    ACTIVE_CLASS_IDS,
    CLASS_NAMES,
    COCO_ANN,
    CONF_A,
    CONF_B,
    IMGSZ,
    WEIGHTS_DIR,
    YOLO_ROOT,
    build_sequences,
    ds_wbf_fuse,
    dets_to_boxmot,
    free_gpu,
    load_models,
    make_tracker,
    make_yoloow_predict,
    yolo_predict_frame,
)

RUNS = RUNS_OUT / "seadronessee"
OUT = RUNS / "yolosea_yoloow_dswbf"
VID = OUT / "videos"
PLOTS = OUT / "plots"
HOTA = OUT / "hota"
OUT.mkdir(parents=True, exist_ok=True)
VID.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)
HOTA.mkdir(exist_ok=True)

# YOLO-SEA Soft-NMS (Bodla et al. / paper §2.2 Soft-NMS)
SOFTNMS_SIGMA = 0.5
SOFTNMS_IOU = 0.50
SOFTNMS_MIN = 0.05
FOVEA_ENABLE = True
FOVEA_MAX_AREA = 80 * 80
FOVEA_SCALE = 1.8
MAX_VAL = 250
SEQ_IDS = (1, 19)
MAX_FRAMES = {1: 55, 19: 40}
FPS = 10
PANEL_W, PANEL_H, HDR = 640, 360, 32

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# YOLO-SEA pieces (paper)
# ---------------------------------------------------------------------------
def simam(x: torch.Tensor, e_lambda: float = 1e-4) -> torch.Tensor:
    """SimAM parameter-free (Yang et al.; paper eqs. 5–7). x: BCHW."""
    n = x.shape[2] * x.shape[3] - 1
    d = x - x.mean(dim=[2, 3], keepdim=True)
    v = (d.pow(2).sum(dim=[2, 3], keepdim=True) / n) + e_lambda
    e_inv = d.pow(2) / (4 * v) + 0.5
    return x * torch.sigmoid(e_inv)


def _iou_mat(a, b):
    x1 = np.maximum(a[None, 0], b[:, 0])
    y1 = np.maximum(a[None, 1], b[:, 1])
    x2 = np.minimum(a[None, 2], b[:, 2])
    y2 = np.minimum(a[None, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ua + ub - inter + 1e-6)


def soft_nms(dets, sigma=SOFTNMS_SIGMA, iou_thr=SOFTNMS_IOU, min_score=SOFTNMS_MIN):
    """Soft-NMS gaussiano por classe (paper YOLO-SEA, substitui NMS duro)."""
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), np.float32)
    dets = np.asarray(dets, dtype=np.float32)
    keep = []
    for cls in np.unique(dets[:, 5].astype(int)):
        d = dets[dets[:, 5].astype(int) == cls].copy()
        while len(d):
            i = int(np.argmax(d[:, 4]))
            cur = d[i].copy()
            keep.append(cur)
            d = np.delete(d, i, 0)
            if len(d) == 0:
                break
            iou = _iou_mat(cur[:4], d[:, :4])
            decay = np.exp(-(iou ** 2) / sigma)
            # só amortece se overlap relevante
            decay = np.where(iou >= iou_thr, decay, 1.0)
            d[:, 4] *= decay
            d = d[d[:, 4] >= min_score]
    if not keep:
        return np.empty((0, 6), np.float32)
    out = np.stack(keep, 0)
    return out[np.argsort(-out[:, 4])]


def pred_yolosea(model_a, frame, w, h):
    """A: YOLOv8s-Sea + Soft-NMS (YOLO-SEA post-process do paper)."""
    # iou alto p/ entregar candidatos; Soft-NMS faz o decay
    r = model_a.predict(source=frame, conf=CONF_A, imgsz=IMGSZ, iou=0.85, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 6), np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()[:, None]
    clss = r.boxes.cls.cpu().numpy()[:, None]
    raw = np.concatenate([xyxy, confs, clss], 1).astype(np.float32)
    return soft_nms(raw)


def fovea_refine(predict_fn, frame, dets):
    if not FOVEA_ENABLE or dets is None or len(dets) == 0:
        return dets
    h, w = frame.shape[:2]
    extra = [dets]
    for row in dets:
        x1, y1, x2, y2 = row[:4]
        area = max(0.0, (x2 - x1) * (y2 - y1))
        if area <= 0 or area > FOVEA_MAX_AREA:
            continue
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        bw = max((x2 - x1) * FOVEA_SCALE, 32)
        bh = max((y2 - y1) * FOVEA_SCALE, 32)
        xa, ya = int(np.clip(cx - bw / 2, 0, w - 1)), int(np.clip(cy - bh / 2, 0, h - 1))
        xb, yb = int(np.clip(cx + bw / 2, 1, w)), int(np.clip(cy + bh / 2, 1, h))
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            continue
        try:
            loc = predict_fn(crop, crop.shape[1], crop.shape[0])
        except Exception:
            continue
        if loc is None or len(loc) == 0:
            continue
        loc = loc.copy()
        loc[:, [0, 2]] += xa
        loc[:, [1, 3]] += ya
        extra.append(loc)
    pool = np.concatenate(extra, 0)
    return ds_wbf_fuse(pool, np.empty((0, 6), np.float32), w, h)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def load_gt(stem, w, h):
    p = YOLO_ROOT / "labels" / f"{stem}.txt"
    if not p.exists():
        return np.empty((0, 5))
    gts = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in ACTIVE_CLASS_IDS:
            continue
        xc, yc, wn, hn = map(float, parts[1:5])
        gts.append([(xc - wn / 2) * w, (yc - hn / 2) * h, (xc + wn / 2) * w, (yc + hn / 2) * h, cls])
    return np.array(gts, np.float32) if gts else np.empty((0, 5))


def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def evaluate(paths, predict_fn, tag, iou_thr=0.5):
    tp = fp = fn = 0
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p in tqdm(paths, desc=f"val-{tag}"):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        dets = predict_fn(frame, w, h)
        if len(dets):
            dets = dets[np.isin(dets[:, 5].astype(int), ACTIVE_CLASS_IDS)]
        gts = load_gt(p.stem, w, h)
        matched = set()
        for d in dets:
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched or int(d[5]) != int(g[4]):
                    continue
                iou = iou_box(d[:4], g[:4])
                if iou > best:
                    best, bj = iou, j
            if best >= iou_thr and bj >= 0:
                tp += 1
                per[int(d[5])]["tp"] += 1
                matched.add(bj)
            else:
                fp += 1
                per[int(d[5])]["fp"] += 1
        for j, g in enumerate(gts):
            if j not in matched:
                fn += 1
                per[int(g[4])]["fn"] += 1
    P = tp / (tp + fp + 1e-6)
    R = tp / (tp + fn + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)
    Acc = tp / (tp + fp + fn + 1e-6)
    sm = {"P": P, "R": R, "F1": F1, "Acc": Acc, "TP": tp, "FP": fp, "FN": fn}
    rows = []
    for c, v in per.items():
        pc = v["tp"] / (v["tp"] + v["fp"] + 1e-6)
        rc = v["tp"] / (v["tp"] + v["fn"] + 1e-6)
        rows.append({"model": tag, "class": CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c),
                     "P": pc, "R": rc, "F1": 2 * pc * rc / (pc + rc + 1e-6), **v})
    rows.append({"model": tag, "class": "GERAL", **sm, "tp": tp, "fp": fp, "fn": fn})
    return rows, sm


def make_plots(summaries, df_rows):
    tags = list(summaries)
    labels = {"A_YOLOSEA": "A  YOLO-SEA", "B_YoloOW": "B  YoloOW", "Fusao_DSWBF": "Fusão DS-WBF"}
    colors = ["#1f77b4", "#9467bd", "#2ca02c"]
    lab = [labels.get(t, t) for t in tags]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(tags))
    w = 0.18
    for i, m in enumerate(["P", "R", "F1", "Acc"]):
        ax.bar(x + (i - 1.5) * w, [summaries[t][m] for t in tags], w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Detecção IoU≥0.5  ·  YOLO-SEA  |  YoloOW  |  DS-WBF (A↔B + conflito)")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(PLOTS / "bar_PRF1_Acc.png", dpi=170)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(lab, [summaries[t]["F1"] for t in tags], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("F1 — late fusion DS-WBF")
    for i, t in enumerate(tags):
        ax.text(i, summaries[t]["F1"] + 0.02, f"{summaries[t]['F1']:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(PLOTS / "bar_F1.png", dpi=170)
    plt.close()

    df = pd.DataFrame(df_rows)
    df.to_csv(PLOTS / "yolosea_yoloow_metrics.csv", index=False)
    df.to_csv(OUT / "yolosea_yoloow_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.8, 2.4))
    ax.axis("off")
    cell, col = [], ["modelo", "P", "R", "F1", "Acc", "TP", "FP", "FN"]
    for t in tags:
        s = summaries[t]
        cell.append([labels.get(t, t), f"{s['P']:.3f}", f"{s['R']:.3f}", f"{s['F1']:.3f}",
                     f"{s['Acc']:.3f}", int(s["TP"]), int(s["FP"]), int(s["FN"])])
    tbl = ax.table(cellText=cell, colLabels=col, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    for (i, j), c in tbl.get_celld().items():
        if i == 0:
            c.set_facecolor("#1f4e79")
            c.set_text_props(color="white", weight="bold")
        elif i == 3:
            c.set_facecolor("#e8f5e9")
    ax.set_title("YOLO-SEA (Soft-NMS) + YoloOW + DS-WBF  ·  IoU≥0.5", pad=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "tabela_metricas.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("[ok] plots", PLOTS)


# ---------------------------------------------------------------------------
# Tracking + vídeo (bloco da 4K, streaming)
# ---------------------------------------------------------------------------
def id_color(tid):
    rng = np.random.RandomState((int(tid) + 1) * 9973)
    hsv = np.uint8([[[rng.randint(0, 180), 210, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def crop_win(H, W, boxes, pad=80, min_w=1000, min_h=640):
    if boxes is None or len(boxes) == 0:
        cx, cy = W // 2, H // 2
        x1, y1 = max(0, cx - min_w // 2), max(0, cy - min_h // 2)
        return x1, y1, min(W, x1 + min_w), min(H, y1 + min_h)
    b = np.asarray(boxes)
    x1 = int(b[:, 0].min()) - pad
    y1 = int(b[:, 1].min()) - pad
    x2 = int(b[:, 2].max()) + pad
    y2 = int(b[:, 3].max()) + pad
    if x2 - x1 < min_w:
        extra = min_w - (x2 - x1)
        x1 -= extra // 2
        x2 += extra - extra // 2
    if y2 - y1 < min_h:
        extra = min_h - (y2 - y1)
        y1 -= extra // 2
        y2 += extra - extra // 2
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def draw_on_crop(crop, tracks, trails, ox, oy):
    vis = crop
    for tid, pts in trails.items():
        color = id_color(tid)
        pc = [(p[0] - ox, p[1] - oy) for p in pts[-28:]]
        for a, b in zip(pc, pc[1:]):
            cv2.line(vis, a, b, color, 2, cv2.LINE_AA)
    if tracks is None or len(tracks) == 0:
        return vis
    t = np.asarray(tracks).reshape(-1, tracks.shape[-1] if getattr(tracks, "ndim", 1) == 2 else 7)
    for row in t:
        x1, y1, x2, y2 = int(row[0] - ox), int(row[1] - oy), int(row[2] - ox), int(row[3] - oy)
        tid = int(row[4])
        conf = float(row[5]) if row.shape[0] > 5 else 1.0
        cls = int(row[6]) if row.shape[0] > 6 else -1
        color = id_color(tid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else "obj"
        lab = f"ID {tid} {name} {conf:.2f}"
        y0 = max(0, y1 - 16)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(vis, (x1, y0), (x1 + tw + 4, y0 + th + 4), color, -1)
        cv2.putText(vis, lab, (x1 + 2, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def header(text, w, bg=(31, 78, 121)):
    bar = np.full((HDR, w, 3), bg, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(bar, text, ((w - tw) // 2, (HDR + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def run_track_and_video(predict_fns, seqs):
    names = {"A_YOLOSEA": "A YOLO-SEA", "B_YoloOW": "B YoloOW", "Fusao_DSWBF": "Fusao DS-WBF"}
    for vid in SEQ_IDS:
        rows = [r for r in seqs.get(vid, []) if r["split"] == "val"] or seqs.get(vid, [])
        cap = MAX_FRAMES.get(vid)
        if cap:
            rows = rows[:cap]
        if not rows:
            print("[skip] seq", vid)
            continue
        print(f"\n### track seq={vid} frames={len(rows)}  BoT-SORT+CMC per_class")
        trackers = {t: make_tracker() for t in predict_fns}
        mot = {t: [] for t in predict_fns}
        trails = {t: defaultdict(list) for t in predict_fns}
        n_fail = 0
        path_out = VID / f"track_compare_seq{vid}.mp4"
        vw = None
        thumbs = []

        for r in tqdm(rows, desc=f"track-seq{vid}"):
            frame = cv2.imread(str(r["path"]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            fidx = int(r["frame_index"])
            if fidx <= 0:
                fidx = 1
            store = {}
            for tag, fn in predict_fns.items():
                try:
                    dets = fn(frame, w, h)
                except Exception as e:
                    print(f"    [aviso] predict {tag} fidx={fidx}: {e}")
                    dets = np.empty((0, 6), np.float32)
                if len(dets):
                    dets = dets[np.isin(dets[:, 5].astype(int), ACTIVE_CLASS_IDS)]
                    if FOVEA_ENABLE and tag.startswith("Fusao"):
                        try:
                            dets = fovea_refine(fn, frame, dets)
                        except Exception as e:
                            print(f"    [aviso] fovea fidx={fidx}: {e}")
                try:
                    tracks = trackers[tag].update(
                        np.ascontiguousarray(dets_to_boxmot(dets), dtype=np.float64),
                        frame,
                    )
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 8:
                        print(f"    [erro] tracker.update {tag} fidx={fidx}: {type(e).__name__}: {e}")
                    tracks = np.empty((0, 8))
                t = np.asarray(tracks)
                rows_t = []
                if t.size and t.ndim == 1:
                    t = t.reshape(1, -1)
                if t.size and t.ndim == 2 and t.shape[1] >= 6:
                    for rowt in t:
                        x1, y1, x2, y2 = map(float, rowt[:4])
                        tid = int(rowt[4]) if rowt[4] == rowt[4] else -1
                        if tid < 0:
                            continue
                        conf = float(rowt[5])
                        cls = int(rowt[6]) if rowt.shape[0] > 6 else 0
                        rows_t.append([x1, y1, x2, y2, tid, conf, cls])
                        mot[tag].append(f"{fidx},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},-1,-1,-1")
                        trails[tag][tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                store[tag] = np.array(rows_t, np.float32) if rows_t else np.empty((0, 7), np.float32)

            tr_f = store.get("Fusao_DSWBF", np.empty((0, 7)))
            x1, y1, x2, y2 = crop_win(h, w, tr_f[:, :4] if len(tr_f) else None)
            panels = []
            for tag in predict_fns:
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    crop = frame.copy()
                    ox = oy = 0
                else:
                    ox, oy = x1, y1
                vis = draw_on_crop(crop, store[tag] if len(store[tag]) else None, trails[tag], ox, oy)
                panel = cv2.resize(vis, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
                panels.append(np.vstack([header(names[tag], PANEL_W), panel]))
            gap = np.full((PANEL_H + HDR, 6, 3), 255, np.uint8)
            rowv = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
            foot = np.full((24, rowv.shape[1], 3), 255, np.uint8)
            cv2.putText(foot, f"seq {vid}  frame_index={fidx}  BoT-SORT+CMC  per_class  DS-WBF A<->B",
                        (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
            canvas = np.vstack([rowv, foot])
            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(path_out), fourcc, FPS, (canvas.shape[1], canvas.shape[0]))
                if not vw.isOpened():
                    path_out = path_out.with_suffix(".avi")
                    vw = cv2.VideoWriter(str(path_out), cv2.VideoWriter_fourcc(*"MJPG"), FPS,
                                         (canvas.shape[1], canvas.shape[0]))
            vw.write(canvas)
            if len(thumbs) < 5:
                thumbs.append(cv2.resize(canvas, (960, 220)))

        if vw is not None:
            vw.release()
            print(f"  [ok] video {path_out}  ({path_out.stat().st_size/1e6:.1f} MB)  update_fail={n_fail}")
        if thumbs:
            cv2.imwrite(str(PLOTS / f"strip_seq{vid}.png"), np.vstack(thumbs))
        for tag in predict_fns:
            d = HOTA / f"pred_{tag}"
            d.mkdir(exist_ok=True)
            (d / f"{vid}.txt").write_text("\n".join(mot[tag]) + ("\n" if mot[tag] else ""))
            print(f"  [{tag}] seq={vid} mot_lines={len(mot[tag])}")
            del trackers[tag]
        free_gpu()


def main():
    print("=" * 64)
    print("YOLO-SEA (Soft-NMS) + YoloOW + DS-WBF + BoT-SORT/CMC")
    print("=" * 64)
    model_a, model_b, device = load_models()
    pred_b = make_yoloow_predict(model_b, device)

    def pred_A(frame, w, h):
        return pred_yolosea(model_a, frame, w, h)

    def pred_B(frame, w, h):
        return pred_b(frame, conf=CONF_B)

    def pred_F(frame, w, h):
        return ds_wbf_fuse(pred_A(frame, w, h), pred_B(frame, w, h), w, h)

    predict_fns = {"A_YOLOSEA": pred_A, "B_YoloOW": pred_B, "Fusao_DSWBF": pred_F}

    img_dir = YOLO_ROOT / "images"
    val_imgs = sorted([p for p in img_dir.glob("val_*.jpg")
                       if (YOLO_ROOT / "labels" / f"{p.stem}.txt").exists()])[:MAX_VAL]
    print(f"[val] n={len(val_imgs)}")
    all_rows, summaries = [], {}
    for tag, fn in predict_fns.items():
        rows, sm = evaluate(val_imgs, fn, tag)
        all_rows.extend(rows)
        summaries[tag] = sm
        print(f"  [{tag:14s}] P={sm['P']:.3f} R={sm['R']:.3f} F1={sm['F1']:.3f} Acc={sm['Acc']:.3f}")
        free_gpu()
    with open(OUT / "metrics_val.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in summaries.items()}, f, indent=2)
    make_plots(summaries, all_rows)

    seqs = build_sequences()
    run_track_and_video(predict_fns, seqs)
    print("\n[done] ", OUT)
    for p in sorted(list(PLOTS.glob("*")) + list(VID.glob("*"))):
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size/1e3:.0f} kB")


if __name__ == "__main__":
    main()
