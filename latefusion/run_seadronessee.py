#!/usr/bin/env python3
"""Validação + teste + K-Fold temporal + tracking da LATE FUSION apenas.

Dataset: seadronessee_15gb (val+test+train, todos os frames, ordem frame_index).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import (  # noqa: E402
    COCO_ANN,
    OUT,
    PESOS_OUT,
    SEADRONESSEE_15GB,
    WEIGHTS,
    setup_sys_path,
)

setup_sys_path()

from latefusion import ds_wbf_fuse, soft_nms  # noqa: E402
from tracking_sahi_paper_blocks import (  # noqa: E402
    ACTIVE_CLASS_IDS,
    CLASS_NAMES,
    CONF_A,
    CONF_B,
    IMGSZ,
    dets_to_boxmot,
    free_gpu,
    load_models,
    make_tracker,
    make_yoloow_predict,
)
from yolosea_yoloow_dswbf import pred_yolosea  # noqa: E402

DATA = SEADRONESSEE_15GB
MET = OUT / "seadronessee" / "metricas"
PLOT = OUT / "seadronessee" / "plots_artigo"
TRK = OUT / "seadronessee" / "tracking"
PESOS = PESOS_OUT
KFOLD_W = PESOS / "kfold"
for d in (MET, PLOT, TRK, KFOLD_W):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
})


def load_gt(stem, w, h):
    p = DATA / "labels" / f"{stem}.txt"
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


def eval_paths(paths, pred_F, tag):
    tp = fp = fn = 0
    for p in tqdm(paths, desc=tag):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        dets = pred_F(frame, w, h)
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
            if best >= 0.5 and bj >= 0:
                tp += 1
                matched.add(bj)
            else:
                fp += 1
        fn += len(gts) - len(matched)
    P = tp / (tp + fp + 1e-6)
    R = tp / (tp + fn + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)
    Acc = tp / (tp + fp + fn + 1e-6)
    sm = {"split": tag, "P": P, "R": R, "F1": F1, "Acc": Acc, "TP": int(tp), "FP": int(fp), "FN": int(fn), "n": len(paths)}
    print(f"  FUSAO [{tag}] n={len(paths)} P={P:.3f} R={R:.3f} F1={F1:.3f} Acc={Acc:.3f} TP={tp} FP={fp} FN={fn}")
    return sm


def stem_to_video():
    m = {}
    for name in ("instances_train_objects_in_water.json",
                 "instances_val_objects_in_water.json",
                 "instances_test_objects_in_water.json"):
        split = "train" if "train" in name else ("val" if "val" in name else "test")
        coco = json.loads((COCO_ANN / name).read_text())
        for im in coco["images"]:
            m[f"{split}_{Path(im['file_name']).stem}"] = {
                "video_id": int(im["video_id"]),
                "frame_index": int(im.get("frame_index") or 0),
            }
    return m


def list_split(prefix):
    img = DATA / "images"
    return sorted(p for p in img.glob(f"{prefix}_*.jpg") if (DATA / "labels" / f"{p.stem}.txt").exists())


def id_color(tid):
    rng = np.random.RandomState((int(tid) + 1) * 9973)
    hsv = np.uint8([[[rng.randint(0, 180), 210, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def crop_win(H, W, boxes, pad=70, min_w=900, min_h=560):
    if boxes is None or len(boxes) == 0:
        cx, cy = W // 2, H // 2
        x1, y1 = max(0, cx - min_w // 2), max(0, cy - min_h // 2)
        return x1, y1, min(W, x1 + min_w), min(H, y1 + min_h)
    b = np.asarray(boxes)
    x1, y1 = int(b[:, 0].min()) - pad, int(b[:, 1].min()) - pad
    x2, y2 = int(b[:, 2].max()) + pad, int(b[:, 3].max()) + pad
    if x2 - x1 < min_w:
        e = min_w - (x2 - x1)
        x1 -= e // 2
        x2 += e - e // 2
    if y2 - y1 < min_h:
        e = min_h - (y2 - y1)
        y1 -= e // 2
        y2 += e - e // 2
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def track_sequences(pred_F, meta, prefixes=("val", "test")):
    """Tracking completo val+test, ordem temporal por video_id / frame_index."""
    seqs = defaultdict(list)
    for pref in prefixes:
        for p in list_split(pref):
            info = meta.get(p.stem)
            if not info:
                continue
            seqs[info["video_id"]].append({"path": p, "frame_index": info["frame_index"], "stem": p.stem})
    counts = {}
    paper_panels = []  # (seq, t_frac, crop_bgr)
    for vid, rows in sorted(seqs.items()):
        rows = sorted(rows, key=lambda r: r["frame_index"])
        tracker = make_tracker()
        lines = []
        n_fail = 0
        want = set(int(x) for x in np.linspace(0, max(0, len(rows) - 1), 5))
        print(f"  track vid={vid} frames={len(rows)} split={prefixes}")
        for i, r in enumerate(tqdm(rows, desc=f"trk-{vid}", leave=False)):
            frame = cv2.imread(str(r["path"]))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            fidx = int(r["frame_index"]) if r["frame_index"] > 0 else i + 1
            try:
                dets = pred_F(frame, w, h)
                if len(dets):
                    dets = dets[np.isin(dets[:, 5].astype(int), ACTIVE_CLASS_IDS)]
                tracks = tracker.update(np.ascontiguousarray(dets_to_boxmot(dets), np.float64), frame)
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    [erro] vid={vid} fidx={fidx}: {type(e).__name__}: {e}")
                tracks = np.empty((0, 8))
            t = np.asarray(tracks)
            boxes = []
            if t.size:
                if t.ndim == 1:
                    t = t.reshape(1, -1)
                if t.shape[1] >= 6:
                    for rowt in t:
                        x1, y1, x2, y2 = map(float, rowt[:4])
                        tid = int(rowt[4]) if rowt[4] == rowt[4] else -1
                        if tid < 0:
                            continue
                        conf = float(rowt[5])
                        lines.append(f"{fidx},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},-1,-1,-1")
                        boxes.append([x1, y1, x2, y2, tid, conf])
            if i in want and len(paper_panels) < 40:
                vis = frame.copy()
                arr = np.array(boxes, np.float32) if boxes else np.empty((0, 6))
                for rowt in arr:
                    xa, ya, xb, yb = map(int, rowt[:4])
                    col = id_color(int(rowt[4]))
                    cv2.rectangle(vis, (xa, ya), (xb, yb), col, 2)
                    cv2.putText(vis, f"ID {int(rowt[4])}", (xa, max(16, ya - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                x1, y1, x2, y2 = crop_win(h, w, arr[:, :4] if len(arr) else None)
                crop = vis[y1:y2, x1:x2]
                if crop.size:
                    paper_panels.append((vid, i / max(1, len(rows) - 1), cv2.resize(crop, (480, 320))))
        outp = TRK / f"{vid}.txt"
        outp.write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[int(vid)] = {"frames": len(rows), "mot": len(lines), "fail": n_fail}
        print(f"    vid={vid} mot={len(lines)} fail={n_fail}")
        del tracker
        free_gpu()
    (MET / "tracking_counts.json").write_text(json.dumps(counts, indent=2))
    return paper_panels


def save_paper_grid(panels):
    if not panels:
        return
    # grid: até 4 colunas (posições no vídeo), linhas = sequências
    by = defaultdict(list)
    for vid, frac, im in panels:
        by[vid].append((frac, im))
    vids = list(by.keys())[:6]
    cols = 5
    pw, ph = 480, 320
    gap = 6
    canvas_w = cols * pw + (cols + 1) * gap
    canvas_h = len(vids) * ph + (len(vids) + 1) * gap + 36
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    cv2.putText(canvas, "Late fusion DS-WBF + BoT-SORT  |  blocos temporais (inicio -> fim)",
                (gap, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    for ri, vid in enumerate(vids):
        items = sorted(by[vid], key=lambda x: x[0])[:cols]
        y = 36 + gap + ri * (ph + gap)
        for ci, (frac, im) in enumerate(items):
            x = gap + ci * (pw + gap)
            canvas[y:y + ph, x:x + pw] = im
            cv2.putText(canvas, f"seq {vid}  t={frac:.2f}", (x + 8, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    path = PLOT / "fig_temporal_blocks.png"
    cv2.imwrite(str(path), canvas)
    print("[ok]", path)
    # também painéis individuais
    for i, (vid, frac, im) in enumerate(panels[:24]):
        cv2.imwrite(str(PLOT / f"seq{vid}_t{int(frac*100):02d}.jpg"), im)


def make_metric_plots(rows):
    df = pd.DataFrame(rows)
    df.to_csv(MET / "latefusion_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labs = [r["split"] for r in rows]
    x = np.arange(len(labs))
    w = 0.18
    for i, m in enumerate(["P", "R", "F1", "Acc"]):
        ax.bar(x + (i - 1.5) * w, [r[m] for r in rows], w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("Late fusion DS-WBF — somente fusão (IoU≥0.5)")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(PLOT / "bar_latefusion_PRF1.png", dpi=170)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 0.55 * (len(rows) + 3)))
    ax.axis("off")
    cell = [[r["split"], f"{r['P']:.3f}", f"{r['R']:.3f}", f"{r['F1']:.3f}",
             f"{r['Acc']:.3f}", r["TP"], r["FP"], r["FN"], r.get("n", "")] for r in rows]
    tbl = ax.table(cellText=cell, colLabels=["split", "P", "R", "F1", "Acc", "TP", "FP", "FN", "n"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for (i, j), c in tbl.get_celld().items():
        if i == 0:
            c.set_facecolor("#1f4e79")
            c.set_text_props(color="white", weight="bold")
    ax.set_title("Late fusion DS-WBF — validação / teste / K-Fold temporal")
    fig.tight_layout()
    fig.savefig(PLOT / "tabela_latefusion.png", dpi=170, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 64)
    print("yolo-git  ·  LATE FUSION only  ·  val+test+kfold temporal")
    print("=" * 64)
    PESOS.mkdir(parents=True, exist_ok=True)
    src_a = WEIGHTS / "yolov8s_seadronessee.pt"
    src_b = WEIGHTS / "YoloOW.pt"
    if src_a.is_file():
        shutil.copy2(src_a, PESOS / "A_yolov8s_seadronessee.pt")
    if src_b.is_file():
        shutil.copy2(src_b, PESOS / "B_YoloOW.pt")
    src_f1 = _REPO.parent / "grok_yolo_blue_eyes" / "results" / "fold_1_best.pt"
    if not src_f1.is_file():
        src_f1 = WEIGHTS.parent / "grok_yolo_blue_eyes" / "results" / "fold_1_best.pt"
    if src_f1.is_file():
        shutil.copy2(src_f1, KFOLD_W / "fold_1_best.pt")

    model_a, model_b, device = load_models()
    pred_b = make_yoloow_predict(model_b, device)

    def pred_F(frame, w, h):
        da = pred_yolosea(model_a, frame, w, h)
        db = pred_b(frame, conf=CONF_B)
        return ds_wbf_fuse(da, db, w, h)

    dummy = np.zeros((640, 640, 3), np.uint8)
    _ = pred_F(dummy, 640, 640)
    print("[GATE] late fusion pronta")

    val_p = list_split("val")
    test_p = list_split("test")
    train_p = list_split("train")
    print(f"val={len(val_p)} test={len(test_p)} train={len(train_p)}")

    rows = []
    rows.append(eval_paths(val_p, pred_F, "val_completo"))
    free_gpu()
    rows.append(eval_paths(test_p, pred_F, "test_completo"))
    free_gpu()

    # K-Fold temporal: GroupKFold por video_id em train+val, TODOS os frames
    meta = stem_to_video()
    pool = train_p + val_p
    groups, keep = [], []
    for p in pool:
        info = meta.get(p.stem)
        if info is None:
            continue
        keep.append(p)
        groups.append(info["video_id"])
    print(f"[kfold] n={len(keep)} videos={len(set(groups))}")
    gkf = GroupKFold(n_splits=5)
    kf_rows = []
    for fi, (_, te) in enumerate(gkf.split(keep, groups=groups), 1):
        te_paths = [keep[i] for i in te]
        te_vids = sorted({groups[i] for i in te})
        sm = eval_paths(te_paths, pred_F, f"kfold_f{fi}")
        sm["val_videos"] = te_vids
        kf_rows.append(sm)
        rows.append(sm)
        free_gpu()
    (MET / "kfold_latefusion.json").write_text(json.dumps(kf_rows, indent=2, default=str))
    f1s = [r["F1"] for r in kf_rows]
    summary_kf = {"F1_mean": float(np.mean(f1s)), "F1_std": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0),
                  "folds": kf_rows}
    (MET / "kfold_summary.json").write_text(json.dumps(summary_kf, indent=2, default=str))
    print(f"K-FOLD F1 = {summary_kf['F1_mean']:.3f} ± {summary_kf['F1_std']:.3f}")

    (MET / "latefusion_all.json").write_text(json.dumps(rows, indent=2, default=str))
    make_metric_plots(rows)

    print("\n=== TRACKING val+test temporal ===")
    panels = track_sequences(pred_F, meta, prefixes=("val", "test"))
    save_paper_grid(panels)

    md = ["# Métricas late fusion (só fusão)\n\n",
          f"Dataset 15GB: val={len(val_p)} test={len(test_p)} train={len(train_p)}\n\n",
          "| split | P | R | F1 | Acc | TP | FP | FN | n |\n|---|---|---|---|---|---|---|---|---|\n"]
    for r in rows:
        md.append(f"| {r['split']} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} | {r['Acc']:.3f} | {r['TP']} | {r['FP']} | {r['FN']} | {r.get('n','')} |\n")
    md.append(f"\nK-Fold 5 temporal (GroupKFold video) F1 = {summary_kf['F1_mean']:.3f} ± {summary_kf['F1_std']:.3f}\n")
    (MET / "NOTAS.md").write_text("".join(md))
    print("[done]", OUT / "seadronessee")


if __name__ == "__main__":
    main()
