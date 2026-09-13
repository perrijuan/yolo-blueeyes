#!/usr/bin/env python3
"""Renderiza vídeos LONGOS de tracking a partir dos MOT já gerados (val+test).

Não re-roda o detector. Recorta o bloco de ação da 4K, desenha boxes + IDs
(sem rastro: no mar o trail polui a cena), grava 10 fps em runs/videos/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import COCO_ANN, OUT as RUNS_OUT, SEADRONESSEE_15GB  # noqa: E402

DATA = SEADRONESSEE_15GB
COCO = COCO_ANN
MOT = RUNS_OUT / "seadronessee" / "tracking"
OUT = RUNS_OUT / "seadronessee" / "videos"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 10
W, H = 1280, 720


def stem_meta():
    m = {}
    for name in ("instances_train_objects_in_water.json",
                 "instances_val_objects_in_water.json",
                 "instances_test_objects_in_water.json"):
        split = "train" if "train" in name else ("val" if "val" in name else "test")
        coco = json.loads((COCO / name).read_text())
        for im in coco["images"]:
            m[f"{split}_{Path(im['file_name']).stem}"] = (
                int(im["video_id"]), int(im.get("frame_index") or 0)
            )
    return m


def seq_frames(vid, meta):
    rows = []
    img = DATA / "images"
    for p in img.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        pref = p.name.split("_")[0]
        if pref not in ("val", "test"):
            continue
        info = meta.get(p.stem)
        if not info or info[0] != vid:
            continue
        rows.append((info[1], p))
    rows.sort(key=lambda x: x[0])
    return rows


def load_mot(path):
    by = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        a = line.split(",")
        fidx = int(float(a[0]))
        tid = int(float(a[1]))
        x, y, bw, bh = map(float, a[2:6])
        conf = float(a[6]) if len(a) > 6 else 1.0
        by[fidx].append([x, y, x + bw, y + bh, tid, conf])
    return {k: np.array(v, np.float32) for k, v in by.items()}


def id_color(tid):
    rng = np.random.RandomState((int(tid) + 1) * 9973)
    hsv = np.uint8([[[rng.randint(0, 180), 210, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def crop_win(h, w, boxes, pad=80, min_w=1100, min_h=700):
    if boxes is None or len(boxes) == 0:
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - min_w // 2), max(0, cy - min_h // 2)
        return np.array([x1, y1, min(w, x1 + min_w), min(h, y1 + min_h)], np.float32)
    b = np.asarray(boxes)
    x1 = float(b[:, 0].min()) - pad
    y1 = float(b[:, 1].min()) - pad
    x2 = float(b[:, 2].max()) + pad
    y2 = float(b[:, 3].max()) + pad
    if x2 - x1 < min_w:
        e = min_w - (x2 - x1)
        x1 -= e / 2
        x2 += e / 2
    if y2 - y1 < min_h:
        e = min_h - (y2 - y1)
        y1 -= e / 2
        y2 += e / 2
    return np.array([max(0, x1), max(0, y1), min(w, x2), min(h, y2)], np.float32)


def render(vid, meta):
    motp = MOT / f"{vid}.txt"
    if not motp.is_file():
        print("[skip] sem MOT", vid)
        return
    frames = seq_frames(vid, meta)
    mot = load_mot(motp)
    if not frames:
        print("[skip] sem frames val/test", vid)
        return
    out = OUT / f"track_fusao_seq{vid}.mp4"
    part = out.with_suffix(".part.mp4")
    vw = None
    ema = None
    print(f"[video] seq={vid} frames={len(frames)} mot_keys={len(mot)} -> {out.name}")
    for fidx, path in tqdm(frames, desc=f"vid{vid}"):
        im = cv2.imread(str(path))
        if im is None:
            continue
        h, w = im.shape[:2]
        tr = mot.get(int(fidx), np.empty((0, 6)))
        win = crop_win(h, w, tr[:, :4] if len(tr) else None)
        ema = win if ema is None else 0.35 * win + 0.65 * ema
        x1, y1, x2, y2 = map(int, ema)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = im[y1:y2, x1:x2].copy()
        if crop.size == 0:
            crop = im.copy()
            x1 = y1 = 0
        for row in tr:
            tid = int(row[4])
            col = id_color(tid)
            xa, ya, xb, yb = int(row[0] - x1), int(row[1] - y1), int(row[2] - x1), int(row[3] - y1)
            cv2.rectangle(crop, (xa, ya), (xb, yb), col, 2)
            lab = f"ID {tid} {float(row[5]):.2f}"
            y0 = max(0, ya - 16)
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(crop, (xa, y0), (xa + tw + 6, y0 + th + 6), col, -1)
            cv2.putText(crop, lab, (xa + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panel = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
        bar = np.full((32, W, 3), (31, 78, 121), np.uint8)
        cv2.putText(bar, f"Late fusion DS-WBF + BoT-SORT/CMC   seq {vid}   frame_index={fidx}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.vstack([bar, panel])
        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(part), fourcc, FPS, (canvas.shape[1], canvas.shape[0]))
            if not vw.isOpened():
                part = out.with_suffix(".part.avi")
                vw = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"MJPG"), FPS,
                                     (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
    if vw:
        vw.release()
        final = out.with_suffix(part.suffix.replace(".part", "")) if part.suffix != ".mp4" else out
        if part.suffix == ".avi":
            final = out.with_suffix(".avi")
        part.replace(final)
        print(f"  [ok] {final}  {final.stat().st_size/1e6:.1f} MB")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", type=int, default=0)
    args = ap.parse_args()
    meta = stem_meta()
    vids = [args.vid] if args.vid else sorted(int(p.stem) for p in MOT.glob("*.txt"))
    print("seqs", vids)
    for vid in vids:
        render(vid, meta)
    print("[done]", OUT)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name:32s} {p.stat().st_size/1e6:6.1f} MB")


if __name__ == "__main__":
    main()
