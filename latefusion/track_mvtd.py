#!/usr/bin/env python3
"""Tracking BoT-SORT + vídeos no MVTD (saida em runs/mvtd/videos/).

A = YOLOv8s MVTD + Soft-NMS. B = YoloOW (4 classes se B_YoloOW_mvtd.pt existir,
senão boat-only do peso Sea). Fusão DS-WBF. Tracker: BoT-SORT per_class,
frame 1-based na ordem do nome do jpg (não enumerate).
Sem rastro. Crop do bloco de ação. MOT + mp4.
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

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import MVTD_YOLO, OUT, setup_sys_path  # noqa: E402

setup_sys_path()

from latefusion import ds_wbf_fuse  # noqa: E402
from run_mvtd import (  # noqa: E402
    CLASS_NAMES,
    load_models,
    make_pred_b,
    pred_a,
)

YOLO_ROOT = MVTD_YOLO
TRK = OUT / "mvtd" / "tracking"
VID = OUT / "mvtd" / "videos"
MET = OUT / "mvtd" / "metricas"
PLOT = OUT / "mvtd" / "plots_artigo"
FPS = 10
OUT_W, OUT_H = 1280, 720
FRACS = (0.0, 0.25, 0.50, 0.75, 1.0)


def id_color(tid: int):
    rng = np.random.RandomState((int(tid) + 1) * 9973)
    hsv = np.uint8([[[rng.randint(0, 180), 210, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def crop_win(h, w, boxes, pad=80, min_w=1100, min_h=700):
    if boxes is None or len(boxes) == 0:
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - min_w // 2), max(0, cy - min_h // 2)
        return np.array([x1, y1, min(w, x1 + min_w), min(h, y1 + min_h)], np.int32)
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
    return np.array([max(0, x1), max(0, y1), min(w, x2), min(h, y2)], np.int32)


def seqs_in_split(split: str) -> dict[str, list[Path]]:
    by = defaultdict(list)
    d = YOLO_ROOT / "images" / split
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        seq = p.stem.rsplit("_", 1)[0]
        by[seq].append(p)
    for seq in by:
        by[seq].sort(key=lambda x: x.stem)
    return dict(by)


def make_tracker():
    from boxmot.trackers.registry import TRACKER_DEFINITIONS, create_tracker

    kind = "botsort" if "botsort" in TRACKER_DEFINITIONS else "ocsort"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tracker = create_tracker(
        tracker_type=kind,
        device=device,
        half=False,
        per_class=True,
        tracker_backend="python",
    )
    for attr, val in [("det_thresh", 0.25), ("det_threshold", 0.25), ("per_class", True)]:
        if hasattr(tracker, attr):
            try:
                setattr(tracker, attr, val)
            except Exception:
                pass
    return tracker


def dets_to_boxmot(dets):
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), np.float32)
    d = np.asarray(dets, np.float32)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.shape[1] > 6:
        d = d[:, :6]
    elif d.shape[1] < 6:
        d = np.concatenate([d, np.zeros((d.shape[0], 6 - d.shape[1]), np.float32)], 1)
    return d


def track_seq(seq: str, frames: list[Path], pred_F, tracker) -> tuple[list[str], int]:
    lines = []
    n_fail = 0
    for i, p in enumerate(frames, 1):  # MOT frame 1-based = ordem do jpg
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        dets = pred_F(frame, w, h)
        try:
            tracks = tracker.update(dets_to_boxmot(dets), frame)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"    [erro] {seq} f={i}: {type(e).__name__}: {e}")
            tracks = np.empty((0, 8))
        t = np.asarray(tracks)
        if t.size:
            if t.ndim == 1:
                t = t.reshape(1, -1)
            if t.shape[1] >= 6:
                for row in t:
                    x1, y1, x2, y2 = map(float, row[:4])
                    tid = int(row[4]) if row[4] == row[4] else -1
                    if tid < 0:
                        continue
                    conf = float(row[5])
                    lines.append(
                        f"{i},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1"
                    )
    return lines, n_fail


def render_video(seq: str, frames: list[Path], mot_lines: list[str], out_mp4: Path | None, method: str):
    by = defaultdict(list)
    for line in mot_lines:
        a = line.split(",")
        fidx = int(float(a[0]))
        tid = int(float(a[1]))
        x, y, bw, bh = map(float, a[2:6])
        by[fidx].append([x, y, x + bw, y + bh, tid])
    n = len(frames)
    mid = max(1, n // 2)
    seed = by.get(mid) or next((by[k] for k in sorted(by) if by[k]), None)
    PLOT.mkdir(parents=True, exist_ok=True)
    want = {max(1, min(n, int(round(f * (n - 1)) + 1))) for f in FRACS}
    part = out_mp4.with_suffix(".part.mp4") if out_mp4 else None
    vw = None
    crop_xy = None
    for i, p in enumerate(tqdm(frames, desc=f"video {seq}"), 1):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = np.array(by.get(i, []), np.float32)
        if crop_xy is None:
            crop_xy = crop_win(h, w, boxes[:, :4] if len(boxes) else seed)
        x1, y1, x2, y2 = crop_xy
        vis = im.copy()
        for row in boxes:
            xa, ya, xb, yb = map(int, row[:4])
            tid = int(row[4])
            col = id_color(tid)
            cv2.rectangle(vis, (xa, ya), (xb, yb), col, 2)
            cv2.putText(
                vis, f"ID {tid}", (xa, max(16, ya - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
            )
        crop = vis[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (OUT_W, OUT_H))
        if i in want:
            pct = int(round(100 * (i - 1) / max(1, n - 1)))
            jpg = PLOT / f"{seq}_{method}_t{pct:02d}.jpg"
            cv2.imwrite(str(jpg), crop)
        if out_mp4 is None:
            continue
        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(part), fourcc, FPS, (OUT_W, OUT_H))
        vw.write(crop)
    if vw is not None:
        vw.release()
        part.replace(out_mp4)
        print(f"[ok] {out_mp4}")
    print(f"[frames] {PLOT}/{seq}_{method}_tXX.jpg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("val", "test", "both"))
    ap.add_argument("--seq", nargs="*", default=None, help="ex: 1-Boat 2-ship 9-SailBoat")
    ap.add_argument("--max-seq", type=int, default=6, help="0 = todas as seq. do split")
    ap.add_argument("--method", choices=("fusao", "A", "B"), default="fusao")
    ap.add_argument("--a-only", action="store_true", help="alias de --method A")
    ap.add_argument("--no-video", action="store_true", help="so MOT + frames t00..t100")
    args = ap.parse_args()
    if args.a_only:
        args.method = "A"
    TRK.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)

    model_a, model_b, device = load_models()
    fn_b = make_pred_b(model_b, device)

    def pred_F(frame, w, h):
        da = pred_a(model_a, frame)
        if args.method == "A":
            return da
        db = fn_b(frame)
        if args.method == "B":
            return db
        return ds_wbf_fuse(da, db, w, h)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    counts = {}
    for sp in splits:
        by = seqs_in_split(sp)
        names = args.seq if args.seq else sorted(by, key=lambda s: -len(by[s]))
        n_done = 0
        for seq in names:
            if seq not in by:
                print(f"[skip] {seq} nao esta em {sp}")
                continue
            if args.seq is None and args.max_seq > 0 and n_done >= args.max_seq:
                break
            frames = by[seq]
            print(f"\n=== {sp}/{seq} method={args.method} frames={len(frames)} ===")
            tracker = make_tracker()
            lines, n_fail = track_seq(seq, frames, pred_F, tracker)
            mot = TRK / f"{seq}_{args.method}.txt"
            mot.write_text("\n".join(lines) + ("\n" if lines else ""))
            counts[f"{sp}/{seq}/{args.method}"] = {
                "split": sp, "frames": len(frames), "mot": len(lines), "fail": n_fail, "method": args.method,
            }
            print(f"    mot={len(lines)} fail={n_fail} -> {mot.name}")
            mp4 = None if args.no_video else VID / f"track_{args.method}_{seq}.mp4"
            render_video(seq, frames, lines, mp4, args.method)
            del tracker
            n_done += 1
            torch.cuda.empty_cache()
    (MET / "tracking_counts.json").write_text(json.dumps(counts, indent=2))
    print("[done] MOT", TRK, "| videos", VID, "| frames", PLOT)


if __name__ == "__main__":
    main()
