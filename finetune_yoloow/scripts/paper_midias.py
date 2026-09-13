#!/usr/bin/env python3
"""Stills + vídeos A / B / fusão para o artigo (val e test MVTD).

Mesmo crop, sem rastro na água. Sai em finetuningmvtd/paper_compare/.
Não treina. Usa o B copiado (4-cls 2GB) + A YOLOv8s MVTD + DS-WBF.
"""
from __future__ import annotations

import argparse
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
PAPER = FT / "paper_compare"
YOLO = MVTD_YOLO

from latefusion import ds_wbf_fuse  # noqa: E402
from run_mvtd import load_models as load_ab, make_pred_b, pred_a  # noqa: E402
import run_mvtd as mv  # noqa: E402

FRACS = (0.0, 0.25, 0.50, 0.75, 1.0)
FPS = 10
OUT_W, OUT_H = 1280, 720
COLORS = {0: (60, 180, 75), 1: (0, 165, 255), 2: (255, 180, 0), 3: (180, 70, 220)}


def cls_of(seq: str) -> str:
    s = seq.lower()
    if "sail" in s:
        return "sailboat"
    if "ship" in s:
        return "ship"
    if "usv" in s:
        return "usv"
    return "boat"


def seqs_in(split: str) -> dict[str, list[Path]]:
    by = defaultdict(list)
    d = YOLO / "images" / split
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            by[p.stem.rsplit("_", 1)[0]].append(p)
    for s in by:
        by[s].sort(key=lambda x: x.stem)
    return dict(by)


def pick_seqs(split: str, k: int) -> list[str]:
    by = seqs_in(split)
    want = ["sailboat", "ship", "boat", "usv"]
    chosen, used = [], set()
    for w in want:
        cands = sorted((s for s in by if cls_of(s) == w), key=lambda s: -len(by[s]))
        if cands:
            chosen.append(cands[0])
            used.add(cands[0])
        if len(chosen) >= k:
            return chosen[:k]
    rest = sorted((s for s in by if s not in used), key=lambda s: -len(by[s]))
    for s in rest:
        if len(chosen) >= k:
            break
        chosen.append(s)
    return chosen[:k]


def crop_win(h, w, boxes, pad=80, min_w=1100, min_h=700):
    if boxes is None or len(boxes) == 0:
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - min_w // 2), max(0, cy - min_h // 2)
        return int(x1), int(y1), int(min(w, x1 + min_w)), int(min(h, y1 + min_h))
    b = np.asarray(boxes)
    x1, y1 = float(b[:, 0].min()) - pad, float(b[:, 1].min()) - pad
    x2, y2 = float(b[:, 2].max()) + pad, float(b[:, 3].max()) + pad
    if x2 - x1 < min_w:
        e = min_w - (x2 - x1)
        x1 -= e / 2
        x2 += e / 2
    if y2 - y1 < min_h:
        e = min_h - (y2 - y1)
        y1 -= e / 2
        y2 += e / 2
    return int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))


def draw(crop, dets, ox, oy):
    vis = crop.copy()
    if dets is None or len(dets) == 0:
        return vis
    for row in np.asarray(dets):
        x1, y1, x2, y2 = [int(v) for v in row[:4]]
        cls = int(row[5]) if row.shape[0] > 5 else 0
        conf = float(row[4]) if row.shape[0] > 4 else 0
        c = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(vis, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), c, 2)
        cv2.putText(vis, f"{cls} {conf:.2f}", (x1 - ox, max(14, y1 - oy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return vis


def header(text, w, h=28, bg=(31, 78, 121)):
    bar = np.full((h, w, 3), bg, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def save_strip(frame, dets_map, path: Path):
    h, w = frame.shape[:2]
    boxes = []
    for d in dets_map.values():
        if d is not None and len(d):
            boxes.append(np.asarray(d)[:, :4])
    union = np.concatenate(boxes, 0) if boxes else None
    x1, y1, x2, y2 = crop_win(h, w, union)
    names = [("A", "A YOLOv8s-MVTD"), ("B", "B YoloOW-2GB"), ("F", "Fusao DS-WBF")]
    pw, ph = 640, 360
    panels = []
    for k, title in names:
        vis = draw(frame[y1:y2, x1:x2], dets_map.get(k), x1, y1)
        vis = cv2.resize(vis, (pw, ph), interpolation=cv2.INTER_AREA)
        panels.append(np.vstack([header(title, pw), vis]))
    gap = np.full((ph + 28, 6, 3), 255, np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, gap, p])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), row)


def write_mp4(path: Path, frames, fps=FPS):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    tmp = path.with_suffix(".part.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        tmp = path.with_suffix(".part.avi")
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for fr in frames:
        if fr.shape[1] != w or fr.shape[0] != h:
            fr = cv2.resize(fr, (w, h))
        vw.write(fr)
    vw.release()
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=FT / "pesos" / "B_YoloOW_mvtd_2gb.pt")
    ap.add_argument("--max-seq-val", type=int, default=2)
    ap.add_argument("--max-seq-test", type=int, default=2)
    ap.add_argument("--max-video-frames", type=int, default=80)
    args = ap.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"peso nao encontrado: {args.weights}")

    still_dir = PAPER / "stills"
    vid_dir = PAPER / "videos"
    still_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    model_a, _old, device = load_ab()
    mv.B_CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    ckpt = torch.load(str(args.weights), map_location=device, weights_only=False)
    model_b = ckpt.get("ema") or ckpt.get("model") or ckpt if isinstance(ckpt, dict) else ckpt
    if hasattr(model_b, "float"):
        model_b = model_b.float()
    model_b.to(device).eval()
    fn_b = make_pred_b(model_b, device)

    def predA(frame):
        return pred_a(model_a, frame)

    def predB(frame):
        return fn_b(frame)

    def predF(frame):
        h, w = frame.shape[:2]
        return ds_wbf_fuse(predA(frame), predB(frame), w, h)

    jobs = []
    for split, k in (("val", args.max_seq_val), ("test", args.max_seq_test)):
        for seq in pick_seqs(split, k):
            jobs.append((split, seq))

    for split, seq in jobs:
        frames = seqs_in(split)[seq]
        n = len(frames)
        idxs = sorted({min(n - 1, max(0, int(round(f * (n - 1))))) for f in FRACS})
        print(f"[paper] {split}/{seq} frames={n} stills={idxs}")
        # stills + strip (same crop from fusion boxes)
        for i in idxs:
            im = cv2.imread(str(frames[i]))
            if im is None:
                continue
            h, w = im.shape[:2]
            da, db, df = predA(im), predB(im), predF(im)
            dets = {"A": da, "B": db, "F": df}
            frac = int(round(100 * i / max(1, n - 1)))
            union = df[:, :4] if len(df) else (da[:, :4] if len(da) else None)
            x1, y1, x2, y2 = crop_win(h, w, union)
            for tag, d in (("A", da), ("B", db), ("fusao", df)):
                vis = draw(im[y1:y2, x1:x2], d, x1, y1)
                vis = cv2.resize(vis, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(still_dir / f"{seq}_{tag}_t{frac:02d}.jpg"), vis)
            save_strip(im, dets, still_dir / f"{seq}_ABF_t{frac:02d}.jpg")

        # short video per method, same crop (fusion union of first frame with dets)
        cap = min(n, args.max_video_frames)
        stride = max(1, n // cap)
        sub = frames[::stride][:cap]
        buf = {"A": [], "B": [], "fusao": []}
        crop = None
        for p in tqdm(sub, desc=f"vid {seq}"):
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            da, db, df = predA(im), predB(im), predF(im)
            if crop is None:
                union = df[:, :4] if len(df) else (da[:, :4] if len(da) else None)
                crop = crop_win(h, w, union)
            x1, y1, x2, y2 = crop
            for tag, d in (("A", da), ("B", db), ("fusao", df)):
                vis = draw(im[y1:y2, x1:x2], d, x1, y1)
                vis = cv2.resize(vis, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                buf[tag].append(vis)
        for tag, frs in buf.items():
            write_mp4(vid_dir / f"{seq}_{tag}.mp4", frs)
        torch.cuda.empty_cache()

    print("[ok] stills", still_dir)
    print("[ok] videos", vid_dir)


if __name__ == "__main__":
    main()
