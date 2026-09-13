#!/usr/bin/env python3
"""Smoke de inferência: MVTD val (A + B + DS-WBF) + 2 frames SeaDronesSee (A)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from paths import (  # noqa: E402
    MVTD_YOLO,
    OUT,
    SEADRONESSEE_15GB,
    WEIGHTS,
    setup_sys_path,
)

setup_sys_path()

from latefusion import ds_wbf_fuse, soft_nms  # noqa: E402
from run_mvtd import load_gt, load_models, make_pred_b, match_prf, pred_a  # noqa: E402

CLASS_NAMES_MVTD = ["boat", "ship", "sailboat", "usv"]
CLASS_NAMES_SDS = ["swimmer", "boat", "jetski", "buoy", "life_saving_appliances"]
N_MVTD = 20
COLORS = {
    "A": (80, 180, 255),
    "B": (80, 220, 80),
    "Fusao": (0, 165, 255),
    "GT": (200, 200, 200),
}


def draw(frame, dets, gts, names, tag):
    vis = frame.copy()
    for g in gts:
        x1, y1, x2, y2 = map(int, g[:4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["GT"], 1)
    for d in dets:
        x1, y1, x2, y2 = map(int, d[:4])
        cid = int(d[5])
        conf = float(d[4])
        name = names[cid] if 0 <= cid < len(names) else str(cid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS.get(tag, (0, 255, 255)), 2)
        cv2.putText(
            vis,
            f"{name} {conf:.2f}",
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS.get(tag, (0, 255, 255)),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(vis, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS.get(tag, (0, 255, 255)), 2)
    return vis


def pick_mvtd(n: int) -> list[Path]:
    img_dir = MVTD_YOLO / "images" / "val"
    lab_dir = MVTD_YOLO / "labels" / "val"
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # um frame a cada ~n_seq para cobrir várias sequências, preferindo quem tem GT
    by_seq: dict[str, list[Path]] = {}
    for p in paths:
        seq = p.stem.rsplit("_", 1)[0]
        by_seq.setdefault(seq, []).append(p)
    picked: list[Path] = []
    seqs = sorted(by_seq)
    i = 0
    while len(picked) < n and seqs:
        seq = seqs[i % len(seqs)]
        frames = by_seq[seq]
        idx = (len(picked) // max(len(seqs), 1)) % len(frames)
        p = frames[idx]
        lab = lab_dir / f"{p.stem}.txt"
        if p not in picked and lab.is_file() and lab.stat().st_size > 0:
            picked.append(p)
        i += 1
        if i > n * 20:
            break
    return picked[:n]


def main() -> None:
    out = OUT / "smoke_infer"
    vis_dir = out / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # kernel sem GPU
    a = np.array([[10.0, 10, 50, 50, 0.9, 0]], np.float32)
    b = np.array([[12.0, 12, 52, 52, 0.8, 0]], np.float32)
    fused = ds_wbf_fuse(a, b, 100, 100)
    kernel = {"n_fused": int(len(fused)), "soft_nms": int(len(soft_nms(a)))}
    print("[kernel]", kernel)
    assert len(fused) >= 1, "ds_wbf_fuse devolveu vazio com boxes sobrepostas"

    print("[cuda]", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    model_a, model_b, device = load_models()
    fn_b = make_pred_b(model_b, device)
    print("[models] A ok;", "B ok" if model_b is not None else "B AUSENTE")

    paths = pick_mvtd(N_MVTD)
    print(f"[data] MVTD val n={len(paths)} root={MVTD_YOLO}")
    if not paths:
        raise SystemExit("nenhuma imagem MVTD val com label")

    acc = {k: [0, 0, 0] for k in ("A", "B", "Fusao")}
    per_frame = []
    lab_dir = MVTD_YOLO / "labels" / "val"
    vis_saved = 0
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print("[skip] não leu", p)
            continue
        h, w = frame.shape[:2]
        gts = load_gt(lab_dir / f"{p.stem}.txt", w, h)
        da = pred_a(model_a, frame)
        db = fn_b(frame)
        df = ds_wbf_fuse(da, db, w, h)
        row = {"image": p.name, "w": w, "h": h, "n_gt": int(len(gts))}
        for tag, dets in (("A", da), ("B", db), ("Fusao", df)):
            tp, fp, fn = match_prf(dets, gts)
            acc[tag][0] += tp
            acc[tag][1] += fp
            acc[tag][2] += fn
            row[tag] = {"n": int(len(dets)), "TP": tp, "FP": fp, "FN": fn}
        per_frame.append(row)
        print(
            f"  {p.name} gt={len(gts)} A={len(da)} B={len(db)} F={len(df)} "
            f"A.TP={row['A']['TP']} F.TP={row['Fusao']['TP']}"
        )
        if vis_saved < 3:
            for tag, dets in (("A", da), ("B", db), ("Fusao", df)):
                cv2.imwrite(str(vis_dir / f"{p.stem}_{tag}.jpg"), draw(frame, dets, gts, CLASS_NAMES_MVTD, tag))
            vis_saved += 1

    def pack(tag, tp, fp, fn):
        P = tp / (tp + fp + 1e-6)
        R = tp / (tp + fn + 1e-6)
        F1 = 2 * P * R / (P + R + 1e-6)
        return {
            "method": tag,
            "P": P,
            "R": R,
            "F1": F1,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "n": len(per_frame),
            "protocol": "local IoU>=0.5 same-class smoke (nao e metrica do paper)",
        }

    mvtd = [pack(tag, *acc[tag]) for tag in ("A", "B", "Fusao")]
    for sm in mvtd:
        print(
            f"  {sm['method']:8s} n={sm['n']} P={sm['P']:.3f} R={sm['R']:.3f} "
            f"F1={sm['F1']:.3f} TP={sm['TP']} FP={sm['FP']} FN={sm['FN']}"
        )

    # SeaDronesSee: 2 frames val com o peso A do workspace (não é o pipeline 4K tiled)
    sds_ok = False
    sds_info = {}
    sds_dir = SEADRONESSEE_15GB / "images"
    sds_imgs = sorted(sds_dir.glob("val_*.jpg"))[:2] if sds_dir.is_dir() else []
    sea_w = WEIGHTS / "yolov8s_seadronessee.pt"
    if sds_imgs and sea_w.is_file():
        from ultralytics import YOLO

        sea = YOLO(str(sea_w))
        sea.to("cuda" if torch.cuda.is_available() else "cpu")
        dets_n = []
        for p in sds_imgs:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            r = sea.predict(source=frame, conf=0.15, imgsz=640, verbose=False)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            dets_n.append({"image": p.name, "n_det": n, "shape": list(frame.shape)})
            print(f"  [SDS] {p.name} shape={frame.shape[:2]} dets={n}")
        sds_ok = True
        sds_info = {"weights": str(sea_w), "frames": dets_n}
        del sea
    else:
        print("[SDS] skip (sem imagens ou peso)")

    summary = {
        "kernel": kernel,
        "cuda": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "B_loaded": model_b is not None,
        "mvtd": mvtd,
        "mvtd_frames": per_frame,
        "sds": sds_info,
        "sds_ok": sds_ok,
        "vis": str(vis_dir),
    }
    outp = out / "smoke_infer.json"
    outp.write_text(json.dumps(summary, indent=2))
    print("[ok]", outp)

    if model_b is None:
        raise SystemExit("B (YoloOW) não carregou — inferência de fusão incompleta")
    if all(sm["TP"] == 0 for sm in mvtd):
        raise SystemExit("zero TP em A/B/Fusão no smoke MVTD")


if __name__ == "__main__":
    main()
