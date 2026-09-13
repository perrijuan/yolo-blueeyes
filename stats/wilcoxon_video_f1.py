#!/usr/bin/env python3
"""F1 local por video_id (A vs fusao), Wilcoxon pareado. Pesos e limiares do paper.

Nao e Optuna. Nao re-treina. Nao mexe no operador DS-WBF.
Protocolo: IoU>=0.5, mesma classe, ACTIVE={0,1,4}, Soft-NMS em A.
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=UserWarning)

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import COCO_ANN, DATASETS, FUSAO, OUT as RUNS_OUT, YOLOOW_SRC, setup_sys_path  # noqa: E402

setup_sys_path()
_scripts = FUSAO / "scripts"
if _scripts.is_dir() and str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
if YOLOOW_SRC.is_dir() and str(YOLOOW_SRC) not in sys.path:
    sys.path.insert(0, str(YOLOOW_SRC))

from tracking_sahi_paper_blocks import (  # noqa: E402
    ACTIVE_CLASS_IDS,
    CONF_A,
    CONF_B,
    IMGSZ,
    ds_wbf_fuse,
)
from yolosea_yoloow_dswbf import pred_yolosea  # noqa: E402

from eval_datasets_completos import (  # noqa: E402
    load_gt_yolo,
    load_sds_models,
    match_prf,
)

SDS_YOLO = DATASETS / "seadronessee_yolo"
COCO_VAL = COCO_ANN / "instances_val_objects_in_water.json"
KFOLD_JSON = FUSAO / "resultados_4gb" / "kfold_summary.json"
OUT = RUNS_OUT / "wilcoxon_video"
ACTIVE = set(ACTIVE_CLASS_IDS)
ALPHA = 0.05


def f1_of(tp, fp, fn) -> float:
    P = tp / (tp + fp + 1e-6)
    R = tp / (tp + fn + 1e-6)
    return float(2 * P * R / (P + R + 1e-6))


def stem_to_video() -> dict[str, int]:
    coco = json.loads(COCO_VAL.read_text())
    m = {}
    for im in coco["images"]:
        m[f"val_{Path(im['file_name']).stem}"] = int(im["video_id"])
    return m


def list_val(stride: int) -> dict[int, list[Path]]:
    img_dir = SDS_YOLO / "images"
    lab_dir = SDS_YOLO / "labels"
    vmap = stem_to_video()
    by: dict[int, list[Path]] = defaultdict(list)
    for p in sorted(img_dir.glob("val_*.jpg")):
        if p.stem not in vmap:
            continue
        if not (lab_dir / f"{p.stem}.txt").is_file():
            continue
        by[vmap[p.stem]].append(p)
    out = {}
    for vid, paths in by.items():
        paths = sorted(paths)
        out[vid] = paths[:: max(stride, 1)]
    return dict(sorted(out.items()))


def eval_videos(by_vid: dict[int, list[Path]]) -> list[dict]:
    model_a, pred_b = load_sds_models()
    lab_dir = SDS_YOLO / "labels"
    rows = []
    for vid, paths in by_vid.items():
        acc = {k: [0, 0, 0] for k in ("A", "B", "Fusao")}
        n_ok = 0
        for p in tqdm(paths, desc=f"video {vid}"):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            gts = load_gt_yolo(lab_dir / f"{p.stem}.txt", w, h, ACTIVE)
            da = pred_yolosea(model_a, frame, w, h)
            db = pred_b(frame, conf=CONF_B)
            df = ds_wbf_fuse(da, db, w, h)
            for tag, dets in (("A", da), ("B", db), ("Fusao", df)):
                t, f, n = match_prf(dets, gts, ACTIVE)
                acc[tag][0] += t
                acc[tag][1] += f
                acc[tag][2] += n
            n_ok += 1
        row = {
            "video_id": int(vid),
            "n_frames": n_ok,
            "provenance": "REAL",
        }
        for tag in ("A", "B", "Fusao"):
            tp, fp, fn = acc[tag]
            row[tag] = {
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "P": tp / (tp + fp + 1e-6),
                "R": tp / (tp + fn + 1e-6),
                "F1": f1_of(tp, fp, fn),
            }
        row["delta_F1"] = row["Fusao"]["F1"] - row["A"]["F1"]
        print(
            f"  vid={vid:3d} n={n_ok:4d}  A={row['A']['F1']:.3f}  "
            f"F={row['Fusao']['F1']:.3f}  d={row['delta_F1']:+.3f}"
        )
        rows.append(row)
    del model_a
    torch.cuda.empty_cache()
    return rows


def wilcoxon_report(rows: list[dict]) -> dict:
    d = np.array([r["delta_F1"] for r in rows], dtype=np.float64)
    n = len(d)
    n_win = int((d > 0).sum())
    n_lose = int((d < 0).sum())
    n_tie = int((d == 0).sum())
    median_d = float(np.median(d))
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1) if n > 1 else 0.0)
    cohen_d = float(mean_d / (std_d + 1e-12))
    # Wilcoxon signed-rank on paired F1 (Fusao - A). zeros: Pratt.
    if np.allclose(d, 0):
        p_two = 1.0
        p_greater = 1.0
        stat = 0.0
        note = "todos os deltas sao zero"
    else:
        w = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided", method="auto")
        wg = stats.wilcoxon(d, zero_method="pratt", alternative="greater", method="auto")
        stat = float(w.statistic)
        p_two = float(w.pvalue)
        p_greater = float(wg.pvalue)
        note = "Wilcoxon signed-rank pareado, zero_method=pratt"
    # Shapiro so para decidir se o t pareado e justificavel; n pequeno.
    if n >= 3 and std_d > 0:
        sw = stats.shapiro(d)
        shapiro_p = float(sw.pvalue)
        t_res = stats.ttest_1samp(d, 0.0)
        t_p = float(t_res.pvalue)
        t_stat = float(t_res.statistic)
    else:
        shapiro_p = t_p = t_stat = float("nan")
    claim = (
        "significancia a alpha=0.05 (two-sided)"
        if p_two < ALPHA
        else "NAO ha evidencia suficiente para superioridade (p >= alpha)"
    )
    return {
        "n_videos": n,
        "median_delta": median_d,
        "mean_delta": mean_d,
        "std_delta": std_d,
        "cohen_d_paired": cohen_d,
        "n_fusion_wins": n_win,
        "n_fusion_loses": n_lose,
        "n_tie": n_tie,
        "wilcoxon_stat": stat,
        "p_two_sided": p_two,
        "p_greater": p_greater,
        "alpha": ALPHA,
        "claim": claim,
        "shapiro_p": shapiro_p,
        "ttest_stat": t_stat,
        "ttest_p_two_sided": t_p,
        "note": note,
        "paper_sentence": (
            f"Across n={n} videos, fusion F1 minus A-only F1 has median "
            f"{median_d:.3f}; Wilcoxon signed-rank p={p_two:.3f}. "
            "We do not claim significance unless p is below the pre-declared alpha=0.05."
        ),
    }


def plot_figure(rows: list[dict], report: dict, kfold: dict, out_png: Path) -> None:
    f1_a = np.array([r["A"]["F1"] for r in rows])
    f1_f = np.array([r["Fusao"]["F1"] for r in rows])
    vids = [str(r["video_id"]) for r in rows]
    d = np.array([r["delta_F1"] for r in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)
    colors = ["#4C72B0", "#C44E52"]

    # 1) o que o paper publica (K-fold 4 GB, NAO pareado por video)
    ax = axes[0]
    means = [kfold["A_YOLOv8s"]["F1_mean"], kfold["Fusao_DSWBF"]["F1_mean"]]
    stds = [kfold["A_YOLOv8s"]["F1_std"], kfold["Fusao_DSWBF"]["F1_std"]]
    ax.bar([0, 1], means, yerr=stds, color=colors, width=0.55, capsize=5, error_kw={"linewidth": 1.2})
    ax.set_xticks([0, 1], ["A", "Fusao"])
    ax.set_ylim(0.78, 0.845)
    ax.set_ylabel("F1")
    ax.set_title("A. K-fold 4 GB no paper\nmean ± std, 5 folds (shuffle imagem)")
    ax.axhline(means[0], color=colors[0], lw=0.8, ls=":")
    ax.text(
        0.5,
        0.838,
        f"Δ = {means[1]-means[0]:.3f}  <  1 std do A ({stds[0]:.3f})\n"
        "sem teste pareado neste painel",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        0.03,
        0.03,
        "REAL  resultados_4gb/kfold_summary.json",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )

    # 2) pareado por video
    ax = axes[1]
    x = np.arange(len(vids))
    ax.plot(x, f1_a, "o-", color=colors[0], label="A", ms=5, lw=1.2)
    ax.plot(x, f1_f, "s-", color=colors[1], label="Fusao", ms=5, lw=1.2)
    ax.set_xticks(x, vids, fontsize=8)
    ax.set_xlabel("video_id  (val oficial, stride=5)")
    ax.set_ylabel("F1")
    ax.set_title("B. F1 por video_id (pareado)")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(0.0, 1.05)
    ax.text(0.03, 0.03, "REAL  deteccao neste run", transform=ax.transAxes, fontsize=8, color="#555")

    # 3) deltas + Wilcoxon
    ax = axes[2]
    order = np.argsort(d)
    ax.axvline(0, color="#888", lw=1)
    ax.barh(
        np.arange(len(d)),
        d[order],
        color=["#C44E52" if v > 0 else "#4C72B0" for v in d[order]],
        height=0.72,
    )
    ax.set_yticks(np.arange(len(d)), [vids[i] for i in order], fontsize=8)
    ax.set_xlabel("Δ F1  (fusao − A)")
    p = report["p_two_sided"]
    ax.set_title(
        f"C. Wilcoxon signed-rank\n"
        f"p={p:.3f}  mediana Δ={report['median_delta']:+.3f}  "
        f"{report['n_fusion_wins']}/{report['n_videos']} ganham"
    )
    ax.set_xlim(-0.04, 0.18)

    fig.suptitle(
        "O Δ=0.008 do K-fold 4 GB nao traz p  |  "
        "o teste pareado (val oficial) e outro protocolo",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="1 = todos os frames do val oficial")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    kfold = json.loads(KFOLD_JSON.read_text())
    by_vid = list_val(args.stride)
    n_frames = sum(len(v) for v in by_vid.values())
    print(f"[data] val oficial videos={len(by_vid)} frames={n_frames} stride={args.stride} imgsz={IMGSZ}")
    rows = eval_videos(by_vid)
    report = wilcoxon_report(rows)
    payload = {
        "provenance": {
            "per_video_F1": "REAL",
            "kfold_4gb_means": "REAL (JSON publicado, folds por imagem, nao por video_id)",
            "p_value": "DERIVADO (Wilcoxon nos F1 REAL por video)",
            "optuna": "nao usado neste teste",
        },
        "protocol": {
            "split": "SeaDronesSee official val",
            "iou": 0.5,
            "same_class": True,
            "active": sorted(ACTIVE),
            "imgsz": IMGSZ,
            "conf_a": CONF_A,
            "conf_b": CONF_B,
            "stride": args.stride,
            "weights_a": str(FUSAO / "models" / "yolov8s_seadronessee.pt"),
            "weights_b": str(FUSAO / "models" / "YoloOW.pt"),
            "fusion": "DS-WBF frozen (latefusion.py defaults)",
            "alpha": ALPHA,
        },
        "kfold_4gb_published": {
            "A_F1": kfold["A_YOLOv8s"]["F1_mean"],
            "F_F1": kfold["Fusao_DSWBF"]["F1_mean"],
            "delta": kfold["Fusao_DSWBF"]["F1_mean"] - kfold["A_YOLOv8s"]["F1_mean"],
            "note": "shuffle imagens seed=20, pesos fixos. Nao e teste pareado.",
        },
        "wilcoxon": report,
        "videos": rows,
    }
    out_json = OUT / "wilcoxon_video_f1.json"
    out_json.write_text(json.dumps(payload, indent=2))
    png = OUT / "wilcoxon_video_f1.png"
    plot_figure(rows, report, kfold, png)
    print(json.dumps(report, indent=2))
    print("[ok]", out_json)
    print("[ok]", png)
    print(report["paper_sentence"])


if __name__ == "__main__":
    main()
