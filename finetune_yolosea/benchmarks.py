#!/usr/bin/env python3
"""Benchmarks SOTA (SeaDronesSee) + resultados locais do K-Fold / fusão DS-WBF.

Fontes (protocolos MISTOS — não comparar mAP como se fosse o mesmo split):
  - YoloOW IEEE TGRS 2024
  - RF-DETR / YOLO26 HuggingFace DetectionBench
  - YOLO-SEA Entropy 2025, 27, 667 (dataset próprio do paper, NÃO SeaDronesSee)
  - este projeto: val IoU≥0.5 e K-Fold 5 (GroupKFold vídeo)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import FUSAO, OUT  # noqa: E402

RES = OUT / "yolosea" / "results"
PLOTS = RES / "plots"
OLD_MET = FUSAO / "runs" / "blue_eyes_a_mhaf_dswbf_temporal_4gb" / "metricas"
SEA_RUN = OUT / "seadronessee" / "yolosea_yoloow_dswbf"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

SOTA = [
    {"method": "YoloOW (paper IEEE)", "mAP50": 37.18, "P": None, "R": None, "F1": None, "source": "IEEE TGRS 2024", "protocol": "YoloOW paper"},
    {"method": "YOLOv8s (HF base)", "mAP50": 72.94, "P": 84.52, "R": 71.25, "F1": 77.3, "source": "HF", "protocol": "HF DetectionBench"},
    {"method": "RF-DETR Nano", "mAP50": 72.38, "P": 81.37, "R": 74.08, "F1": None, "source": "DetectionBench", "protocol": "HF DetectionBench"},
    {"method": "RF-DETR Medium", "mAP50": 83.47, "P": 87.01, "R": 83.33, "F1": 85.1, "source": "DetectionBench", "protocol": "HF DetectionBench"},
    {"method": "YOLOv26s", "mAP50": 80.14, "P": 88.5, "R": 77.51, "F1": 82.64, "source": "HF", "protocol": "HF DetectionBench"},
    {"method": "YOLOv26m", "mAP50": 82.38, "P": 90.01, "R": 81.18, "F1": 85.4, "source": "HF", "protocol": "HF DetectionBench"},
    {"method": "YOLO-SEA (paper Entropy)", "mAP50": 88.2, "P": None, "R": None, "F1": None, "source": "Entropy 2025 27:667", "protocol": "dataset YOLO-SEA (8 classes, NAO SDS)"},
]


def load_local():
    rows = []
    # k-fold evaluation-only (pesos fixos, notebook)
    kf = OLD_MET / "kfold_summary.json"
    if kf.is_file():
        d = json.loads(kf.read_text())
        for m, s in d.items():
            rows.append({
                "method": f"{m} K-Fold eval (pesos fixos)",
                "mAP50": None,
                "P": s["P_mean"] * 100, "R": s["R_mean"] * 100, "F1": s["F1_mean"] * 100,
                "source": "notebook Group? shuffle KFold imagens",
                "protocol": "IoU>=0.5 local 5-fold (eval only)",
            })
    # YOLO-SEA Soft-NMS + YoloOW + DS-WBF val
    sea = SEA_RUN / "metrics_val.json"
    if sea.is_file():
        d = json.loads(sea.read_text())
        lab = {"A_YOLOSEA": "A YOLO-SEA Soft-NMS (local)", "B_YoloOW": "B YoloOW (local)", "Fusao_DSWBF": "Fusao DS-WBF (local)"}
        for k, s in d.items():
            rows.append({
                "method": lab.get(k, k),
                "mAP50": None,
                "P": s["P"] * 100, "R": s["R"] * 100, "F1": s["F1"] * 100,
                "source": "grok pipeline",
                "protocol": "IoU>=0.5 val 250 imgs subset 4GB",
            })
    # treino k-fold (se já rodou)
    live = RES / "kfold_live.json"
    if live.is_file():
        for rec in json.loads(live.read_text()):
            if "map50" not in rec:
                continue
            p, r = rec.get("precision") or 0, rec.get("recall") or 0
            f1 = 2 * p * r / (p + r + 1e-9) if p or r else None
            rows.append({
                "method": f"YOLOv8s train fold {rec['fold']}",
                "mAP50": (rec["map50"] * 100) if rec["map50"] <= 1 else rec["map50"],
                "P": (p * 100) if p <= 1 else p,
                "R": (r * 100) if r <= 1 else r,
                "F1": (f1 * 100) if f1 is not None and f1 <= 1 else f1,
                "source": "grok_yolo_blue_eyes K-Fold train",
                "protocol": "Ultralytics val do fold (GroupKFold video)",
            })
    return rows


def main():
    local = load_local()
    df = pd.DataFrame(SOTA + local)
    df.to_csv(RES / "benchmarks_all.csv", index=False)
    df.to_csv(PLOTS / "benchmarks_all.csv", index=False)
    print(df.to_string(index=False))

    # mAP SOTA (só quem tem mAP)
    sub = df[df["mAP50"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sub = sub.sort_values("mAP50")
    colors = ["#2ca02c" if "Fusao" in m or "Ours" in m or "train fold" in m else "#1f77b4" for m in sub["method"]]
    ax.barh(sub["method"], sub["mAP50"], color=colors)
    ax.set_xlabel("mAP@50 (%)")
    ax.set_title("Benchmarks SeaDronesSee / marítimo — protocolos mistos")
    fig.tight_layout()
    fig.savefig(PLOTS / "bar_map50_sota.png", dpi=170)
    plt.close()

    # P/R/F1 local
    loc = df[df["F1"].notna() & df["protocol"].str.contains("IoU", na=False)].copy()
    if len(loc):
        fig, ax = plt.subplots(figsize=(10, 4.4))
        x = np.arange(len(loc))
        w = 0.25
        ax.bar(x - w, loc["P"], w, label="P")
        ax.bar(x, loc["R"], w, label="R")
        ax.bar(x + w, loc["F1"], w, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(loc["method"], rotation=18, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title("Métricas locais IoU≥0.5 — A | YoloOW | Fusão DS-WBF | K-Fold eval")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS / "bar_local_PRF1.png", dpi=170)
        plt.close()

    # tabela imagem
    fig, ax = plt.subplots(figsize=(12.5, 0.45 * (len(df) + 2)))
    ax.axis("off")
    cols = ["method", "mAP50", "P", "R", "F1", "protocol"]
    cell = []
    for _, r in df.iterrows():
        cell.append([
            str(r["method"])[:40],
            "" if pd.isna(r["mAP50"]) else f"{r['mAP50']:.1f}",
            "" if pd.isna(r["P"]) else f"{r['P']:.1f}",
            "" if pd.isna(r["R"]) else f"{r['R']:.1f}",
            "" if pd.isna(r["F1"]) else f"{r['F1']:.1f}",
            str(r["protocol"])[:42],
        ])
    tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for (i, j), c in tbl.get_celld().items():
        if i == 0:
            c.set_facecolor("#1f4e79")
            c.set_text_props(color="white", weight="bold")
    ax.set_title("Benchmarks — mAP de fontes distintas NAO sao o mesmo protocolo", pad=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "tabela_benchmarks.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("[ok]", PLOTS)


if __name__ == "__main__":
    main()
