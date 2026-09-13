#!/usr/bin/env python3
"""Avalia um peso YOLO no val e no test oficiais do MVTD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import MVTD_YOLO, OUT, PESOS_OUT  # noqa: E402

PROJ = _REPO
DATA = MVTD_YOLO / "data.yaml"
METRICAS = OUT / "yolosea" / "metricas"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=PESOS_OUT / "mvtd_yolov8s_v2_best.pt")
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"peso não encontrado: {args.weights}")
    METRICAS.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    splits = ["val", "test"] if args.split == "both" else [args.split]
    out = {"weights": str(args.weights), "imgsz": args.imgsz}
    for sp in splits:
        r = model.val(
            data=str(DATA),
            split=sp,
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            plots=True,
            project=str(OUT / "yolosea" / "mvtd"),
            name=f"eval_{sp}",
            exist_ok=True,
        )
        d = dict(getattr(r, "results_dict", {}) or {})
        out[sp] = {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
        print(sp, out[sp])
    (METRICAS / "eval.json").write_text(json.dumps(out, indent=2))
    print("[ok]", METRICAS / "eval.json")


if __name__ == "__main__":
    main()
