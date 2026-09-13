"""Workspace paths via environment variables only (see env.example).

No machine-specific home directories. Set FUSAO_ROOT (folder with
datasets/ and weights/) or the defaults resolve relative to this repo.
A gitignored .env in the repo root is loaded if present.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    p = _REPO / ".env"
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    if not v:
        return None
    return Path(v).expanduser()


def _first_existing(*cands: Path | None) -> Path:
    kept: list[Path] = []
    for c in cands:
        if c is None:
            continue
        p = Path(c)
        kept.append(p)
        if p.exists():
            return p
    return kept[0] if kept else _REPO


_env_fusao = _env_path("FUSAO_ROOT")
if _env_fusao is not None:
    FUSAO = _env_fusao
elif (_REPO.parent / "datasets").is_dir() or (_REPO.parent / "weights").is_dir():
    FUSAO = _REPO.parent
elif (_REPO.parent.parent / "datasets").is_dir() or (_REPO.parent.parent / "weights").is_dir():
    FUSAO = _REPO.parent.parent
else:
    FUSAO = _REPO.parent

DATASETS = _env_path("DATASETS_ROOT") or (FUSAO / "datasets")
WEIGHTS = _env_path("WEIGHTS_DIR") or (FUSAO / "weights")
OUT = _env_path("YOLO_GIT_OUT") or (_REPO / "runs")
PESOS_OUT = _REPO / "pesos"

MVTD_ROOT = _first_existing(
    _env_path("MVTD_ROOT"),
    FUSAO / "mvtd",
    _REPO.parent / "mvtd",
)
MVTD_YOLO = _env_path("MVTD_YOLO") or (MVTD_ROOT / "yolo")
MVTD_PESOS = _env_path("MVTD_PESOS") or (MVTD_ROOT / "pesos")
MVTD_RAW = _env_path("MVTD_RAW") or (MVTD_ROOT / "Maritime_Visual_Tracking_Dataset_MVTD")

SEADRONESSEE_15GB = _env_path("SDS_15GB") or (DATASETS / "seadronessee_15gb")
SEADRONESSEE_4GB = _env_path("SDS_4GB") or (DATASETS / "seadronessee_reduzido_4gb")
COCO_ANN = _env_path("SDS_COCO") or (DATASETS / "SeaDronesSee_MOT" / "annotations")

COCO_INIT = _first_existing(
    _env_path("COCO_INIT"),
    FUSAO / "yolov8s.pt",
    MVTD_PESOS / "yolov8s.pt",
    PESOS_OUT / "yolov8s.pt",
    WEIGHTS / "yolov8s.pt",
)

YOLOOW_SRC = _first_existing(
    _env_path("YOLOOW_ROOT"),
    _REPO / "finetune_yoloow" / "YoloOW",
    FUSAO / "YoloOW",
)


def setup_sys_path() -> None:
    """Repo + latefusion/ (sibling scripts) + YoloOW (utils.general)."""
    for p in (_REPO, _REPO / "latefusion", YOLOOW_SRC):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
