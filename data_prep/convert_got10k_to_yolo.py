#!/usr/bin/env python3
"""Converte MVTD (GOT-10k / arXiv:2506.02866) para YOLO Ultralytics.

Fonte HF: AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD
  train/  159 sequências oficiais (paper Protocol II)
  test/    23 sequências oficiais

Val NÃO existe no paper. Este script recorta val a partir do train oficial,
por sequência (sem vazar frames do mesmo vídeo), estratificado por classe.

Imagens: symlink (não Path.resolve) para o raw, com caminho contendo /images/
para o Ultralytics trocar /images/ → /labels/.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASS_NAMES = ["boat", "ship", "sailboat", "usv"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
ALIAS = {
    "boat": "boat",
    "ship": "ship",
    "sailboat": "sailboat",
    "sail_boat": "sailboat",
    "usv": "usv",
    "unmannedsurfacevehicle": "usv",
}

SEQ_RE = re.compile(r"^[0-9]+-(.+)$")


def parse_class(seq_name: str) -> str:
    m = SEQ_RE.match(seq_name)
    if not m:
        raise ValueError(f"nome de sequência inesperado: {seq_name!r}")
    raw = re.sub(r"[^a-z0-9]+", "", m.group(1).lower())
    if raw not in ALIAS:
        raise ValueError(f"classe desconhecida em {seq_name!r} ({raw})")
    return ALIAS[raw]


def list_sequences(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def list_frames(seq_dir: Path) -> list[Path]:
    frames = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))
    return frames


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_gt_line(line: str) -> tuple[float, float, float, float] | None:
    line = line.strip()
    if not line:
        return None
    parts = [p for p in re.split(r"[\s,;]+", line) if p]
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = map(float, parts[:4])
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def xywh_pixel_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    if img_w <= 0 or img_h <= 0:
        return None
    x1 = max(0.0, min(float(img_w), x))
    y1 = max(0.0, min(float(img_h), y))
    x2 = max(0.0, min(float(img_w), x + w))
    y2 = max(0.0, min(float(img_h), y + h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    return xc, yc, nw, nh


def stratified_val_split(seq_dirs: list[Path], val_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    by_cls: dict[str, list[Path]] = defaultdict(list)
    for p in seq_dirs:
        by_cls[parse_class(p.name)].append(p)
    rng = random.Random(seed)
    train, val = [], []
    for cls in CLASS_NAMES:
        seqs = sorted(by_cls.get(cls, []), key=lambda p: p.name)
        rng.shuffle(seqs)
        n = len(seqs)
        if n == 0:
            continue
        n_val = max(1, round(n * val_frac)) if n >= 2 else 0
        n_val = min(n_val, n - 1) if n >= 2 else 0
        val.extend(seqs[:n_val])
        train.extend(seqs[n_val:])
    train.sort(key=lambda p: p.name)
    val.sort(key=lambda p: p.name)
    return train, val


def image_size(path: Path, cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    key = str(path.parent)
    if key in cache:
        return cache[key]
    with Image.open(path) as im:
        size = im.size
    cache[key] = size
    return size


def convert_sequence(
    seq_dir: Path,
    dest_images: Path,
    dest_labels: Path,
    class_id: int,
    size_cache: dict[str, tuple[int, int]],
) -> dict:
    frames = list_frames(seq_dir)
    gt = [parse_gt_line(x) for x in read_lines(seq_dir / "groundtruth.txt")]
    absence = read_lines(seq_dir / "absence.label")
    n_ok = 0
    n_empty = 0
    n_skip = 0
    stem_prefix = seq_dir.name  # e.g. 40-Boat
    for i, frame in enumerate(frames):
        stem = f"{stem_prefix}_{frame.stem}"
        img_dst = dest_images / f"{stem}{frame.suffix.lower()}"
        lab_dst = dest_labels / f"{stem}.txt"
        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        # alvo = arquivo real em raw/; o path YOLO (img_dst) NÃO é resolvido
        img_dst.symlink_to(frame.resolve())
        absent = False
        if i < len(absence):
            absent = absence[i].strip() not in ("", "0")
        box = gt[i] if i < len(gt) else None
        if absent or box is None:
            lab_dst.write_text("", encoding="utf-8")
            n_empty += 1
            continue
        try:
            iw, ih = image_size(frame, size_cache)
        except OSError:
            lab_dst.write_text("", encoding="utf-8")
            n_skip += 1
            continue
        yolo = xywh_pixel_to_yolo(*box, iw, ih)
        if yolo is None:
            lab_dst.write_text("", encoding="utf-8")
            n_empty += 1
            continue
        xc, yc, nw, nh = yolo
        lab_dst.write_text(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")
        n_ok += 1
    extra_gt = max(0, len(gt) - len(frames))
    return {
        "sequence": seq_dir.name,
        "class": CLASS_NAMES[class_id],
        "frames": len(frames),
        "gt_lines": len(gt),
        "labeled": n_ok,
        "empty": n_empty,
        "skip": n_skip,
        "gt_without_frame": extra_gt,
    }


def write_list(path: Path, image_dir: Path) -> int:
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # caminhos absolutos COM /images/ (não resolve symlink)
    lines = [str(p) + "\n" for p in files]
    path.write_text("".join(lines), encoding="utf-8")
    return len(files)


def write_data_yaml(yolo_root: Path) -> None:
    text = (
        f"path: {yolo_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (yolo_root / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    _repo = Path(__file__).resolve().parents[1]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    from paths import MVTD_RAW, MVTD_YOLO  # noqa: E402

    ap.add_argument("--raw", type=Path, default=MVTD_RAW)
    ap.add_argument("--out", type=Path, default=MVTD_YOLO)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = args.raw
    out = args.out
    train_official = list_sequences(raw / "train")
    test_official = list_sequences(raw / "test")
    if not train_official or not test_official:
        raise SystemExit(
            f"raw incompleto: train={len(train_official)} test={len(test_official)} em {raw}"
        )

    yolo_train, yolo_val = stratified_val_split(train_official, args.val_frac, args.seed)
    splits = {
        "train": yolo_train,
        "val": yolo_val,
        "test": test_official,
    }

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    size_cache: dict[str, tuple[int, int]] = {}
    reports: dict[str, list[dict]] = {}
    for split, seqs in splits.items():
        reports[split] = []
        img_dir = out / "images" / split
        lab_dir = out / "labels" / split
        for seq in seqs:
            cid = CLASS_TO_ID[parse_class(seq.name)]
            reports[split].append(
                convert_sequence(seq, img_dir, lab_dir, cid, size_cache)
            )
        n_list = write_list(out / f"{split}.txt", img_dir)
        print(f"{split}: {len(seqs)} seqs, {n_list} imagens")

    write_data_yaml(out)

    summary = {
        "source": "AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD",
        "paper": "arXiv:2506.02866",
        "format": "GOT-10k x,y,w,h pixels → YOLO class xc yc w h (normalizado)",
        "classes": CLASS_NAMES,
        "val_policy": (
            "val recortado do train oficial por sequência, estratificado por classe; "
            "test = split oficial do paper. Sem vazamento de vídeo."
        ),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "counts": {
            split: {
                "sequences": len(splits[split]),
                "frames": sum(r["frames"] for r in reports[split]),
                "labeled": sum(r["labeled"] for r in reports[split]),
                "empty": sum(r["empty"] for r in reports[split]),
                "by_class": dict(
                    Counter(parse_class(p.name) for p in splits[split])
                ),
                "sequences_names": [p.name for p in splits[split]],
            }
            for split in ("train", "val", "test")
        },
    }
    (out / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary["counts"][k] for k in ("train", "val", "test")}, indent=2))
    print(f"data.yaml → {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
