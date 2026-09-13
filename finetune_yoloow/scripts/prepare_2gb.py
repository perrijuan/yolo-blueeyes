#!/usr/bin/env python3
"""Subset MVTD ~2 GB para fine-tune do YoloOW (YOLOv7).

Não altera scripts de fusão. Sequências inteiras, classes raras primeiro
(sailboat, ship), depois usv/boat. Paths com /images/ (nunca Path.resolve).
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import MVTD_YOLO, OUT as RUNS_OUT  # noqa: E402

OUT = RUNS_OUT / "yoloow"
DATA = OUT / "data"
TARGET = int(2.0 * (1024 ** 3))
# ~0.4 GB val, resto treino
VAL_BUDGET = int(0.40 * (1024 ** 3))


def cls_of(seq: str) -> str:
    s = seq.lower()
    if "sail" in s:
        return "sailboat"
    if "ship" in s:
        return "ship"
    if "usv" in s:
        return "usv"
    return "boat"


def resolved_size(p: Path) -> int:
    try:
        return p.resolve(strict=False).stat().st_size
    except OSError:
        return p.stat().st_size if p.exists() else 0


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def seqs(split: str) -> dict[str, list[Path]]:
    d = MVTD_YOLO / "images" / split
    by = defaultdict(list)
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        seq = p.stem.rsplit("_", 1)[0]
        by[seq].append(p)
    return {k: sorted(v, key=lambda x: x.stem) for k, v in by.items()}


def pick(items: list[tuple[str, list[Path], int]], budget: int) -> tuple[list, int]:
    """items already ordered. Keep whole sequences until budget."""
    used, sel = 0, []
    for seq, frames, nbytes in items:
        if used + nbytes <= budget:
            sel.append((seq, frames, "FULL"))
            used += nbytes
            print(f"  + FULL {seq} n={len(frames)} total={used/1e9:.2f}GB")
        else:
            keep, cum = [], 0
            for p in frames:
                sz = resolved_size(p)
                if cum + sz > budget - used:
                    break
                keep.append(p)
                cum += sz
            if keep:
                sel.append((seq, keep, "PREFIX"))
                used += cum
                print(f"  + PREFIX {seq} n={len(keep)} total={used/1e9:.2f}GB")
            break
    return sel, used


def materialize(split: str, selected: list) -> list[Path]:
    img_dir = DATA / "images" / split
    lbl_src_root = MVTD_YOLO / "labels" / split
    written = []
    for seq, frames, _mode in selected:
        for src in frames:
            dst = img_dir / src.name
            link(src, dst)
            lbl = lbl_src_root / f"{src.stem}.txt"
            if lbl.is_file():
                dst_l = DATA / "labels" / split / lbl.name
                dst_l.parent.mkdir(parents=True, exist_ok=True)
                if not dst_l.exists():
                    shutil.copy2(lbl, dst_l)
            written.append(dst)
    return written


def write_list(paths: list[Path], out: Path) -> None:
    # absolute, still contain /images/ — do not resolve
    lines = [str(p) + "\n" for p in paths]
    out.write_text("".join(lines))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    train_map = seqs("train")
    val_map = seqs("val")

    def pack(mapping: dict[str, list[Path]]) -> dict[str, list[tuple[str, list[Path], int]]]:
        buckets = defaultdict(list)
        for seq, frames in mapping.items():
            nbytes = sum(resolved_size(p) for p in frames)
            buckets[cls_of(seq)].append((seq, frames, nbytes))
        for c in buckets:
            buckets[c].sort(key=lambda x: -x[2])
        return buckets

    train_b = pack(train_map)
    val_b = pack(val_map)
    rare = ("sailboat", "ship")
    common = ("usv", "boat")

    print("=" * 60)
    print("MVTD 2GB  ·  raro primeiro  ·  seq inteira")
    print("=" * 60)

    def clip(item, max_frames: int):
        seq, frames, _nbytes = item
        frames = frames[:max_frames]
        return seq, frames, sum(resolved_size(p) for p in frames)

    print("[val]")
    val_items = []
    if val_b.get("sailboat"):
        val_items.append(clip(val_b["sailboat"][0], 400))
    if val_b.get("ship"):
        val_items.append(clip(val_b["ship"][0], 200))
    boats_v = [clip(it, 300) for it in val_b.get("boat", [])]
    usvs_v = [clip(it, 300) for it in val_b.get("usv", [])]
    mixed_v = []
    n = max(len(boats_v), len(usvs_v))
    for i in range(n):
        if i < len(boats_v):
            mixed_v.append(boats_v[i])
        if i < len(usvs_v):
            mixed_v.append(usvs_v[i])
    val_items.extend(mixed_v)
    val_sel, val_used = pick(val_items, VAL_BUDGET)

    print("[train]")
    train_budget = TARGET - val_used
    train_items = list(train_b.get("sailboat", []))
    for it in train_b.get("ship", [])[:3]:
        train_items.append(clip(it, 250))
    boats = [clip(it, 400) for it in train_b.get("boat", [])]
    usvs = [clip(it, 400) for it in train_b.get("usv", [])]
    mixed = []
    n = max(len(boats), len(usvs))
    for i in range(n):
        if i < len(boats):
            mixed.append(boats[i])
        if i < len(usvs):
            mixed.append(usvs[i])
    train_items.extend(mixed)
    train_sel, train_used = pick(train_items, train_budget)

    val_paths = materialize("val", val_sel)
    train_paths = materialize("train", train_sel)
    write_list(train_paths, DATA / "train.txt")
    write_list(val_paths, DATA / "val.txt")

    yaml = (
        f"train: {DATA / 'train.txt'}\n"
        f"val: {DATA / 'val.txt'}\n"
        "nc: 4\n"
        "names: ['boat', 'ship', 'sailboat', 'usv']\n"
    )
    (DATA / "yoloow_2gb.yaml").write_text(yaml)
    meta = {
        "target_gb": 2.0,
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "train_bytes": train_used,
        "val_bytes": val_used,
        "gb": (train_used + val_used) / 1e9,
        "train_seqs": [s[0] for s in train_sel],
        "val_seqs": [s[0] for s in val_sel],
        "reason": "2GB cut for YoloOW fine-tune on a 6GB GPU (time/machine)",
    }
    (OUT / "subset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] train={len(train_paths)} val={len(val_paths)} ~{meta['gb']:.2f}GB")
    print(yaml)


if __name__ == "__main__":
    main()
