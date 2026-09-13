#!/usr/bin/env python3
"""GroupKFold 5 por video_id (sem vazamento temporal entre folds)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import GroupKFold

import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from paths import COCO_ANN, OUT, SEADRONESSEE_4GB  # noqa: E402

DATA = SEADRONESSEE_4GB
COCO = COCO_ANN
FOLDS = OUT / "yolosea" / "folds"
K, SEED = 5, 20
NAMES = ["swimmer", "boat", "jetski", "buoy", "life_saving_appliances"]


def stem_to_video():
    m = {}
    for name in (
        "instances_train_objects_in_water.json",
        "instances_val_objects_in_water.json",
        "instances_test_objects_in_water.json",
    ):
        split = "train" if "train" in name else ("val" if "val" in name else "test")
        coco = json.loads((COCO / name).read_text())
        for im in coco["images"]:
            m[f"{split}_{Path(im['file_name']).stem}"] = int(im["video_id"])
    return m


def main():
    FOLDS.mkdir(parents=True, exist_ok=True)
    s2v = stem_to_video()
    rows = []
    for p in sorted((DATA / "images").glob("*.jpg")):
        if not (p.name.startswith("train_") or p.name.startswith("val_")):
            continue
        vid = s2v.get(p.stem)
        if vid is None:
            continue
        rel = str(DATA / "images" / p.name)  # NÃO resolve() — é symlink; YOLO troca /images/ → /labels/
        rows.append((rel, vid, p.stem))
    X = [r[0] for r in rows]
    groups = [r[1] for r in rows]
    print(f"[kfold] n={len(X)} videos={len(set(groups))} K={K} seed={SEED}")

    gkf = GroupKFold(n_splits=K)
    # GroupKFold ignora shuffle; ordem dos grupos é estável
    summary = []
    for i, (tr, te) in enumerate(gkf.split(X, groups=groups), 1):
        d = FOLDS / f"fold_{i}"
        d.mkdir(exist_ok=True)
        train_p, val_p = [X[j] for j in tr], [X[j] for j in te]
        (d / "train.txt").write_text("\n".join(train_p) + "\n")
        (d / "val.txt").write_text("\n".join(val_p) + "\n")
        yml = {
            "path": str(DATA),
            "train": str((d / "train.txt").resolve()),
            "val": str((d / "val.txt").resolve()),
            "nc": 5,
            "names": NAMES,
        }
        (d / "data.yaml").write_text(yaml.safe_dump(yml, sort_keys=False, allow_unicode=True))
        tr_v = sorted({groups[j] for j in tr})
        te_v = sorted({groups[j] for j in te})
        rec = {"fold": i, "n_train": len(train_p), "n_val": len(val_p), "val_videos": te_v, "train_videos": tr_v}
        summary.append(rec)
        print(f"  fold {i}: train={len(train_p)} val={len(val_p)} val_vids={te_v}")

    (FOLDS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[ok]", FOLDS)


if __name__ == "__main__":
    main()
