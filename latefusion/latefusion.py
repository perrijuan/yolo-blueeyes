#!/usr/bin/env python3
"""Núcleo da late fusion DS-WBF (Dempster-Shafer + WBF) + Soft-NMS.

Independente do restante do repo: só numpy + ensemble_boxes (opcional).
"""
from __future__ import annotations

import numpy as np

DST_MATCH_IOU = 0.35
DST_REL_A = 0.95
DST_REL_B = 0.60
DST_CONFLICT = 0.55
WBF_IOU = 0.55
SOFTNMS_SIGMA = 0.5
SOFTNMS_IOU = 0.50
SOFTNMS_MIN = 0.05


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)


def dst_combine(ma, mb):
    ma = float(np.clip(ma, 0.0, 0.999))
    mb = float(np.clip(mb, 0.0, 0.999))
    ua, ub = 1.0 - ma, 1.0 - mb
    numer = ma * mb + ma * ub + ua * mb
    K = ma * mb * abs(ma - mb)
    denom = 1.0 - K
    if denom <= 1e-9:
        return 0.0, 1.0
    return float(np.clip(numer / denom, 0.0, 1.0)), float(K)


def soft_nms(dets, sigma=SOFTNMS_SIGMA, iou_thr=SOFTNMS_IOU, min_score=SOFTNMS_MIN):
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), np.float32)
    dets = np.asarray(dets, np.float32)
    keep = []
    for cls in np.unique(dets[:, 5].astype(int)):
        d = dets[dets[:, 5].astype(int) == cls].copy()
        while len(d):
            i = int(np.argmax(d[:, 4]))
            cur = d[i].copy()
            keep.append(cur)
            d = np.delete(d, i, 0)
            if len(d) == 0:
                break
            x1 = np.maximum(cur[0], d[:, 0])
            y1 = np.maximum(cur[1], d[:, 1])
            x2 = np.minimum(cur[2], d[:, 2])
            y2 = np.minimum(cur[3], d[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            ua = (cur[2] - cur[0]) * (cur[3] - cur[1])
            ub = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
            iou = inter / (ua + ub - inter + 1e-6)
            decay = np.where(iou >= iou_thr, np.exp(-(iou ** 2) / sigma), 1.0)
            d[:, 4] *= decay
            d = d[d[:, 4] >= min_score]
    if not keep:
        return np.empty((0, 6), np.float32)
    out = np.stack(keep, 0)
    return out[np.argsort(-out[:, 4])]


def _xyxy_to_norm(boxes, w, h):
    if len(boxes) == 0:
        return np.empty((0, 4))
    b = boxes.copy()
    b[:, [0, 2]] /= max(w, 1e-6)
    b[:, [1, 3]] /= max(h, 1e-6)
    return np.clip(b, 0, 1)


def nms_pool(dets, w, h, iou_thr=WBF_IOU):
    if dets is None or len(dets) == 0:
        return np.empty((0, 6), np.float32)
    dets = np.asarray(dets, np.float32)
    try:
        from ensemble_boxes import weighted_boxes_fusion
        boxes, scores, labels = weighted_boxes_fusion(
            [_xyxy_to_norm(dets[:, :4], w, h).tolist()],
            [dets[:, 4].tolist()],
            [dets[:, 5].astype(int).tolist()],
            weights=[1.0], iou_thr=iou_thr, skip_box_thr=0.01,
        )
        if len(boxes) == 0:
            return np.empty((0, 6), np.float32)
        b = np.array(boxes, np.float32)
        b[:, [0, 2]] *= w
        b[:, [1, 3]] *= h
        return np.concatenate([b, np.array(scores)[:, None], np.array(labels)[:, None]], 1).astype(np.float32)
    except Exception:
        keep, used = [], np.zeros(len(dets), dtype=bool)
        for i in np.argsort(-dets[:, 4]):
            if used[i]:
                continue
            keep.append(dets[i])
            for j in range(len(dets)):
                if used[j] or int(dets[i, 5]) != int(dets[j, 5]):
                    continue
                if iou_xyxy(dets[i, :4], dets[j, :4]) >= iou_thr:
                    used[j] = True
        return np.stack(keep, 0).astype(np.float32) if keep else np.empty((0, 6), np.float32)


def ds_wbf_fuse(dets_a, dets_b, w, h):
    """Late fusion A+B: match → DST (com conflito) → WBF."""
    if len(dets_a) == 0 and len(dets_b) == 0:
        return np.empty((0, 6), np.float32)
    if len(dets_a) == 0:
        return np.asarray(dets_b, np.float32)
    if len(dets_b) == 0:
        return np.asarray(dets_a, np.float32)
    used_b = set()
    fused, leftover_a = [], []
    leftover_b = set(range(len(dets_b)))
    for da in dets_a:
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(dets_b):
            if j in used_b or int(da[5]) != int(db[5]):
                continue
            iou = iou_xyxy(da[:4], db[:4])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= DST_MATCH_IOU:
            db = dets_b[best_j]
            used_b.add(best_j)
            leftover_b.discard(best_j)
            ma, mb = float(da[4]) * DST_REL_A, float(db[4]) * DST_REL_B
            m_star, K = dst_combine(ma, mb)
            if K >= DST_CONFLICT:
                leftover_a.append(da)
                leftover_b.add(best_j)
                continue
            wa, wb = max(ma, 1e-6), max(mb, 1e-6)
            box = (wa * da[:4] + wb * db[:4]) / (wa + wb)
            cls = da[5] if ma >= mb else db[5]
            fused.append(np.array([*box, m_star, cls], np.float32))
        else:
            leftover_a.append(da)
    parts = []
    if fused:
        parts.append(np.stack(fused, 0))
    if leftover_a:
        parts.append(np.stack(leftover_a, 0).astype(np.float32))
    if leftover_b:
        parts.append(np.stack([dets_b[j] for j in sorted(leftover_b)], 0).astype(np.float32))
    pool = np.concatenate(parts, 0) if parts else np.empty((0, 6), np.float32)
    return nms_pool(pool, w, h, iou_thr=WBF_IOU)
