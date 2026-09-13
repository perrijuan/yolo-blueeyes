# Inferência

Dois níveis: **smoke** (o código carrega e detecta) e **eval** (P/R/F1 no split).

Protocolo local deste repo: IoU ≥ 0.5, mesma classe. Não misturar com mAP do Ultralytics `val`, nem com `Labels=0` do `train.py` YoloOW (quirk do loader YOLOv7, não é métrica).

## Smoke (recomendado depois de clonar)

Requisitos: pesos MVTD + `yolo/images/val` (ver [WEIGHTS.md](WEIGHTS.md), [DATASETS.md](DATASETS.md)).

```bash
source env.sh
python smoke_infer.py
```

O que faz:

1. Kernel DS-WBF em boxes sintéticas (sem GPU).
2. Carrega A (`mvtd_yolov8s_best.pt`) e B (`B_YoloOW_mvtd.pt`) na GPU.
3. 20 frames do MVTD val (sequências diferentes).
4. Pred A (Soft-NMS), pred B (NMS YoloOW), fusão `ds_wbf_fuse`.
5. Opcional: 2 frames SeaDronesSee 4K com `yolov8s_seadronessee.pt` (`imgsz=640`, só sanity).
6. Escreve `runs/smoke_infer/smoke_infer.json` e 3 vis A/B/Fusão (gitignore).

Smoke local (RTX 3060, 2026-09-12), **não citar como resultado de paper**:

| método | n | P | R | F1 | TP | FP | FN |
|--------|---|------|------|------|----|----|-----|
| A | 20 | 0.625 | 0.750 | 0.682 | 15 | 9 | 5 |
| B | 20 | 0.139 | 0.250 | 0.179 | 5 | 31 | 15 |
| Fusão | 20 | 0.320 | 0.800 | 0.457 | 16 | 34 | 4 |

A fusão subiu o recall e baixou o F1 (FP do B + KEEP_BOTH). Esperado no MVTD neste recorte curto.

SeaDronesSee sanity: `val_20706.jpg` e `val_20707.jpg` (3840×2160) → 8 detecções cada.

## Eval MVTD (A vs B vs fusão)

```bash
python latefusion/run_mvtd.py --split val --stride 5
python latefusion/run_mvtd.py --split both --stride 1   # completo; demora
```

Saída: `runs/mvtd/metricas/latefusion_val_test.json`.

`--stride 5` é o molde de paper (subsample temporal). `stride 1` = todos os frames.

## Eval SeaDronesSee

```bash
python latefusion/run_seadronessee.py
```

Usa o 15 GB se `SEADRONESSEE_15GB` existir. Tracking BoT-SORT só na fusão, `frame_index` COCO.

Pipeline paper (Soft-NMS + fóvea + 3 painéis):

```bash
python latefusion/yolosea_yoloow_dswbf.py
```

Tiles 4K + figuras:

```bash
python latefusion/tracking_sahi_paper_blocks.py
```

## Tracking / vídeos

A partir de MOT já gerado (não re-detecta):

```bash
python latefusion/render_tracking_videos.py --vid N   # 0 = todos
python latefusion/track_mvtd.py --split test --max-seq 6 --no-video
```

Escreve `*.part.mp4` e só depois renomeia. Boxes + ID (+ conf). **Sem rastro** (polylines) em água.

## Falhas típicas

| Sintoma | Causa |
|---------|--------|
| `ModuleNotFoundError: paths` | este ficheiro tem de estar na raiz do repo |
| YoloOW não carregou | falta clone em `finetune_yoloow/YoloOW` ou o `.pt` |
| `falta A (YOLOv8s MVTD)` | copiar `mvtd_yolov8s_best.pt` para `$MVTD_PESOS` ou `pesos/` |
| CUDA OOM | fechar outro `train.py`; 6 GB não leva dois treinos |
| Fusão F1 < A no MVTD | leftover B / KEEP_BOTH; documentado, não é crash |
