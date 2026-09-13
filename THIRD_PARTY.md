# Terceiros

Este repositório **não** inclui pesos (`.pt`) nem datasets. O clone do YoloOW também **não** vai para o Git: clone-o à parte (ver `docs/SETUP.md`).

| Componente | Uso aqui | Origem | Notas |
|------------|----------|--------|--------|
| YoloOW | detector B (YOLOv7-style) | [Xjh-UCAS/YoloOW](https://github.com/Xjh-UCAS/YoloOW) | Paper IEEE TGRS 2024. Clone em `finetune_yoloow/YoloOW/`. Pesos: [release v0.1](https://github.com/Xjh-UCAS/YoloOW/releases/download/v0.1/YoloOW.pt). |
| YOLOv7 | base do YoloOW | WongKinYiu / YOLOv7 | GPL-3.0 típico da linhagem. |
| Ultralytics YOLO | detector A (YOLOv8s) | [ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0. Init **COCO** (`yolov8s.pt`), nunca um peso já treinado no mesmo dataset. |
| Soft-NMS | pós-processamento A | Bodla et al., ICCV 2017 | Implementação em `latefusion/latefusion.py`. |
| Weighted Boxes Fusion | pool depois do DST | [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) | Fallback NMS se o pacote não estiver instalado. |
| BoT-SORT + CMC | tracking | [boxmot](https://github.com/mikel-brostrom/boxmot) | `per_class=True`, CMC `sof`. Sem rastro em água. |
| SeaDronesSee MOT | dataset A/B/fusão mar UAV | [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/) | 5 classes; test público **sem** boxes. |
| MVTD | dataset marítimo SOT→YOLO | [AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD](https://huggingface.co/datasets/AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD) | 4 classes: boat, ship, sailboat, usv. |

A combinação DST+WBF deste projeto é **inspirada** em Dempster-Shafer. O conflito `K = m_A m_B |m_A − m_B|` é um gate suave, não o K clássico de Dempster. KEEP_BOTH (sobreviver as duas boxes se `K ≥ 0.55`) é escolha de desenho.
