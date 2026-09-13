# Pesos (não versionados)

**Regra:** nenhum `.pt` / `.pth` / `.onnx` / `.engine` entra no GitHub. O `.gitignore` bloqueia; o CI `no-weights` falha o push se escapar.

Coloque os ficheiros numa destas pastas (a primeira que existir ganha, ver `paths.py` / `run_mvtd.py`):

## SeaDronesSee (A domínio + B open-water)

| Ficheiro | Função | Onde procurar |
|----------|--------|----------------|
| `yolov8s_seadronessee.pt` | A (YOLOv8s fine-tune Sea) | `$WEIGHTS_DIR/` ou `pesos/A_yolov8s_seadronessee.pt` |
| `YoloOW.pt` | B pré-treino open-water (7 classes) | `$WEIGHTS_DIR/` ou `pesos/B_YoloOW.pt` |
| `yolov8s.pt` | init COCO (K-Fold / MVTD A) | `$FUSAO_ROOT/yolov8s.pt` ou `pesos/` |

Download B oficial: [YoloOW.pt (release v0.1)](https://github.com/Xjh-UCAS/YoloOW/releases/download/v0.1/YoloOW.pt).

Init do K-Fold **sempre** COCO. Não usar `yolov8s_seadronessee.pt` como init: o peso já viu o dataset e vaza o fold de validação.

## MVTD (4 classes)

| Ficheiro | Função | Onde procurar |
|----------|--------|----------------|
| `mvtd_yolov8s_best.pt` (ou `mvtd_yolov8s_v2_best.pt`) | A fine-tune MVTD | `$MVTD_PESOS/` ou `pesos/` |
| `B_YoloOW.pt` | B init (peso Sea, head 7→4) | `$MVTD_PESOS/` |
| `B_YoloOW_mvtd.pt` | B fine-tune 4 classes | `$MVTD_PESOS/` ou `runs/yoloow/pesos/` |

Se `B_YoloOW_mvtd.pt` existir, o remap de classes é identidade `{0,1,2,3}`. Se só existir o peso Sea, YoloOW só contribui **boat** (`id 5 → 0`).

## Pasta `pesos/` deste repo

No git há só `pesos/README.md`. Copie os `.pt` para aqui **ou** deixe-os no workspace (`$WEIGHTS_DIR`, `$MVTD_PESOS`). Não faça `git add pesos/*.pt`.

## O que não misturar

- Peso SeaDronesSee (5 classes) como init do MVTD (4 classes).
- Peso já treinado no mesmo dataset como init de K-Fold.
- YOLO-SEA Entropy 2025 (SESA/BiFPN): o paper **não** libera esses pesos. Aqui A = YOLOv8s-Sea + Soft-NMS.
