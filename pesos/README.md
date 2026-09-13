Coloque aqui os checkpoints (esta pasta **não** versiona `.pt`: ver `.gitignore` e `docs/WEIGHTS.md`).

| Ficheiro | Uso |
|----------|-----|
| `A_yolov8s_seadronessee.pt` | YOLO-SEA / YOLOv8s no SeaDronesSee |
| `B_YoloOW.pt` | YoloOW pré-treino open-water |
| `yolov8s.pt` | init COCO (K-Fold e MVTD A) |
| `mvtd_yolov8s_v2_best.pt` / `mvtd_yolov8s_best.pt` | YOLO-SEA fine-tune MVTD |
| `B_YoloOW_mvtd.pt` | YoloOW fine-tune MVTD (4 classes) |

Alternativa: deixar os pesos no workspace (`$WEIGHTS_DIR`, `$MVTD_PESOS`) e só exportar as variáveis em `env.sh`.
