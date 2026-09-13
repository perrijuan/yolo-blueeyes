# Datasets (não versionados)

Imagens, labels YOLO e JSON COCO **não** entram no git (`*.jpg`, `**/images/`, `**/labels/`, `datasets/`).

Defina `$DATASETS_ROOT` e `$MVTD_ROOT` (ver [SETUP.md](SETUP.md)).

## SeaDronesSee MOT

Fonte: [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/). Classes: `swimmer`, `boat`, `jetski`, `buoy`, `life_saving_appliances`. GT activo na fusão: `{0, 1, 4}`.

| Árvore local típica | Conteúdo |
|---------------------|----------|
| `$DATASETS_ROOT/seadronessee_reduzido_4gb` | subset ~4 GB, jpg 3840×2160 (muitas vezes **symlinks**) |
| `$DATASETS_ROOT/seadronessee_15gb` | recorte ~15 GB (val ≠ val oficial) |
| `$DATASETS_ROOT/seadronessee_yolo` | split oficial YOLO: train 27259 / **val 8584** / test 18253 |
| `$DATASETS_ROOT/SeaDronesSee_MOT/annotations` | `instances_{train,val,test}_objects_in_water.json` |

Armadilhas:

- `Path.resolve()` em imagens-symlink aponta para `Compressed/train/1000.jpg` (sem `/images/`) e o Ultralytics diz "No labels found". Use o caminho que ainda contém `/images/`.
- O **test público não tem boxes** (`annotations: null`; labels YOLO vazios). Não reporte P/R/F1 de test.
- Val oficial = **8584** frames (JSON), não o corte 15 GB (4090) nem o 4 GB.

K-Fold: GroupKFold por `video_id` (`finetune_yolosea/prepare_kfold.py`). Frame MOT = `frame_index` COCO (1-based), nunca `enumerate`.

## MVTD

Fonte: [Maritime Visual Tracking Dataset](https://huggingface.co/datasets/AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD) (GOT-10k, 1 bbox/frame).

```bash
python data_prep/convert_got10k_to_yolo.py --raw "$MVTD_RAW" --out "$MVTD_YOLO"
```

Classes: `boat`, `ship`, `sailboat`, `usv`.

- **Test** = split oficial do paper.
- **Val** não existe no paper: o script recorta 20% das sequências de **treino**, estratificado por classe, sem vazar frames do mesmo vídeo.
- SOT: frames consecutivos são quase iguais. Treino A usa stride por classe (sailboat=1, ship=2, boat/usv=5).

Eval local de fusão: IoU ≥ 0.5, mesma classe. **Não** é AUC de tracker do paper MVTD.

YoloOW-Sea (7 classes) no MVTD 4 classes sem fine-tune quase não contribui (F1 residual). Use `B_YoloOW_mvtd.pt`.

## Detecção em 4K

Não redimensionar o frame 3840×2160 para 640 e chamar isso de detecção. O pipeline SDS fatia tiles 1280×1280 (overlap 0.25) + 1 passada full-frame, mapeia boxes, NMS/WBF no referencial original (`tracking_sahi_paper_blocks.py`).
