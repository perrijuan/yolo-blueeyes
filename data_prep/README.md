# Preparação do MVTD

Converte o [MVTD](https://huggingface.co/datasets/AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD)
(GOT-10k, 1 bbox/frame) para YOLO Ultralytics.

Val **não** existe no paper: este script recorta 20% das sequências de treino,
estratificado por classe, sem vazar frames do mesmo vídeo. Test = split oficial.

```bash
python data_prep/convert_got10k_to_yolo.py \
  --raw "$MVTD_RAW" \
  --out "$MVTD_YOLO"
```

Classes: boat, ship, sailboat, usv.
