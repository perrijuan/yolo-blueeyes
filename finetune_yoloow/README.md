# Fine-tune YoloOW

`YoloOW/` **não está neste GitHub** (`.gitignore`). Clone o código de treino/inferência do paper
[YoloOW (IEEE TGRS)](https://ieeexplore.ieee.org/abstract/document/10517350)
([GitHub Xjh-UCAS/YoloOW](https://github.com/Xjh-UCAS/YoloOW)):

```bash
git clone https://github.com/Xjh-UCAS/YoloOW.git finetune_yoloow/YoloOW
```

## SeaDronesSee (7 classes, receita do paper)

```bash
cd finetune_yoloow/YoloOW
python train.py --workers 8 --device 0 --batch-size 4 \
  --data data/sea_drones_see.yaml --img 1280 \
  --cfg cfg/training/yoloOW.yaml --weights /caminho/YoloOW.pt \
  --name yoloOW --hyp data/hyp.scratch.sea.yaml --epoch 300
```

Ajuste `data/sea_drones_see.yaml` para o split local.

## MVTD (4 classes)

Arquitetura: `configs/yoloOW-nc4.yaml` (`nc: 4`). Hipers: `configs/hyp.yoloow.mvtd.yaml`.

```bash
# listas strided + treino (batch 2, imgsz 640, 15 épocas)
bash finetune_yoloow/scripts/train_mvtd.sh

# subset ~2 GB → treino → copia peso → eval A/B/fusão
bash finetune_yoloow/scripts/run_tudo.sh
```

Init: `MVTD_PESOS/B_YoloOW.pt` (peso Sea). Head de 7 → 4 classes; o `yoloOW-nc4.yaml` redefine `nc`.
