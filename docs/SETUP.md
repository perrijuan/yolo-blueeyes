# Setup

## Requisitos

- Python 3.11 (o workspace local usa `.venv` em `$FUSAO_ROOT/.venv`)
- GPU NVIDIA (testado: RTX 3060 Laptop 6 GB). CPU funciona para o kernel DS-WBF, não para eval 4K.
- `git`, `pip`

## 1. Código deste repo

```bash
git clone <repository-url>
cd dswbf-latefusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pacotes principais: `ultralytics`, `torch`, `opencv-python`, `ensemble-boxes`, `boxmot`.

## 2. YoloOW (não está no GitHub deste projeto)

O detector B é o código de [Xjh-UCAS/YoloOW](https://github.com/Xjh-UCAS/YoloOW). A pasta `finetune_yoloow/YoloOW/` está no `.gitignore` (clone local / symlink absoluto não pode ir para o git).

```bash
git clone https://github.com/Xjh-UCAS/YoloOW.git finetune_yoloow/YoloOW
pip install -r finetune_yoloow/YoloOW/requirements.txt
```

`paths.py` também aceita um clone em `$FUSAO_ROOT/YoloOW` se a pasta acima não existir.

PyTorch ≥ 2.6: o `train.py` do YoloOW precisa de `torch.load(..., weights_only=False)` (o código original pode falhar sem monkeypatch). A inferência em `latefusion/run_mvtd.py` já usa `weights_only=False`.

## 3. Caminhos

```bash
cp env.example .env
# edite FUSAO_ROOT, DATASETS_ROOT, WEIGHTS_DIR, MVTD_*
source env.sh
```

Variáveis (todas opcionais; ver `env.sh` e `paths.py`):

| Env | Default | O quê |
|-----|---------|--------|
| `FUSAO_ROOT` | pasta com `datasets/` e `weights/` | workspace |
| `DATASETS_ROOT` | `$FUSAO_ROOT/datasets` | SeaDronesSee |
| `WEIGHTS_DIR` | `$FUSAO_ROOT/weights` | `yolov8s_seadronessee.pt`, `YoloOW.pt` |
| `MVTD_ROOT` | `$FUSAO_ROOT/yolo_blue_eyes/mvtd` | raw + yolo + pesos MVTD |
| `MVTD_YOLO` | `$MVTD_ROOT/yolo` | árvore Ultralytics |
| `MVTD_PESOS` | `$MVTD_ROOT/pesos` | `mvtd_yolov8s_best.pt`, `B_YoloOW_mvtd.pt` |
| `YOLOOW_ROOT` | `./finetune_yoloow/YoloOW` | código B |
| `YOLO_GIT_OUT` | `./runs` | saídas (gitignore) |
| `COCO_INIT` | `$FUSAO_ROOT/yolov8s.pt` | init K-Fold / MVTD A |
| `PYTHON` | `$FUSAO_ROOT/.venv/bin/python` | scripts `.sh` |

Não commitar `.env`.

## 4. Pesos e dados

Ver [WEIGHTS.md](WEIGHTS.md) e [DATASETS.md](DATASETS.md). Nada disso entra no git.

## 5. Verificar

```bash
source env.sh
python -c "from paths import YOLOOW_SRC, MVTD_YOLO, WEIGHTS; print(YOLOOW_SRC, MVTD_YOLO.exists(), (WEIGHTS/'YoloOW.pt').exists())"
python smoke_infer.py
```

## GPU

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` já vai no `env.sh`. Em 6 GB: YOLO-SEA MVTD `imgsz=1280 batch=4`; YoloOW MVTD `imgsz=640 batch=2`. Não lançar um segundo `train.py`.
