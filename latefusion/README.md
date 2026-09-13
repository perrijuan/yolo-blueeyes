# Late fusion DS-WBF

Núcleo em `latefusion.py` (só numpy; `ensemble_boxes` opcional para WBF).

1. Match A↔B, mesma classe, IoU ≥ 0.35
2. Massas `m_A = conf_A × 0.95`, `m_B = conf_B × 0.60`
3. Dempster combina; se conflito `K ≥ 0.55` as duas boxes sobrevivem
4. WBF no pool (fundidos + sobras)
5. Tracker BoT-SORT só na fusão, ordem `frame_index` (nunca `enumerate`)

```bash
python latefusion/run_seadronessee.py
python latefusion/run_mvtd.py --split both --stride 5
python latefusion/yolosea_yoloow_dswbf.py
```

A (YOLO-SEA) usa Soft-NMS gaussiano em vez de NMS duro. Detalhe: `docs/LATEFUSION.md`.
