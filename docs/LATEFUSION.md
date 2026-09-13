# Late Fusion DS-WBF + Tracking — documentação

## 1. Por que late fusion

A (YOLOv8s-Sea / YOLO-SEA Soft-NMS) é o detector de **domínio**: recall alto em nadador, mais FP.
B (YoloOW) é SOTA open-water: **precisão** alta em barco, recall baixo em nadador.

Combinar *scores* com um fator constante (`conf *= 0.775`) ignora conflito. Aqui a evidência A e B é combinada com um gate **inspirado** em Dempster-Shafer e a geometria com **WBF**.

O `K = m_A m_B |m_A − m_B|` **não** é o conflito clássico de Dempster. KEEP_BOTH (`K ≥ 0.55` → as duas boxes sobrevivem) é escolha de desenho. No MVTD isso pode deixar F1 da fusão **abaixo** do A-only.

## 2. Soft-NMS (YOLO-SEA, contribuição 4 do paper)

Em vez de apagar boxes com IoU alto, a confiança decai com gaussiana:

```
s_i ← s_i · exp( −IoU(b*, b_i)² / σ )    se IoU ≥ τ
```

`σ = 0.5`, `τ = 0.50`. Preserva alvos densos (nadadores juntos).

## 3. Dempster–Shafer nas confianças

Para um par A↔B da **mesma classe** com IoU ≥ `DST_MATCH_IOU = 0.35`:

```
m_A = clip(conf_A · REL_A, 0, 0.999)     REL_A = 0.95
m_B = clip(conf_B · REL_B, 0, 0.999)     REL_B = 0.60
u_A = 1 − m_A    (ignorância)
u_B = 1 − m_B

numer = m_A m_B + m_A u_B + u_A m_B
K     = m_A m_B |m_A − m_B|              # conflito suave
m*    = numer / (1 − K)
```

- Se `K ≥ 0.55` (**conflito alto**): as duas boxes **sobrevivem** (não funde).
- Senão: box = `(m_A·box_A + m_B·box_B) / (m_A+m_B)`, score = `m*`.

Isso **não** é um ganho constante: o resultado depende do acordo espacial, da classe e da divergência de confiança.

## 4. Weighted Boxes Fusion

Depois do DST, o pool (fundidos + sobras A + sobras B) passa por WBF (`iou_thr = 0.55`) para fundir duplicatas no referencial da imagem.

## 5. Tracking (temporal)

- Entrada do tracker = **somente** a fusão (não A nem B isolados).
- Sequência ordenada por `frame_index` do COCO (1-based MOTChallenge). **Proibido** `enumerate`.
- BoT-SORT com CMC (`cmc_method=sof`) para câmera UAV.
- `per_class=True`: associação só dentro da mesma classe (`ACTIVE_CLASS_IDS = {0,1,4}`).
- `det_thresh ≈ 0.25`, alinhado à detecção (não 0.6).
- Fovéa: segunda passada em boxes pequenas (`área ≤ 80²`, crop ×1.8) e re-fusão.
- `tracker.update` **loga** exceção; não engole erro.

Formato MOT: `frame,id,x,y,w,h,conf,-1,-1,-1`

## 6. K-Fold temporal

GroupKFold **por `video_id`**: um vídeo inteiro cai em treino ou val, nunca os dois. Todos os frames do vídeo entram, em ordem temporal. Assim não vaza o futuro do mesmo voo.

## 7. Arquivos de pesos

| arquivo | origem |
|---------|--------|
| `pesos/A_yolov8s_seadronessee.pt` | detector de domínio |
| `pesos/B_YoloOW.pt` | YoloOW |
| `pesos/fusion_config.yaml` | limiares DST/WBF/tracker |
| `pesos/kfold/fold_k_best.pt` | YOLOv8s treinado no fold k (init COCO) |
