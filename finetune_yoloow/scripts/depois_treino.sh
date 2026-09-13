#!/usr/bin/env bash
# Copia o peso do YoloOW 2GB e corre val+test. Não treina.
# SÓ corre se FROM_WATCH_TREINO=1 (watch_treino.sh ou run_tudo.sh).
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"

if [ "${FROM_WATCH_TREINO:-}" != "1" ]; then
  echo "[recusado] depois_treino.sh só corre via watch_treino.sh ou run_tudo.sh"
  exit 1
fi

FT="$YOLO_GIT_OUT/yoloow"
PAPER="$FT/paper_compare"
mkdir -p "$FT/pesos" "$FT/metricas" "$PAPER" "$YOLO_GIT_ROOT/pesos"

LOCK="$FT/metricas/depois_treino.lock"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" || true)
  if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
    echo "[skip] depois_treino já a correr pid=$old"
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

BEST="$FT/yoloow_2gb/weights/best.pt"
LAST="$FT/yoloow_2gb/weights/last.pt"
SRC=""
if [ -f "$BEST" ]; then SRC="$BEST"; echo "[peso] best.pt"
elif [ -f "$LAST" ]; then SRC="$LAST"; echo "[peso] last.pt (best ainda não existe)"
else
  echo "[falha] nem best.pt nem last.pt em $FT/yoloow_2gb/weights"
  ls -la "$FT/yoloow_2gb/weights" || true
  exit 1
fi

DST_FT="$FT/pesos/B_YoloOW_mvtd_2gb.pt"
DST_REPO="$YOLO_GIT_ROOT/pesos/B_YoloOW_mvtd.pt"
cp -f "$SRC" "$DST_FT"
cp -f "$SRC" "$DST_REPO"
cp -f "$SRC" "$PAPER/B_YoloOW_mvtd_2gb.pt"
if [ -f "$FT/yoloow_2gb/results.txt" ]; then
  cp -f "$FT/yoloow_2gb/results.txt" "$FT/metricas/train_results.txt"
  cp -f "$FT/yoloow_2gb/results.txt" "$PAPER/train_results.txt"
fi
ls -lh "$DST_FT" "$DST_REPO"
echo "[ok] pesos copiados"

EVAL_JSON="$FT/metricas/eval_mvtd.json"
if [ -f "$EVAL_JSON" ] && [ "$EVAL_JSON" -nt "$DST_FT" ]; then
  echo "[skip] eval já existe e é mais novo que o peso"
else
  echo "======== val + test MVTD (A vs B_2gb vs fusao, stride 5) ========"
  "$PYTHON" -u "$YOLO_GIT_ROOT/finetune_yoloow/scripts/eval_mvtd.py" \
    --weights "$DST_FT" \
    --stride 5 \
    --split both \
    2>&1 | tee "$FT/metricas/eval.log"
fi

cp -f "$FT/metricas/eval_mvtd.json" "$PAPER/eval_mvtd.json" 2>/dev/null || true
cp -f "$FT/metricas/NOTAS.md" "$PAPER/NOTAS.md" 2>/dev/null || true

echo "======== stills + videos A/B/fusao (val+test) ========"
"$PYTHON" -u "$YOLO_GIT_ROOT/finetune_yoloow/scripts/paper_midias.py" \
  --weights "$DST_FT" \
  --max-seq-val 2 \
  --max-seq-test 2 \
  --max-video-frames 80 \
  2>&1 | tee "$FT/metricas/paper_midias.log"

echo "[done] paper_compare -> $PAPER"
