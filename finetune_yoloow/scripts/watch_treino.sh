#!/usr/bin/env bash
# Espera o train.py (YoloOW 2GB) acabar e depois publica pesos + val/test.
set -euo pipefail
source "$(cd "$(dirname "$0")/../.." && pwd)/env.sh"

FT="$YOLO_GIT_OUT/yoloow"
LOG="$FT/metricas/watch.log"
mkdir -p "$FT/metricas"

train_alive() {
  pgrep -f "name yoloow_2gb" >/dev/null 2>&1
}

echo "[$(date +%H:%M:%S)] watch_treino: à espera do train.py sair..." | tee -a "$LOG"

if ! train_alive; then
  echo "[$(date +%H:%M:%S)] train.py não está a correr agora. Se o treino já acabou, publico na mesma." | tee -a "$LOG"
else
  while train_alive; do
    echo "[$(date +%H:%M:%S)] treino vivo; próximo check em 60s" | tee -a "$LOG"
    sleep 60
  done
  echo "[$(date +%H:%M:%S)] train.py saiu. à espera de 15s (flush de best.pt)..." | tee -a "$LOG"
  sleep 15
fi

while pgrep -f "finetune_yoloow/scripts/run_all.sh" >/dev/null 2>&1; do
  echo "[$(date +%H:%M:%S)] run_all.sh ainda activo; espero 30s" | tee -a "$LOG"
  sleep 30
done

echo "[$(date +%H:%M:%S)] a publicar pesos + val/test" | tee -a "$LOG"
export FROM_WATCH_TREINO=1
bash "$YOLO_GIT_ROOT/finetune_yoloow/scripts/depois_treino.sh" 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] watch_treino done" | tee -a "$LOG"
