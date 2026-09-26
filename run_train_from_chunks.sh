#!/bin/bash

# Останавливаемся при первой ошибке
set -e

CHUNKS_DIR="data/chunks"
AI_DATA_DIR="ai_data"
COMBINED="$AI_DATA_DIR/combined_chunks.parquet"
EPOCHS_BASELINE="${EPOCHS_BASELINE:-200}"
VAL_FRAC="${VAL_FRAC:-0.1}"

mkdir -p "$AI_DATA_DIR"

echo "🚀 Старт обучения из готовых чанков ($CHUNKS_DIR)..."

# [1/5] Собираем все parquet-чанки
echo "📦 [1/5] Ищем чанки в $CHUNKS_DIR..."
CHUNKS=( "$CHUNKS_DIR"/chunk-*.parquet )
if [ ${#CHUNKS[@]} -lt 2 ]; then
  echo "❌ Нужно как минимум 2 чанка в $CHUNKS_DIR (найдено: ${#CHUNKS[@]})"
  exit 1
fi
echo "   Найдено ${#CHUNKS[@]} чанков:"
printf '     - %s\n' "${CHUNKS[@]}"

# [2/5] Проверяем целостность по .sha256 (если есть)
echo "🔐 [2/5] Проверяем контрольные суммы..."
shopt -s nullglob
SUMS=( "$CHUNKS_DIR"/*.sha256 )
shopt -u nullglob
if [ ${#SUMS[@]} -gt 0 ]; then
  ( cd "$CHUNKS_DIR" && sha256sum -c --quiet "${SUMS[@]##*/}" ) \
    && echo "   ✅ Все суммы совпали" \
    || { echo "❌ Контрольная сумма не сошлась"; exit 1; }
else
  echo "   ⚠️  .sha256 файлов нет, пропускаем"
fi

# [3/5] Склеиваем чанки в один parquet
echo "🧬 [3/5] Объединяем чанки -> $COMBINED..."
cargo run --release -p lunar-ai-cli -- \
  combine \
  --inputs "${CHUNKS[@]}" \
  --output "$COMBINED"

# [4/5] Baseline-обучение с валидацией
echo "🧠 [4/5] Baseline-обучение ($EPOCHS_BASELINE эпох, val-frac=$VAL_FRAC)..."
cargo run --release -p lnai -- \
  --data "$COMBINED" \
  --epochs "$EPOCHS_BASELINE" \
  --val-frac "$VAL_FRAC"

# [5/5] Holdout-проверка (если файл существует)
HOLDOUT="$AI_DATA_DIR/holdout.parquet"
if [ -f "$HOLDOUT" ]; then
  echo "📊 [5/5] Проверка на holdout-датасете..."
  cargo run --release -p lnai -- \
    --data "$COMBINED" \
    --holdout "$HOLDOUT"
else
  echo "⏭️  [5/5] $HOLDOUT не найден — пропускаем holdout-этап"
  echo "   (добавь holdout.parquet в $AI_DATA_DIR/ для независимой проверки)"
fi

echo "👑 Готово! Контрольный набор не включался в обучение; веса см. в каталоге обучения (.bpk)."
