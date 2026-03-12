#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Конфигурация
RESULTS_DIR="data/eda/evaluation_results"
mkdir -p "$RESULTS_DIR"

# Batch sizes для валидации (из таблицы)
declare -A VAL_BATCHES=(
    ["resnet18_age_db"]=512
    ["resnet18_appa_real"]=512
    ["resnet18_cacd"]=1024
    ["resnet18_imdb_clean"]=512
    ["resnet18_morph_2"]=1024

    ["resnet50_age_db"]=256
    ["resnet50_appa_real"]=256
    ["resnet50_cacd"]=512
    ["resnet50_imdb_clean"]=256
    ["resnet50_morph_2"]=512

    ["convnext_paper_age_db"]=256
    ["convnext_paper_appa_real"]=128
    ["convnext_paper_cacd"]=512
    ["convnext_paper_imdb_clean"]=256
    ["convnext_paper_morph_2"]=512

    ["transformer_paper_age_db"]=512
    ["transformer_paper_appa_real"]=512
    ["transformer_paper_cacd"]=1024
    ["transformer_paper_imdb_clean"]=512
    ["transformer_paper_morph_2"]=1024

    ["hybrid_paper_age_db"]=128
    ["hybrid_paper_appa_real"]=128
    ["hybrid_paper_cacd"]=256
    ["hybrid_paper_imdb_clean"]=128
    ["hybrid_paper_morph_2"]=256
)

# Модели и их типы
declare -A MODEL_TYPES=(
    ["resnet18"]="resnet18"
    ["resnet50"]="resnet50"
    ["convnext_paper"]="convnext"
    ["transformer_paper"]="transformer"
    ["hybrid_paper"]="hybrid"
)

# Датасеты
DATASETS=("cacd" "imdb_clean" "morph_2" "age_db" "appa_real")
SPLITS=("train" "val" "test")

# Чекпоинты для оцениваемых моделей (фиксированные)
MODEL_CHECKPOINTS=(
    "resnet18=checkpoints/resnet18_pretrained.ckpt.ckpt"
    "resnet50=checkpoints/resnet50_pretrained.ckpt.ckpt"
    "convnext_paper=checkpoints/convnext_paper.ckpt"
    "transformer_paper=checkpoints/transformer_paper.ckpt"
    "hybrid_paper=checkpoints/hybrid_paper.ckpt"
)

# Функция для получения batch size
get_batch_size() {
    local model=$1
    local dataset=$2
    echo "${VAL_BATCHES[${model}_${dataset}]}"
}

# Основной файл результатов
RESULTS_CSV="$RESULTS_DIR/all_results.csv"
echo "model,dataset,split,loss,mae,mape,mse" > "$RESULTS_CSV"

# Функция для оценки одной модели
evaluate_model() {
    local model_key=$1
    local model_type=${MODEL_TYPES[$model_key]}
    local dataset=$2
    local split=$3
    local checkpoint=$4
    local batch_size
    batch_size=$(get_batch_size "$model_key" "$dataset")

    echo -e "\n${YELLOW}Оценка: $model_key на $dataset ($split), batch=$batch_size${NC}"

    # Формируем команду
    CMD="uv run age-by-face evaluate \
        model=$model_type \
        dataset=$dataset \
        eval.split=$split \
        eval.verbose=true \
        dataset.val_batch_size=$batch_size \
        dataset.num_workers=8"

    # Добавляем чекпоинт
    if [ -n "$checkpoint" ]; then
        CMD="$CMD model.checkpoint_path=$checkpoint"
    fi

    # Запускаем
    echo "$CMD"
    OUTPUT=$(eval "$CMD" 2>&1)

    # Извлекаем метрики из вывода
    LOSS=$(echo "$OUTPUT" | grep -oP 'test_loss: \K[\d.]+')
    MAE=$(echo "$OUTPUT" | grep -oP 'test_mae: \K[\d.]+')
    MAPE=$(echo "$OUTPUT" | grep -oP 'test_mape: \K[\d.]+')
    MSE=$(echo "$OUTPUT" | grep -oP 'test_mse: \K[\d.]+')

    # Сохраняем
    echo "$model_key,$dataset,$split,$LOSS,$MAE,$MAPE,$MSE" >> "$RESULTS_CSV"
    echo -e "${GREEN}  MAE=$MAE, MAPE=$MAPE${NC}"
}

# Основной цикл
echo "========================================="
echo "Запуск оценки всех моделей"
echo "========================================="

# Оцениваем все модели (на всех датасетах)
for model_entry in "${MODEL_CHECKPOINTS[@]}"; do
    model_key="${model_entry%%=*}"
    ckpt="${model_entry#*=}"

    if [ -f "$ckpt" ]; then
        for dataset in "${DATASETS[@]}"; do
            for split in "${SPLITS[@]}"; do
                evaluate_model "$model_key" "$dataset" "$split" "$ckpt"
            done
        done
    else
        echo -e "${RED}Чекпоинт не найден: $ckpt${NC}"
    fi
done

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ Оценка завершена${NC}"
echo -e "${GREEN}=========================================${NC}"
echo "Результаты сохранены в: $RESULTS_CSV"

# Показываем сводку
echo -e "\n${YELLOW}Сводка по test MAE:${NC}"
column -s, -t < "$RESULTS_CSV" | grep "test" | sort -t, -k3 -n | head -20
