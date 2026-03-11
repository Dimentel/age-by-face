#!/bin/bash

# Собираем метрики для всех моделей и датасетов
file_name=${1:-"all_metrics_base.csv"}
RESULTS_FILE="data/eda/evaluation_results/${file_name}"
echo "model,dataset,split,loss,mae,mape,mse" > "$RESULTS_FILE"

# Функция для поиска последнего лога
find_latest_log() {
    local model=$1
    local dataset=$2
    local split=$3

    # Ищем в logs/eval/
    find "logs/eval" -maxdepth 1 -name "${model}_*_${dataset}_${split}_*" -type d | sort | tail -1
}

# Функция для извлечения метрик из CSV
extract_metrics() {
    local log_dir=$1
    local csv_file="$log_dir/metrics.csv"

    if [ -f "$csv_file" ]; then
        # Берём последнюю строку (итоговые метрики)
        tail -1 "$csv_file" | awk -F, '{print $2","$3","$4","$5}'
    else
        echo ",,,"
    fi
}

# Все модели
MODELS=(
#    "resnet18"
#    "resnet50"
    "convnext_paper"
    "transformer_paper"
    "hybrid_paper"
)

# Все датасеты
DATASETS=("morph_2")  #"cacd" "imdb_clean"  "age_db" "appa_real"
SPLITS=("test") # "train" "val"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for split in "${SPLITS[@]}"; do
            log_dir=$(find_latest_log "$model" "$dataset" "$split")
            if [ -n "$log_dir" ]; then
                metrics=$(extract_metrics "$log_dir")
                echo "$model,$dataset,$split,$metrics" >> "$RESULTS_FILE"
                echo "✅ $model $dataset $split"
            else
                echo "❌ $model $dataset $split - лог не найден"
            fi
        done
    done
done

echo -e "\n✅ Метрики сохранены в $RESULTS_FILE"
