#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

RESULTS_DIR="data/eda/evaluation_results"
LOGS_DIR="logs/eval"
mkdir -p "$RESULTS_DIR"

# Batch sizes
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

# Models
declare -A MODEL_TYPES=(
    ["resnet18"]="resnet18"
    ["resnet50"]="resnet50"
    ["convnext_paper"]="convnext"
    ["transformer_paper"]="transformer"
    ["hybrid_paper"]="hybrid"
)

# Datasets
DATASETS=("cacd" "imdb_clean" "morph_2" "age_db" "appa_real")
SPLITS=("train" "val" "test")

# Checkpoints to evaluate
MODEL_CHECKPOINTS=(
    "resnet18=checkpoints/resnet18_pretrained.ckpt"
    "resnet50=checkpoints/resnet50_pretrained.ckpt"
    "convnext_paper=checkpoints/convnext_paper.ckpt"
    "transformer_paper=checkpoints/transformer_paper.ckpt"
    "hybrid_paper=checkpoints/hybrid_paper.ckpt"
)

# Function to get batch size
get_batch_size() {
    echo "${VAL_BATCHES[${1}_${2}]}"
}

# Function to find latest log directory
find_latest_log() {
    find "$LOGS_DIR" -maxdepth 1 -name "${1}_*_${2}_${3}_*" -type d | sort | tail -1
}

# Function to extract metrics from log file
extract_metrics_from_log() {
    local log_dir=$1
    local csv_file="$log_dir/metrics.csv"

    if [ ! -f "$csv_file" ]; then
        echo ",,,"
        return
    fi

    # Get last line test_loss, test_mae, test_mape, test_mse (columns 3-6)
    tail -1 "$csv_file" | awk -F, '{print $3","$4","$5","$6}'
}

# Function to evaluate model
evaluate_model() {
    local model_key=$1
    local model_type=${MODEL_TYPES[$model_key]}
    local dataset=$2
    local split=$3
    local checkpoint=$4
    local batch_size
    batch_size=$(get_batch_size "$model_key" "$dataset")

    echo -e "\n${YELLOW}Оценка: $model_key на $dataset ($split), batch=$batch_size${NC}"

    # Make command
    CMD="uv run age-by-face evaluate \
        model=$model_type \
        dataset=$dataset \
        eval.split=$split \
        eval.verbose=true \
        dataset.val_batch_size=$batch_size \
        dataset.num_workers=8"

    [ -n "$checkpoint" ] && CMD="$CMD model.checkpoint_path=$checkpoint"

    # Time evaluation
    local start_time
    local end_time
    local total_time
    local time_per_image
    local num_samples

    # Set number of samples based on dataset and split
    case "$dataset:$split" in
        "cacd:train") num_samples=111984 ;;
        "cacd:val") num_samples=19696 ;;
        "cacd:test") num_samples=23235 ;;
        "imdb_clean:train") num_samples=122287 ;;
        "imdb_clean:val") num_samples=30732 ;;
        "imdb_clean:test") num_samples=37276 ;;
        "morph_2:train") num_samples=40012 ;;
        "morph_2:val") num_samples=5001 ;;
        "morph_2:test") num_samples=5002 ;;
        "age_db:train") num_samples=12959 ;;
        "age_db:val") num_samples=1625 ;;
        "age_db:test") num_samples=1904 ;;
        "appa_real:train") num_samples=4113 ;;
        "appa_real:val") num_samples=1500 ;;
        "appa_real:test") num_samples=1978 ;;
        *) num_samples=1000 ;;
    esac

    start_time=$(date +%s.%N)
    # Start evaluation
    echo "$CMD"
    eval "$CMD" > /dev/null 2>&1
    end_time=$(date +%s.%N)

    total_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "0")
    time_per_image=$(echo "scale=2; $total_time * 1000 / $num_samples" | bc 2>/dev/null || echo "0")

    # Search for log and extract metrics
    local log_dir
    log_dir=$(find_latest_log "$model_type" "$dataset" "$split")

    if [ -n "$log_dir" ]; then
        local metrics
        metrics=$(extract_metrics_from_log "$log_dir")

        # Combine metrics and timing
        printf "%s,%s,%s,%s,%s,%s\n" \
            "$model_key" "$dataset" "$split" \
            "$metrics" "$total_time" "$time_per_image" >> "$RESULTS_CSV"

        echo -e "${GREEN}  ✓ Метрики сохранены из: $log_dir${NC}"

        # Show metrics and timing
        IFS=',' read -r loss mae mape mse <<< "$metrics"
        echo -e "${BLUE}     loss=$loss, mae=$mae, mape=$mape, mse=$mse${NC}"
        printf "${BLUE}     время: %.2f сек (%.2f мс/изобр)${NC}\n" "$total_time" "$time_per_image"
    else
        echo -e "${RED}  ✗ Лог не найден для $model_key $dataset $split${NC}"
        echo "$model_key,$dataset,$split,,,,,,$total_time,$time_per_image" >> "$RESULTS_CSV"
    fi
}

main() {
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local results_file="evaluation_results_${timestamp}.csv"
    RESULTS_CSV="$RESULTS_DIR/$results_file"

    echo "model,dataset,split,loss,mae,mape,mse,total_time_sec,time_per_image_ms" > "$RESULTS_CSV"

    echo "========================================="
    echo "ЗАПУСК ОЦЕНКИ ВСЕХ МОДЕЛЕЙ"
    echo "========================================="
    echo "Результаты будут сохранены в: $RESULTS_CSV"
    echo ""

    local total=0
    local completed=0

    # Evaluate each model
    for model_entry in "${MODEL_CHECKPOINTS[@]}"; do
        model_key="${model_entry%%=*}"
        ckpt="${model_entry#*=}"

        if [ ! -f "$ckpt" ]; then
            echo -e "${RED}Чекпоинт не найден: $ckpt${NC}"
            continue
        fi

        echo -e "\n${GREEN}>>> Модель: $model_key${NC}"

        for dataset in "${DATASETS[@]}"; do
            for split in "${SPLITS[@]}"; do
                ((total++))
                evaluate_model "$model_key" "$dataset" "$split" "$ckpt"
                ((completed++))
            done
        done
    done

    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}✅ ОЦЕНКА ЗАВЕРШЕНА${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo "Всего оценок: $total"
    echo "Успешно: $completed"
    echo ""
    echo "Результаты сохранены в:"
    echo "  - $RESULTS_CSV"

    # Show summary of results for test splits
    echo -e "\n${YELLOW}СВОДКА ПО ТЕСТОВЫМ СПЛИТАМ (MAE):${NC}"
    echo "----------------------------------------"
    column -s, -t < "$RESULTS_CSV" | grep ",test," | sort -t, -k3 -n | head -20
}

main "$@"
