#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# datasets batch sizes
declare -A BATCH_SIZES=(
    ["resnet18_mixed"]=512
    ["resnet50_mixed"]=256
    ["convnext_paper_mixed"]=256
    ["transformer_paper_mixed"]=512
    ["hybrid_paper_mixed"]=256
)

# Model key (in script) -> model type
declare -A MODEL_TYPES=(
    ["resnet18"]="resnet18"
    ["resnet50"]="resnet50"
    ["convnext_paper"]="convnext"
    ["transformer_paper"]="transformer"
    ["hybrid_paper"]="hybrid"
)

MODEL_CHECKPOINTS=(
    "resnet18=checkpoints/resnet18_pretrained.ckpt"
    "resnet50=checkpoints/resnet50_pretrained.ckpt"
    "convnext_paper=checkpoints/convnext_paper.ckpt"
    "transformer_paper=checkpoints/transformer_paper.ckpt"
    "hybrid_paper=checkpoints/hybrid_paper.ckpt"
)

DATASET="mixed"

# Общие параметры обучения
MAX_EPOCHS=100

echo "========================================="
echo "Начало обучения моделей на общем датасете"
echo "========================================="

# Function for training a single model
train_model() {
    local model_key=$1
    local model_type=${MODEL_TYPES[$model_key]}
    local checkpoint=$2
    local batch_size=${BATCH_SIZES[${model_key}_${DATASET}]}
    local output_name="${model_key}_all_data.ckpt"

    echo -e "\n${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}Обучение: $model_key на $DATASET${NC}"
    echo -e "${YELLOW}  Batch size: $batch_size${NC}"
    echo -e "${YELLOW}=========================================${NC}"

    # Make the command
    CMD="uv run age-by-face train \
        model=$model_type \
        dataset=$DATASET \
        dataset.train_batch_size=$batch_size \
        dataset.val_batch_size=$((batch_size * 2)) \
        dataset.num_workers=8 \
        training.max_epochs=$MAX_EPOCHS"

    # add checkpoint if exists for fine-tuning
    if [ -f "$checkpoint" ]; then
        CMD="$CMD model.checkpoint_path=$checkpoint"
        echo "  Режим: fine-tuning из $checkpoint"
    else
        echo "  Режим: обучение с нуля (чекпойнт не найден: $checkpoint)"
    fi

    # start training
    echo -e "\n${GREEN}Запуск команды:${NC}"
    echo "$CMD"
    echo ""

    eval "$CMD"

    # Take best from checkpoints directory
    local checkpoint_dir="checkpoints/${model_type}_${DATASET}"
    local best_ckpt="$checkpoint_dir/best.ckpt"

    if [ -f "$best_ckpt" ]; then
        local final_path="checkpoints/${output_name}"
        cp "$best_ckpt" "$final_path"
        echo -e "${GREEN}Лучший чекпоинт скопирован: $final_path${NC}"
    else
        echo -e "${YELLOW}Лучший чекпоинт не найден в $checkpoint_dir${NC}"
    fi

    echo "----------------------------------------"
}

# Main cycle
for model_entry in "${MODEL_CHECKPOINTS[@]}"; do
    model_key="${model_entry%%=*}"
    checkpoint="${model_entry#*=}"

    train_model "$model_key" "$checkpoint"
done

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Обучение всех моделей завершено${NC}"
echo -e "${GREEN}=========================================${NC}"

# Показываем итоговые чекпоинты
echo -e "\n${YELLOW}Итоговые чекпоинты:${NC}"
ls -lh checkpoints/*_all_data.ckpt 2>/dev/null || echo "Чекпоинты не найдены"
