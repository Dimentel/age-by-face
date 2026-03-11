#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Конфигурация датасетов и batch size для обучения
declare -A BATCH_SIZES=(
    ["resnet18_cacd"]=512
    ["resnet18_imdb_clean"]=256
    ["resnet18_morph_2"]=512
    ["resnet50_cacd"]=256
    ["resnet50_imdb_clean"]=128
    ["resnet50_morph_2"]=256
)

# Датасеты по порядку (важно для fine-tuning)
DATASETS=("cacd" "imdb_clean" "morph_2")

echo "========================================="
echo "Начало обучения ResNet моделей"
echo "========================================="

# Обучаем ResNet18
echo -e "\n${GREEN}>>> ResNet18${NC}"
PREV_CHECKPOINT=""

for dataset in "${DATASETS[@]}"; do
    echo -e "\n${YELLOW}--- Обучение на $dataset ---${NC}"

    # Формируем команду
    CMD="uv run age-by-face train \
        model=resnet18 \
        dataset=$dataset \
        dataset.train_batch_size=${BATCH_SIZES[resnet18_${dataset}]} \
        dataset.val_batch_size=$(( ${BATCH_SIZES[resnet18_${dataset}]} * 2 )) \
        dataset.num_workers=8 \
        training.max_epochs=100"

    # Если есть предыдущий чекпоинт - дообучаем
    if [ -n "$PREV_CHECKPOINT" ]; then
        echo "Дообучение с чекпоинта: $PREV_CHECKPOINT"
        CMD="$CMD model.checkpoint_path=$PREV_CHECKPOINT"
    fi

    # Запускаем
    echo "$CMD"
    eval "$CMD"

    # После обучения берём лучший чекпоинт для следующего датасета
    PREV_CHECKPOINT="checkpoints/resnet18_${dataset}/best.ckpt"
done

# Копируем финальный чекпоинт
cp "checkpoints/resnet18_morph_2/best.ckpt" "checkpoints/resnet18_pretrained.ckpt"
echo -e "${GREEN}✅ ResNet18 финальный чекпоинт: checkpoints/resnet18_pretrained.ckpt${NC}"

# Сбрасываем для ResNet50
PREV_CHECKPOINT=""

# Обучаем ResNet50
echo -e "\n${GREEN}>>> ResNet50${NC}"

for dataset in "${DATASETS[@]}"; do
    echo -e "\n${YELLOW}--- Обучение на $dataset ---${NC}"

    CMD="uv run age-by-face train \
        model=resnet50 \
        dataset=$dataset \
        dataset.train_batch_size=${BATCH_SIZES[resnet50_${dataset}]} \
        dataset.val_batch_size=$(( ${BATCH_SIZES[resnet50_${dataset}]} * 2 )) \
        dataset.num_workers=8 \
        training.max_epochs=100"

    if [ -n "$PREV_CHECKPOINT" ]; then
        echo "Дообучение с чекпоинта: $PREV_CHECKPOINT"
        CMD="$CMD model.checkpoint_path=$PREV_CHECKPOINT"
    fi

    echo "$CMD"
    eval "$CMD"

    PREV_CHECKPOINT="checkpoints/resnet50_${dataset}/best.ckpt"
done

cp "checkpoints/resnet50_morph_2/best.ckpt" "checkpoints/resnet50_pretrained.ckpt"
echo -e "${GREEN}✅ ResNet50 финальный чекпоинт: checkpoints/resnet50_pretrained.ckpt${NC}"

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ Обучение всех ResNet моделей завершено${NC}"
echo -e "${GREEN}=========================================${NC}"
