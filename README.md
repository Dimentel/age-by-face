# Определение возраста по фотографиям лица

## Постановка задачи

Создать алгоритм, который по фотографии лица сможет предсказывать возраст как
число.

## Формат входных и выходных данных

На входе имеются 7591 изображений лиц с различным разрешением от 47x47 до
4466x4466 (не все изображения квадратные, имеются аугментации в виде поворотов).
Также имеются файлы csv с информацией о реальном и видимом (на основе опроса
людей) возрасте. На выходе системы одно число - возраст в годах.

## Метрики

В ходе обучения модели считается функция потерь MSE, а также вычисляются метрики
MAE и MAPE.

## Данные

Датасет взят из
[исследования](https://ieeexplore.ieee.org/abstract/document/7961727) и содержит
7591 изображений лиц с соответствующими метками реального возраста. Все данные
содержатс в папке `data`. Изображения разделены на выборки: 4113 для обучения,
1500 для валидации и 1978 для тестирования. Выборки размещены в папках
`data/train/`, `data/valid/` и `data/test/`. Наименование файла с изображением
имеет формат `dddddd.jpg_face.jpg`, где d - цифра от 0 до 9. Изображение
содержит обрезанное и повернутое лицо с отступом 40%, полученное с помощью
детектора лиц
[Mathias](http://markusmathias.bitbucket.org/2014_eccv_face_detection/) при
множественных поворотах.

Данные о названии файла (колонка `file_name`, формат `dddddd.jpg`), о реальном
возрасте (колонка `real_age`) представлены в файлах `gt_avg_train.csv`,
`gt_avg_valid.csv` и `gt_avg_test.csv` в папке `data`.

## Моделирование

### Базовая модель

Голова Resnet18 ([статья](https://arxiv.org/abs/1512.03385),
[реализация](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18)
в Pytorch), все блоки до avg_pool плюс свой блок avg_pool, набор скрытых
полносвязных слоёв (по умолчанию 1 слой 512 нейронов, количество слоёв и
нейронов конфигурируется в файле конфигурации модели) и выходной полносвязный
слой с одним выходом. Активация слоёв (по умолчанию - ReLU), наличие и параметр
p dropout (по умолчанию имеется на последнем скрытом слое, p=0.3)
конфигурируются в файлах конфигурации моделей.

### Основная модель

На текущий момент - голова Resnet18 ([статья](https://arxiv.org/abs/1512.03385),
[реализация](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
в Pytorch), все блоки до avg_pool плюс свой блок avg_pool, набор скрытых
полносвязных слоёв (по умолчанию 1 слой 2048 нейронов, количество слоёв и
нейронов конфигурируется в файле конфигурации модели) и выходной полносвязный
слой с одним выходом. Активация слоёв (по умолчанию - ReLU), наличие и параметр
p dropout (по умолчанию имеется на последнем скрытом слое, p=0.3)
конфигурируются в файлах конфигурации моделей.

### Другие параметры

Файлы конфигурации позволяют выбрать различные параметры:

- размеры батчей в режиме обучения, валидации и теста,
- размеры картинок на входе в модель и параметры масштабирования,
- параметры оптимизатора и scheduler шага обучения,
- другие параметры (см. директорию `conf/`)

## Setup

- Предварительные требования:
  - Git, Python 3.12
  - Установленный uv (менеджер окружений и зависимостей). Инструкции:
    https://docs.astral.sh/uv/getting-started/installation/
- Клонируйте репозиторий:

```bash
git clone https://github.com/Dimentel/age-by-face.git && cd age-by-face
```

- Создайте и активируйте виртуальное окружение (опционально):

```bash
uv venv -p 3.12
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

- Установите зависимости проекта:

```bash
uv sync
```

- Проверьте, что CLI-скрипт доступен (из venv):

```bash
uv run age-by-face --help
```

- DVC и данные:
  - DVC уже настроен в проекте; бакет S3 доступен для чтения.
  - Обучение автоматически выполнит dvc pull перед стартом.
  - Если хотите забрать данные заранее:

```bash
uv run dvc pull
```

- MLflow:
  - По умолчанию логирование MLflow включено (conf/logging/logging.yaml). Если
    сервер MLflow не запущен на tracking_uri, либо запустите свой сервер, либо
    временно отключите логирование:

```bash
uv run age-by-face train logging.mlflow.enabled=false ...
```

- Проверка установки:

```bash
uv run python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
uv run dvc --version
```

## Train

- Базовый запуск (одна тренировка)
  - Рекомендуется явно задать директорию чекпоинтов без ссылок на hydra:\* (для
    совместимости с Fire+Compose). Обучение:

```bash
uv run age-by-face train \
  'training.checkpoint.dirpath=checkpoints/${model.type}' \
  training.max_epochs=3 \
  model.type=resnet18
```

- Пояснения:
  - training.checkpoint.dirpath — каталог, куда будут сохранены last.ckpt и
    лучшие .ckpt. Используем форму checkpoints/${model.type}, чтобы для каждой
    модели была своя папка.
  - model.type — архитектура из конфигов (resnet18 или resnet50).
  - training.max_epochs — число эпох.

- Выбор/переопределение параметров
  - Датасет:
    - По умолчанию используется папка data (конфиг conf/dataset/dataset.yaml).
      Можно переопределить:

```bash
uv run age-by-face train 'training.checkpoint.dirpath=checkpoints/${model.type}' dataset.data_dir=/abs/path/to/data
```

- Отключить MLflow (если сервер не запущен):

```bash
uv run age-by-face train 'training.checkpoint.dirpath=checkpoints/${model.type}' logging.mlflow.enabled=false
```

- Сменить модель:

```bash
uv run age-by-face train 'training.checkpoint.dirpath=checkpoints/${model.type}' model.type=resnet50
```

- Прочие оверрайды возможны для любых ключей конфигурации (conf/\*).

- Что делает команда обучения
  - Выполняет dvc pull (один раз) для загрузки данных.
  - Создаёт DataModule (тренировочные и валидационные лоадеры).
  - Строит модель по конфигу (build_model) и LightningModule.
  - Настраивает оптимизатор AdamW и ReduceLROnPlateau по конфигу.
  - Ведёт логирование:
    - MLflow (если включено): метрики, гиперпараметры, git_commit/git_dirty и
      лучший чекпоинт как артефакт checkpoints/best.ckpt.
    - Иначе — стандартный TensorBoard.
  - Сохраняет чекпоинты:
    - last.ckpt — последний
    - Лучшие по метрике (настройки в training.checkpoint.\*). Для удобства
      скрипт также копирует лучший чекпоинт в checkpoints/best.ckpt.

- Грид-свипы (несколько запусков)
  - Для перебора значений (Hydra multirun) используйте команду sweep — она
    проксирует в нативный Hydra с -m:

```bash
uv run age-by-face sweep \
  "model.type=resnet18,resnet50" \
  "training.max_epochs=5" \
  "training.checkpoint.dirpath=checkpoints/\${model.type}"
```

- Обратите внимание: строки с запятыми и с ${...} нужно брать в кавычки, чтобы
  оболочка не подставляла переменные.

- Инференс и визуализация
  - По умолчанию скрипт infer возьмёт checkpoints/${model.type}/best.ckpt (если
    нет — last.ckpt), выполнит тестовую оценку и построит сетку 3x3 с
    предсказанными возрастами (predictions_grid.png):

```bash
uv run age-by-face infer
```

- Если чекпоинт в другом месте, укажите явный путь:

```bash
uv run age-by-face infer ckpt_path=/abs/path/to/best.ckpt
```

- Если вы запускаете инференс на “чистой” машине без предварительного обучения,
  предварительно скачайте данные:

```bash
uv run dvc pull
```

Примечания

- Обёртка age-by-face реализована через Fire и Hydra Compose API для одиночных
  запусков и через нативный Hydra (@hydra.main) для свипов (команда sweep).
- В командах с ${...} используйте одинарные кавычки (bash/zsh), чтобы избежать
  подстановок оболочки.
- Если используете MLflow по умолчанию (enabled: true), убедитесь, что
  tracking_uri доступен, иначе отключите логирование через override
  logging.mlflow.enabled=false.
