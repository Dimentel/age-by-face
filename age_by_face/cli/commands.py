# ruff: noqa: RUF002
import subprocess
import sys
from collections.abc import Sequence

import fire
from hydra import compose, initialize

from age_by_face.inference.evaluate import evaluate as evaluate_impl
from age_by_face.inference.infer import infer as infer_impl
from age_by_face.training.train import train as train_impl


def _compose_cfg(overrides: Sequence[str] | None = None):
    """
    Собирает конфигурацию Hydra через Compose API.

    Аргументы:
      overrides: последовательность оверрайдов в формате "key=value".
                 Пример: ["model.type=resnet50", "training.max_epochs=5"]

    Возвращает:
      DictConfig — собранный конфиг проекта.

    Примечания:
      - Compose API НЕ меняет рабочую директорию и НЕ поддерживает Hydra multirun (-m).
      - Для свипов используйте команду `sweep`, которая проксирует запуск в нативный Hydra-скрипт.
    """
    with initialize(version_base=None, config_path="../../conf"):
        return compose(config_name="config", overrides=list(overrides or []))


class Commands:
    def train(self, *overrides: str) -> None:
        """
        Запуск обучения модели с оверрайдами конфигурации.

        Что делает:
          - Выполняет загрузку данных через DVC.
          - Собирает DataModule, модель и LightningModule по конфигу.
          - Запускает обучение с логированием (MLflow или TensorBoard).
          - Сохраняет last.ckpt и лучшие чекпоинты; копирует лучший в best.ckpt.

        Полезные оверрайды:
          - model.type=resnet18|resnet50
          - training.max_epochs=EPOCHS
          - logging.mlflow.enabled=true|false
          - dataset.data_dir=/abs/path/to/data
          - 'training.checkpoint.dirpath=checkpoints/${model.type}'  (обратите внимание на кавычки)

        Примеры (PowerShell):
          age-by-face train 'training.checkpoint.dirpath=checkpoints/${model.type}'
                              model.type=resnet18 training.max_epochs=3
          age-by-face train logging.mlflow.enabled=false training.max_epochs=1

        Примеры (bash/zsh):
          age-by-face train 'training.checkpoint.dirpath=checkpoints/${model.type}'
                             model.type=resnet50 training.max_epochs=5

        Примечания:
          - Значения с ${...} заключайте в одинарные кавычки, чтобы оболочка не подставляла
            переменные.
          - Для перебора нескольких значений (свип) используйте команду `sweep`.
        """
        cfg = _compose_cfg(overrides)
        train_impl(cfg)

    def infer(self, *overrides: str) -> None:
        """
        Инференс/оценка модели и визуализация предсказаний.

        Что делает:
          - Ищет чекпоинт: сначала best.ckpt, затем last.ckpt в training.checkpoint.dirpath.
          - Загружает модель и выполняет тестовую оценку.
          - Строит сетку 3×3 изображений с предсказанными возрастами (predictions_grid.png).

        Полезные оверрайды:
          - ckpt_path=/abs/path/to/model.ckpt      (если чекпоинт не в стандартном месте)
          - model.type=resnet18|resnet50           (должен соответствовать обучению)
          - dataset.data_dir=/abs/path/to/data     (если данные не в стандартном каталоге)
          - 'training.checkpoint.dirpath=checkpoints/${model.type}'

        Примеры (PowerShell):
          age-by-face infer
          age-by-face infer ckpt_path='C:\\path\\to\\best.ckpt'

        Примеры (bash/zsh):
          age-by-face infer
          age-by-face infer ckpt_path=/abs/path/to/best.ckpt

        Примечания:
          - Инференс предполагает, что данные уже доступны локально (при необходимости
            выполните `dvc pull`).
        """
        cfg = _compose_cfg(overrides)
        infer_impl(cfg)

    def evaluate(self, *overrides: str, split: str = "test") -> None:
        """Evaluate a trained model on a specified dataset split.

        This command composes a Hydra configuration with the given overrides,
        resolves the checkpoint path (best.ckpt → last.ckpt → cfg.ckpt_path),
        and runs evaluation on the requested split.

        Args:
            *overrides: Hydra configuration overrides in "key=value" format.
                        Examples:
                        - "model.type=resnet50"
                        - "training.checkpoint.dirpath=checkpoints/resnet18"
            split: Dataset split to evaluate on. Must be one of 'train', 'val', or 'test'.
                   Defaults to 'test'.

        Raises:
            ValueError: If split is not one of 'train', 'val', or 'test'.

        Examples:
            # Evaluate best model on test set (default)
            age-by-face evaluate

            # Evaluate on validation set
            age-by-face evaluate split=val

            # Evaluate a specific model checkpoint
            age-by-face evaluate 'training.checkpoint.dirpath=checkpoints/resnet50' split=test

            # Evaluate with different model architecture
            age-by-face evaluate model.type=resnet50 split=val

        Notes:
            - The checkpoint resolution logic is shared with the `infer` command.
            - Metrics are printed to console and returned via MLflow logging
              (if enabled in the configuration).
            - The model architecture specified in config must match the checkpoint.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'test' or 'val', got {split}")
        cfg = _compose_cfg(overrides)
        evaluate_impl(cfg, split=split)

    def sweep(self, *overrides: str) -> int:
        """
        Мультизапуск (свип) через проксирование в нативный Hydra-скрипт с флагом -m.

        Что делает:
          - Вызывает `python -m age_by_face.train_cli -m ...`, где работают все механизмы Hydra:
            ${hydra:job.num}, hydra.sweep.dir/subdir и пр.
          - Полезно для перебора нескольких значений гиперпараметров.

        Как задавать списки значений:
          - Используйте запятую без пробелов в значении: key=val1,val2
          - Всю строку с запятой берите в кавычки, чтобы передалось одним аргументом.

        Примеры (PowerShell):
          age-by-face sweep "model.type=resnet18,resnet50" "training.max_epochs=5"
          age-by-face sweep "dataset.random_horizontal_flip_prob=0.0,0.5" "training.max_epochs=3"

        Примеры (bash/zsh):
          age-by-face sweep "model.type=resnet18,resnet50" "training.max_epochs=5"
          age-by-face sweep "dataset.random_horizontal_flip_prob=0.0,0.5" "training.max_epochs=3"

        Возвращает:
          Код возврата подпроцесса Hydra (0 при успехе).

        Примечания:
          - Путь к конфигам и все hydra: переменные обрабатываются в age_by_face/train_cli.py.
        """
        cmd = [sys.executable, "-m", "age_by_face.train_cli", "-m", *overrides]
        completed = subprocess.run(cmd, check=False)
        return completed.returncode


def main() -> None:
    """
    Точка входа CLI.

    Команды:
      - train — одиночный запуск обучения с оверрайдами (Compose API).
      - infer — инференс/оценка и визуализация (Compose API).
      - sweep — мультизапуск через нативный Hydra (-m).

    Справка:
      age-by-face -- --help
      age-by-face train -- --help
      age-by-face infer -- --help
      age-by-face sweep -- --help
    """
    fire.Fire(Commands)
