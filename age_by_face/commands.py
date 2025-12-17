import subprocess
import sys
from collections.abc import Sequence

import fire
from hydra import compose, initialize

from age_by_face.infer import infer as infer_impl
from age_by_face.train import train as train_impl


def _compose_cfg(overrides: Sequence[str] | None = None):
    # config_path относительно текущего файла: ../conf (из корня проекта)
    with initialize(version_base=None, config_path="../conf"):
        return compose(config_name="config", overrides=list(overrides or []))


class Commands:
    def train(self, *overrides: str) -> None:
        """
        Примеры:
          age-by-face train training.max_epochs=3 model.type=resnet50
          age-by-face train dataset.data_dir=data
        """
        cfg = _compose_cfg(overrides)
        train_impl(cfg)

    def infer(self, *overrides: str) -> None:
        """
        Примеры:
          age-by-face infer
          age-by-face infer ckpt_path=/abs/path/to/best.ckpt
        """
        cfg = _compose_cfg(overrides)
        infer_impl(cfg)

    def sweep(self, *overrides: str) -> int:
        """
        Проксирует мультизапуск (-m) в нативный Hydra-скрипт.
        Примеры:
          age-by-face sweep "model.type=resnet18,resnet50" "training.max_epochs=5"
          age-by-face sweep "training.max_epochs=3" "dataset.random_horizontal_flip_prob=0.0,0.5"
        """
        cmd = [sys.executable, "-m", "age_by_face.train_cli", "-m", *overrides]
        completed = subprocess.run(cmd, check=False)
        return completed.returncode


def main() -> None:
    fire.Fire(Commands)
