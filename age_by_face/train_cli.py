import hydra
from omegaconf import DictConfig

from age_by_face.train import train as train_impl


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train_impl(cfg)
