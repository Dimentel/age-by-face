from contextlib import suppress
from pathlib import Path

from dvc.repo import Repo
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig


def download_data_dvc(cfg: DictConfig) -> Path:
    """
    Make sure data is available with dvc:
    - call `dvc pull` from repo root;
    - make sure that cfg.dataset.data_dir absolute path.
    """
    repo_root = Path(get_original_cwd())
    abs_data_dir = Path(to_absolute_path(str(getattr(cfg.dataset, "data_dir", "data"))))

    # If no dvc, just return absolute path
    if not (repo_root / ".dvc").exists():
        cfg.dataset.data_dir = str(abs_data_dir)
        return abs_data_dir

    # Pull all data
    with Repo(repo_root) as repo:
        repo.pull()

    if not abs_data_dir.exists():
        raise FileNotFoundError(f"Data directory not found after dvc pull: {abs_data_dir}")
    # Pass if data_dir is already absolute
    with suppress(Exception):
        cfg.dataset.data_dir = str(abs_data_dir)

    return abs_data_dir
