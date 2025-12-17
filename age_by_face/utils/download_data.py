from contextlib import suppress
from pathlib import Path

from dvc.repo import Repo
from omegaconf import DictConfig


def _find_repo_root(start: Path) -> Path:
    """
    find repo root looking for: .dvc, pyproject.toml, .git.
    """
    p = start
    for candidate in [p, *p.parents]:
        if (
            (candidate / ".dvc").exists()
            or (candidate / "pyproject.toml").exists()
            or (candidate / ".git").exists()
        ):
            return candidate
    return start


def download_data_dvc(cfg: DictConfig) -> Path:
    """
    Make sure data is available with dvc:
    - call `dvc pull` from repo root;
    - make sure that cfg.dataset.data_dir absolute path.
    """
    repo_root = _find_repo_root(Path(__file__).resolve())

    # Make abs path from root
    data_dir_cfg = str(getattr(cfg.dataset, "data_dir", "data"))
    abs_data_dir = (
        Path(data_dir_cfg) if Path(data_dir_cfg).is_absolute() else (repo_root / data_dir_cfg)
    ).resolve()

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
