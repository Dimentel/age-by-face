import logging
import subprocess
from pathlib import Path

from dvc.repo import Repo
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _find_repo_root(start: Path) -> Path:
    """
    find repo root looking for: .dvc, pyproject.toml, .git.
    """
    p = start
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return start


def _data_is_up_to_date(repo_root: Path) -> bool:
    """Check if data is up to date with dvc."""
    result = subprocess.run(
        ["dvc", "status"], check=False, cwd=repo_root, capture_output=True, text=True
    )

    return "Data and pipelines are up to date" in result.stdout


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
    logger.info(f"Data directory: {abs_data_dir}")

    # Fast check with dvc status
    if _data_is_up_to_date(repo_root):
        logger.info("Data is up to date with DVC")
        return abs_data_dir

    # Pull all data
    with Repo(repo_root) as repo:
        logger.info("Pulling all data with DVC...")
        repo.pull()
        logger.info("DVC pull completed")

    return abs_data_dir
