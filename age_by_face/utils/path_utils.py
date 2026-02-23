from pathlib import Path

from age_by_face.utils.types import Directory, File


def ensure_path(path: Directory | File) -> Path:
    """
    Ensure the input is a Path object.

    Args:
        path: String or Path object

    Returns:
        Path object
    """
    if isinstance(path, str):
        return Path(path)
    if isinstance(path, Path):
        return path

    raise TypeError(f"Expected str or Path, got {type(path)}")
