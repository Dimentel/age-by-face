from pathlib import Path


def ensure_path(path: str | Path) -> Path:
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
