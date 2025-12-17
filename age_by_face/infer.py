from collections.abc import Sequence
from pathlib import Path

import lightning as l
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from age_by_face.data import AgeDataModule
from age_by_face.model import build_model
from age_by_face.module import AgeRegressionModule


def _resolve_ckpt_path(cfg: DictConfig) -> Path:
    """
    Приоритет:
    1) <cfg.training.checkpoint.dirpath>/best.ckpt
    2) <cfg.training.checkpoint.dirpath>/last.ckpt
    3) Явно заданный cfg.ckpt_path
    """
    ckpt_dir = Path(str(cfg.training.checkpoint.dirpath))
    best_local = ckpt_dir / "best.ckpt"
    if best_local.exists():
        return best_local

    last_local = ckpt_dir / "last.ckpt"
    if last_local.exists():
        return last_local

    explicit = getattr(cfg, "ckpt_path", None)
    if explicit:
        p = Path(str(explicit))
        if p.exists():
            return p
        raise FileNotFoundError(f"Указанный ckpt_path не найден: {p}")

    raise FileNotFoundError(
        "best.ckpt, last.ckpt not found"
        f"Expected: {ckpt_dir}. Or set ckpt_path=/abs/path/to/model.ckpt"
    )


def _unnormalize(img: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """
    img: Tensor [C,H,W] в нормализованном виде
    return: Tensor [H,W,C] в диапазоне [0,1]
    """
    out = img.detach().cpu().clone()
    for channel in range(min(3, out.shape[0])):
        out[channel] = out[channel] * float(std[channel]) + float(mean[channel])
    out = out.clamp(0.0, 1.0)
    return out.permute(1, 2, 0)


def _show_predictions_grid(  # noqa: PLR0913
    images: torch.Tensor,
    preds: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    grid_h: int = 3,
    grid_w: int = 3,
    save_path: str = "predictions_grid.png",
) -> None:
    """
    Make grid_h x grid_w with photos and with ages as titles;
    images: [B, C, H, W], preds: [B]
    """
    n = min(images.size(0), grid_h * grid_w)
    _, axes = plt.subplots(grid_h, grid_w, figsize=(3.5 * grid_w, 3.5 * grid_h))
    axes = axes.flatten()

    for i in range(grid_h * grid_w):
        axes[i].axis("off")

    for i in range(n):
        img = _unnormalize(images[i], mean=mean, std=std).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"{preds[i].item():.1f}", fontsize=12, pad=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def infer(cfg: DictConfig) -> None:
    l.seed_everything(int(getattr(cfg, "seed", 0)), workers=True)

    # Путь к чекпоинту: best.ckpt -> last.ckpt -> cfg.ckpt_path
    ckpt_path = _resolve_ckpt_path(cfg)

    # Данные: используем тестовый набор и трансформации из конфигурации
    datamodule = AgeDataModule(cfg.dataset)
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # Восстановление модуля из чекпоинта
    model = build_model(cfg.model)
    module = AgeRegressionModule.load_from_checkpoint(
        str(ckpt_path), model=model, cfg=cfg, map_location="cpu"
    )
    module.eval()

    # Берём один батч и считаем предсказания
    batch = next(iter(test_loader))
    images, _ = batch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)

    with torch.no_grad():
        preds = module(images.to(device)).squeeze(1).cpu()

    # Visualize
    _show_predictions_grid(
        images=images,
        preds=preds,
        mean=list(getattr(cfg.dataset, "normalize_means", [0.5, 0.5, 0.5])),
        std=list(getattr(cfg.dataset, "normalize_stds", [0.5, 0.5, 0.5])),
        grid_h=3,
        grid_w=3,
        save_path="predictions_grid.png",
    )
