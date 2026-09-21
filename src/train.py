"""Train the SubPipe U-Net (Stage 4).

Run:
    python -m src.train --config configs/config.yaml
Colab (T4 GPU): Runtime -> Change runtime type -> T4, then the same command.

What happens each epoch:
  1. TRAIN loop: for each batch, predict mask -> BCE loss -> backprop ->
     Adam step. Model LEARNS here (weights change).
  2. VAL loop: same predictions on held-out images, NO learning
     (torch.no_grad). This measures generalisation — would it work on a
     new stretch of pipe?
  3. Checkpoint: save `best.pth` when val IoU improves, plus `last.pth`
     every epoch so a Colab timeout never loses everything.

Loss vs metrics (defend this in your viva):
  - LOSS (BCEWithLogits) is what the optimizer minimises. It must be smooth
    and differentiable, so gradients exist for every pixel.
  - METRICS (IoU, Dice) are what HUMANS judge by: after thresholding at 0.5,
    how much does the predicted pipe overlap the true pipe?
  - We do NOT use pixel accuracy: ~90% of pixels are background, so a model
    predicting "no pipe anywhere" scores 90% accuracy while finding nothing.
    IoU/Dice ignore the easy background and score only the pipe overlap.

Checkpoints save: model weights + optimizer state + epoch + best val IoU +
the config, so Stage 5 eval can reload everything reproducibly.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_model, count_parameters
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Metrics — overlap scores for the PIPE class only (background ignored)
# ---------------------------------------------------------------------------
def _binarise(logits: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Logits -> {0,1} predictions. Sigmoid squashes to probability,
    threshold at 0.5 = "more likely pipe than not"."""
    return (torch.sigmoid(logits) > thresh).float()


def iou_score(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> float:
    """IoU = overlap / union. 1.0 = perfect, 0.0 = no overlap.

    Example: predict half the pipe correctly, miss the rest, no false
    alarms -> overlap=50, union=100 -> IoU=0.5. The SubPipe paper reports
    this, so it is our headline number and checkpoint criterion.
    """
    preds = _binarise(logits)
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return float(((intersection + eps) / (union + eps)).mean())


def dice_score(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> float:
    """Dice = 2*overlap / (pred + true). Same idea as IoU, kinder to small
    or thin objects (like our pipe): Dice is always >= IoU for the same
    prediction, so report BOTH and never compare one paper's Dice with
    another's IoU."""
    preds = _binarise(logits)
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return float(((2 * intersection + eps) / (denom + eps)).mean())


# ---------------------------------------------------------------------------
# One epoch of training / validation
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train mode: gradients ON, augmentations ON (from dataset.py)."""
    model.train()
    tot_loss, tot_iou, tot_dice, n = 0.0, 0.0, 0.0, 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)  # (B,1,H,W) raw scores
        loss = criterion(logits, masks)
        loss.backward()  # how should each weight move to reduce loss?
        optimizer.step()  # move them (Adam: adaptive step sizes)
        b = images.size(0)
        tot_loss += loss.item() * b
        tot_iou += iou_score(logits.detach(), masks) * b
        tot_dice += dice_score(logits.detach(), masks) * b
        n += b
    return tot_loss / n, tot_iou / n, tot_dice / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Val mode: NO gradients, NO learning — pure measurement."""
    model.eval()
    tot_loss, tot_iou, tot_dice, n = 0.0, 0.0, 0.0, 0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        tot_loss += criterion(logits, masks).item() * images.size(0)
        tot_iou += iou_score(logits, masks) * images.size(0)
        tot_dice += dice_score(logits, masks) * images.size(0)
        n += images.size(0)
    return tot_loss / n, tot_iou / n, tot_dice / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train SubPipe U-Net.")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["data"].get("seed", 42)
    set_seed(seed)  # same shuffle, same init, same augments -> reproducible
    # Extra determinism (tiny slowdown, worth it for a portfolio):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (Colab: check Runtime -> T4 GPU if you see 'cpu').")

    train_loader, val_loader = get_dataloaders(
        root=cfg["data"]["root"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_split"],
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
    )

    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        in_channels=cfg["model"].get("in_channels", 3),
        classes=cfg["model"].get("classes", 1),
    ).to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {cfg['model']['architecture']}/{cfg['model']['encoder']} "
          f"params={total:,} trainable={trainable:,}")

    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one stable op.
    # Right choice for 1-channel binary masks; one weight per pixel.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_iou = 0.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, tr_iou, tr_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_iou, va_dice = validate(model, val_loader, criterion, device)
        print(f"epoch {epoch:02d}/{cfg['train']['epochs']} "
              f"train loss={tr_loss:.4f} iou={tr_iou:.4f} dice={tr_dice:.4f} | "
              f"val loss={va_loss:.4f} iou={va_iou:.4f} dice={va_dice:.4f}")

        # Always keep `last` (Colab may disconnect); keep `best` on val IoU.
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "val_iou": va_iou, "config": cfg},
            ckpt_dir / "last.pth",
        )
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "val_iou": va_iou, "config": cfg},
                ckpt_dir / "best.pth",
            )
            print(f"  -> new best val IoU {best_iou:.4f}, saved best.pth")

    print(f"Done. Best val IoU={best_iou:.4f}. "
          f"Weights: {ckpt_dir/'best.pth'} (Stage 5 loads this).")


if __name__ == "__main__":
    main()
