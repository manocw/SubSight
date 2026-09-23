"""Train the LIACI multi-label U-Net (Phase 2).

Run from repo root:
    python tasks/hull-defect/src/train.py

Multi-label: 10 sigmoid channels, BCE loss with pos_weight so rare
classes (defect 4%, corrosion 11%, tiny anodes) are not averaged away.
Metric: per-class IoU accumulated over val (micro), checkpoint on the
macro mean over classes present in val. Empty val classes report nan,
never a fake 1.0.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks" / "hull-defect" / "src"))

from liaci_dataset import CLASSES, get_dataloaders  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.utils import set_seed  # noqa: E402

# Pixel prevalence measured on full set (non-empty frac x cover),
# neg/pos capped at 50 so defect does not nuke the loss.
POS_WEIGHT = [0.5, 50.0, 10.0, 34.0, 50.0, 50.0, 10.0, 8.0, 50.0, 50.0]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot, n = 0.0, 0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        tot += loss.item() * images.size(0)
        n += images.size(0)
    return tot / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Returns loss, per-class IoU, per-class Dice (nan if class absent)."""
    model.eval()
    tot, n = 0.0, 0
    inter = torch.zeros(len(CLASSES), device=device)
    union = torch.zeros(len(CLASSES), device=device)
    pred_sum = torch.zeros(len(CLASSES), device=device)
    true_sum = torch.zeros(len(CLASSES), device=device)
    for images, masks in tqdm(loader, desc="val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        tot += criterion(logits, masks).item() * images.size(0)
        n += images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        tgt = (masks > 0.5).float()
        inter += (preds * tgt).sum(dim=(0, 2, 3))
        pred_sum += preds.sum(dim=(0, 2, 3))
        true_sum += tgt.sum(dim=(0, 2, 3))
    union = pred_sum + true_sum - inter
    eps = 1e-6
    iou = torch.where(union > 0, inter / (union + eps),
                      torch.full_like(inter, float("nan")))
    dice = torch.where(union > 0,
                       2 * inter / (pred_sum + true_sum + eps),
                       torch.full_like(inter, float("nan")))
    return tot / n, iou.cpu(), dice.cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LIACI U-Net.")
    ap.add_argument("--config", default="tasks/hull-defect/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["data"].get("seed", 42))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, _ = get_dataloaders(
        root=cfg["data"]["root"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        in_channels=3,
        classes=len(CLASSES),
    ).to(device)
    total, trainable = count_parameters(model)
    print(f"Model params={total:,} trainable={trainable:,}")

    pos_weight = torch.tensor(POS_WEIGHT, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion,
                                  optimizer, device)
        va_loss, va_iou, va_dice = validate(model, val_loader, criterion,
                                            device)
        macro = float(torch.nanmean(va_iou))
        print(f"epoch {epoch:02d}/{cfg['train']['epochs']} "
              f"train loss={tr_loss:.4f} | val loss={va_loss:.4f} "
              f"macro IoU={macro:.4f}")
        for c, v in zip(CLASSES, va_iou.tolist()):
            print(f"    {c}: IoU={v:.4f}")
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "val_iou": va_iou.tolist(), "config": cfg},
            ckpt_dir / "last.pth",
        )
        if macro > best:
            best = macro
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "val_iou": va_iou.tolist(), "config": cfg},
                ckpt_dir / "best.pth",
            )
            print(f"  -> new best macro IoU {best:.4f}, saved best.pth")
    print(f"Done. Best macro IoU={best:.4f}.")


if __name__ == "__main__":
    main()
