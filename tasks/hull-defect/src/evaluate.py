"""Evaluate a LIACI checkpoint: per-class table + sample figure.

Run from repo root:
    python tasks/hull-defect/src/evaluate.py --checkpoint \
        tasks/hull-defect/checkpoints/best.pth --num-images 4

Figure: input | ground truth (rare channels) | prediction (rare
channels). Rare = anode, corrosion, paint peel, defect.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks" / "hull-defect" / "src"))

from liaci_dataset import CLASSES, get_dataloaders  # noqa: E402
from src.model import build_model  # noqa: E402


def _load_validate():
    """Import validate() from the sibling train.py without name clash
    with src/train.py (explicit file-location import)."""
    spec = importlib.util.spec_from_file_location(
        "hull_train", ROOT / "tasks" / "hull-defect" / "src" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate, mod.POS_WEIGHT


FOCUS = ["anode", "corrosion", "paint_peel", "defect"]
COLOURS = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

_MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
_STD = np.array([0.229, 0.224, 0.225])[None, None, :]


def overlay_multi(img: np.ndarray, masks: np.ndarray,
                  idx: list[int]) -> np.ndarray:
    """Tint focus channels onto RGB img. masks: (C,H,W) {0,1}."""
    out = img.copy()
    for k, i in enumerate(idx):
        m = masks[i] > 0.5
        tint = np.zeros_like(out)
        tint[m] = COLOURS[k]
        out = np.where(m[..., None], 0.5 * out + 0.5 * tint, out)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Eval LIACI checkpoint.")
    ap.add_argument("--checkpoint",
                    default="tasks/hull-defect/checkpoints/best.pth")
    ap.add_argument("--config", default="tasks/hull-defect/config.yaml")
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--out", default="tasks/hull-defect/outputs/eval.png")
    args = ap.parse_args()

    validate, pos_weight = _load_validate()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", yaml.safe_load(open(args.config)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=3,
        classes=len(CLASSES),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, val_loader, _, _ = get_dataloaders(
        root=cfg["data"]["root"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    _, va_iou, va_dice = validate(model, val_loader, criterion, device)
    print(f"{'class':<18}{'IoU':>8}{'Dice':>8}")
    for c, i, d in zip(CLASSES, va_iou.tolist(), va_dice.tolist()):
        print(f"{c:<18}{i:>8.4f}{d:>8.4f}")
    macro = float(torch.nanmean(va_iou))
    print(f"{'macro (present)':<18}{macro:>8.4f}")

    # Figure on first N val images: rare-channel overlays.
    idx = [CLASSES.index(c) for c in FOCUS]
    batches = next(iter(val_loader))
    nimgs = min(args.num_images, batches[0].size(0))
    logits = model(batches[0][:nimgs].to(device))
    probs = torch.sigmoid(logits).cpu().numpy()
    gts = batches[1][:nimgs].numpy()
    ims = batches[0][:nimgs].numpy()
    fig, axes = plt.subplots(nimgs, 3, figsize=(9, 3 * nimgs),
                             squeeze=False)
    for r in range(nimgs):
        img = np.clip(np.moveaxis(ims[r], 0, -1) * _STD + _MEAN, 0, 1)
        axes[r][0].imshow(img)
        axes[r][0].set_title("input")
        axes[r][1].imshow(overlay_multi(img, gts[r], idx))
        axes[r][1].set_title("truth (anode/corrosion/peel/defect)")
        axes[r][2].imshow(overlay_multi(img, probs[r] > 0.5, idx))
        axes[r][2].set_title("prediction (same colours)")
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(f"LIACI val overlays, macro IoU {macro:.3f}", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved figure -> {args.out}")


if __name__ == "__main__":
    main()
