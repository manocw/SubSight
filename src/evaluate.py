"""Evaluate a trained checkpoint + save overlay figures (Stage 5).

Run (after Stage 4 produced checkpoints/best.pth):
    python -m src.evaluate --checkpoint checkpoints/best.pth
    python -m src.evaluate --checkpoint checkpoints/best.pth --num-images 8

What it does:
  1. Reloads the model + the config saved inside the checkpoint (so eval
     always matches how it was trained — no copy-paste mismatch).
  2. Runs the VAL split (same seed => same images as training's val set),
     reporting mean IoU / Dice. These numbers go in your README table.
  3. Saves `outputs/eval_examples.png`: rows = best / median / worst
     predictions by IoU, columns = image | true mask | predicted mask.
     Showing failures (e.g. sand-covered pipe) is a FEATURE for your
     portfolio: it proves you can read results like an engineer.

Overlays: green = ground-truth pipe, red = predicted pipe, both drawn
semi-transparent on the raw image so you can see alignment at a glance.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe (Colab has no display)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.dataset import get_dataloaders, get_full_loader
from src.model import build_model
from src.train import dice_score, iou_score

# ImageNet stats (must match dataset.py) — needed to UNDO normalisation
# before displaying: show = clip(img * std + mean).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])[None, None, :]
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])[None, None, :]


def denormalise(img_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) displayable RGB in [0,1]."""
    img = np.moveaxis(img_chw, 0, -1)  # -> (H, W, 3)
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)


def overlay(image_rgb: np.ndarray, mask: np.ndarray,
            colour: tuple = (1, 0, 0)) -> np.ndarray:
    """Blend a binary mask onto an RGB image. `mask`: (H,W) {0,1}."""
    out = image_rgb.copy()
    m = mask > 0.5
    tint = np.zeros_like(out)
    tint[m] = colour
    return np.where(m[..., None], 0.5 * out + 0.5 * tint, out)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Eval SubPipe checkpoint.")
    ap.add_argument("--checkpoint", default="checkpoints/best.pth")
    ap.add_argument("--config", default="configs/config.yaml",
                    help="Fallback if checkpoint has no saved config.")
    ap.add_argument("--num-images", type=int, default=6,
                    help="Rows in the figure (best..worst by IoU).")
    ap.add_argument("--out", default="outputs/eval_examples.png")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="sigmoid threshold for pipe/no-pipe.")
    ap.add_argument("--data-root", default=None,
                    help="Override config root, e.g. data/Chunk1/Segmentation.")
    ap.add_argument("--full-chunk", action="store_true",
                    help="Score every frame in data-root, no split.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Prefer the config FROZEN in the checkpoint (reproducibility); fall back
    # to the live yaml only for old checkpoints.
    cfg = ckpt.get("config", yaml.safe_load(open(args.config)))
    seed = cfg["data"].get("seed", 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
          f"saved val IoU {ckpt.get('val_iou', float('nan')):.4f})")
    print(f"Device: {device}")

    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None,  # weights come from checkpoint, not ImageNet
        in_channels=cfg["model"].get("in_channels", 3),
        classes=cfg["model"].get("classes", 1),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Same seed => same val images Stage 4 validated on. Shuffle=False so
    # row i in the figure is a reproducible sample, not random.
    # --data-root + --full-chunk scores a fresh chunk with no retrain.
    eval_root = args.data_root or cfg["data"]["root"]
    if args.full_chunk:
        val_loader, _ = get_full_loader(
            root=eval_root,
            image_size=tuple(cfg["data"]["image_size"]),
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
        )
        print(f"Cross-chunk eval: full chunk in {eval_root} "
              "(no retrain).")
    else:
        _, val_loader = get_dataloaders(
            root=eval_root,
            image_size=tuple(cfg["data"]["image_size"]),
            batch_size=cfg["train"]["batch_size"],
            train_ratio=cfg["data"].get("train_split", 0.8),
            num_workers=cfg["data"]["num_workers"],
            seed=seed,
            split=cfg["data"].get("split", "random"),
        )

    # --- Score the whole val set, keeping per-image IoU for the figure ---
    all_ious, all_dices, saved = [], [], []
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        all_ious.append(
            [iou_score(logits[i:i + 1], masks[i:i + 1])
             for i in range(images.size(0))]
        )
        all_dices.append(
            [dice_score(logits[i:i + 1], masks[i:i + 1])
             for i in range(images.size(0))]
        )
        # Keep raw tensors for the figure (CPU, small val set so fine).
        probs = torch.sigmoid(logits).cpu()
        saved.append((images.cpu(), masks.cpu(), probs))
    ious = np.concatenate([np.array(b) for b in all_ious])
    dices = np.concatenate([np.array(b) for b in all_dices])
    print(f"VAL  n={len(ious)}  mean IoU={ious.mean():.4f}  "
          f"mean Dice={dices.mean():.4f}")
    print("-> Put these two numbers in your README results table (Stage 6).")
    print(f"-> Worst val IoU={ious.min():.4f} (likely sand-covered / faint "
          "pipe — discuss as a limitation, not a failure).")

    # --- Figure: best / median / worst rows so failures are visible ---
    flat_imgs = torch.cat([s[0] for s in saved])
    flat_gts = torch.cat([s[1] for s in saved])
    flat_pr = torch.cat([s[2] for s in saved])
    order = np.argsort(ious)  # ascending: worst first
    picks = np.linspace(0, len(ious) - 1, min(args.num_images, len(ious)),
                        dtype=int)
    # Reorder so the figure reads best -> worst top to bottom.
    picks = np.sort(order[picks])[::-1]

    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n),
                             squeeze=False)
    for r, idx in enumerate(picks):
        img = denormalise(flat_imgs[idx].numpy())
        gt = flat_gts[idx, 0].numpy() > 0.5
        pr = (flat_pr[idx, 0].numpy() > args.thresh)
        axes[r][0].imshow(img)
        axes[r][0].set_title("input image")
        axes[r][1].imshow(overlay(img, gt, colour=(0, 1, 0)))
        axes[r][1].set_title("ground truth (green)")
        axes[r][2].imshow(overlay(img, pr, colour=(1, 0, 0)))
        axes[r][2].set_title(f"prediction (red) IoU={ious[idx]:.2f} "
                             f"Dice={dices[idx]:.2f}")
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(f"SubPipe val overlays — mean IoU {ious.mean():.3f}, "
                 f"Dice {dices.mean():.3f} (best top, worst bottom)",
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved figure -> {args.out}. Check: red should sit on the pipe; "
          "where it doesn't, note WHY (sand? blur? edge?) for Stage 6.")


if __name__ == "__main__":
    main()
