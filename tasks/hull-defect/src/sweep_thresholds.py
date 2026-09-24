"""Sweep sigmoid thresholds for production classes (Phase 2).

Run from repo root:
    python tasks/hull-defect/src/sweep_thresholds.py --checkpoint \
        tasks/hull-defect/checkpoints_bce_dice_40/best.pth

Default classes are the shippable subset: ship_hull, propeller.
Research classes stay out of the operator view. Prints per-threshold
precision, recall, IoU so thresholds are picked from val, not guessed.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks" / "hull-defect" / "src"))

from liaci_dataset import CLASSES, get_dataloaders  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import set_seed  # noqa: E402


def find_checkpoint(arg: str) -> str:
    """Accept an exact path, else search Kaggle inputs for the file name."""
    if Path(arg).exists():
        return arg
    name = Path(arg).name
    cands = list(Path("/kaggle/input").rglob(name)) if Path(
        "/kaggle/input").exists() else []
    if cands:
        print(f"Using Kaggle input: {cands[0]}")
        return str(cands[0])
    raise FileNotFoundError(f"No checkpoint at {arg}.")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold sweep.")
    ap.add_argument("--checkpoint",
                    default="tasks/hull-defect/checkpoints_bce_dice_40/best.pth")
    ap.add_argument("--config", default="tasks/hull-defect/config.yaml")
    ap.add_argument("--thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--classes", default="ship_hull,propeller")
    args = ap.parse_args()

    ckpt_path = find_checkpoint(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", yaml.safe_load(open(args.config)))
    set_seed(cfg["data"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None, in_channels=3, classes=len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, val_loader, _, _ = get_dataloaders(
        root=cfg["data"]["root"],
        image_size=tuple(cfg["data"]["image_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    thresh = [float(t) for t in args.thresholds.split(",")]
    want = [c.strip() for c in args.classes.split(",") if c.strip()]
    idx = [CLASSES.index(c) for c in want]

    tp = torch.zeros(len(want), len(thresh))
    fp = torch.zeros(len(want), len(thresh))
    fn = torch.zeros(len(want), len(thresh))
    for images, masks in tqdm(val_loader, desc="sweep", leave=False):
        images = images.to(device)
        tgt = (masks > 0.5).float()
        probs = torch.sigmoid(model(images)).cpu()
        for j, t in enumerate(thresh):
            pred = (probs > t).float()
            tp[:, j] += (pred[:, idx] * tgt[:, idx]).sum(dim=(0, 2, 3))
            fp[:, j] += (pred[:, idx] * (1 - tgt[:, idx])).sum(dim=(0, 2, 3))
            fn[:, j] += ((1 - pred[:, idx]) * tgt[:, idx]).sum(dim=(0, 2, 3))
    print(f"{'class':<18}{'thresh':>8}{'prec':>8}{'rec':>8}{'IoU':>8}")
    for i, c in enumerate(want):
        for j, t in enumerate(thresh):
            p = float(tp[i, j] / (tp[i, j] + fp[i, j] + 1e-6))
            r = float(tp[i, j] / (tp[i, j] + fn[i, j] + 1e-6))
            u = float(tp[i, j] / (tp[i, j] + fp[i, j] + fn[i, j] + 1e-6))
            print(f"{c:<18}{t:>8.2f}{p:>8.4f}{r:>8.4f}{u:>8.4f}")


if __name__ == "__main__":
    main()
