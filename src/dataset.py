"""PyTorch Dataset / DataLoader for SubPipe pipeline segmentation (Chunk0).

What this does (read top to bottom):
  1. `find_pairs()` — discovers image + mask file pairs on disk.
  2. `SubPipeSegDataset` — loads ONE pair (RGB image + binary pipe mask).
  3. `train_val_split()` — reproducibly splits pairs into train / val.
  4. `get_*_transforms()` — augmentation pipelines (Albumentations).
  5. `get_dataloaders()` — one-call helper used by train.py (Stage 4).

Expected layout (see data/README.md):
    data/Chunk0/Segmentation/
        <timestamp>.png          <- input image (RGB)
        <timestamp>_label.png    <- mask (grayscale)

Classes (from the SubPipe paper, Alvarez-Tunon et al. 2024):
    - background (seabed, water, sand) vs. `pipeline` (pipe + clamp as ONE class).
    - So this is BINARY segmentation: 1 foreground class.

Pixel encoding — VERIFY on your data (Stage 2 inspection script):
    - The repo/paper do NOT document exact values. LabelMe masks are
      usually 0 = background, 255 = pipe (sometimes 0/1).
    - The code below accepts BOTH: anything > 1 is divided by 255,
      then thresholded at 0.5. Run notebooks/inspect_masks.py and
      check the printed "unique values" to confirm.
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Pair discovery
# ---------------------------------------------------------------------------
def find_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Find (image, mask) pairs under `root`.

    Masks end in `_label.png`; the image is the same name minus `_label`.
    Pairs with a missing image are skipped with a warning so one corrupt
    download does not kill training.
    """
    root = Path(root)
    mask_paths = sorted(root.glob("*_label.png"))
    if not mask_paths:
        raise FileNotFoundError(
            f"No '*_label.png' masks found in {root.resolve()}. "
            "Did you unzip Chunk0/Segmentation there? See data/README.md."
        )
    pairs: list[tuple[Path, Path]] = []
    for mask_path in mask_paths:
        # e.g. 168070..._label.png -> 168070....png
        image_path = mask_path.with_name(mask_path.name.replace("_label", ""))
        if not image_path.exists():
            print(f"[warn] mask without image, skipping: {mask_path.name}")
            continue
        pairs.append((image_path, mask_path))
    print(f"Found {len(pairs)} image+mask pairs in {root} "
          f"({len(mask_paths) - len(pairs)} masks skipped).")
    return pairs


# ---------------------------------------------------------------------------
# 2. The Dataset — teaches PyTorch how to load ONE sample
# ---------------------------------------------------------------------------
class SubPipeSegDataset(Dataset):
    """Loads one RGB image + its binary pipe mask as tensors.

    Returns:
        image: float32 tensor, shape (3, H, W), range [0, 1] then
               ImageNet-normalized (because our encoder is pretrained).
        mask:  float32 tensor, shape (1, H, W), values {0.0, 1.0}.

    Why `transform` handles both: geometric ops (flip/rotate/crop) MUST be
    applied identically to image and mask, or the mask no longer lines up
    with the pipe. Albumentations does this for us when we pass
    `transform(image=..., mask=...)` — colour ops touch only the image.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        transform: A.Compose | None = None,
    ) -> None:
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        # OpenCV loads BGR; convert to RGB so colours are correct.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask as grayscale: shape (H, W), values e.g. {0, 255} or {0, 1}.
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Normalise to {0.0, 1.0}: handles both 0/255 and 0/1 encodings.
        # VERIFY with inspect_masks.py — if you see values like {0, 128, 255},
        # stop: that would mean multi-class and this threshold is wrong.
        if mask.max() > 1:
            mask = (mask / 255.0 > 0.5).astype(np.float32)
        else:
            mask = (mask > 0.5).astype(np.float32)

        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # ToTensorV2 already gives (C, H, W) float; mask needs a channel dim.
        if mask.ndim == 2:
            mask = mask[None, ...]  # (H, W) -> (1, H, W)
        return image, mask.float()


# ---------------------------------------------------------------------------
# 3. Train/val split — fixed seed => same split every run
# ---------------------------------------------------------------------------
def train_val_split(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    """Deterministically shuffle with `seed`, then split 80/20.

    NOTE for your defence: the SubPipe paper used a CHRONOLOGICAL split
    (first 60% train / next 20% val / last 20% test) because consecutive
    video frames look almost identical — a random split can leak "same
    scene" into both sets and inflate scores. We use random+seed here for
    simplicity on Chunk0 (portfolio-friendly), but mention the paper's
    approach and its reason as a limitation/future fix.
    """
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]


# ---------------------------------------------------------------------------
# 4. Augmentations — what we apply and why
# ---------------------------------------------------------------------------
def get_train_transforms(image_size: tuple[int, int] = (256, 256)) -> A.Compose:
    """Training augmentations (mirrors the paper + Colab constraints).

    Geometry (applied to image AND mask — simulates new viewpoints):
      - HorizontalFlip / VerticalFlip: pipe can run either direction.
      - ShiftScaleRotate (small): simulates AUV drift/altitude change.
    Appearance (image ONLY — simulates water/lighting; mask untouched):
      - RandomBrightnessContrast, HueSaturationValue: underwater colour
        shifts, turbidity, GoPro auto-exposure.
    Always: Resize to image_size (fits T4 GPU memory), ImageNet Normalize
    (required because the encoder was pretrained on ImageNet), ToTensorV2.
    """
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: tuple[int, int] = (256, 256)) -> A.Compose:
    """Validation: NO augmentation — just Resize + Normalize + tensor.

    We measure on clean, deterministic inputs so val IoU/Dice reflects
    the model, not random flips.
    """
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# 5. One-call helper for training
# ---------------------------------------------------------------------------
def get_dataloaders(
    root: str | Path = "data/Chunk0/Segmentation",
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 8,
    train_ratio: float = 0.8,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders.

    DataLoader = the "batcher": shuffles train, stacks samples into
    (B, C, H, W) batches, loads in parallel via num_workers.
    """
    pairs = find_pairs(root)
    train_pairs, val_pairs = train_val_split(pairs, train_ratio, seed)

    train_ds = SubPipeSegDataset(train_pairs, get_train_transforms(image_size))
    val_ds = SubPipeSegDataset(val_pairs, get_val_transforms(image_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Split: {len(train_ds)} train / {len(val_ds)} val "
          f"(seed={seed}, ratio={train_ratio}).")
    return train_loader, val_loader
