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

Pixel encoding — VERIFIED on real Chunk0 data (Stage 2 inspection):
    masks hold exactly THREE values: 0 = background (87.98% of pixels),
    1 = pipe body (10.47%), 128 = pipe boundary (~1.56%, hugs the pipe
    edges — confirm on outputs/mask_values.png). There is NO 255 encoding.
    `decode_mask()` below maps this explicitly and FAILS LOUD on anything
    unexpected, instead of a silent threshold that would corrupt training.
"""

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Mask decoding — the ONLY place raw pixel values are interpreted
# ---------------------------------------------------------------------------
PIPE_VALUE = 1
BOUNDARY_VALUE = 128
# Boundary pixels ARE pipe edge, so they train as pipe. Flip to False only
# if your mask_values.png shows 128 somewhere other than pipe edges.
BOUNDARY_AS_PIPE = True


def decode_mask(raw: np.ndarray) -> np.ndarray:
    """Raw SubPipe mask (values {0, 1, 128}) -> binary {0.0, 1.0} float32.

    Explicit mapping beats a threshold: a threshold can't tell "pipe=1"
    from "background", and it silently passes new values through. This
    raises ValueError on anything outside {0, 1, 128} so a changed encoding
    (e.g. a new chunk with 255s) stops loudly instead of training wrong.

    IMPORTANT: pass RAW indices (PIL), not cv2 output. These masks are
    palette-mode PNGs: PIL returns indices {0,1,128}, but cv2 expands the
    palette and grayscales, turning index 1 into grey ~38. Same file,
    two readings — always PIL here.
    """
    unexpected = set(np.unique(raw).tolist()) - {0, PIPE_VALUE, BOUNDARY_VALUE}
    if unexpected:
        raise ValueError(
            f"Unexpected mask values {sorted(unexpected)}. "
            "Encoding changed? Update PIPE_VALUE/BOUNDARY_VALUE above."
        )
    mask = (raw == PIPE_VALUE).astype(np.float32)
    if BOUNDARY_AS_PIPE:
        mask[raw == BOUNDARY_VALUE] = 1.0
    return mask


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
        # Mask e.g. 1693573934.247_label.png -> image 1693573934.247.<ext>.
        # Images are mixed .png AND .jpg in SubPipeMini, so try the mask's
        # own suffix first, then common photo suffixes.
        stem = mask_path.name[: -len("_label" + mask_path.suffix)]
        candidates = [mask_path.with_name(stem + ext)
                      for ext in dict.fromkeys(
                          [mask_path.suffix, ".png", ".jpg", ".jpeg"])]
        image_path = next((c for c in candidates if c.exists()), None)
        if image_path is None:
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

        # Mask via PIL (NOT cv2): palette-mode PNGs must stay raw indices.
        # See decode_mask() docstring for why cv2 corrupts them.
        # Decoded to {0.0: background, 1.0: pipe incl. boundary}.
        with Image.open(mask_path) as im:
            if im.mode not in ("L", "P"):
                im = im.convert("L")  # RGB/RGBA mask -> luminance, then decode
            raw = np.array(im)
        if raw.ndim == 3:  # safety net, should not happen
            raw = raw[..., 0]
        mask = decode_mask(raw)

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
