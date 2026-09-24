"""LIACI multi-label dataset (Phase 2, hull-defect head).

Layout (SINTEF pretty version):
    <root>/images/image_NNNN.jpg          (1920x1080 RGB)
    <root>/masks/<class>/image_NNNN.bmp   (1-bit bitmap per class)
    <root>/train_test_split.csv           (file_name, split)

Target: (10, H, W) float32, one binary channel per class in CLASSES
order. Multi-label, not one-hot: growth sits on hull, so a pixel can
carry two labels. Loss is BCE per channel with pos_weight for rares.

Split: the official csv, seed only shuffles within train. Never train
on the eval split.
"""

import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ["ship_hull", "anode", "marine_growth", "paint_peel", "corrosion",
           "defect", "propeller", "sea_chest_grating", "over_board_valves",
           "bilge_keel"]

# Prevalence on full set (non-empty frames): hull 88%, growth 46%,
# peel 45%, anode 28%, propeller 26%, grating 22%, valves 12%,
# corrosion 11%, keel 10%, defect 4%. Rare rows get loss weight.


def read_split(root: str | Path) -> tuple[list[str], list[str]]:
    """Official file lists. Returns (train_names, test_names)."""
    rows = list(csv.DictReader(open(Path(root) / "train_test_split.csv")))
    train = [r["file_name"] for r in rows if r["split"].lower() == "train"]
    test = [r["file_name"] for r in rows if r["split"].lower() != "train"]
    if not train or not test:
        raise ValueError("Split csv gave empty train or test. Check values.")
    return train, test


class LiaciDataset(Dataset):
    """One image + stacked mask. Mask (H,W,C) for C in class_idx.
    Default None keeps all 10 CLASSES (baseline contract)."""

    def __init__(self, root: str | Path, names: list[str],
                 transform: A.Compose | None = None,
                 class_idx: list[int] | None = None) -> None:
        self.root = Path(root)
        self.names = names
        self.transform = transform
        self.class_idx = class_idx if class_idx is not None else list(
            range(len(CLASSES)))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        stem = Path(name).stem
        img = cv2.imread(str(self.root / "images" / name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        chans = []
        for i in self.class_idx:
            c = CLASSES[i]
            m = np.array(Image.open(self.root / "masks" / c / (stem + ".bmp")))
            chans.append((m > 0).astype(np.float32))
        mask = np.stack(chans, axis=-1)  # (H, W, C)
        if self.transform is not None:
            s = self.transform(image=img, mask=mask)
            img, mask = s["image"], s["mask"]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(mask, -1, 0)))
        elif (isinstance(mask, torch.Tensor) and mask.ndim == 3
                and mask.shape[0] != len(self.class_idx)
                and mask.shape[-1] == len(self.class_idx)):
            mask = mask.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img, mask.float()


def get_train_transforms(image_size: tuple[int, int] = (256, 256)) -> A.Compose:
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
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
    h, w = image_size
    return A.Compose(
        [
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_dataloaders(
    root: str | Path,
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 8,
    num_workers: int = 2,
    classes: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Train loader from official train list, val loader from test list.
    classes subsets the channels (default all 10, baseline contract)."""
    names = classes if classes is not None else CLASSES
    idx = [CLASSES.index(c) for c in names]
    train_names, test_names = read_split(root)
    train_ds = LiaciDataset(root, train_names,
                            get_train_transforms(image_size), idx)
    val_ds = LiaciDataset(root, test_names,
                          get_val_transforms(image_size), idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"LIACI: {len(train_ds)} train / {len(val_ds)} val (official split).")
    print(f"Channels: {names}")
    return train_loader, val_loader, train_names, test_names
