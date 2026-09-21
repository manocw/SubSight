"""Inspect SubPipe Chunk0 masks — RUN THIS FIRST (Colab-friendly).

Purpose: answer definitively "what are the classes and pixel encoding?"
The paper says binary (background vs `pipeline`, clamp included) but does
NOT document pixel values, so we measure them here.

Run (Colab, after downloading Chunk0 per data/README.md):
    !python notebooks/inspect_masks.py --root data/Chunk0/Segmentation --n 5

What it prints per mask:
  - image size + mask size (should match each other)
  - unique pixel values, e.g. [0, 255]  -> binary 0/255 encoding
                           e.g. [0, 1]    -> binary 0/1 encoding
                           e.g. [0, 128, 255] -> MULTI-CLASS, stop and tell us
  - pipe pixel fraction (how much of the frame is pipe — expect small, ~5-20%)

It also saves outputs/inspect_examples.png: image | mask | overlay,
so you can SEE the alignment with your own eyes.

Needs only: Pillow, numpy, matplotlib (no torch — runs anywhere).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # no display needed (Colab / headless safe)
import matplotlib.pyplot as plt


def inspect_mask(mask_path: Path) -> dict:
    """Load one mask, return size, unique values, pipe fraction."""
    m = np.array(Image.open(mask_path))  # keep raw values, no conversion
    unique = np.unique(m).tolist()
    # Pipe fraction: assume the NON-ZERO value(s) are pipe. For binary
    # masks this is exact; for anything else it is a rough signal.
    pipe_frac = float((m > 0).mean())
    return {
        "shape": m.shape,
        "mode": Image.open(mask_path).mode,
        "unique": unique,
        "pipe_frac": pipe_frac,
        "array": m,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect SubPipe masks.")
    ap.add_argument("--root", default="data/Chunk0/Segmentation",
                    help="Folder with <ts>.png + <ts>_label.png pairs.")
    ap.add_argument("--n", type=int, default=5,
                    help="How many examples to print + plot.")
    ap.add_argument("--out", default="outputs/inspect_examples.png",
                    help="Where to save the figure.")
    args = ap.parse_args()

    root = Path(args.root)
    masks = sorted(root.glob("*_label.png"))
    if not masks:
        raise FileNotFoundError(
            f"No '*_label.png' in {root.resolve()}. "
            "Download Chunk0 first — see data/README.md."
        )
    print(f"Found {len(masks)} masks in {root}.\n")

    # --- 1. Global sweep: are ALL masks the same encoding? ---
    global_unique: set = set()
    for mp in masks:
        global_unique.update(np.unique(np.array(Image.open(mp))).tolist())
    print(f"Masks scanned: {len(masks)}")
    print(f"UNIQUE VALUES across ALL masks: {sorted(global_unique)}")
    if sorted(global_unique) in ([0, 255], [0, 1], [0], [1]):
        print("=> Binary mask confirmed: background + ONE pipe class.\n")
    else:
        print("=> WARNING: more than 2 values! Possible multi-class or "
              "anti-aliased edges — inspect the figure and report back.\n")

    # --- 2. Per-example detail + figure ---
    show = masks[: args.n]
    fig, axes = plt.subplots(len(show), 3, figsize=(9, 3 * len(show)))
    if len(show) == 1:
        axes = axes[None, :]
    for row, mp in enumerate(show):
        img_path = mp.with_name(mp.name.replace("_label", ""))
        info = inspect_mask(mp)
        has_img = img_path.exists()
        print(f"{mp.name}: shape={info['shape']} mode={info['mode']} "
              f"unique={info['unique']} pipe_frac={info['pipe_frac']:.3f} "
              f"image_found={has_img}")
        if has_img:
            img = np.array(Image.open(img_path).convert("RGB"))
            if img.shape[:2] != info["shape"]:
                print(f"  [warn] image {img.shape[:2]} != mask {info['shape']}")
        else:
            img = np.zeros((*info["shape"], 3), dtype=np.uint8)

        # Normalise mask to 0/1 for display whatever the encoding.
        disp = (np.array(info["array"]) > 0).astype(float)
        overlay = img.copy()
        overlay[disp > 0.5] = (overlay[disp > 0.5] * 0.5
                               + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("image")
        axes[row, 1].imshow(disp, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"mask unique={info['unique']}")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"overlay pipe={info['pipe_frac']:.1%}")
        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved figure -> {args.out}. Open it and confirm the red "
          "overlay sits exactly on the pipe.")


if __name__ == "__main__":
    main()
