"""Render a prediction video: raw frame + predicted pipe overlay (Stage 5+).

Run (Colab, AFTER training — needs checkpoints/best.pth + Chunk0 data):
    python -m src.make_video --checkpoint checkpoints/best.pth
    python -m src.make_video --checkpoint checkpoints/best.pth --max-frames 200 --fps 10

What it does: takes the VAL pairs, restores TIMESTAMP order (filenames are
timestamps, so sorting = video order), runs the model frame by frame, and
writes outputs/pred_video.mp4: left = raw image, right = overlay
(green = ground truth, red = prediction) with per-frame IoU stamped on.
Watch the red track the pipe — where it flickers or vanishes (sand-covered
stretches), note the timestamps for your limitations section.

Needs: torch + opencv (both in requirements.txt). CPU is fine for a short
clip; GPU is faster.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.dataset import (
    decode_mask,
    find_pairs,
    get_val_transforms,
    read_mask_raw,
    train_val_split,
)
from src.model import build_model

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalise(img_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) normalised -> (H,W,3) uint8 BGR for cv2."""
    rgb = np.moveaxis(img_chw, 0, -1) * _IMAGENET_STD + _IMAGENET_MEAN
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)[..., ::-1]  # RGB -> BGR


def overlay_bgr(frame_bgr: np.ndarray, mask_hw: np.ndarray,
                colour_bgr: tuple) -> np.ndarray:
    """Blend a binary mask onto a BGR frame."""
    out = frame_bgr.copy()
    m = mask_hw > 0.5
    tint = np.zeros_like(out)
    tint[m] = colour_bgr
    return np.where(m[..., None],
                    (0.5 * out + 0.5 * tint).astype(np.uint8), out)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Render SubPipe prediction video.")
    ap.add_argument("--checkpoint", default="checkpoints/best.pth")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="outputs/pred_video.mp4")
    ap.add_argument("--fps", type=int, default=5,
                    help="Val frames are sparse; 5 fps looks natural.")
    ap.add_argument("--max-frames", type=int, default=100,
                    help="Cap length for portfolio clips.")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", yaml.safe_load(open(args.config)))
    seed = cfg["data"].get("seed", 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}). Device: {device}")

    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None,  # weights come from the checkpoint
        in_channels=cfg["model"].get("in_channels", 3),
        classes=cfg["model"].get("classes", 1),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Same val set as training, but sorted by filename (= timestamp order).
    pairs = find_pairs(cfg["data"]["root"])
    _, val_pairs = train_val_split(pairs, cfg["data"]["train_split"], seed)
    val_pairs = sorted(val_pairs, key=lambda p: p[0].name)[:args.max_frames]
    transform = get_val_transforms(tuple(cfg["data"]["image_size"]))
    print(f"Rendering {len(val_pairs)} frames in timestamp order...")

    h, w = cfg["data"]["image_size"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps, (w * 2, h))  # side-by-side: raw | overlay

    for img_path, mask_path in val_pairs:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = decode_mask(read_mask_raw(mask_path))  # PIL mode-aware, see dataset
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        sample = transform(image=rgb, mask=gt.astype(np.float32))
        logits = model(sample["image"][None].to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = prob > args.thresh

        frame = denormalise(sample["image"].numpy())  # BGR, (H, W, 3)
        ov = overlay_bgr(frame, sample["mask"].numpy(), (0, 255, 0))  # GT green
        ov = overlay_bgr(ov, pred, (0, 0, 255))  # pred red on top
        inter = (pred & (sample["mask"].numpy() > 0.5)).sum()
        union = (pred | (sample["mask"].numpy() > 0.5)).sum()
        iou = inter / max(union, 1)
        cv2.putText(ov, f"IoU {iou:.2f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(np.concatenate([frame, ov], axis=1))

    writer.release()
    print(f"Saved video -> {out_path}. If red flickers over sand, "
          "those are your limitation timestamps.")


if __name__ == "__main__":
    main()
