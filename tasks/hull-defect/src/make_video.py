"""Render a hull-defect prediction clip: raw frame + overlay.

Run from repo root:
    python tasks/hull-defect/src/make_video.py --checkpoint \
        tasks/hull-defect/checkpoints/best.pth --max-frames 60

Left = raw image, right = overlay (anode red, corrosion yellow,
peel magenta, defect cyan). Val frames in filename order with
per-frame macro IoU stamped on.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks" / "hull-defect" / "src"))

from liaci_dataset import CLASSES, LiaciDataset, get_val_transforms, read_split  # noqa: E402
from src.model import build_model  # noqa: E402

FOCUS = ["anode", "corrosion", "paint_peel", "defect"]
COLOURS = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def overlay(frame_bgr: np.ndarray, masks: np.ndarray,
            idx: list[int]) -> np.ndarray:
    out = frame_bgr.copy()
    for k, i in enumerate(idx):
        m = masks[i] > 0.5
        tint = np.zeros_like(out)
        tint[m] = COLOURS[k]
        out = np.where(m[..., None],
                       (0.5 * out + 0.5 * tint).astype(np.uint8), out)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Render LIACI prediction clip.")
    ap.add_argument("--checkpoint",
                    default="tasks/hull-defect/checkpoints/best.pth")
    ap.add_argument("--config", default="tasks/hull-defect/config.yaml")
    ap.add_argument("--out", default="tasks/hull-defect/outputs/hull_pred.mp4")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", yaml.safe_load(open(args.config)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None, in_channels=3, classes=len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, val_names = read_split(cfg["data"]["root"])
    names = sorted(val_names)[:args.max_frames]
    ds = LiaciDataset(cfg["data"]["root"], names,
                      get_val_transforms(tuple(cfg["data"]["image_size"])))
    idx = [CLASSES.index(c) for c in FOCUS]
    h, w = cfg["data"]["image_size"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w * 2, h))
    for n in range(len(ds)):
        img_t, gt_t = ds[n]
        logits = model(img_t[None].to(device))
        pred = (torch.sigmoid(logits)[0].cpu().numpy() > args.thresh)
        gt = gt_t.numpy() > 0.5
        rgb = np.moveaxis(img_t.numpy(), 0, -1) * _STD + _MEAN
        bgr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]
        ov = overlay(overlay(bgr, gt, idx), pred, idx)
        inter = (pred & gt).sum(axis=(1, 2))
        union = (pred | gt).sum(axis=(1, 2))
        ious = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        macro = float(np.nanmean(ious))
        lines = [f"macro {macro:.2f}"]
        for k, i in enumerate(idx):
            v = ious[i]
            tag = f"{FOCUS[k][:6]} {v:.2f}" if not np.isnan(v) else f"{FOCUS[k][:6]} n/a"
            lines.append(tag)
        for j, t in enumerate(lines):
            cv2.putText(ov, t, (10, 22 + 18 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(np.concatenate([bgr, ov], axis=1))
    writer.release()
    print(f"Saved clip -> {out_path} ({len(ds)} frames).")


if __name__ == "__main__":
    main()
