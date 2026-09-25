"""Export trained checkpoints to ONNX for the web app.

Run on a GPU box (needs torch + transformers + the repo):
    python web/export_onnx.py --out web/models
    python web/export_onnx.py --out web/models --hull-checkpoint \
        tasks/hull-defect/checkpoints_bce_dice_40/best.pth --expect-hull 10

Reads checkpoints/best.pth (pipe) and
tasks/hull-defect/checkpoints_prod3/best.pth (hull, 3ch prod head),
writes pipe.onnx and hull.onnx per web/CONTRACT.md. Channel order is
the checkpoint config data.classes. The 10ch file in
checkpoints_bce_dice_40 stays on disk as fallback and is never
overwritten. The web app never trains and never imports torch.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml


def export_pipe(out_dir: Path, ckpt_path: Path) -> None:
    from src.model import build_model

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No pipe checkpoint at {ckpt_path}. "
            f"Pass --pipe-checkpoint explicitly.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.zeros(1, 3, 256, 256)
    path = out_dir / "pipe.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["input"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"},
                                    "logits": {0: "batch"}})
    print(f"wrote {path}")


def export_hull(out_dir: Path, ckpt_path: Path) -> int:
    sys.path.insert(0, str(ROOT / "tasks" / "hull-defect" / "src"))
    from src.model import build_model

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No hull checkpoint at {ckpt_path}. Train prod3 first or "
            f"pass --hull-checkpoint explicitly.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    # Channel order is the contract: config data.classes.
    channels = cfg["data"].get("classes", ["ship_hull"] * 10)
    print(f"hull channels ({len(channels)}): {channels}")
    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None, in_channels=3, classes=len(channels))
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.zeros(1, 3, 256, 256)
    path = out_dir / "hull.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["input"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"},
                                    "logits": {0: "batch"}})
    print(f"wrote {path}")
    return len(channels)


def verify(out_dir: Path, hull_chans: int) -> None:
    import onnxruntime as ort

    for name, chans in [("pipe.onnx", 1), ("hull.onnx", hull_chans)]:
        sess = ort.InferenceSession(str(out_dir / name))
        x = np.zeros((1, 3, 256, 256), dtype=np.float32)
        y = sess.run(None, {"input": x})[0]
        assert y.shape == (1, chans, 256, 256), y.shape
        print(f"{name} ok, out {y.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ONNX models.")
    ap.add_argument("--out", default="web/models")
    ap.add_argument("--skip", default="",
                    help="comma list of pipe,hull to skip")
    ap.add_argument("--pipe-checkpoint", default="checkpoints/best.pth")
    ap.add_argument("--hull-checkpoint",
                    default="tasks/hull-defect/checkpoints_prod3/best.pth")
    ap.add_argument("--expect-hull", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    if "pipe" not in skip:
        export_pipe(out, ROOT / args.pipe_checkpoint)
    hull_chans = args.expect_hull
    if "hull" not in skip:
        hull_chans = export_hull(out, ROOT / args.hull_checkpoint)
        assert hull_chans == args.expect_hull, \
            f"hull chans {hull_chans} vs expected {args.expect_hull}"
    verify(out, hull_chans)


if __name__ == "__main__":
    main()
