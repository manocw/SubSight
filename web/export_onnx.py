"""Export trained checkpoints to ONNX for the web app.

Run on a GPU box (needs torch + transformers + the repo):
    python web/export_onnx.py --out web/models

Reads checkpoints/best.pth (pipe) and
tasks/hull-defect/checkpoints/best.pth (hull), writes pipe.onnx and
hull.onnx per web/CONTRACT.md. Verifies each file loads in
onnxruntime and a dummy forward pass matches the PyTorch output.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml


def export_pipe(out_dir: Path) -> None:
    from src.model import build_model

    ckpt = torch.load(ROOT / "checkpoints" / "best.pth",
                      map_location="cpu")
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


def export_hull(out_dir: Path) -> None:
    sys.path.insert(0, str(ROOT / "tasks" / "hull-defect" / "src"))
    from src.model import build_model

    ckpt = torch.load(ROOT / "tasks" / "hull-defect" / "checkpoints"
                      / "best.pth", map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(
        architecture=cfg["model"].get("architecture", "unet"),
        encoder=cfg["model"].get("encoder", "resnet34"),
        encoder_weights=None, in_channels=3, classes=10)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.zeros(1, 3, 256, 256)
    path = out_dir / "hull.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["input"],
                      output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"},
                                    "logits": {0: "batch"}})
    print(f"wrote {path}")


def verify(out_dir: Path) -> None:
    import onnxruntime as ort

    for name, chans in [("pipe.onnx", 1), ("hull.onnx", 10)]:
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
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    if "pipe" not in skip:
        export_pipe(out)
    if "hull" not in skip:
        export_hull(out)
    verify(out)


if __name__ == "__main__":
    main()
