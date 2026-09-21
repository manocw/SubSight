"""U-Net model for SubPipe binary pipe segmentation (Stage 3).

READ THIS FIRST (the whole idea in 30 seconds):
  Segmentation = classify EVERY pixel: pipe (1) or background (0).
  U-Net does this with two halves:
    - Encoder (down path): a ResNet-34 pretrained on ImageNet squeezes the
      256x256 image down, learning "WHAT is where" (edges -> textures ->
      pipe shapes). Pretrained = transfer learning: we reuse vision features
      learned from millions of photos instead of starting from zero.
    - Decoder (up path): upsamples back to full resolution, learning
      "WHERE exactly" the pipe boundaries are.
    - Skip connections: shortcuts that pipe fine detail straight from the
      encoder to the decoder, so thin pipe edges survive the squeeze.
      That is THE U-Net trick — without them, masks come out blobby.

Why this default: `resnet34 + imagenet` is light (~24M params), fits a
Colab T4 at 256x256 batch 8, and is a strong, defensible baseline before
trying anything heavier.

Output convention (must match dataset.py + train.py):
    model(image [B,3,H,W]) -> logits [B,1,H,W] (RAW scores, no sigmoid).
    Loss (Stage 4): BCEWithLogitsLoss. Eval: prob = sigmoid(logit) > 0.5.
    We use 1 channel, not 2, because it is binary: one "pipe-ness" score
    per pixel is enough.

SegFormer swap path (paper used it — you asked to keep the door open):
    Keep calling `build_model(cfg)`; only the inside changes. When ready:
      1. `pip install transformers` (deliberately NOT in requirements yet
         so you can defend U-Net first),
      2. set `model.architecture: segformer` in configs/config.yaml,
      3. the branch below loads HuggingFace SegFormer with 1 output class.
    The training loop never changes because input/output shapes are identical.
"""

import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # friendly error for Colab/pip issues
    raise ImportError(
        "segmentation-models-pytorch not found. Run: "
        "pip install -r requirements.txt"
    ) from exc


def build_model(
    architecture: str = "unet",
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Build the segmentation model. Single entry point for train/eval.

    Args:
        architecture: "unet" (implemented) or "segformer" (swap path, Stage 3+).
        encoder: smp encoder name, e.g. "resnet34", "efficientnet-b0".
        encoder_weights: "imagenet" (transfer learning) or None (from scratch,
            like the paper's SegFormer — slower, only for comparison).
        in_channels: 3 for RGB.
        classes: 1 = one pipe logit per pixel (binary). Do NOT set 2.

    Returns:
        nn.Module mapping (B,3,H,W) -> (B,1,H,W) logits.
    """
    architecture = architecture.lower()
    if architecture == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,  # None => from scratch
            in_channels=in_channels,
            classes=classes,  # 1 logit => BCEWithLogitsLoss in Stage 4
        )
        return model

    if architecture == "segformer":
        # --- Future swap path: same I/O, different inside. ---
        # SegFormer ( transformer encoder + MLP decoder ) sees the whole
        # image at once (global context: long straight pipe), where U-Net
        # sees mostly local neighbourhoods. Paper trained it from scratch.
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "You selected architecture='segformer' but `transformers` "
                "is not installed. Run `pip install transformers` first "
                "(kept out of requirements.txt until you defend U-Net)."
            ) from exc
        hf_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-224",
            num_labels=1,  # binary: one logit, same as U-Net above
        )
        # HuggingFace returns (B,1,H/4,W/4) logits + dict wrapper; unwrap and
        # upsample to full resolution so the rest of the code is unchanged.
        return _HFLogitWrapper(hf_model)

    raise ValueError(
        f"Unknown architecture={architecture!r}. Use 'unet' or 'segformer'."
    )


class _HFLogitWrapper(nn.Module):
    """Unwrap HF SegFormer output -> full-resolution (B,1,H,W) logits.

    You never call this directly; build_model() returns it for 'segformer'.
    Kept tiny on purpose: the training loop sees the same tensor shape
    either way, which is exactly what makes the swap a one-line config change.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: nn.Module) -> nn.Module:
        import torch.nn.functional as F

        out = self.hf_model(pixel_values=x).logits  # (B,1,h,w), small
        return F.interpolate(
            out, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) params — handy for your report/defence."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# Quick smoke test: `python -m src.model` (CPU, no data needed).
if __name__ == "__main__":
    import torch

    m = build_model()  # defaults: unet / resnet34 / imagenet / 1 class
    m.eval()
    with torch.no_grad():
        fake = torch.zeros(1, 3, 256, 256)  # one fake image
        logits = m(fake)
    total, trainable = count_parameters(m)
    print(f"arch=unet encoder=resnet34 params={total:,} trainable={trainable:,}")
    print(f"in {tuple(fake.shape)} -> out {tuple(logits.shape)} "
          "(expect (1,1,256,256) raw logits)")
