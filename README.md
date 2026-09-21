# SubPipe — Automatic Segmentation of Subsea Pipelines

Automatic pixel-level segmentation of subsea pipelines in underwater images,
using the public **SubPipe dataset** (real AUV inspection data, Chunk0).
Built as a portfolio project by a first-year mechanical engineering student
aiming for subsea engineering — every stage is small, commented, and defensible.

**Pipeline:** raw image → U-Net → binary pipe mask → IoU/Dice + overlay figures.

## What this does

1. Loads image + mask pairs from SubPipe `Segmentation/` (Chunk0 only).
2. Trains a **U-Net** (ResNet-34 encoder, ImageNet-pretrained) to predict a
   binary mask: pipe (1) vs background (0).
3. Reports **IoU / Dice** on a held-out val split, with fixed seeds and
   `best.pth` / `last.pth` checkpointing.
4. Saves overlay figures (image + true vs predicted mask) ordered best → worst,
   so failure modes are visible, not hidden.

## Dataset

- Docs: https://github.com/remaro-network/SubPipe-dataset
- Download: https://zenodo.org/doi/10.5281/zenodo.10053564 (see `data/README.md`)
- Paper: Alvarez-Tunon et al., *SubPipe: A Submarine Pipeline Inspection
  Dataset for Segmentation and Visual-inertial Localization* (arXiv:2401.17907)
- License: **GPL-3.0**. Raw data is NOT committed (see `.gitignore`).
  For any reuse/publication include the required credit: SubPipe is a public
  dataset of a submarine outfall pipeline, property of Oceanscan-MST, acquired
  with a Light AUV within H2020 REMARO (grant No. 956200).
- Scope here: **Chunk0, `Segmentation/` only** — `<timestamp>.png` images with
  `<timestamp>_label.png` masks. Classes: single foreground class `pipeline`
  (pipe body + clamp as one class) vs background = binary segmentation.
  Pixel encoding, verified on real data: masks hold exactly {0, 1, 128} =
  background (88%) / pipe body (10.5%) / pipe boundary (1.5%, trained as
  pipe). `src/dataset.py::decode_mask` maps this explicitly and raises on
  anything unexpected.

## Method

- **Preprocessing:** resize to 256×256 (Colab T4 memory), ImageNet normalise
  (required by the pretrained encoder). Train augments mirror the paper:
  flips, small shift/scale/rotate (AUV viewpoint change) + brightness/contrast
  and hue/saturation (water, turbidity, auto-exposure). Geometry hits image
  AND mask identically; colour hits the image only (`src/dataset.py`).
- **Model:** U-Net via `segmentation-models-pytorch`, encoder `resnet34`,
  `encoder_weights=imagenet` (transfer learning), 1 output channel = one
  pipe logit per pixel, `BCEWithLogitsLoss` (`src/model.py`). One function,
  `build_model()`, so swapping to HuggingFace SegFormer later (the paper's
  model) is a config change, not a rewrite.
- **Training:** Adam (lr 3e-4, 20 epochs, batch 8), 80/20 random train/val
  split with seed 42 (`configs/config.yaml`, `src/train.py`). Note for
  defence: the paper used a *chronological* 60/20/20 split because consecutive
  video frames look alike — random splits can leak scenes and flatter scores.
- **Metrics:** IoU = overlap/union (paper's headline number, checkpoint
  criterion); Dice = 2·overlap/(pred+true), kinder to thin pipes. Pixel
  accuracy is deliberately NOT used (~90% background makes it meaningless).

## How to run (Google Colab, free T4 GPU)

```python
!pip install -r requirements.txt
```

```bash
# 1. Inspect masks FIRST — confirm encoding with your own eyes
python notebooks/inspect_masks.py --root data/Chunk0/Segmentation --n 5
# -> prints UNIQUE VALUES + pipe fraction, saves outputs/inspect_examples.png

# 2. Train (~minutes on T4 at 256x256)
python -m src.train --config configs/config.yaml
# -> checkpoints/best.pth (best val IoU) + checkpoints/last.pth

# 3. Evaluate + figures
python -m src.evaluate --checkpoint checkpoints/best.pth --num-images 6
# -> prints mean IoU/Dice, saves outputs/eval_examples.png

# 4. Prediction video (val frames in timestamp order, raw | overlay)
python -m src.make_video --checkpoint checkpoints/best.pth --max-frames 100
# -> outputs/pred_video.mp4 (green = truth, red = prediction, IoU stamped)
```

## Results

> Fill this in after YOUR run — paste the two console numbers and commit the
> figure. Do not copy anyone else's scores.

| Split (seed 42) | IoU | Dice | Notes |
|---|---|---|---|
| Val (Chunk0, 80/20 random) | _TODO_ | _TODO_ | U-Net resnet34, 256px, 20 epochs |
| Paper reference (SegFormer/DeepLabV3, SubPipeMini) | — | — | Paper reports IoU "with room for improvement"; compare qualitatively, not numerically (different split/data) |

Example overlays (`outputs/eval_examples.png`, best row on top, worst at bottom):

![eval overlays](outputs/eval_examples.png)

*Green = ground truth, red = prediction. If the figure is missing, run Stage 5
first — the path above is where it lands.*

## Limitations (read before trusting the model)

- **Sand-covered / buried sections:** the pipe vanishes under sand (the dataset
  is a real outfall survey, not a clean tank). Expect the worst-IoU rows to be
  exactly these — partial occlusion means even humans labelled from
  contrast-enhanced images. Treat low scores there as physics, not just bugs.
- **Blur, marine growth, lighting:** forward motion + turbidity + GoPro
  auto-exposure shift colours frame to frame; 256px resize also erases thin
  edges. Hue/brightness augments help but do not fix genuinely invisible pipe.
- **Chunk0 + random split only:** consecutive frames correlate, so the random
  80/20 val score is optimistic vs new surveys. The honest follow-ups are the
  paper's chronological split, then testing on Chunk1–4.
- **Single-class, single-sensor:** clamp = pipe (by annotation), no defect
  classes, RGB only (sonar untouched). Good for detection, not for condition
  assessment yet.

## Why this matters for automated subsea pipeline inspection

Today, pipeline surveys mean hours of AUV/diver video reviewed by eye — slow,
expensive, and inconsistent in murky water. A reliable pipe mask is the
foundation for automating that: once every frame knows *where the pipe is*,
you can track continuity over kilometres, flag buried or spanning sections,
measure lateral deviation, and cue human inspectors only to the hard frames
instead of all of them. This project is a minimal, honest version of that
first step on real survey data — including its failure cases.

## Repo layout

```text
SubPipe/
├── configs/config.yaml      # all hyperparameters, one place
├── data/README.md           # Chunk0 download instructions (data not in git)
├── notebooks/inspect_masks.py  # Stage 2: verify mask encoding + alignment
├── src/
│   ├── dataset.py           # pairs, split, augmentations, DataLoaders
│   ├── model.py             # U-Net + SegFormer swap path
│   ├── train.py             # loss, IoU/Dice, val loop, checkpointing
│   ├── evaluate.py          # metrics + best→worst overlay figures
│   └── utils.py             # fixed seeds
├── checkpoints/             # best.pth / last.pth (gitignored)
├── outputs/                 # inspect + eval figures (gitignored)
└── requirements.txt         # Colab-safe deps (no transformers until SegFormer)
```

## Roadmap / next steps

- [x] Stages 1–5: skeleton → dataset → U-Net → training → eval figures
- [x] Stage 6: this README (results table = your numbers)
- [ ] Verify mask encoding on YOUR download, paste unique values into results
- [ ] Chronological split + test on Chunk1 (honest generalisation check)
- [ ] SegFormer swap (`pip install transformers`, `architecture: segformer`)
- [ ] Larger input (512px) / longer schedule once the baseline is defended
