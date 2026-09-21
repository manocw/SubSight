# Finding a subsea pipeline with a neural net

I'm a first-year mechanical engineering student and I want to work in subsea
engineering. For my portfolio I taught a U-Net to find a pipeline in murky
underwater photos, using real AUV survey data. It gets IoU 0.75 on held-out
frames. This repo is the whole thing: code, trained setup, results, and the
failure cases I don't want to hide.

Raw image in, binary pipe mask out, overlays to prove it.

## What it does

1. Reads image + mask pairs from the SubPipe `Segmentation/` folder (Chunk0).
2. Trains a U-Net (ResNet-34 encoder, pretrained on ImageNet) to output one
   mask per image: pipe or background.
3. Scores it with IoU and Dice on a held-out split, seed fixed, best weights
   kept in `best.pth`.
4. Saves overlay pictures sorted best to worst, so the misses are on display
   next to the wins.

## Dataset

Docs: https://github.com/remaro-network/SubPipe-dataset

Download: https://zenodo.org/doi/10.5281/zenodo.10053564 (I used
`SubPipeMini.zip`, about 6 GB — the full dump is ~80 GB and I didn't need it)

Paper: Alvarez-Tunon et al., *SubPipe: A Submarine Pipeline Inspection
Dataset for Segmentation and Visual-inertial Localization* (arXiv:2401.17907)

The data is GPL-3.0 and stays out of git. If you reuse it, include this:
SubPipe is a public dataset of a submarine outfall pipeline, property of
Oceanscan-MST. This dataset was acquired with a Light Autonomous Underwater
Vehicle by Oceanscan-MST, within the scope of Challenge Camp 1 of the H2020
REMARO project.

I only use Chunk0's `Segmentation/` folder: timestamped photos with matching
`<timestamp>_label.png` masks, one foreground class (`pipeline`, clamp
included). The masks nearly tricked me twice. They hold exactly {0, 1, 128}:
background (88%), pipe body (10.5%), pipe edge (1.5%, trained as pipe). And
106 of the 647 masks are RGB with the annotation hiding in the red channel,
which crashed my loader until I read them per-channel. `decode_mask()` in
`src/dataset.py` handles all of it and throws on anything it doesn't
recognise, because silent misreads are how you train on garbage for a week.

## How I built it

Resize everything to 256x256 (a free Colab T4 can't swallow more) and
normalise for the pretrained encoder. Training copies get flipped, slightly
rotated and shifted (an AUV never flies straight), plus brightness and colour
jitter for the water. Geometry applies to image and mask together or the mask
drifts off the pipe. Colour touches the image only.

The model is a plain U-Net from `segmentation-models-pytorch`, ResNet-34
backbone, one output channel per pixel, BCE loss. I kept one `build_model()`
function so swapping in SegFormer (what the paper used) is a config change
later, not a rewrite.

Training: Adam, lr 3e-4, 20 epochs, batch 8, 80/20 random split on seed 42.
One thing I'd flag to anyone reading closely: the paper splits chronologically
(60/20/20) because neighbouring video frames look alike, and they're right.
My random split probably flatters the score a bit. Chronological is on the
to-do list.

I score IoU (overlap over union, the paper's headline number, picks the
checkpoint) and Dice alongside it. I skipped pixel accuracy on purpose: 90% of
every frame is background, so predicting "no pipe anywhere" scores 90% and
finds nothing.

## Running it (free Colab T4)

```python
!pip install -r requirements.txt
```

```bash
# 1. Look at the masks first — I learned this the hard way
python notebooks/inspect_masks.py --root data/Chunk0/Segmentation --n 5
# -> prints UNIQUE VALUES + pipe fraction, saves outputs/inspect_examples.png

# 2. Train (minutes on a T4 at 256px)
python -m src.train --config configs/config.yaml
# -> checkpoints/best.pth (best val IoU) + checkpoints/last.pth

# 3. Score it + pictures
python -m src.evaluate --checkpoint checkpoints/best.pth --num-images 6
# -> prints mean IoU/Dice, saves outputs/eval_examples.png

# 4. Prediction video (val frames back in timestamp order, raw next to overlay)
python -m src.make_video --checkpoint checkpoints/best.pth --max-frames 100
# -> outputs/pred_video.mp4 (green = human label, red = model, IoU stamped)
```

## Results

| Split (seed 42) | IoU | Dice | Notes |
|---|---|---|---|
| Val (Chunk0, 80/20 random, seed 42) | 0.7459 | 0.8368 | U-Net resnet34, 256px, 20 epochs (best epoch 12, 517 train / 130 val) |
| Paper reference (SegFormer/DeepLabV3, SubPipeMini) | — | — | The paper says IoU has "room for improvement". Don't compare numbers directly, different split and data. |

0.75 on murky water with a tiny baseline. I'll take it.

Overlays (`outputs/eval_examples.png`, best row first, worst last):

![eval overlays](outputs/eval_examples.png)

Green is the human label, red is my model. If the picture 404s, run step 3 above, that's where it gets made.

## Where it fails

Sand buries the pipe. This is a real outfall survey, not a test tank, and
whole stretches vanish under sand that even the annotators only found via
contrast-enhanced images. My worst val frame scores IoU 0.0000, a total miss
on a buried stretch. That's physics, not a bug, and it's staying in the
report.

Blur, growth, and the GoPro's auto-exposure swing colours frame to frame, and
256px eats thin edges. The colour jitter helps but can't rebuild an invisible
pipe.

Chunk0 with a random split is the other asterisk. Neighbouring frames
correlate, so 0.75 is optimistic against fresh surveys. Chronological split,
then Chunk1-4, is the honest next test.

Single class, RGB only. Clamp counts as pipe, there are no defect labels, and
I never touched the sonar. It finds pipe. It doesn't assess condition. Yet.

## Why bother

Someone still watches hours of AUV footage by eye: slow, pricey, and two
people disagree in bad visibility. A pipe mask per frame is the boring
foundation that automates the rest. Know where the pipe is in every frame and
you can track kilometres of it, pick out buried or spanning sections, measure
drift, and send humans only the hard frames. This is that first step, built
on real survey data, misses included.

## Repo layout

```text
SubPipe/
├── configs/config.yaml      # every hyperparameter lives here
├── data/README.md           # where Chunk0 comes from (data itself not in git)
├── notebooks/inspect_masks.py  # look at the masks before trusting them
├── src/
│   ├── dataset.py           # pairs, split, augments, loaders, mask decoding
│   ├── model.py             # U-Net, plus the SegFormer door left open
│   ├── train.py             # loss, IoU/Dice, val loop, checkpointing
│   ├── evaluate.py          # scores + best-to-worst overlay pictures
│   ├── make_video.py        # val frames as raw-next-to-overlay video
│   └── utils.py             # seeds
├── checkpoints/             # best.pth / last.pth (gitignored, too big anyway)
├── outputs/                 # pictures + video (gitignored, except the README figure)
└── requirements.txt         # Colab-safe deps
```

## Next

- [x] Stages 1-5: skeleton, dataset, U-Net, training, pictures
- [x] Stage 6: this writeup with real numbers
- [x] Mask encoding saga: {0,1,128} plus red-channel RGB variants, all handled
- [ ] Chronological split + Chunk1 test
- [ ] SegFormer swap (`pip install transformers`, `architecture: segformer`)
- [ ] 512px inputs once the baseline is defended
