# Hull-defect head (Phase 2)

Multi-label segmentation on ship-hull stills: one channel per class,
sigmoid, BCE with rare-class weights. Growth sits on hull, so pixels
keep both tags.

## Data

LIACI semantic segmentation set, SINTEF: 1,893 frames at 1920x1080,
10 classes with one bitmap each plus merged masks, COCO labels and an
official train/test split (1,370 train / 523 val, seed 42 for shuffle
only). Licence CC BY-NC-SA 4.0: portfolio and research use fine,
credit SINTEF, commercial use forbidden. If you reuse it, cite Waszak
et al., IEEE Journal of Oceanic Engineering 2022.

Prevalence (non-empty frames): hull 88%, growth 46%, peel 45%, anode
28%, propeller 26%, grating 22%, valves 12%, corrosion 11%, keel 10%,
defect 4%. Anodes cover 1.9% of pixels when present.

## Model

U-Net ResNet-34, ImageNet start, 10 output channels, 256px, batch 8,
20 epochs, lr 3e-4. pos_weight per class from measured pixel
prevalence, capped at 50. Checkpoint on macro IoU over classes
present in val. Same backbone family as pipe-seg for a fair read.

## Results

| Class | IoU | Dice |
|---|---|---|
| ship_hull | 0.6214 | 0.7665 |
| anode | 0.0878 | 0.1615 |
| marine_growth | 0.2998 | 0.4613 |
| paint_peel | 0.0831 | 0.1535 |
| corrosion | 0.0550 | 0.1043 |
| defect | 0.0000 | 0.0000 |
| propeller | 0.4928 | 0.6603 |
| sea_chest_grating | 0.5115 | 0.6768 |
| over_board_valves | 0.2555 | 0.4069 |
| bilge_keel | 0.0838 | 0.1546 |
| macro (present) | 0.2491 | — |

Big structures score, small rare ones do not. Defect at 4% prevalence
never registers. That is the baseline to beat, not a bug report.

Overlays (`outputs/eval.png`, anode/corrosion/peel/defect in colour):

![hull eval](outputs/eval.png)

## Next

- Focal or dice-weighted loss for the tail classes
- Longer schedule once the baseline is defended
- MobileNetV2 encoder per the paper's real-time pick
