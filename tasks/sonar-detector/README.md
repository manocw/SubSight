# Sonar detector (Phase 3)

Boxes on forward-scan sonar waterfall for pipe. ROV video stays the
input for optical heads, sonar gets its own head.

## Data

SubPipe MiniSSS, 669 paired HF frames with YOLO boxes, one class
(Pipeline). Chrono split by filename timestamp, seed 42 for nothing
but shuffling inside train. 356 extra box files reference frames
outside the Mini and are excluded. Licence per the Zenodo record,
same as SubPipe. Proxy note: real operator sonar differs, this is
labelled as proxy.

## Model

YOLOv8 nano at 256px first, 50 epochs, batch 16. Then 640px once the
small run is defended.

## Results

| Split | mAP50 | Notes |
|---|---|---|
| val | 0.514 | 100 frames chrono, precision 0.78 recall 0.52 |
| test (locked) | 0.859 | 101 frames chrono, precision 0.935 recall 0.708, scored once |

YOLOv8n 3M params, 256px, 50 epochs, seed 42, batch 16.
Test split runs easier than val, both reported as is.

## Next

- 640px run once nano is defended
- Cross-set drop if a second sonar set lands
