# Web/model contract (both sessions honor this, change by agreement)

## Model files (gitignored, live in web/models/)

| File | Input | Output |
|---|---|---|
| pipe.onnx | (1,3,256,256) float32, RGB, ImageNet-normalized | (1,1,256,256) float32 logits |
| hull.onnx | (1,3,256,256) float32, RGB, ImageNet-normalized | (1,10,256,256) float32 logits |

Hull channel order (fixed): ship_hull, anode, marine_growth,
paint_peel, corrosion, defect, propeller, sea_chest_grating,
over_board_valves, bilge_keel.

Produced by web/export_onnx.py on a GPU box from the .pth
checkpoints. The web app never trains and never imports torch.

## Scores file (one per uploaded video)

`web/data/scores/<video_id>.json`:

```json
{
  "fps": 2.0,
  "frames": [
    {"t": 0.0, "pipe": 0.11, "macro": 0.21,
     "classes": {"anode": 0.0, "corrosion": 0.03}},
    "... one entry per sampled frame, t in video seconds ..."
  ]
}
```

Values are predicted-positive pixel fractions per class (plus pipe
and macro). A frame is flagged when any value tops its threshold
(defaults in web/app.py, tunable per class).

## Endpoints (web/app.py)

- POST /api/videos (multipart file) -> {"id": ...}
- GET /api/jobs -> list with status (queued/working/done/error)
- GET /api/videos/{id}/scores -> scores file above
- GET / serves the viewer (upload, player, flag timeline)
